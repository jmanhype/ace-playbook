"""
ANE LoRA Gradient Kernels — Conv-based MIL dispatch via ane_bridge.

Computes LoRA gradients on the Apple Neural Engine using conv ops
(the only MIL op the ANE private compiler actually supports for compute).

Conv requires weights baked at compile time via BLOBFILE. For LoRA gradients
where weights change each step, we compile fresh kernels per step and free
them immediately. Compile budget (~119 per process) is managed by running
gradient dispatch in a subprocess (multiprocessing spawn) to get a fresh
bridge per step.

The 4 LoRA gradient matmuls expressed as conv:
  Conv computes: Y[out_ch, spatial] = W[out_ch, in_ch] @ X[in_ch, spatial]

  Step 1: tmp = dy @ B^T => (B @ dy^T)^T => conv(W=B[r,out], X=dy^T[out,seq])^T
  Step 2: d_A = x^T @ tmp               => conv(W=x^T[in,seq], X=tmp[seq,r])
  Step 3: ax  = x @ A  => (A^T @ x^T)^T => conv(W=A^T[r,in], X=x^T[in,seq])^T
  Step 4: d_B = ax^T @ dy               => conv(W=ax^T[r,seq], X=dy[seq,out])

ANE spatial constraints:
  - spatial must be >= 16 and a multiple of 16
  - Handled automatically by _pad_spatial() in _conv_matmul()
  - Only Step 2 needs this (spatial=rank=8 -> padded to 16)

MIL format: program(1.3), func main<ios18>, fp32 I/O, internal fp16 casting.
Matches proven patterns from training/ane_mil_gen.h (conv variant).
"""
import ctypes
import struct
import sys
import numpy as np
import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Tuple

# Build info matching coremltools 9.0 output (required by ANE compiler)
BUILD_INFO = (
    '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]'
)

# ANE spatial dimension constraint:
#   - Must be >= 16
#   - Must be a multiple of 16
# This applies to the spatial (last) dimension of conv inputs.
ANE_SPATIAL_ALIGN = 16
ANE_SPATIAL_MIN = 16


def _pad_spatial(spatial: int) -> int:
    """Round spatial to next valid ANE dimension (multiple of 16, min 16)."""
    if spatial < ANE_SPATIAL_MIN:
        return ANE_SPATIAL_MIN
    # Round up to next multiple of 16
    return ((spatial + ANE_SPATIAL_ALIGN - 1) // ANE_SPATIAL_ALIGN) * ANE_SPATIAL_ALIGN


def _build_weight_blob(weights_f32: np.ndarray) -> bytes:
    """Build ANE weight blob: 128-byte header + fp16 data.

    Global header (64 bytes):
      [0:4]   version = 1
      [4:8]   type = 2 (fp16)
    Chunk header (64 bytes at offset 64):
      [0:4]   magic = 0xDEADBEEF (LE)
      [4:8]   chunk_count = 1
      [8:12]  data_size
      [16:20] data_offset = 128
    """
    fp16_data = weights_f32.astype(np.float16).tobytes()
    wsize = len(fp16_data)
    header = bytearray(128)
    # Global header
    header[0:4] = b'\x01\x00\x00\x00'  # version
    header[4:8] = b'\x02\x00\x00\x00'  # type fp16
    # Chunk header
    header[64:68] = b'\xEF\xBE\xAD\xDE'  # DEADBEEF magic (LE)
    header[68:72] = b'\x01\x00\x00\x00'  # chunk count
    struct.pack_into('<I', header, 72, wsize)   # data_size
    struct.pack_into('<I', header, 80, 128)     # data_offset
    return bytes(header) + fp16_data


def _gen_conv_mil(in_ch: int, out_ch: int, spatial: int) -> str:
    """Generate MIL for conv: W[out_ch, in_ch, 1, 1] * x[1, in_ch, 1, spatial].

    Output: [1, out_ch, 1, spatial] fp32.
    This is equivalent to matmul A[out_ch, in_ch] @ B[in_ch, spatial].
    """
    return f"""program(1.3)
{BUILD_INFO}
{{
    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {spatial}]> x) {{
        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        tensor<fp16, [1, {in_ch}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];
        tensor<fp16, [{out_ch}, {in_ch}, 1, 1]> W = const()[name = string("W"), val = tensor<fp16, [{out_ch}, {in_ch}, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
        tensor<fp16, [1, {out_ch}, 1, {spatial}]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string("conv")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
        tensor<fp32, [1, {out_ch}, 1, {spatial}]> y = cast(dtype = to_fp32, x = y16)[name = string("cast_out")];
    }} -> (y);
}}
"""


def _conv_matmul(lib, W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute y = W @ x using ANE conv.

    W: [out_ch, in_ch] weight matrix (baked into conv kernel)
    x: [in_ch, spatial] input data (passed via IOSurface)
    Returns: [out_ch, spatial]

    Handles ANE spatial alignment automatically:
    - Pads spatial to next multiple of 16 (min 16) if needed
    - Truncates output back to original spatial dimension
    """
    out_ch, in_ch = W.shape
    orig_spatial = x.shape[1] if x.ndim == 2 else x.shape[0]
    if x.ndim == 1:
        orig_spatial = 1
        x = x.reshape(in_ch, 1)

    # Pad spatial for ANE alignment
    spatial = _pad_spatial(orig_spatial)
    if spatial > orig_spatial:
        x_padded = np.zeros((in_ch, spatial), dtype=np.float32)
        x_padded[:, :orig_spatial] = x
        x = x_padded

    # Build weight blob
    W_4d = np.ascontiguousarray(W.reshape(out_ch, in_ch, 1, 1), dtype=np.float32)
    weight_blob = _build_weight_blob(W_4d)

    # Generate and compile MIL
    mil_text = _gen_conv_mil(in_ch, out_ch, spatial)
    mil_bytes = mil_text.encode('utf-8')

    wb = (ctypes.c_uint8 * len(weight_blob))(*weight_blob)
    in_sz = (ctypes.c_size_t * 1)(1 * in_ch * 1 * spatial * 4)
    out_sz = (ctypes.c_size_t * 1)(1 * out_ch * 1 * spatial * 4)

    kernel = lib.ane_bridge_compile(
        mil_bytes, len(mil_bytes),
        wb, len(weight_blob),
        1, in_sz, 1, out_sz
    )
    if not kernel:
        raise RuntimeError(f"ANE conv compile failed: W[{out_ch},{in_ch}] x[{in_ch},{spatial}]")

    # Write input: x reshaped to [1, in_ch, 1, spatial]
    x_4d = np.ascontiguousarray(x.reshape(1, in_ch, 1, spatial), dtype=np.float32)
    x_buf = x_4d.tobytes()
    lib.ane_bridge_write_input(kernel, 0, ctypes.c_char_p(x_buf), ctypes.c_size_t(len(x_buf)))

    # Eval
    ok = lib.ane_bridge_eval(kernel)
    if not ok:
        lib.ane_bridge_free(kernel)
        raise RuntimeError(f"ANE eval failed: W[{out_ch},{in_ch}] x[{in_ch},{spatial}]")

    # Read output
    out_4d = np.zeros((1, out_ch, 1, spatial), dtype=np.float32)
    lib.ane_bridge_read_output(kernel, 0, out_4d.ctypes.data, ctypes.c_size_t(out_4d.nbytes))

    # Free kernel (we recompile each time since weights change)
    lib.ane_bridge_free(kernel)

    # Truncate to original spatial dimension
    result = out_4d.reshape(out_ch, spatial)
    if spatial > orig_spatial:
        result = result[:, :orig_spatial]
    return result


# ------------------------------------------------------------------ #
#  Subprocess worker for ANE gradient computation                     #
# ------------------------------------------------------------------ #

def _ane_gradient_worker(bridge_path: str, data_dir: str, result_dir: str):
    """Subprocess entry point: compile conv kernels, compute LoRA gradients.

    Runs in a fresh process with fresh ANE compile budget.
    Reads input arrays from data_dir, writes results to result_dir.
    """
    import ctypes as ct

    lib = ct.CDLL(bridge_path)
    lib.ane_bridge_init.restype = ct.c_int
    lib.ane_bridge_compile.restype = ct.c_void_p
    lib.ane_bridge_compile.argtypes = [
        ct.c_char_p, ct.c_size_t,
        ct.POINTER(ct.c_uint8), ct.c_size_t,
        ct.c_int, ct.POINTER(ct.c_size_t),
        ct.c_int, ct.POINTER(ct.c_size_t),
    ]
    lib.ane_bridge_eval.restype = ct.c_bool
    lib.ane_bridge_eval.argtypes = [ct.c_void_p]
    lib.ane_bridge_write_input.argtypes = [
        ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_size_t]
    lib.ane_bridge_read_output.argtypes = [
        ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_size_t]
    lib.ane_bridge_free.argtypes = [ct.c_void_p]

    rc = lib.ane_bridge_init()
    if rc != 0:
        # Write error marker
        np.save(os.path.join(result_dir, "error.npy"), np.array([rc]))
        return

    # Read module data
    n_modules = int(np.load(os.path.join(data_dir, "n_modules.npy")))
    compiles = 0
    dispatches = 0

    for m in range(n_modules):
        prefix = f"m{m}_"
        try:
            dy = np.load(os.path.join(data_dir, f"{prefix}dy.npy"))
            x = np.load(os.path.join(data_dir, f"{prefix}x.npy"))
            lora_a = np.load(os.path.join(data_dir, f"{prefix}a.npy"))
            lora_b = np.load(os.path.join(data_dir, f"{prefix}b.npy"))
        except FileNotFoundError:
            continue

        seq = dy.shape[0]
        out_dim = dy.shape[1]
        in_dim = x.shape[1]
        rank = lora_a.shape[1]

        # Pad seq to next multiple of 16 (ANE spatial alignment)
        padded_seq = _pad_spatial(seq)
        if padded_seq > seq:
            dy_pad = np.zeros((padded_seq, out_dim), dtype=np.float32)
            dy_pad[:seq] = dy
            dy = dy_pad
            x_pad = np.zeros((padded_seq, in_dim), dtype=np.float32)
            x_pad[:seq] = x
            x = x_pad

        try:
            # Step 1: tmp = dy @ B^T -> [seq, rank]
            # (dy @ B^T)^T = B @ dy^T, so conv(W=B[rank,out], X=dy^T[out,seq])
            tmp = _conv_matmul(lib, lora_b, dy.T).T  # [rank,seq]^T = [seq,rank]
            compiles += 1
            dispatches += 1

            # Step 2: d_A = x^T @ tmp -> [in_dim, rank]
            # conv(W=x^T[in,padded_seq], X=tmp[padded_seq,rank]) -> [in,rank]
            # Note: spatial=rank=8 will be auto-padded to 16 by _conv_matmul
            d_lora_a = _conv_matmul(lib, x.T, tmp)  # x.T[in,padded_seq] @ tmp[padded_seq,rank]
            compiles += 1
            dispatches += 1

            # Step 3: ax = x @ A -> [seq, rank]
            # (x @ A)^T = A^T @ x^T, so conv(W=A^T[rank,in], X=x^T[in,padded_seq])
            ax = _conv_matmul(lib, lora_a.T, x.T).T  # [rank,padded_seq]^T = [padded_seq,rank]
            compiles += 1
            dispatches += 1

            # Step 4: d_B = ax^T @ dy -> [rank, out_dim]
            # conv(W=ax^T[rank,padded_seq], X=dy[padded_seq,out]) -> [rank,out]
            d_lora_b = _conv_matmul(lib, ax.T, dy)  # ax^T[rank,padded_seq] @ dy[padded_seq,out]
            compiles += 1
            dispatches += 1

            # Trim padding from results
            d_lora_a_final = d_lora_a[:in_dim, :rank]
            d_lora_b_final = d_lora_b[:rank, :out_dim]

            np.save(os.path.join(result_dir, f"{prefix}d_a.npy"), d_lora_a_final)
            np.save(os.path.join(result_dir, f"{prefix}d_b.npy"), d_lora_b_final)

        except RuntimeError as e:
            # Write error for this module, continue with others
            np.save(os.path.join(result_dir, f"{prefix}error.npy"),
                    np.array([ord(c) for c in str(e)[:100]], dtype=np.uint8))

    # Write stats
    np.save(os.path.join(result_dir, "stats.npy"),
            np.array([compiles, dispatches], dtype=np.int32))


class ANELoRAKernels:
    """Dispatch LoRA gradient computation to ANE via conv kernels.

    Uses subprocess isolation for compile budget management.
    Each training step spawns a fresh process that:
    1. Inits ANE bridge
    2. For each LoRA module: compiles 4 conv kernels with current weights
    3. Dispatches gradient matmuls via conv
    4. Returns gradient arrays via temp files
    """

    def __init__(self, bridge_path: str):
        self.bridge_path = bridge_path
        self._total_dispatches = 0
        self._total_compiles = 0
        self._total_steps = 0
        self._lib = None

    def _ensure_lib(self):
        """Lazy init for verification only (main process bridge)."""
        if self._lib is None:
            self._lib = ctypes.CDLL(self.bridge_path)
            self._lib.ane_bridge_init.restype = ctypes.c_int
            self._lib.ane_bridge_compile.restype = ctypes.c_void_p
            self._lib.ane_bridge_compile.argtypes = [
                ctypes.c_char_p, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
            ]
            self._lib.ane_bridge_eval.restype = ctypes.c_bool
            self._lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]
            self._lib.ane_bridge_write_input.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
            self._lib.ane_bridge_read_output.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
            self._lib.ane_bridge_free.argtypes = [ctypes.c_void_p]
            self._lib.ane_bridge_get_compile_count.restype = ctypes.c_int
            rc = self._lib.ane_bridge_init()
            if rc != 0:
                raise RuntimeError(f"ANE bridge init failed: {rc}")

    def verify_conv(self) -> float:
        """Verify ANE conv works. Returns max error vs numpy.

        Uses 16x16 with spatial=32 which is within ANE constraints.
        """
        self._ensure_lib()
        W = np.random.randn(16, 16).astype(np.float32) * 0.1
        x = np.random.randn(16, 32).astype(np.float32)
        try:
            result = _conv_matmul(self._lib, W, x)
            expected = W @ x
            return float(np.max(np.abs(result - expected)))
        except RuntimeError:
            return float('inf')

    @property
    def compile_count(self) -> int:
        if self._lib:
            return self._lib.ane_bridge_get_compile_count()
        return 0

    @property
    def kernel_count(self) -> int:
        return 0  # kernels are compiled/freed per step in subprocess

    @property
    def total_dispatches(self) -> int:
        return self._total_dispatches

    @property
    def total_compiles(self) -> int:
        return self._total_compiles

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def compute_lora_gradients(
        self,
        modules: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Compute LoRA gradients for all modules on ANE.

        Args:
            modules: List of (dy, x, lora_a, lora_b) tuples per LoRA module.
                dy: [seq, out_dim] upstream gradient
                x:  [seq, in_dim]  layer input activation
                lora_a: [in_dim, rank]
                lora_b: [rank, out_dim]

        Returns:
            List of (d_lora_a, d_lora_b) gradient tuples.
            Falls back to numpy computation on ANE failure.
        """
        data_dir = tempfile.mkdtemp(prefix="ane_lora_data_")
        result_dir = tempfile.mkdtemp(prefix="ane_lora_result_")

        try:
            # Write module data
            np.save(os.path.join(data_dir, "n_modules.npy"), len(modules))
            for i, (dy, x, lora_a, lora_b) in enumerate(modules):
                prefix = f"m{i}_"
                np.save(os.path.join(data_dir, f"{prefix}dy.npy"),
                        np.ascontiguousarray(dy, dtype=np.float32))
                np.save(os.path.join(data_dir, f"{prefix}x.npy"),
                        np.ascontiguousarray(x, dtype=np.float32))
                np.save(os.path.join(data_dir, f"{prefix}a.npy"),
                        np.ascontiguousarray(lora_a, dtype=np.float32))
                np.save(os.path.join(data_dir, f"{prefix}b.npy"),
                        np.ascontiguousarray(lora_b, dtype=np.float32))

            # Build subprocess script inline — runs _ane_gradient_worker
            # in a fresh process with fresh ANE compile budget
            module_dir = os.path.dirname(os.path.abspath(__file__))
            script = f"""
import sys
sys.path.insert(0, {module_dir!r})
from ane_lora_kernels import _ane_gradient_worker
_ane_gradient_worker({self.bridge_path!r}, {data_dir!r}, {result_dir!r})
"""
            proc = subprocess.run(
                [sys.executable, "-c", script],
                timeout=30,
                capture_output=True,
                text=True
            )

            if proc.returncode != 0:
                stderr = proc.stderr[:200] if proc.stderr else "no stderr"
                print(f"[ANE-LORA] Subprocess failed (exit={proc.returncode}): {stderr}")
                return self._numpy_fallback(modules)

            # Check for global error
            error_path = os.path.join(result_dir, "error.npy")
            if os.path.exists(error_path):
                print(f"[ANE-LORA] Bridge init failed in subprocess, falling back to numpy")
                return self._numpy_fallback(modules)

            # Read stats
            stats_path = os.path.join(result_dir, "stats.npy")
            if os.path.exists(stats_path):
                stats = np.load(stats_path)
                self._total_compiles += int(stats[0])
                self._total_dispatches += int(stats[1])

            # Read results
            results = []
            for i in range(len(modules)):
                prefix = f"m{i}_"
                da_path = os.path.join(result_dir, f"{prefix}d_a.npy")
                db_path = os.path.join(result_dir, f"{prefix}d_b.npy")
                err_path = os.path.join(result_dir, f"{prefix}error.npy")

                if os.path.exists(err_path):
                    # This module failed, use numpy fallback
                    dy, x, a, b = modules[i]
                    d_a, d_b = self._numpy_grad_single(dy, x, a, b)
                    results.append((d_a, d_b))
                elif os.path.exists(da_path) and os.path.exists(db_path):
                    d_a = np.load(da_path)
                    d_b = np.load(db_path)
                    results.append((d_a, d_b))
                else:
                    dy, x, a, b = modules[i]
                    d_a, d_b = self._numpy_grad_single(dy, x, a, b)
                    results.append((d_a, d_b))

            self._total_steps += 1
            return results

        finally:
            # Cleanup temp dirs
            import shutil
            shutil.rmtree(data_dir, ignore_errors=True)
            shutil.rmtree(result_dir, ignore_errors=True)

    @staticmethod
    def _numpy_grad_single(dy, x, lora_a, lora_b):
        """Compute LoRA gradients using numpy (fallback)."""
        tmp = dy @ lora_b.T        # [seq, rank]
        d_a = x.T @ tmp            # [in, rank]
        ax = x @ lora_a            # [seq, rank]
        d_b = ax.T @ dy            # [rank, out]
        return d_a, d_b

    @staticmethod
    def _numpy_fallback(modules):
        """Compute all module gradients using numpy."""
        results = []
        for dy, x, a, b in modules:
            d_a, d_b = ANELoRAKernels._numpy_grad_single(dy, x, a, b)
            results.append((d_a, d_b))
        return results


# ------------------------------------------------------------------ #
#  Global kernel instance + mx.custom_function VJP                    #
# ------------------------------------------------------------------ #

_ane_kernels_global: Optional[ANELoRAKernels] = None
_ane_dispatch_stats = {"ane_dispatches": 0, "fallback_dispatches": 0}


def set_ane_kernels(kernels: ANELoRAKernels):
    """Set the global ANE kernel instance for use in custom VJP."""
    global _ane_kernels_global
    _ane_kernels_global = kernels


def get_dispatch_stats() -> dict:
    """Get ANE dispatch statistics."""
    stats = dict(_ane_dispatch_stats)
    if _ane_kernels_global:
        stats["total_ane_compiles"] = _ane_kernels_global.total_compiles
        stats["total_ane_dispatches"] = _ane_kernels_global.total_dispatches
        stats["total_ane_steps"] = _ane_kernels_global.total_steps
    return stats


# Try to import MLX for the custom VJP layer
try:
    import mlx.core as mx
    import mlx.nn as nn

    # -------------------------------------------------------------- #
    #  Custom forward function with ANE VJP for LoRA gradients        #
    # -------------------------------------------------------------- #

    @mx.custom_function
    def _ane_lora_part(x, lora_a, lora_b, scale_arr):
        """LoRA component: scale * (x @ A) @ B.

        Separated from the base linear layer so the custom VJP
        only intercepts LoRA gradients (base layer is frozen).
        All computation in float32, output cast to x.dtype.
        """
        # Promote to float32 for stable computation
        x_f32 = x.astype(mx.float32)
        z = (x_f32 @ lora_a) @ lora_b
        return (scale_arr * z).astype(x.dtype)

    @_ane_lora_part.vjp
    def _ane_lora_part_vjp(primals, cotangent, output):
        """VJP: Route LoRA gradients to ANE via conv kernels.

        dx stays on MLX GPU (needed for chain rule through earlier layers).
        d_lora_a and d_lora_b dispatched to ANE via subprocess conv kernels.
        Falls back to MLX GPU if ANE dispatch fails.
        """
        x, lora_a, lora_b, scale_arr = primals
        dy = cotangent.astype(mx.float32)  # promote for stable grads
        scale_f = float(scale_arr.item())

        # dx on MLX (GPU) — chain rule for earlier layers
        x_f32 = x.astype(mx.float32)
        dx = (scale_f * ((dy @ lora_b.T) @ lora_a.T)).astype(x.dtype)

        # LoRA gradients: try ANE, fallback to MLX
        if _ane_kernels_global is not None:
            try:
                # Flatten batch dims
                x_2d = x_f32.reshape(-1, x_f32.shape[-1])
                dy_2d = dy.reshape(-1, dy.shape[-1])

                # Force evaluation and convert to numpy
                mx.eval(x_2d, dy_2d, lora_a, lora_b)
                x_np = np.array(x_2d, copy=False).astype(np.float32)
                dy_np = np.array(dy_2d, copy=False).astype(np.float32)
                a_np = np.array(lora_a, copy=False).astype(np.float32)
                b_np = np.array(lora_b, copy=False).astype(np.float32)

                # Single module dispatch to ANE
                gradients = _ane_kernels_global.compute_lora_gradients(
                    [(dy_np, x_np, a_np, b_np)]
                )
                d_a_np, d_b_np = gradients[0]

                d_lora_a = mx.array(d_a_np * scale_f).astype(lora_a.dtype)
                d_lora_b = mx.array(d_b_np * scale_f).astype(lora_b.dtype)
                _ane_dispatch_stats["ane_dispatches"] += 1
            except Exception as e:
                print(f"[ANE-LORA] VJP fallback to MLX: {e}")
                tmp = dy @ lora_b.T
                d_lora_a = (scale_f * (x_f32.T @ tmp)).reshape(lora_a.shape).astype(lora_a.dtype)
                ax = x_f32 @ lora_a
                d_lora_b = (scale_f * (ax.T @ dy)).reshape(lora_b.shape).astype(lora_b.dtype)
                _ane_dispatch_stats["fallback_dispatches"] += 1
        else:
            # No ANE available
            tmp = dy @ lora_b.T
            d_lora_a = (scale_f * (x_f32.T @ tmp)).reshape(lora_a.shape).astype(lora_a.dtype)
            ax = x_f32 @ lora_a
            d_lora_b = (scale_f * (ax.T @ dy)).reshape(lora_b.shape).astype(lora_b.dtype)
            _ane_dispatch_stats["fallback_dispatches"] += 1

        d_scale = mx.zeros_like(scale_arr)
        return (dx, d_lora_a, d_lora_b, d_scale)

    # -------------------------------------------------------------- #
    #  ANELoRALinear: drop-in replacement for LoRALinear               #
    # -------------------------------------------------------------- #

    class ANELoRALinear(nn.Module):
        """LoRA linear with ANE-dispatched gradient computation.

        Drop-in replacement for mlx_lm's LoRALinear. Forward pass is
        identical. Backward pass routes LoRA gradient matmuls to ANE
        via conv kernels in a subprocess.

        Compatible with both nn.Linear and nn.QuantizedLinear base layers.
        """

        def __init__(self, input_dims: int, output_dims: int, r: int = 8,
                     scale: float = 20.0):
            super().__init__()
            self.scale = scale
            self.linear = nn.Linear(input_dims, output_dims)
            self.dropout = nn.Dropout(p=0.0)
            self.lora_a = mx.random.normal((input_dims, r)) * (1.0 / r)
            self.lora_b = mx.zeros((r, output_dims))

        @classmethod
        def from_lora(cls, lora_linear):
            """Convert an existing LoRALinear to ANE-backed version.

            Shares all parameters (no copy), just changes the class
            so __call__ uses the ANE-dispatched VJP.
            """
            obj = cls.__new__(cls)
            nn.Module.__init__(obj)
            obj.linear = lora_linear.linear  # base layer (may be QuantizedLinear)
            obj.lora_a = lora_linear.lora_a
            obj.lora_b = lora_linear.lora_b
            obj.scale = lora_linear.scale
            obj.dropout = lora_linear.dropout
            obj.freeze(keys=["linear"])  # freeze the base layer
            return obj

        def __call__(self, x):
            """Forward: base_linear(x) + scale * (x @ A) @ B.

            The LoRA part uses a custom function whose VJP dispatches
            LoRA gradient computation to ANE.
            """
            base_out = self.linear(x)
            lora_out = _ane_lora_part(
                x, self.lora_a, self.lora_b,
                mx.array(float(self.scale)))
            return (base_out + lora_out).astype(x.dtype)

    # -------------------------------------------------------------- #
    #  Helper: replace LoRA layers in model with ANE-backed versions   #
    # -------------------------------------------------------------- #

    def replace_lora_with_ane(model) -> int:
        """Replace all LoRALinear layers with ANELoRALinear."""
        from mlx_lm.tuner.lora import LoRALinear as _LoRALinear

        replaced = 0
        if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
            print("[ANE-LORA] Model structure not recognized, skipping replacement")
            return 0

        for i, layer in enumerate(model.model.layers):
            if not hasattr(layer, 'self_attn'):
                continue
            attn = layer.self_attn
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                proj = getattr(attn, proj_name, None)
                if isinstance(proj, _LoRALinear) and not isinstance(proj, ANELoRALinear):
                    ane_proj = ANELoRALinear.from_lora(proj)
                    setattr(attn, proj_name, ane_proj)
                    replaced += 1

        return replaced

except ImportError:
    # MLX not available (e.g., running in subprocess without mlx)
    ANELoRALinear = None
    replace_lora_with_ane = None
    _ane_lora_fwd = None
