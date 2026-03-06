"""
ANE LoRA Gradient Kernels — Real MIL program generation + dispatch via ane_bridge.

Computes LoRA gradients on the Apple Neural Engine using matmul ops:
  Step 1: tmp = dy @ lora_b^T     [seq, rank]
  Step 2: d_lora_a = x^T @ tmp    [in_dim, rank]
  Step 3: ax = x @ lora_a         [seq, rank]
  Step 4: d_lora_b = ax^T @ dy    [rank, out_dim]

MIL format: program(1.3) with func main<ios18>, fp32 I/O, internal fp16 casting.
Matches proven patterns from training/ane_mil_gen.h (matmul variant).
Seq length padded to power-of-2, kernels cached per (M, K, N) shape.
"""
import ctypes
import numpy as np
from typing import Dict, List, Optional, Tuple

# Compile budget: ANE leaks ~1 handle per compile, fails silently after ~119
MAX_COMPILE_COUNT = 100

# Power-of-2 seq length variants for kernel caching
SEQ_POWERS = [8, 16, 32, 64, 128, 256, 512]

# Build info matching coremltools 9.0 output (required by ANE compiler)
BUILD_INFO = (
    '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]'
)


class ANELoRAKernels:
    """Compile and dispatch LoRA gradient matmuls on the Apple Neural Engine.

    Uses the matmul MIL op (not conv) so both operands are passed as inputs
    via IOSurfaces. This avoids recompilation when weight values change --
    only shape changes require new kernels.

    Kernel cache key: (M, K, N) = output shape dimensions.
    Seq dimension padded to power-of-2 -> ~7 variants x ~5 unique shapes = ~35 kernels.
    """

    def __init__(self, bridge_lib):
        """
        Args:
            bridge_lib: loaded ctypes.CDLL of libane_bridge.dylib
        """
        self.lib = bridge_lib
        self._kernels: Dict[Tuple[int, int, int], ctypes.c_void_p] = {}
        self._compile_count = 0
        self._setup_bridge_signatures()

    def _setup_bridge_signatures(self):
        """Set ctypes argtypes/restypes for bridge functions used by this module."""
        self.lib.ane_bridge_compile.restype = ctypes.c_void_p
        self.lib.ane_bridge_compile.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t,              # mil_text, mil_len
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,  # weight_data, weight_len
            ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),    # n_inputs, input_sizes
            ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),    # n_outputs, output_sizes
        ]
        self.lib.ane_bridge_eval.restype = ctypes.c_bool
        self.lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]
        self.lib.ane_bridge_write_input.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
        ]
        self.lib.ane_bridge_read_output.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
        ]
        self.lib.ane_bridge_free.argtypes = [ctypes.c_void_p]
        self.lib.ane_bridge_get_compile_count.restype = ctypes.c_int

    # ------------------------------------------------------------------ #
    #  MIL generation                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _gen_matmul_mil(out_ch: int, in_ch: int, spatial: int) -> str:
        """Generate MIL for matmul: W[1, out_ch, in_ch] @ x[1, in_ch, spatial].

        Output: [1, out_ch, spatial] fp32.
        fp32 inputs -> cast to fp16 -> matmul -> cast back to fp32.
        Matches the proven mil_gen_matmul() pattern from ane_mil_gen.h.
        """
        return (
            f'program(1.3)\n'
            f'{BUILD_INFO}\n'
            f'{{\n'
            f'    func main<ios18>('
            f'tensor<fp32, [1, {in_ch}, {spatial}]> x, '
            f'tensor<fp32, [1, {out_ch}, {in_ch}]> W'
            f') {{\n'
            f'        string to_fp16 = const()[name = string("to_fp16"), '
            f'val = string("fp16")];\n'
            f'        tensor<fp16, [1, {in_ch}, {spatial}]> x16 = '
            f'cast(dtype = to_fp16, x = x)[name = string("cast_x")];\n'
            f'        tensor<fp16, [1, {out_ch}, {in_ch}]> W16 = '
            f'cast(dtype = to_fp16, x = W)[name = string("cast_W")];\n'
            f'        bool tx = const()[name = string("tx"), val = bool(false)];\n'
            f'        bool ty = const()[name = string("ty"), val = bool(false)];\n'
            f'        tensor<fp16, [1, {out_ch}, {spatial}]> y16 = '
            f'matmul(transpose_x = tx, transpose_y = ty, x = W16, y = x16)'
            f'[name = string("mm")];\n'
            f'        string to_fp32 = const()[name = string("to_fp32"), '
            f'val = string("fp32")];\n'
            f'        tensor<fp32, [1, {out_ch}, {spatial}]> y = '
            f'cast(dtype = to_fp32, x = y16)[name = string("cast_out")];\n'
            f'    }} -> (y);\n'
            f'}}\n'
        )

    # ------------------------------------------------------------------ #
    #  Seq padding                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pad_seq(seq: int) -> int:
        """Round seq to next power-of-2 from SEQ_POWERS (capped at 512)."""
        for p in SEQ_POWERS:
            if seq <= p:
                return p
        return 512

    # ------------------------------------------------------------------ #
    #  Kernel compilation + cache                                          #
    # ------------------------------------------------------------------ #

    def _get_or_compile(self, M: int, K: int, N: int) -> ctypes.c_void_p:
        """Get cached kernel handle or compile a new one for matmul(M, K, N).

        The matmul computes C[M,N] = W[M,K] @ x[K,N].
        MIL signature: x=[1,K,N], W=[1,M,K] -> y=[1,M,N].
        """
        key = (M, K, N)
        if key in self._kernels:
            return self._kernels[key]

        if self._compile_count >= MAX_COMPILE_COUNT:
            raise RuntimeError(
                f"ANE compile budget exhausted ({self._compile_count}/{MAX_COMPILE_COUNT}). "
                f"Save adapter and restart daemon to reset."
            )

        mil_text = self._gen_matmul_mil(out_ch=M, in_ch=K, spatial=N)
        mil_bytes = mil_text.encode('utf-8')

        # Two fp32 inputs: x[1,K,N] and W[1,M,K]
        x_bytes = 1 * K * N * 4
        w_bytes = 1 * M * K * 4
        out_bytes = 1 * M * N * 4

        in_sizes = (ctypes.c_size_t * 2)(x_bytes, w_bytes)
        out_sizes = (ctypes.c_size_t * 1)(out_bytes)

        kernel = self.lib.ane_bridge_compile(
            mil_bytes, len(mil_bytes),
            None, 0,       # no baked weights -- both operands are inputs
            2, in_sizes,   # 2 inputs (x, W)
            1, out_sizes,  # 1 output (y)
        )

        if not kernel:
            raise RuntimeError(
                f"ANE compile failed for matmul({M},{K},{N}). "
                f"Check MIL syntax or ANE availability."
            )

        self._kernels[key] = kernel
        self._compile_count += 1
        print(f"[ANE-LORA] Compiled matmul({M},{K},{N}) "
              f"[{self._compile_count}/{MAX_COMPILE_COUNT} budget]")
        return kernel

    # ------------------------------------------------------------------ #
    #  ANE matmul dispatch                                                 #
    # ------------------------------------------------------------------ #

    def _ane_matmul(self, A: np.ndarray, B: np.ndarray,
                    M: int, K: int, N: int) -> np.ndarray:
        """Compute C[M,N] = A[M,K] @ B[K,N] on ANE.

        Both operands written as fp32 to IOSurfaces. ANE casts to fp16
        internally, computes matmul, casts output back to fp32.

        Returns fp32 numpy array of shape [M, N].
        """
        kernel = self._get_or_compile(M, K, N)

        # Prepare inputs: contiguous fp32, reshaped to [1, dim1, dim2]
        x_data = np.ascontiguousarray(B.reshape(1, K, N), dtype=np.float32)
        W_data = np.ascontiguousarray(A.reshape(1, M, K), dtype=np.float32)

        # Write input 0: x[1,K,N] (matches first param in MIL func signature)
        x_buf = x_data.tobytes()
        self.lib.ane_bridge_write_input(
            kernel, 0,
            ctypes.c_char_p(x_buf),
            ctypes.c_size_t(len(x_buf))
        )

        # Write input 1: W[1,M,K]
        w_buf = W_data.tobytes()
        self.lib.ane_bridge_write_input(
            kernel, 1,
            ctypes.c_char_p(w_buf),
            ctypes.c_size_t(len(w_buf))
        )

        # Execute on ANE
        ok = self.lib.ane_bridge_eval(kernel)
        if not ok:
            raise RuntimeError(f"ANE eval failed for matmul({M},{K},{N})")

        # Read output: y[1,M,N] fp32
        out_nbytes = 1 * M * N * 4
        out_buf = (ctypes.c_uint8 * out_nbytes)()
        self.lib.ane_bridge_read_output(
            kernel, 0,
            ctypes.cast(out_buf, ctypes.c_void_p),
            ctypes.c_size_t(out_nbytes)
        )

        return np.frombuffer(out_buf, dtype=np.float32).reshape(M, N).copy()

    # ------------------------------------------------------------------ #
    #  LoRA gradient computation (Phase 2b: full ANE gradient path)        #
    # ------------------------------------------------------------------ #

    def compute_lora_gradients(
        self,
        dy: np.ndarray,      # [seq, out_dim] upstream gradient (fp32)
        x: np.ndarray,       # [seq, in_dim] input activation (fp32)
        lora_a: np.ndarray,  # [in_dim, rank] (fp32)
        lora_b: np.ndarray,  # [rank, out_dim] (fp32)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute LoRA gradients on ANE via 4 matmul dispatches.

        Returns (d_lora_a[in_dim, rank], d_lora_b[rank, out_dim]) as fp32.
        Falls back to CPU numpy if any ANE dispatch fails.
        """
        seq = dy.shape[0]
        out_dim = dy.shape[1]
        in_dim = x.shape[1]
        rank = lora_a.shape[1]

        # Pad seq to power-of-2 for kernel cache reuse
        padded_seq = self._pad_seq(seq)

        try:
            # Zero-pad inputs along seq dimension
            if padded_seq > seq:
                dy_pad = np.zeros((padded_seq, out_dim), dtype=np.float32)
                dy_pad[:seq] = dy
                x_pad = np.zeros((padded_seq, in_dim), dtype=np.float32)
                x_pad[:seq] = x
            else:
                dy_pad = np.ascontiguousarray(dy, dtype=np.float32)
                x_pad = np.ascontiguousarray(x, dtype=np.float32)

            # Step 1: tmp = dy @ lora_b^T  ->  [padded_seq, rank]
            # A=dy_pad[padded_seq, out_dim] @ B=lora_b^T[out_dim, rank]
            bT = np.ascontiguousarray(lora_b.T, dtype=np.float32)
            tmp = self._ane_matmul(dy_pad, bT, padded_seq, out_dim, rank)

            # Step 2: d_lora_a = x^T @ tmp  ->  [in_dim, rank]
            # A=x_pad^T[in_dim, padded_seq] @ B=tmp[padded_seq, rank]
            xT = np.ascontiguousarray(x_pad.T, dtype=np.float32)
            d_lora_a = self._ane_matmul(xT, tmp, in_dim, padded_seq, rank)

            # Step 3: ax = x @ lora_a  ->  [padded_seq, rank]
            # A=x_pad[padded_seq, in_dim] @ B=lora_a[in_dim, rank]
            a_cont = np.ascontiguousarray(lora_a, dtype=np.float32)
            ax = self._ane_matmul(x_pad, a_cont, padded_seq, in_dim, rank)

            # Step 4: d_lora_b = ax^T @ dy  ->  [rank, out_dim]
            # A=ax^T[rank, padded_seq] @ B=dy_pad[padded_seq, out_dim]
            axT = np.ascontiguousarray(ax.T, dtype=np.float32)
            d_lora_b = self._ane_matmul(axT, dy_pad, rank, padded_seq, out_dim)

            return d_lora_a, d_lora_b

        except Exception as e:
            # CPU fallback -- always correct, just not on ANE
            print(f"[ANE-LORA] Dispatch failed ({e}), CPU fallback")
            tmp = dy @ lora_b.T
            d_lora_a = x.T @ tmp
            ax = x @ lora_a
            d_lora_b = ax.T @ dy
            return d_lora_a, d_lora_b

    # ------------------------------------------------------------------ #
    #  Verification matmul (Phase 2: prove ANE pipeline works)             #
    # ------------------------------------------------------------------ #

    def verify_matmul(self, A: np.ndarray, B: np.ndarray) -> float:
        """Compute A @ B on ANE, compare with numpy, return max abs error.

        Used during Phase 2 to prove the ANE dispatch pipeline is correct
        before graduating to full gradient computation.
        """
        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Shape mismatch: A[{M},{K}] @ B[{K2},{N}]"

        ane_result = self._ane_matmul(A, B, M, K, N)
        cpu_result = A @ B
        max_err = float(np.max(np.abs(ane_result - cpu_result)))
        return max_err

    # ------------------------------------------------------------------ #
    #  Pre-compilation                                                     #
    # ------------------------------------------------------------------ #

    def precompile_for_shapes(
        self,
        in_dim: int,
        out_dims: List[int],
        rank: int,
        seq_lengths: Optional[List[int]] = None,
    ) -> int:
        """Pre-compile kernels for expected LoRA gradient shapes.

        Call at daemon startup to front-load compile latency.

        For Qwen2.5-3B with rank=8, last 4 layers:
          in_dim=2048, out_dims=[2048, 256], rank=8

        Returns number of kernels compiled.
        """
        if seq_lengths is None:
            seq_lengths = SEQ_POWERS

        compiled = 0
        for seq in seq_lengths:
            for out_dim in out_dims:
                shapes = set()
                # Step 1: dy[seq, out_dim] @ B^T[out_dim, rank] -> (seq, out_dim, rank)
                shapes.add((seq, out_dim, rank))
                # Step 2: x^T[in_dim, seq] @ tmp[seq, rank] -> (in_dim, seq, rank)
                shapes.add((in_dim, seq, rank))
                # Step 3: x[seq, in_dim] @ A[in_dim, rank] -> (seq, in_dim, rank)
                shapes.add((seq, in_dim, rank))
                # Step 4: ax^T[rank, seq] @ dy[seq, out_dim] -> (rank, seq, out_dim)
                shapes.add((rank, seq, out_dim))

                for M, K, N in shapes:
                    try:
                        self._get_or_compile(M, K, N)
                        compiled += 1
                    except RuntimeError as e:
                        print(f"[ANE-LORA] Pre-compile stopped: {e}")
                        return compiled

        print(f"[ANE-LORA] Pre-compiled {compiled} kernels "
              f"(cache: {len(self._kernels)} unique shapes)")
        return compiled

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def cleanup(self):
        """Free all compiled kernels."""
        freed = 0
        for key, kernel in self._kernels.items():
            try:
                self.lib.ane_bridge_free(kernel)
                freed += 1
            except Exception:
                pass
        self._kernels.clear()
        self._compile_count = 0
        print(f"[ANE-LORA] Freed {freed} kernels")

    @property
    def compile_count(self) -> int:
        return self._compile_count

    @property
    def kernel_count(self) -> int:
        return len(self._kernels)


# ================================================================== #
#  Phase 2b: ANE-Dispatched Autograd via mx.custom_function           #
#                                                                      #
#  Intercepts dy (upstream gradient) in the backward pass and routes   #
#  all 4 LoRA gradient matmuls per module to the Neural Engine.        #
#  Falls back to MLX GPU transparently on any ANE failure.             #
# ================================================================== #

try:
    import mlx.core as mx
    import mlx.nn as nn
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

if _HAS_MLX:
    # Global state for ANE dispatch inside VJP
    _ane_kernels_global: Optional["ANELoRAKernels"] = None
    _ane_dispatch_stats = {
        "dispatches": 0,    # number of ANE matmul dispatches
        "fallbacks": 0,     # number of MLX fallbacks
        "last_error": None, # last error message
    }

    def set_ane_kernels(kernels: "ANELoRAKernels"):
        """Set the global ANE kernel reference used by custom VJP."""
        global _ane_kernels_global
        _ane_kernels_global = kernels

    def get_ane_dispatch_stats() -> dict:
        """Get dispatch statistics since last reset."""
        return dict(_ane_dispatch_stats)

    def reset_ane_dispatch_stats():
        """Reset dispatch counters (call before each training step)."""
        _ane_dispatch_stats["dispatches"] = 0
        _ane_dispatch_stats["fallbacks"] = 0
        _ane_dispatch_stats["last_error"] = None

    # -------------------------------------------------------------- #
    #  Custom forward + VJP for LoRA linear with ANE gradient dispatch #
    # -------------------------------------------------------------- #

    @mx.custom_function
    def _ane_lora_fwd(x, weight, lora_a, lora_b, scale_arr):
        """Forward: y = x @ W^T + scale * (x @ A @ B).

        Identical output to standard LoRALinear. The magic is in the VJP
        below which routes LoRA gradient computation to the Neural Engine.
        """
        y = x @ weight.T
        z = (x @ lora_a) @ lora_b
        return y + scale_arr * z

    @_ane_lora_fwd.vjp
    def _ane_lora_bwd(primals, cotangent, output):
        """Custom backward: dispatch LoRA gradient matmuls to ANE.

        Called during mx.eval() when the backward pass reaches this node.
        primals and cotangent are evaluated arrays at this point.

        Computes:
          dx        = dy @ W + scale * dy @ B^T @ A^T   (MLX — chain rule for earlier layers)
          d_lora_a  = scale * x^T @ (dy @ B^T)          (ANE — 2 matmuls)
          d_lora_b  = scale * (x @ A)^T @ dy             (ANE — 2 matmuls)
          d_weight  = zeros                               (frozen)
          d_scale   = 0                                   (not trainable)
        """
        x, weight, lora_a, lora_b, scale_arr = primals
        dy = cotangent

        # Extract scale as Python float
        scale_f = float(scale_arr.item()) if hasattr(scale_arr, 'item') else float(scale_arr)

        # dx for chain rule — always on MLX (feeds earlier layers)
        dx = dy @ weight + scale_f * (dy @ (lora_b.T @ lora_a.T))

        # LoRA gradients — try ANE, fall back to MLX
        ane_ok = False
        if _ane_kernels_global is not None:
            try:
                # Flatten batch dims: [batch, seq, dim] -> [batch*seq, dim]
                x_2d = x.reshape(-1, x.shape[-1])
                dy_2d = dy.reshape(-1, dy.shape[-1])

                # Convert to contiguous fp32 numpy (triggers eval if lazy)
                x_np = np.ascontiguousarray(np.array(x_2d), dtype=np.float32)
                dy_np = np.ascontiguousarray(np.array(dy_2d), dtype=np.float32)
                a_np = np.ascontiguousarray(np.array(lora_a), dtype=np.float32)
                b_np = np.ascontiguousarray(np.array(lora_b), dtype=np.float32)

                # 4 matmul dispatches to Neural Engine
                d_a_np, d_b_np = _ane_kernels_global.compute_lora_gradients(
                    dy_np, x_np, a_np, b_np)

                # Apply scale and convert back to MLX
                d_lora_a = mx.array(d_a_np * scale_f)
                d_lora_b = mx.array(d_b_np * scale_f)
                ane_ok = True
                _ane_dispatch_stats["dispatches"] += 4  # 4 matmuls per module
            except Exception as e:
                _ane_dispatch_stats["fallbacks"] += 1
                _ane_dispatch_stats["last_error"] = str(e)

        if not ane_ok:
            # MLX GPU fallback — mathematically identical
            x_2d = x.reshape(-1, x.shape[-1])
            dy_2d = dy.reshape(-1, dy.shape[-1])
            d_lora_a = scale_f * (x_2d.T @ (dy_2d @ lora_b.T))
            d_lora_b = scale_f * ((x_2d @ lora_a).T @ dy_2d)

        d_weight = mx.zeros_like(weight)
        d_scale = mx.array(0.0)

        return (dx, d_weight, d_lora_a, d_lora_b, d_scale)

    # -------------------------------------------------------------- #
    #  ANELoRALinear: drop-in replacement for LoRALinear               #
    # -------------------------------------------------------------- #

    class ANELoRALinear(nn.Module):
        """LoRA linear with ANE-dispatched gradient computation.

        Drop-in replacement for mlx_lm's LoRALinear. Forward pass is
        identical. Backward pass routes the 4 LoRA gradient matmuls to
        the Neural Engine via the custom VJP above.

        Falls back to MLX GPU transparently if ANE dispatch fails.
        """

        def __init__(self, input_dims: int, output_dims: int, r: int = 8,
                     scale: float = 20.0):
            super().__init__()
            self.scale = scale
            # Placeholders — normally populated via from_lora()
            self.weight = mx.zeros((output_dims, input_dims))
            self.lora_a = mx.random.normal((input_dims, r)) * (1.0 / r)
            self.lora_b = mx.zeros((r, output_dims))

        @classmethod
        def from_lora(cls, lora_linear):
            """Convert an existing LoRALinear to ANE-backed version.

            Shares parameter arrays (no copy). Preserves parameter paths
            so model.load_weights() and mx.savez() work identically.
            """
            from mlx_lm.tuner.lora import LoRALinear as _LoRALinear

            obj = cls.__new__(cls)
            nn.Module.__init__(obj)
            # Share arrays — no copy, same memory
            obj.weight = lora_linear.weight
            obj.lora_a = lora_linear.lora_a
            obj.lora_b = lora_linear.lora_b
            obj.scale = lora_linear.scale
            if hasattr(lora_linear, 'bias') and lora_linear.bias is not None:
                obj.bias = lora_linear.bias
            # Keep base weight frozen (only lora_a, lora_b trainable)
            obj.freeze(keys=["weight"])
            return obj

        def __call__(self, x):
            """Forward pass using custom function with ANE VJP."""
            y = _ane_lora_fwd(
                x, self.weight, self.lora_a, self.lora_b,
                mx.array(float(self.scale)))
            if "bias" in self:
                y = y + self.bias
            return y

    # -------------------------------------------------------------- #
    #  Helper: replace LoRA layers in model with ANE-backed versions   #
    # -------------------------------------------------------------- #

    def replace_lora_with_ane(model) -> int:
        """Replace all LoRALinear layers with ANELoRALinear.

        Walks model.model.layers[*].self_attn.{q,k,v,o}_proj and swaps
        LoRALinear instances in-place. Parameter paths are preserved.

        Returns count of layers replaced.
        """
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
