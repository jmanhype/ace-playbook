"""Test dimension workarounds for ANE conv eval constraints.

Key findings so far:
- Square-ish dims work: 8x8, 16x16, 64x64, 256x256, 768x768
- Asymmetric dims FAIL: 8x2048, 2048x8
- Very small (4x4) also fails
- spatial=8 with small channels fails

Hypothesis tests:
1. Is it spatial too small? (pad spatial)
2. Is it channel ratio too extreme? (pad channels)
3. Is there a minimum channel size?
4. What's the max ratio that works?
"""
import ctypes
import numpy as np
import sys
sys.path.insert(0, '/Users/batmanosama/ANE-backup')
from ane_lora_kernels import _gen_conv_mil, _build_weight_blob

BRIDGE_PATH = '/Users/batmanosama/ANE-backup/bridge/libane_bridge.dylib'
lib = ctypes.CDLL(BRIDGE_PATH)
lib.ane_bridge_init.restype = ctypes.c_int
lib.ane_bridge_compile.restype = ctypes.c_void_p
lib.ane_bridge_compile.argtypes = [
    ctypes.c_char_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
]
lib.ane_bridge_eval.restype = ctypes.c_bool
lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]
lib.ane_bridge_write_input.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
lib.ane_bridge_read_output.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
lib.ane_bridge_free.argtypes = [ctypes.c_void_p]

rc = lib.ane_bridge_init()
print(f"Init: {rc}\n")

def test_conv(in_ch, out_ch, spatial, name=""):
    """Test a conv kernel. Returns (compiled, eval_ok, error) or (False, False, None)."""
    W = np.random.randn(out_ch, in_ch).astype(np.float32) * 0.01
    W_4d = np.ascontiguousarray(W.reshape(out_ch, in_ch, 1, 1), dtype=np.float32)
    blob = _build_weight_blob(W_4d)
    mil = _gen_conv_mil(in_ch, out_ch, spatial)
    mil_b = mil.encode('utf-8')
    wb = (ctypes.c_uint8 * len(blob))(*blob)
    in_sz = (ctypes.c_size_t * 1)(1 * in_ch * 1 * spatial * 4)
    out_sz = (ctypes.c_size_t * 1)(1 * out_ch * 1 * spatial * 4)
    k = lib.ane_bridge_compile(mil_b, len(mil_b), wb, len(blob), 1, in_sz, 1, out_sz)
    if not k:
        return False, False, None
    x_in = np.random.randn(1, in_ch, 1, spatial).astype(np.float32)
    lib.ane_bridge_write_input(k, 0, x_in.ctypes.data, ctypes.c_size_t(x_in.nbytes))
    ok = lib.ane_bridge_eval(k)
    err = None
    if ok:
        out = np.zeros((1, out_ch, 1, spatial), dtype=np.float32)
        lib.ane_bridge_read_output(k, 0, out.ctypes.data, ctypes.c_size_t(out.nbytes))
        expected = W @ x_in.reshape(in_ch, spatial)
        err = np.max(np.abs(out.reshape(out_ch, spatial) - expected))
    lib.ane_bridge_free(k)
    return True, ok, err

# ============================================================
# Test 1: Is spatial=8 the problem?
# ============================================================
print("=" * 60)
print("TEST 1: Spatial dimension sweep (with 8x8 channels)")
print("=" * 60)
for sp in [4, 8, 12, 16, 24, 32, 64]:
    compiled, ok, err = test_conv(8, 8, sp, f"8x8 sp={sp}")
    status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
    print(f"  sp={sp:4d}  8->8   : {status}")

# ============================================================
# Test 2: Asymmetric channels with larger spatial
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 2: Asymmetric channels (8<->2048) with varying spatial")
print("=" * 60)
for sp in [16, 32, 64, 128, 256]:
    compiled, ok, err = test_conv(8, 2048, sp, f"8->2048 sp={sp}")
    status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
    print(f"  sp={sp:4d}  8->2048: {status}")

    compiled, ok, err = test_conv(2048, 8, sp, f"2048->8 sp={sp}")
    status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
    print(f"  sp={sp:4d}  2048->8: {status}")

# ============================================================
# Test 3: What channel ratio breaks things?
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 3: Channel ratio sweep (spatial=32)")
print("=" * 60)
for ratio in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    compiled, ok, err = test_conv(8, 8*ratio, 32)
    status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
    print(f"  8->{8*ratio:5d} (ratio {ratio:4d}x): {status}")

# ============================================================
# Test 4: Pad out_ch to make it less asymmetric
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 4: Padded asymmetric - pad small dim to 16/32/64")
print("=" * 60)
for pad_to in [16, 32, 64, 128, 256]:
    # Simulate LoRA: we need 8->2048, pad 8 to pad_to
    compiled, ok, err = test_conv(pad_to, 2048, 32)
    status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
    print(f"  {pad_to:4d}->2048 sp=32: {status}")

    compiled, ok, err = test_conv(2048, pad_to, 32)
    status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
    print(f"  2048->{pad_to:4d} sp=32: {status}")

# ============================================================
# Test 5: Actual LoRA shapes with padding
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 5: Actual LoRA gradient shapes (rank=8, dim=2048, seq=32)")
print("=" * 60)
seq = 32
rank = 8
dim = 2048

# Step 1: dy @ B^T -> conv(input=[1,dim,1,seq], W=[rank,dim,1,1])
compiled, ok, err = test_conv(dim, rank, seq, "step1: dy@B^T")
status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
print(f"  Step 1 (dy@B^T):   in={dim}, out={rank}, sp={seq} : {status}")

# Padded version: pad rank to 64
pad_rank = 64
compiled, ok, err = test_conv(dim, pad_rank, seq, "step1 padded")
status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
print(f"  Step 1 PADDED:     in={dim}, out={pad_rank}, sp={seq} : {status}")

# Step 2: x^T @ tmp -> conv(input=[1,seq,1,rank], W=[dim,seq,1,1])
compiled, ok, err = test_conv(seq, dim, rank, "step2: x^T@tmp")
status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
print(f"  Step 2 (x^T@tmp):  in={seq}, out={dim}, sp={rank} : {status}")

# Padded version: pad rank(spatial) to 32
compiled, ok, err = test_conv(seq, dim, 32, "step2 padded sp")
status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
print(f"  Step 2 PAD sp=32:  in={seq}, out={dim}, sp=32 : {status}")

# Step 3: x @ A -> conv(input=[1,dim,1,seq], W=[rank,dim,1,1])
compiled, ok, err = test_conv(dim, rank, seq, "step3: x@A")
status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
print(f"  Step 3 (x@A):      in={dim}, out={rank}, sp={seq} : {status}")

# Padded version
compiled, ok, err = test_conv(dim, pad_rank, seq, "step3 padded")
status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
print(f"  Step 3 PADDED:     in={dim}, out={pad_rank}, sp={seq} : {status}")

# Step 4: ax^T @ dy -> conv(input=[1,seq,1,dim], W=[rank,seq,1,1])
compiled, ok, err = test_conv(seq, rank, dim, "step4: ax^T@dy")
status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
print(f"  Step 4 (ax^T@dy):  in={seq}, out={rank}, sp={dim} : {status}")

# Padded version
compiled, ok, err = test_conv(seq, pad_rank, dim, "step4 padded")
status = f"PASS err={err:.6f}" if ok else ("EVAL_FAIL" if compiled else "COMPILE_FAIL")
print(f"  Step 4 PADDED:     in={seq}, out={pad_rank}, sp={dim} : {status}")

# ============================================================
# Test 6: Minimum viable dimensions
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 6: Minimum viable dimensions")
print("=" * 60)
for in_ch in [4, 8, 16]:
    for out_ch in [4, 8, 16]:
        for sp in [4, 8, 16, 32]:
            compiled, ok, err = test_conv(in_ch, out_ch, sp)
            if not ok:
                status = "EVAL_FAIL" if compiled else "COMPILE_FAIL"
                print(f"  FAIL: in={in_ch:3d} out={out_ch:3d} sp={sp:3d} : {status}")
            # Only print passes for small dims
            elif in_ch <= 8 and out_ch <= 8 and sp <= 16:
                print(f"  PASS: in={in_ch:3d} out={out_ch:3d} sp={sp:3d} : err={err:.6f}")

print("\nDone.")
