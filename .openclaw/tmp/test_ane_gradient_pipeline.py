"""Test full ANE LoRA gradient pipeline with spatial alignment.

Tests:
1. verify_conv — basic sanity
2. Single LoRA module gradient (rank=8, dim=2048) via direct lib (no subprocess)
3. Full compute_lora_gradients via subprocess
"""
import sys
import numpy as np
sys.path.insert(0, '/Users/batmanosama/ANE-backup')

BRIDGE_PATH = '/Users/batmanosama/ANE-backup/bridge/libane_bridge.dylib'

from ane_lora_kernels import (
    ANELoRAKernels, _conv_matmul, _pad_spatial, _build_weight_blob, _gen_conv_mil
)
import ctypes

print("=" * 60)
print("ANE LoRA Gradient Pipeline Test")
print("=" * 60)

# ============================================================
# Test 0: Verify spatial padding logic
# ============================================================
print("\nTest 0: Spatial padding")
for sp in [1, 4, 8, 12, 15, 16, 17, 24, 31, 32, 48, 64, 100, 256, 512]:
    padded = _pad_spatial(sp)
    print(f"  {sp:4d} -> {padded:4d} (align={padded % 16 == 0}, >= 16={padded >= 16})")

# ============================================================
# Test 1: verify_conv
# ============================================================
print("\nTest 1: verify_conv")
kernels = ANELoRAKernels(BRIDGE_PATH)
err = kernels.verify_conv()
print(f"  Max error: {err:.8f}")
print(f"  Status: {'PASS' if err < 0.01 else 'FAIL'}")

# ============================================================
# Test 2: Direct _conv_matmul with LoRA-relevant shapes
# ============================================================
print("\nTest 2: Direct _conv_matmul with spatial alignment")

# Init bridge for direct testing
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
print(f"  Bridge init: {rc}")

np.random.seed(42)

# These are the actual shapes from LoRA gradient steps with spatial padding:
test_cases = [
    # (W_shape, x_cols, description)
    # Step 1: W=B^T[rank,out], x=dy^T[out,seq] -> spatial=seq
    ((8, 2048), 32, "Step1: B^T[8,2048] x dy^T[2048,32]"),
    # Step 2: W=x^T[in,seq], x=tmp[seq,rank] -> spatial=rank=8 (padded to 16!)
    ((2048, 32), 8, "Step2: x^T[2048,32] x tmp[32,8] (spatial=8->16)"),
    # Step 3: W=A^T[rank,in], x=x^T[in,seq] -> spatial=seq
    ((8, 2048), 32, "Step3: A^T[8,2048] x x^T[2048,32]"),
    # Step 4: W=ax^T[rank,seq], x=dy[seq,out] -> spatial=out
    ((8, 32), 2048, "Step4: ax^T[8,32] x dy[32,2048]"),
    # k_proj/v_proj variant: out_dim=256
    ((8, 2048), 32, "Step1_kv: B^T[8,2048] x dy^T[2048,32]"),
    ((2048, 32), 8, "Step2_kv: x^T[2048,32] x tmp[32,8]"),
    ((8, 256), 32, "Step4_kv: ax^T[8,32] x dy[32,256]"),
]

all_pass = True
for W_shape, x_cols, desc in test_cases:
    W = np.random.randn(*W_shape).astype(np.float32) * 0.01
    x = np.random.randn(W_shape[1], x_cols).astype(np.float32)
    try:
        result = _conv_matmul(lib, W, x)
        expected = W @ x
        err = np.max(np.abs(result - expected))
        status = "PASS" if err < 0.01 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {desc}: {status} err={err:.6f}")
    except RuntimeError as e:
        all_pass = False
        print(f"  {desc}: EXCEPTION {e}")

print(f"\n  All direct tests: {'PASS' if all_pass else 'FAIL'}")

# ============================================================
# Test 3: Full LoRA gradient computation (direct, no subprocess)
# ============================================================
print("\nTest 3: Full LoRA gradient (direct, rank=8, dim=2048, seq=32)")
np.random.seed(123)
seq, in_dim, rank, out_dim = 32, 2048, 8, 2048

dy = np.random.randn(seq, out_dim).astype(np.float32) * 0.01
x = np.random.randn(seq, in_dim).astype(np.float32) * 0.01
lora_a = np.random.randn(in_dim, rank).astype(np.float32) * 0.1
lora_b = np.random.randn(rank, out_dim).astype(np.float32) * 0.1

# Numpy reference
tmp_ref = dy @ lora_b.T              # [seq, rank]
d_a_ref = x.T @ tmp_ref              # [in, rank]
ax_ref = x @ lora_a                  # [seq, rank]
d_b_ref = ax_ref.T @ dy              # [rank, out]

# ANE computation (manual steps using _conv_matmul)
# Pad seq
padded_seq = _pad_spatial(seq)
dy_p = dy if padded_seq == seq else np.zeros((padded_seq, out_dim), dtype=np.float32)
x_p = x if padded_seq == seq else np.zeros((padded_seq, in_dim), dtype=np.float32)
if padded_seq > seq:
    dy_p[:seq] = dy
    x_p[:seq] = x

try:
    # Step 1: tmp = dy @ B^T => conv(W=lora_b[rank,out], input=dy^T[out,padded_seq])
    # Note: lora_b is [rank, out_dim] already. We want lora_b @ dy^T = [rank, padded_seq]
    # Then transpose to get [padded_seq, rank]
    # Actually: dy @ B^T = (B @ dy^T)^T, so conv(W=B, x=dy^T) gives B@dy^T, then transpose.
    tmp_ane = _conv_matmul(lib, lora_b, dy_p.T).T  # [rank,padded_seq]^T = [padded_seq,rank]
    tmp_ane = tmp_ane[:seq]  # trim padding
    err1 = np.max(np.abs(tmp_ane - tmp_ref))
    print(f"  Step 1 (dy@B^T): err={err1:.6f}")

    # Step 2: d_A = x^T @ tmp => conv(W=x^T[in,padded_seq], input=tmp^T_padded[padded_seq,rank])
    # _conv_matmul handles spatial padding (rank=8 -> 16)
    d_a_ane = _conv_matmul(lib, x_p.T, tmp_ane)  # x^T[in,padded_seq] @ tmp[padded_seq,rank]
    # But wait: tmp_ane is [seq,rank], we need tmp as [padded_seq,rank] for weight dim to match
    # Actually the weight is x_p.T which is [in_dim, padded_seq].
    # The input to conv is tmp_ane which is [seq, rank] -> transposed? No.
    # Conv: W[out_ch, in_ch] @ X[in_ch, spatial]
    # We want: x^T @ tmp = x^T[in, seq] @ tmp[seq, rank] = [in, rank]
    # So: W = x_p.T[in, padded_seq], X = tmp_padded[padded_seq, rank]
    # in_ch = padded_seq, spatial = rank = 8 -> needs padding to 16

    # Recompute with proper padded tmp
    tmp_padded = np.zeros((padded_seq, rank), dtype=np.float32)
    tmp_padded[:seq] = tmp_ane
    d_a_ane = _conv_matmul(lib, x_p.T, tmp_padded)  # [in, rank]
    err2 = np.max(np.abs(d_a_ane - d_a_ref))
    print(f"  Step 2 (x^T@tmp): err={err2:.6f}")

    # Step 3: ax = x @ A => conv(W=A^T[rank,in], input=x^T[in,padded_seq])
    ax_ane = _conv_matmul(lib, lora_a.T, x_p.T).T  # [rank,padded_seq]^T = [padded_seq,rank]
    ax_ane = ax_ane[:seq]
    err3 = np.max(np.abs(ax_ane - ax_ref))
    print(f"  Step 3 (x@A): err={err3:.6f}")

    # Step 4: d_B = ax^T @ dy => conv(W=ax^T[rank,padded_seq], input=dy[padded_seq,out])
    ax_padded = np.zeros((padded_seq, rank), dtype=np.float32)
    ax_padded[:seq] = ax_ane
    d_b_ane = _conv_matmul(lib, ax_padded.T, dy_p)  # [rank,padded_seq] @ [padded_seq,out] = [rank,out]
    err4 = np.max(np.abs(d_b_ane - d_b_ref))
    print(f"  Step 4 (ax^T@dy): err={err4:.6f}")

    max_err = max(err1, err2, err3, err4)
    print(f"\n  Overall max error: {max_err:.6f}")
    print(f"  Status: {'PASS' if max_err < 0.01 else 'FAIL'}")
    print(f"  (fp16 intermediate precision, errors < 0.01 expected)")

except RuntimeError as e:
    print(f"  EXCEPTION: {e}")

# ============================================================
# Test 4: k_proj/v_proj variant (out_dim=256)
# ============================================================
print("\nTest 4: LoRA gradient with k/v_proj (rank=8, in=2048, out=256, seq=32)")
np.random.seed(456)
seq, in_dim, rank, out_dim = 32, 2048, 8, 256
dy = np.random.randn(seq, out_dim).astype(np.float32) * 0.01
x = np.random.randn(seq, in_dim).astype(np.float32) * 0.01
lora_a = np.random.randn(in_dim, rank).astype(np.float32) * 0.1
lora_b = np.random.randn(rank, out_dim).astype(np.float32) * 0.1

# Numpy reference
d_a_ref = x.T @ (dy @ lora_b.T)
d_b_ref = (x @ lora_a).T @ dy

padded_seq = _pad_spatial(seq)
dy_p = np.zeros((padded_seq, out_dim), dtype=np.float32)
dy_p[:seq] = dy
x_p = np.zeros((padded_seq, in_dim), dtype=np.float32)
x_p[:seq] = x

try:
    # All 4 steps
    tmp = _conv_matmul(lib, lora_b, dy_p.T).T[:seq]
    tmp_p = np.zeros((padded_seq, rank), dtype=np.float32)
    tmp_p[:seq] = tmp
    d_a_ane = _conv_matmul(lib, x_p.T, tmp_p)

    ax = _conv_matmul(lib, lora_a.T, x_p.T).T[:seq]
    ax_p = np.zeros((padded_seq, rank), dtype=np.float32)
    ax_p[:seq] = ax
    d_b_ane = _conv_matmul(lib, ax_p.T, dy_p)

    err_a = np.max(np.abs(d_a_ane - d_a_ref))
    err_b = np.max(np.abs(d_b_ane - d_b_ref))
    print(f"  d_lora_a err: {err_a:.6f}")
    print(f"  d_lora_b err: {err_b:.6f}")
    print(f"  Status: {'PASS' if max(err_a, err_b) < 0.01 else 'FAIL'}")
except RuntimeError as e:
    print(f"  EXCEPTION: {e}")

print("\nDone.")
