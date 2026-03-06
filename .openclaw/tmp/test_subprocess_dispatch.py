"""Test subprocess-based ANE gradient dispatch (compute_lora_gradients).

This tests the full pipeline: parent process serializes data to tmpdir,
subprocess inits ANE bridge, compiles conv kernels, computes gradients,
parent reads results.
"""
import sys
import numpy as np
import time
sys.path.insert(0, '/Users/batmanosama/ANE-backup')

BRIDGE_PATH = '/Users/batmanosama/ANE-backup/bridge/libane_bridge.dylib'

from ane_lora_kernels import ANELoRAKernels

print("=" * 60)
print("ANE Subprocess Gradient Dispatch Test")
print("=" * 60)

kernels = ANELoRAKernels(BRIDGE_PATH)

# Test 1: Single module (q_proj)
print("\nTest 1: Single q_proj module (rank=8, in=2048, out=2048, seq=32)")
np.random.seed(42)
seq, in_dim, rank, out_dim = 32, 2048, 8, 2048
dy = np.random.randn(seq, out_dim).astype(np.float32) * 0.01
x = np.random.randn(seq, in_dim).astype(np.float32) * 0.01
lora_a = np.random.randn(in_dim, rank).astype(np.float32) * 0.1
lora_b = np.random.randn(rank, out_dim).astype(np.float32) * 0.1

# Numpy reference
d_a_ref = x.T @ (dy @ lora_b.T)
d_b_ref = (x @ lora_a).T @ dy

t0 = time.time()
results = kernels.compute_lora_gradients([(dy, x, lora_a, lora_b)])
t1 = time.time()

d_a, d_b = results[0]
err_a = np.max(np.abs(d_a - d_a_ref))
err_b = np.max(np.abs(d_b - d_b_ref))
print(f"  d_lora_a err: {err_a:.6f}")
print(f"  d_lora_b err: {err_b:.6f}")
print(f"  Time: {t1-t0:.3f}s")
print(f"  Stats: compiles={kernels.total_compiles}, dispatches={kernels.total_dispatches}, steps={kernels.total_steps}")
print(f"  Status: {'PASS' if max(err_a, err_b) < 0.01 else 'FAIL'}")

# Test 2: Multiple modules (like real 4-layer LoRA with q,k,v,o projections)
print("\nTest 2: 16 LoRA modules (4 layers x 4 projections)")
np.random.seed(123)
modules = []
refs = []

for layer in range(4):
    for proj, (in_d, out_d) in enumerate([
        (2048, 2048),  # q_proj
        (2048, 256),   # k_proj
        (2048, 256),   # v_proj
        (2048, 2048),  # o_proj
    ]):
        dy_i = np.random.randn(seq, out_d).astype(np.float32) * 0.01
        x_i = np.random.randn(seq, in_d).astype(np.float32) * 0.01
        a_i = np.random.randn(in_d, rank).astype(np.float32) * 0.1
        b_i = np.random.randn(rank, out_d).astype(np.float32) * 0.1
        modules.append((dy_i, x_i, a_i, b_i))
        refs.append((
            x_i.T @ (dy_i @ b_i.T),
            (x_i @ a_i).T @ dy_i
        ))

t0 = time.time()
results = kernels.compute_lora_gradients(modules)
t1 = time.time()

max_err = 0
all_pass = True
for i, ((d_a, d_b), (d_a_ref, d_b_ref)) in enumerate(zip(results, refs)):
    ea = np.max(np.abs(d_a - d_a_ref))
    eb = np.max(np.abs(d_b - d_b_ref))
    err = max(ea, eb)
    max_err = max(max_err, err)
    if err >= 0.01:
        all_pass = False
        print(f"  Module {i:2d}: FAIL err={err:.6f}")

print(f"  Overall max error: {max_err:.6f}")
print(f"  Time: {t1-t0:.3f}s")
print(f"  Stats: compiles={kernels.total_compiles}, dispatches={kernels.total_dispatches}, steps={kernels.total_steps}")
print(f"  Status: {'PASS' if all_pass else 'FAIL'}")

print("\nDone.")
