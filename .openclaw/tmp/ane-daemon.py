#!/usr/bin/env python3
"""
ANE Real-Time Fine-Tuning Daemon
Based on Ex0byt's pipeline architecture:
  1. MLX inference with live LoRA adapter (GPU/unified memory)
  2. SSE streaming responses back to client
  3. Background ANE training on user+assistant pairs (Neural Engine @ 2.8W)
  4. LoRA adapter hot-swapped in-memory for next inference

Port 8766 — daemon accepts POST /chat with JSON {messages: [...]}
"""
import os
import sys
import json
import time
import ctypes
import struct
import threading
import queue
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from mlx_lm import load, generate
from mlx_lm.tuner.lora import LoRALinear
import numpy as np

# ---------- Config ----------
MODEL_NAME = os.environ.get("ANE_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")
LORA_RANK = int(os.environ.get("ANE_LORA_RANK", "8"))
LORA_LAYERS = int(os.environ.get("ANE_LORA_LAYERS", "4"))  # last N layers
PORT = int(os.environ.get("ANE_PORT", "8766"))
MAX_TOKENS = int(os.environ.get("ANE_MAX_TOKENS", "512"))
BRIDGE_PATH = os.environ.get("ANE_BRIDGE_PATH",
    os.path.expanduser("~/ANE-backup/bridge/libane_bridge.dylib"))
TRAINING_DIR = os.path.expanduser("~/ane-training-data")
ADAPTER_PATH = os.path.expanduser("~/ane-lora-adapter")

# ---------- Training queue ----------
training_queue = queue.Queue()
training_pairs = []  # persistent conversation pairs

# ---------- ANE Bridge (ctypes) ----------
class ANEBridge:
    """Python wrapper around libane_bridge.dylib for ANE kernel dispatch."""

    def __init__(self, dylib_path):
        self.lib = None
        self.available = False
        try:
            self.lib = ctypes.CDLL(dylib_path)
            self._setup_signatures()
            rc = self.lib.ane_bridge_init()
            if rc == 0:
                self.available = True
                print(f"[ANE] Bridge initialized from {dylib_path}")
            else:
                print(f"[ANE] Bridge init failed (rc={rc}) — ANE training disabled")
        except OSError as e:
            print(f"[ANE] Cannot load bridge: {e} — ANE training disabled")

    def _setup_signatures(self):
        self.lib.ane_bridge_init.restype = ctypes.c_int
        # compile: mil_text, mil_len, weight_data, weight_len,
        #          n_inputs, input_sizes, n_outputs, output_sizes
        self.lib.ane_bridge_compile.restype = ctypes.c_void_p
        self.lib.ane_bridge_compile.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.ane_bridge_compile_multi_weights.restype = ctypes.c_void_p
        self.lib.ane_bridge_eval.restype = ctypes.c_bool
        self.lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]
        self.lib.ane_bridge_write_input.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.ane_bridge_read_output.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.ane_bridge_free.argtypes = [ctypes.c_void_p]
        self.lib.ane_bridge_get_compile_count.restype = ctypes.c_int
        self.lib.ane_bridge_reset_compile_count.restype = None
        self.lib.ane_bridge_build_weight_blob.restype = ctypes.POINTER(ctypes.c_uint8)
        self.lib.ane_bridge_build_weight_blob.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t)]
        self.lib.ane_bridge_free_blob.argtypes = [ctypes.c_void_p]

    @property
    def compile_count(self):
        if not self.available:
            return 0
        return self.lib.ane_bridge_get_compile_count()


ane = ANEBridge(BRIDGE_PATH)

# ---------- Model loading ----------
print(f"[MLX] Loading model: {MODEL_NAME}")
model, tokenizer = load(MODEL_NAME)
print(f"[MLX] Model loaded. Applying LoRA (rank={LORA_RANK}) to last {LORA_LAYERS} layers...")

# Apply LoRA to attention layers
def apply_lora(model, rank, num_layers):
    """Apply LoRA adapters to the last N transformer layers."""
    layers = model.model.layers
    total = len(layers)
    start = max(0, total - num_layers)

    for i in range(start, total):
        layer = layers[i]
        attn = layer.self_attn

        # Replace Q, K, V, O projections with LoRA
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if hasattr(attn, proj_name):
                orig = getattr(attn, proj_name)
                if isinstance(orig, nn.Linear):
                    lora = LoRALinear.from_base(orig, r=rank)
                    setattr(attn, proj_name, lora)

    # Count LoRA params
    lora_params = sum(
        p.size for k, p in mlx.utils.tree_flatten(model.trainable_parameters())
    )
    total_params = sum(p.size for _, p in mlx.utils.tree_flatten(model.parameters()))
    print(f"[LoRA] {lora_params:,} trainable / {total_params:,} total "
          f"({100*lora_params/total_params:.2f}%)")
    return model

model = apply_lora(model, LORA_RANK, LORA_LAYERS)

# Phase 2b: Replace LoRA layers with ANE-backed versions
ane_kernels = None
if ane.available:
    try:
        from ane_lora_kernels import (
            ANELoRAKernels, set_ane_kernels, replace_lora_with_ane)

        # Initialize kernel dispatcher + pre-compile for Qwen2.5-3B shapes
        ane_kernels = ANELoRAKernels(ane.lib)
        compiled = ane_kernels.precompile_for_shapes(
            in_dim=2048, out_dims=[2048, 256], rank=LORA_RANK,
            seq_lengths=[8, 16, 32, 64, 128, 256, 512])

        # Wire kernels into custom VJP
        set_ane_kernels(ane_kernels)

        # Swap LoRALinear → ANELoRALinear (shares params, no copy)
        replaced = replace_lora_with_ane(model)
        print(f"[ANE] {replaced} LoRA layers → ANE gradient dispatch active "
              f"({compiled} kernels pre-compiled)")
    except Exception as e:
        print(f"[ANE] Layer replacement failed ({e}), using MLX GPU fallback")
        ane_kernels = None

# Load saved adapter if exists
if os.path.exists(ADAPTER_PATH):
    try:
        model.load_weights(ADAPTER_PATH, strict=False)
        print(f"[LoRA] Loaded saved adapter from {ADAPTER_PATH}")
    except Exception as e:
        print(f"[LoRA] Could not load adapter: {e}")

# ---------- Background training thread ----------
def ane_training_worker():
    """Background thread: trains LoRA on user+assistant pairs.

    If ANELoRALinear layers are active, LoRA gradient computation is
    automatically dispatched to the Neural Engine via the custom VJP.
    Otherwise, everything runs on MLX GPU. The training loop is identical
    in both cases — ANE dispatch is transparent.
    """
    os.makedirs(TRAINING_DIR, exist_ok=True)

    while True:
        pair = training_queue.get()
        if pair is None:
            break

        user_text, assistant_text = pair
        training_pairs.append(pair)

        # Save training pair to disk
        pair_file = os.path.join(TRAINING_DIR, f"pair_{int(time.time()*1000)}.json")
        with open(pair_file, "w") as f:
            json.dump({"user": user_text, "assistant": assistant_text}, f)

        try:
            loss = _finetune_step(user_text, assistant_text)
        except Exception as e:
            print(f"[FT] Error: {e}")

        # Save adapter after each training step
        try:
            os.makedirs(ADAPTER_PATH, exist_ok=True)
            mx.savez(
                os.path.join(ADAPTER_PATH, "adapters.npz"),
                **dict(mlx.utils.tree_flatten(model.trainable_parameters()))
            )
        except Exception as e:
            print(f"[LoRA] Save error: {e}")


def _finetune_step(user_text, assistant_text):
    """Single gradient step. ANE dispatch is transparent via custom VJP.

    If ANELoRALinear layers are in the model, the backward pass
    automatically routes LoRA gradient matmuls to the Neural Engine.
    If ANE is not available or dispatch fails, MLX GPU computes them.
    The training loop is identical either way.
    """
    # Tokenize
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text},
         {"role": "assistant", "content": assistant_text}],
        tokenize=False, add_generation_prompt=False
    )
    tokens = mx.array(tokenizer.encode(prompt))

    def loss_fn(model, tokens):
        logits = model(tokens[None, :-1])
        targets = tokens[1:]
        return nn.losses.cross_entropy(logits.squeeze(0), targets, reduction="mean")

    # Reset ANE dispatch stats for this step
    ane_stats = None
    if ane.available:
        try:
            from ane_lora_kernels import reset_ane_dispatch_stats, get_ane_dispatch_stats
            reset_ane_dispatch_stats()
        except ImportError:
            pass

    # Forward + backward (ANE dispatch happens automatically in VJP!)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, tokens)
    mx.eval(loss, grads)

    # SGD update
    lr = 1e-4
    trainable = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
    grad_flat = dict(mlx.utils.tree_flatten(grads))

    updates = {k: trainable[k] - lr * grad_flat[k]
               for k in trainable if k in grad_flat}

    if updates:
        model.load_weights(list(updates.items()))
        mx.eval(model.parameters())

    loss_val = float(loss)

    # Log with ANE stats if available
    if ane.available:
        try:
            from ane_lora_kernels import get_ane_dispatch_stats
            stats = get_ane_dispatch_stats()
            dispatches = stats["dispatches"]
            fallbacks = stats["fallbacks"]
            engine = "ANE" if dispatches > 0 else "MLX"
            print(f"[{engine}-FT] loss={loss_val:.4f} | "
                  f"dispatches={dispatches} fallbacks={fallbacks}")
        except ImportError:
            print(f"[MLX-FT] loss={loss_val:.4f}")
    else:
        print(f"[MLX-FT] loss={loss_val:.4f}")

    return loss_val


# Start training thread
training_thread = threading.Thread(target=ane_training_worker, daemon=True)
training_thread.start()

# ---------- HTTP Server ----------
class DaemonHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/chat":
            self._handle_chat()
        elif self.path == "/status":
            self._handle_status()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/status":
            self._handle_status()
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_error(404)

    def _handle_status(self):
        # ANE kernel + dispatch stats
        ane_kernel_count = 0
        ane_kernel_compile_count = 0
        ane_dispatch_stats = {}
        if ane_kernels is not None:
            ane_kernel_count = ane_kernels.kernel_count
            ane_kernel_compile_count = ane_kernels.compile_count
        try:
            from ane_lora_kernels import get_ane_dispatch_stats
            ane_dispatch_stats = get_ane_dispatch_stats()
        except ImportError:
            pass

        status = {
            "model": MODEL_NAME,
            "lora_rank": LORA_RANK,
            "lora_layers": LORA_LAYERS,
            "ane_available": ane.available,
            "ane_phase": "2b" if ane_kernels is not None else ("fallback" if not ane.available else "init"),
            "ane_compile_count": ane.compile_count,
            "ane_kernel_count": ane_kernel_count,
            "ane_kernel_compile_count": ane_kernel_compile_count,
            "ane_dispatch_stats": ane_dispatch_stats,
            "training_pairs_total": len(training_pairs),
            "training_queue_size": training_queue.qsize(),
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())

    def _handle_chat(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))
        messages = body.get("messages", [])
        stream = body.get("stream", True)

        if not messages:
            self.send_error(400, "No messages provided")
            return

        # Extract last user message for training pair
        user_msg = None
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m["content"]
                break

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if stream:
            self._stream_response(prompt, user_msg)
        else:
            self._batch_response(prompt, user_msg)

    def _stream_response(self, prompt, user_msg):
        """SSE streaming response."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        tokens = tokenizer.encode(prompt)
        full_response = []

        try:
            response_text = generate(
                model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS,
                verbose=False
            )
            # Send as SSE chunks (word by word for streaming feel)
            words = response_text.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == 0 else " " + word
                data = json.dumps({"content": chunk, "done": False})
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()
                full_response.append(chunk)

            # Send done signal
            done_data = json.dumps({"content": "", "done": True})
            self.wfile.write(f"data: {done_data}\n\n".encode())
            self.wfile.flush()

            # Queue training pair
            assistant_text = "".join(full_response)
            if user_msg and assistant_text:
                training_queue.put((user_msg, assistant_text))

        except Exception as e:
            error_data = json.dumps({"error": str(e), "done": True})
            self.wfile.write(f"data: {error_data}\n\n".encode())
            self.wfile.flush()

    def _batch_response(self, prompt, user_msg):
        """Non-streaming response."""
        try:
            response_text = generate(
                model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS,
                verbose=False
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "content": response_text,
                "model": MODEL_NAME,
                "ane_training": ane.available
            }).encode())

            # Queue training pair
            if user_msg and response_text:
                training_queue.put((user_msg, response_text))

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def log_message(self, format, *args):
        print(f"[HTTP] {args[0]} {args[1]} {args[2]}")


# ---------- Main ----------
if __name__ == "__main__":
    ane_mode = "Phase 2b — GRADIENT DISPATCH" if ane_kernels else (
        "FALLBACK (MLX GPU)" if not ane.available else "INIT FAILED")
    print(f"\n{'='*60}")
    print(f"  ANE Real-Time Fine-Tuning Daemon")
    print(f"  Model:     {MODEL_NAME}")
    print(f"  LoRA:      rank={LORA_RANK}, layers={LORA_LAYERS}")
    print(f"  ANE:       {ane_mode}")
    if ane_kernels:
        print(f"  Kernels:   {ane_kernels.kernel_count} cached, "
              f"{ane_kernels.compile_count}/{100} budget")
    print(f"  Port:      {PORT}")
    print(f"  Adapter:   {ADAPTER_PATH}")
    print(f"{'='*60}\n")

    server = HTTPServer(("0.0.0.0", PORT), DaemonHandler)
    print(f"[DAEMON] Listening on 0.0.0.0:{PORT}")
    print(f"[DAEMON] POST /chat  — inference + background training")
    print(f"[DAEMON] GET  /status — daemon status")
    print(f"[DAEMON] GET  /health — health check")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[DAEMON] Shutting down...")
        training_queue.put(None)  # signal training thread to exit
        training_thread.join(timeout=5)
        server.server_close()
