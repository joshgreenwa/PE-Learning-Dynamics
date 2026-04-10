"""
MQAR Training Script — Colab-friendly
======================================
Trains RoPE, GOAT, p-RoPE (p=0.5), ALiBi, and NoPE models on the MQAR
synthetic task from the Zoology paper (Arora et al., 2023).

Tracks per model:
  - Train / validation loss
  - Attention entropy per head per layer
  - MLP gradient norm per layer
  - MLP output norm per layer
  - Attention pattern snapshots (raw matrices on a fixed validation batch)

Saves all logs and final model checkpoints to Google Drive.

Usage:
  1. Open in Google Colab
  2. Run cells top-to-bottom (or run as script)
  3. Results saved to /content/drive/MyDrive/PE_learning_dynamics/
"""

# ── 0. Setup ─────────────────────────────────────────────────────────────────

import subprocess, sys, os

# Mount Google Drive
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

DRIVE_DIR = "/content/drive/MyDrive/PE_learning_dynamics"
os.makedirs(DRIVE_DIR, exist_ok=True)

# Clone repo (public)
REPO_DIR = "/content/PE-Learning-Dynamics"
if not os.path.isdir(REPO_DIR):
    subprocess.check_call(
        ["git", "clone", "https://github.com/joshgreenwa/PE-Learning-Dynamics.git", REPO_DIR]
    )
else:
    # Pull latest changes if already cloned
    subprocess.check_call(["git", "-C", REPO_DIR, "pull"])

# Add repo root to path (mqar_rope.py lives at repo root)
sys.path.insert(0, REPO_DIR)

import math
import json
import copy
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mqar_rope import (
    generate_mqar_batch,
    build_rope_model,
    build_prope_model,
    build_goat_model,
    build_alibi_model,
    RoPETransformer,
    pRoPETransformer,
    GOATTransformer,
    ALiBiTransformer,
)

# ── 1. Hyperparameters ───────────────────────────────────────────────────────

CFG = dict(
    # Data
    vocab_size    = 8192,
    seq_len       = 128,
    num_kv_pairs  = 8,
    alpha         = 0.1,

    # Model
    d_model       = 64,
    n_heads       = 2,        # >= 2 needed for p-RoPE split
    n_layers      = 2,
    rope_base     = 10_000.0,

    # Training
    lr            = 3e-3,
    weight_decay  = 0.1,
    warmup_frac   = 0.10,
    epochs        = 64,
    batch_size    = 64,
    train_batches = 128,      # batches per epoch
    val_batches   = 32,

    # Logging
    attn_snapshot_epochs = [0, 1, 4, 16, 32, 63],  # when to save raw attention
    seed          = 42,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# Save config
with open(f"{DRIVE_DIR}/config.json", "w") as f:
    json.dump(CFG, f, indent=2)


# ── 2. Data generation helpers ───────────────────────────────────────────────

def make_batch(seed=None):
    """Generate one training batch."""
    return generate_mqar_batch(
        vocab_size=CFG["vocab_size"],
        seq_len=CFG["seq_len"],
        num_kv_pairs=CFG["num_kv_pairs"],
        alpha=CFG["alpha"],
        batch_size=CFG["batch_size"],
        random_seed=seed,
    )


# Fixed validation set (same across all models for fair comparison)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
VAL_BATCHES = [make_batch(seed=CFG["seed"] + i) for i in range(CFG["val_batches"])]
# A single small batch for attention snapshots
SNAPSHOT_BATCH = generate_mqar_batch(
    vocab_size=CFG["vocab_size"],
    seq_len=CFG["seq_len"],
    num_kv_pairs=CFG["num_kv_pairs"],
    alpha=CFG["alpha"],
    batch_size=8,
    random_seed=CFG["seed"] + 9999,
)


# ── 3. Model factory ────────────────────────────────────────────────────────

def make_models():
    """Return a dict of {name: model} for all five PE variants."""
    common = dict(
        vocab_size=CFG["vocab_size"],
        d_model=CFG["d_model"],
        n_heads=CFG["n_heads"],
        n_layers=CFG["n_layers"],
        max_seq_len=CFG["seq_len"],
    )
    models = {
        "RoPE": build_rope_model(**common, rope_base=CFG["rope_base"]),
        "GOAT": build_goat_model(**common, rope_base=CFG["rope_base"]),
        "pRoPE_0.5": build_prope_model(**common, rope_base=CFG["rope_base"], p=0.5),
        "ALiBi": build_alibi_model(
            vocab_size=common["vocab_size"],
            d_model=common["d_model"],
            n_heads=common["n_heads"],
            n_layers=common["n_layers"],
            max_seq_len=common["max_seq_len"],
        ),
        "NoPE": build_prope_model(**common, rope_base=CFG["rope_base"], p=1.0),
    }
    for m in models.values():
        m.to(DEVICE)
    return models


# ── 4. Metric helpers ────────────────────────────────────────────────────────

def get_attn_modules(model):
    """Return list of (layer_idx, attn_module) for any model variant."""
    modules = []
    for i, block in enumerate(model.blocks):
        modules.append((i, block.attn))
    return modules


def get_mlp_modules(model):
    """Return list of (layer_idx, mlp_module) for any model variant."""
    modules = []
    for i, block in enumerate(model.blocks):
        modules.append((i, block.mlp))
    return modules


def compute_attention_entropy(attn_weights):
    """Compute per-head entropy from attention weights.

    Args:
        attn_weights: (B, n_heads, T, T) after softmax
    Returns:
        (n_heads,) average entropy per head
    """
    # Clamp for numerical stability in log
    aw = attn_weights.clamp(min=1e-12)
    # Entropy per position: -sum_j p(j) log p(j), shape (B, n_heads, T)
    ent = -(aw * aw.log()).sum(dim=-1)
    # Average over batch and query positions -> (n_heads,)
    return ent.mean(dim=(0, 2)).cpu()


def enable_attn_storage(model, enable=True):
    """Toggle attention weight caching on all attention layers."""
    for _, attn in get_attn_modules(model):
        attn.store_attn = enable


def collect_attn_weights(model):
    """After a forward pass with store_attn=True, collect cached weights."""
    results = {}
    for i, attn in get_attn_modules(model):
        if attn._attn_weights is not None:
            results[i] = attn._attn_weights.cpu()
    return results


class MLPHookManager:
    """Context manager that registers forward hooks on MLP modules to capture output norms,
    and backward hooks to capture gradient norms."""

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.output_norms = {}   # layer_idx -> float
        self.grad_norms = {}     # layer_idx -> float

    def __enter__(self):
        for i, mlp in get_mlp_modules(self.model):
            # Forward hook: capture output norm
            def fwd_hook(module, inp, out, layer_idx=i):
                self.output_norms[layer_idx] = out.detach().norm().item()
            self.handles.append(mlp.register_forward_hook(fwd_hook))

            # Backward hook on the last linear layer (w2) for gradient norm
            last_linear = mlp.w2
            def bwd_hook(module, grad_in, grad_out, layer_idx=i):
                if grad_out[0] is not None:
                    self.grad_norms[layer_idx] = grad_out[0].detach().norm().item()
            self.handles.append(last_linear.register_full_backward_hook(bwd_hook))

        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ── 5. Training loop ────────────────────────────────────────────────────────

def compute_loss(model, inputs, labels):
    """Forward pass + cross-entropy on query positions only."""
    logits = model(inputs)
    return F.cross_entropy(
        logits.view(-1, CFG["vocab_size"]),
        labels.view(-1),
        ignore_index=-100,
    )


def evaluate(model, val_batches):
    """Compute average validation loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inp, lab in val_batches:
            inp, lab = inp.to(DEVICE), lab.to(DEVICE)
            total_loss += compute_loss(model, inp, lab).item()
    return total_loss / len(val_batches)


def lr_schedule(step, total_steps, warmup_steps, peak_lr):
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_model(name, model):
    """Train a single model and return its full log dict."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )

    total_steps = CFG["epochs"] * CFG["train_batches"]
    warmup_steps = int(CFG["warmup_frac"] * total_steps)

    log = {
        "train_loss": [],       # per epoch
        "val_loss": [],         # per epoch
        "attn_entropy": [],     # per epoch: list of {layer: [per-head entropies]}
        "mlp_grad_norm": [],    # per epoch: {layer: norm}
        "mlp_output_norm": [],  # per epoch: {layer: norm}
        "attn_snapshots": {},   # epoch -> {layer: (B, n_heads, T, T) numpy}
    }

    global_step = 0
    train_seed_offset = 10000

    for epoch in range(CFG["epochs"]):
        model.train()
        epoch_loss = 0.0

        # We'll accumulate MLP metrics from the last batch of the epoch
        mlp_manager = None

        for batch_idx in range(CFG["train_batches"]):
            # Learning rate schedule
            lr_now = lr_schedule(global_step, total_steps, warmup_steps, CFG["lr"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            # Generate fresh training data each batch
            inp, lab = make_batch(seed=train_seed_offset + global_step)
            inp, lab = inp.to(DEVICE), lab.to(DEVICE)

            # On last batch of epoch, attach MLP hooks
            if batch_idx == CFG["train_batches"] - 1:
                mlp_manager = MLPHookManager(model)
                mlp_manager.__enter__()

            optimizer.zero_grad()
            loss = compute_loss(model, inp, lab)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if mlp_manager is not None and batch_idx == CFG["train_batches"] - 1:
                mlp_manager.__exit__(None, None, None)

        avg_train = epoch_loss / CFG["train_batches"]
        log["train_loss"].append(avg_train)

        # MLP metrics from last batch
        if mlp_manager is not None:
            log["mlp_grad_norm"].append(
                {k: v for k, v in mlp_manager.grad_norms.items()}
            )
            log["mlp_output_norm"].append(
                {k: v for k, v in mlp_manager.output_norms.items()}
            )
        else:
            log["mlp_grad_norm"].append({})
            log["mlp_output_norm"].append({})

        # Validation loss
        val_loss = evaluate(model, VAL_BATCHES)
        log["val_loss"].append(val_loss)

        # Attention entropy (on first val batch)
        model.eval()
        enable_attn_storage(model, True)
        with torch.no_grad():
            v_inp, v_lab = VAL_BATCHES[0]
            _ = model(v_inp.to(DEVICE))
        attn_dict = collect_attn_weights(model)
        entropy_dict = {}
        for layer_idx, aw in attn_dict.items():
            entropy_dict[layer_idx] = compute_attention_entropy(aw).tolist()
        log["attn_entropy"].append(entropy_dict)
        enable_attn_storage(model, False)

        # Attention snapshots at selected epochs
        if epoch in CFG["attn_snapshot_epochs"]:
            model.eval()
            enable_attn_storage(model, True)
            with torch.no_grad():
                snap_inp, snap_lab = SNAPSHOT_BATCH
                _ = model(snap_inp.to(DEVICE))
            snap_dict = collect_attn_weights(model)
            log["attn_snapshots"][epoch] = {
                k: v.numpy() for k, v in snap_dict.items()
            }
            enable_attn_storage(model, False)

        # Print progress
        if epoch % 4 == 0 or epoch == CFG["epochs"] - 1:
            ent_str = ""
            if entropy_dict:
                for li in sorted(entropy_dict.keys()):
                    heads = entropy_dict[li]
                    ent_str += f" L{li}={[f'{e:.2f}' for e in heads]}"
            print(
                f"  Epoch {epoch:3d} | "
                f"train {avg_train:.4f} | val {val_loss:.4f} | "
                f"entropy{ent_str}"
            )

    return log


# ── 6. Run all models ───────────────────────────────────────────────────────

models = make_models()
all_logs = {}

for name, model in models.items():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    log = train_one_model(name, model)
    all_logs[name] = log

    # Save model checkpoint
    ckpt_path = f"{DRIVE_DIR}/{name}_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")


# ── 7. Save training logs ───────────────────────────────────────────────────

def make_serializable(obj):
    """Recursively convert numpy arrays / tensors for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    return obj


# Save scalar logs as JSON (attention snapshots saved separately as .npz)
scalar_logs = {}
for name, log in all_logs.items():
    scalar_logs[name] = {
        "train_loss": log["train_loss"],
        "val_loss": log["val_loss"],
        "attn_entropy": make_serializable(log["attn_entropy"]),
        "mlp_grad_norm": make_serializable(log["mlp_grad_norm"]),
        "mlp_output_norm": make_serializable(log["mlp_output_norm"]),
    }

with open(f"{DRIVE_DIR}/training_logs.json", "w") as f:
    json.dump(scalar_logs, f, indent=2)
print(f"\nScalar logs saved to {DRIVE_DIR}/training_logs.json")

# Save attention snapshots as .npz files (one per model)
for name, log in all_logs.items():
    if log["attn_snapshots"]:
        snapshot_data = {}
        for epoch, layer_dict in log["attn_snapshots"].items():
            for layer_idx, arr in layer_dict.items():
                key = f"epoch{epoch}_layer{layer_idx}"
                snapshot_data[key] = arr
        npz_path = f"{DRIVE_DIR}/{name}_attn_snapshots.npz"
        np.savez_compressed(npz_path, **snapshot_data)
        print(f"Attention snapshots saved to {npz_path}")


# ── 8. Summary plots ────────────────────────────────────────────────────────

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# --- Train loss ---
ax = axes[0, 0]
for name, log in scalar_logs.items():
    ax.plot(log["train_loss"], label=name)
ax.set_title("Train Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Val loss ---
ax = axes[0, 1]
for name, log in scalar_logs.items():
    ax.plot(log["val_loss"], label=name)
ax.set_title("Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Attention entropy (head 0, layer 0) ---
ax = axes[0, 2]
for name, log in scalar_logs.items():
    ent_series = []
    for epoch_data in log["attn_entropy"]:
        if "0" in epoch_data and len(epoch_data["0"]) > 0:
            ent_series.append(epoch_data["0"][0])
        elif 0 in epoch_data and len(epoch_data[0]) > 0:
            ent_series.append(epoch_data[0][0])
        else:
            ent_series.append(float("nan"))
    ax.plot(ent_series, label=name)
ax.set_title("Attn Entropy (Layer 0, Head 0)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Entropy (nats)")
ax.legend()
ax.grid(True, alpha=0.3)

# --- MLP gradient norm (layer 0) ---
ax = axes[1, 0]
for name, log in scalar_logs.items():
    gn = []
    for epoch_data in log["mlp_grad_norm"]:
        if "0" in epoch_data:
            gn.append(epoch_data["0"])
        elif 0 in epoch_data:
            gn.append(epoch_data[0])
        else:
            gn.append(float("nan"))
    ax.plot(gn, label=name)
ax.set_title("MLP Grad Norm (Layer 0)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Norm")
ax.legend()
ax.grid(True, alpha=0.3)

# --- MLP output norm (layer 0) ---
ax = axes[1, 1]
for name, log in scalar_logs.items():
    on = []
    for epoch_data in log["mlp_output_norm"]:
        if "0" in epoch_data:
            on.append(epoch_data["0"])
        elif 0 in epoch_data:
            on.append(epoch_data[0])
        else:
            on.append(float("nan"))
    ax.plot(on, label=name)
ax.set_title("MLP Output Norm (Layer 0)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Norm")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Final val loss bar chart ---
ax = axes[1, 2]
names = list(scalar_logs.keys())
final_vals = [scalar_logs[n]["val_loss"][-1] for n in names]
bars = ax.bar(names, final_vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
ax.set_title("Final Validation Loss")
ax.set_ylabel("Loss")
ax.grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, final_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(f"{DRIVE_DIR}/training_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nPlot saved to {DRIVE_DIR}/training_summary.png")

# ── 9. Attention pattern visualization (final epoch snapshot) ────────────────

last_snap_epoch = max(CFG["attn_snapshot_epochs"])
n_models = len(all_logs)
n_layers = CFG["n_layers"]
n_heads = CFG["n_heads"]

fig, axes = plt.subplots(
    n_models, n_layers * n_heads,
    figsize=(5 * n_layers * n_heads, 4 * n_models),
    squeeze=False,
)

for row, (name, log) in enumerate(all_logs.items()):
    snap = log["attn_snapshots"].get(last_snap_epoch, {})
    col = 0
    for layer_idx in range(n_layers):
        arr = snap.get(layer_idx, None)
        for head_idx in range(n_heads):
            ax = axes[row, col]
            if arr is not None:
                # Show first example in the snapshot batch
                attn_map = arr[0, head_idx]  # (T, T)
                im = ax.imshow(attn_map, aspect="auto", cmap="viridis")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_title(f"{name} L{layer_idx} H{head_idx}", fontsize=9)
            if col == 0:
                ax.set_ylabel("Query pos")
            ax.set_xlabel("Key pos")
            col += 1

plt.tight_layout()
plt.savefig(f"{DRIVE_DIR}/attn_patterns_epoch{last_snap_epoch}.png",
            dpi=150, bbox_inches="tight")
plt.show()
print(f"Attention patterns saved to {DRIVE_DIR}/attn_patterns_epoch{last_snap_epoch}.png")

print(f"\nAll results saved to {DRIVE_DIR}/")
print("Done!")
