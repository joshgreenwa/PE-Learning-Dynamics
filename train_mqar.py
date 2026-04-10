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
  - Attention pattern snapshots (raw attention matrices on a fixed validation batch)

Saves all logs and final model checkpoints to Google Drive.

Usage:
  1. Open in Google Colab
  2. Run cells top-to-bottom (or run as script)
  3. Results saved to /content/drive/MyDrive/PE_learning_dynamics/scale_{L,XL}/
"""

# ── 0. Setup ─────────────────────────────────────────────────────────────────

import subprocess, sys, os

# Mount Google Drive
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

DRIVE_ROOT = "/content/drive/MyDrive/PE_learning_dynamics"
os.makedirs(DRIVE_ROOT, exist_ok=True)

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
import matplotlib.pyplot as plt

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── 1. Scale configs ─────────────────────────────────────────────────────────

# Predefined scale configs (theoretically motivated, see Zoology paper Sec 4.3).
# Holding d_model fixed while increasing seq_len and D isolates the PE method
# as the bottleneck — attention solves MQAR at constant width (Prop 4.3), so
# any degradation at larger N implicates the positional encoding.
#
#   S  — paper baseline, fast iteration
#   M  — 2x distances, same d
#   L  — 4x distances, d held constant: PE quality matters most here
#   XL — denser recall load, needs more capacity

SCALE_CONFIGS = {
    "S": dict(
        seq_len=128, num_kv_pairs=8,  d_model=64,  batch_size=64,
        epochs=64,  train_batches=128, val_batches=32,
        attn_snapshot_epochs=[0, 1, 4, 16, 32, 63],
    ),
    "M": dict(
        seq_len=256, num_kv_pairs=16, d_model=64,  batch_size=16,
        epochs=96,  train_batches=192, val_batches=48,
        attn_snapshot_epochs=[0, 1, 8, 24, 48, 95],
    ),
    "L": dict(
        seq_len=512, num_kv_pairs=32, d_model=64,  batch_size=8,
        epochs=128, train_batches=256, val_batches=64,
        attn_snapshot_epochs=[0, 1, 8, 32, 64, 127],
    ),
    "XL": dict(
        seq_len=512, num_kv_pairs=64, d_model=128, batch_size=8,
        epochs=128, train_batches=256, val_batches=64,
        attn_snapshot_epochs=[0, 1, 8, 32, 64, 127],
    ),
}

# ── SELECT SCALES TO RUN (in order) ──
SCALES_TO_RUN = ["L", "XL"]
# ──────────────────────────────────────


# ── 2. Shared helpers (scale-independent) ────────────────────────────────────

def get_attn_modules(model):
    """Return list of (layer_idx, attn_module) for any model variant."""
    return [(i, block.attn) for i, block in enumerate(model.blocks)]


def get_mlp_modules(model):
    """Return list of (layer_idx, mlp_module) for any model variant."""
    return [(i, block.mlp) for i, block in enumerate(model.blocks)]


def compute_attention_entropy(attn_weights):
    """Compute per-head entropy from attention weights.

    Args:
        attn_weights: (B, n_heads, T, T) after softmax
    Returns:
        (n_heads,) average entropy per head
    """
    aw = attn_weights.clamp(min=1e-12)
    ent = -(aw * aw.log()).sum(dim=-1)
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
        self.output_norms = {}
        self.grad_norms = {}

    def __enter__(self):
        for i, mlp in get_mlp_modules(self.model):
            def fwd_hook(module, inp, out, layer_idx=i):
                self.output_norms[layer_idx] = out.detach().norm().item()
            self.handles.append(mlp.register_forward_hook(fwd_hook))

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


def lr_schedule(step, total_steps, warmup_steps, peak_lr):
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── 3. Per-scale training run ────────────────────────────────────────────────

def run_scale(scale_name: str):
    """Run the full training pipeline for one scale config."""

    _scale = SCALE_CONFIGS[scale_name]

    CFG = dict(
        vocab_size    = 8192,
        seq_len       = _scale["seq_len"],
        num_kv_pairs  = _scale["num_kv_pairs"],
        alpha         = 0.1,
        d_model       = _scale["d_model"],
        n_heads       = 2,
        n_layers      = 2,
        rope_base     = 10_000.0,
        lr            = 3e-3,
        weight_decay  = 0.1,
        warmup_frac   = 0.10,
        epochs        = _scale["epochs"],
        batch_size    = _scale["batch_size"],
        train_batches = _scale["train_batches"],
        val_batches   = _scale["val_batches"],
        attn_snapshot_epochs = _scale["attn_snapshot_epochs"],
        seed          = 42,
        scale         = scale_name,
    )

    DRIVE_DIR = f"{DRIVE_ROOT}/scale_{scale_name}"
    os.makedirs(DRIVE_DIR, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# SCALE {scale_name}  |  seq_len={CFG['seq_len']}  d_model={CFG['d_model']}  "
          f"KV={CFG['num_kv_pairs']}  epochs={CFG['epochs']}  batch={CFG['batch_size']}")
    print(f"# Saving to: {DRIVE_DIR}")
    print(f"{'#'*70}")

    with open(f"{DRIVE_DIR}/config.json", "w") as f:
        json.dump(CFG, f, indent=2)

    # ── Data generation ──────────────────────────────────────────────────

    def make_batch(seed=None):
        return generate_mqar_batch(
            vocab_size=CFG["vocab_size"],
            seq_len=CFG["seq_len"],
            num_kv_pairs=CFG["num_kv_pairs"],
            alpha=CFG["alpha"],
            batch_size=CFG["batch_size"],
            random_seed=seed,
        )

    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    val_batches = [make_batch(seed=CFG["seed"] + i) for i in range(CFG["val_batches"])]
    snapshot_batch = generate_mqar_batch(
        vocab_size=CFG["vocab_size"],
        seq_len=CFG["seq_len"],
        num_kv_pairs=CFG["num_kv_pairs"],
        alpha=CFG["alpha"],
        batch_size=8,
        random_seed=CFG["seed"] + 9999,
    )

    # ── Model factory ────────────────────────────────────────────────────

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

    # ── Training ─────────────────────────────────────────────────────────

    def compute_loss(model, inputs, labels):
        logits = model(inputs)
        return F.cross_entropy(
            logits.view(-1, CFG["vocab_size"]),
            labels.view(-1),
            ignore_index=-100,
        )

    def evaluate(model):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inp, lab in val_batches:
                inp, lab = inp.to(DEVICE), lab.to(DEVICE)
                total_loss += compute_loss(model, inp, lab).item()
        return total_loss / len(val_batches)

    def train_one_model(name, model):
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
            "train_loss": [],
            "val_loss": [],
            "attn_entropy": [],
            "mlp_grad_norm": [],
            "mlp_output_norm": [],
            "attn_snapshots": {},
        }

        global_step = 0
        train_seed_offset = 10000

        for epoch in range(CFG["epochs"]):
            model.train()
            epoch_loss = 0.0
            mlp_manager = None

            for batch_idx in range(CFG["train_batches"]):
                lr_now = lr_schedule(global_step, total_steps, warmup_steps, CFG["lr"])
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

                inp, lab = make_batch(seed=train_seed_offset + global_step)
                inp, lab = inp.to(DEVICE), lab.to(DEVICE)

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

            if mlp_manager is not None:
                log["mlp_grad_norm"].append(dict(mlp_manager.grad_norms))
                log["mlp_output_norm"].append(dict(mlp_manager.output_norms))
            else:
                log["mlp_grad_norm"].append({})
                log["mlp_output_norm"].append({})

            val_loss = evaluate(model)
            log["val_loss"].append(val_loss)

            # Attention entropy
            model.eval()
            enable_attn_storage(model, True)
            with torch.no_grad():
                v_inp, v_lab = val_batches[0]
                _ = model(v_inp.to(DEVICE))
            attn_dict = collect_attn_weights(model)
            entropy_dict = {}
            for layer_idx, aw in attn_dict.items():
                entropy_dict[layer_idx] = compute_attention_entropy(aw).tolist()
            log["attn_entropy"].append(entropy_dict)
            enable_attn_storage(model, False)

            # Attention snapshots
            if epoch in CFG["attn_snapshot_epochs"]:
                model.eval()
                enable_attn_storage(model, True)
                with torch.no_grad():
                    snap_inp, snap_lab = snapshot_batch
                    _ = model(snap_inp.to(DEVICE))
                snap_dict = collect_attn_weights(model)
                log["attn_snapshots"][epoch] = {
                    k: v.numpy() for k, v in snap_dict.items()
                }
                enable_attn_storage(model, False)

            # Print progress
            if epoch % 8 == 0 or epoch == CFG["epochs"] - 1:
                ent_str = ""
                if entropy_dict:
                    for li in sorted(entropy_dict.keys()):
                        heads = entropy_dict[li]
                        ent_str += f" L{li}={[f'{e:.2f}' for e in heads]}"
                print(
                    f"  Epoch {epoch:3d}/{CFG['epochs']} | "
                    f"train {avg_train:.4f} | val {val_loss:.4f} | "
                    f"entropy{ent_str}"
                )

        return log

    # ── Run all models ───────────────────────────────────────────────────

    all_logs = {}
    for name, model in models.items():
        torch.manual_seed(CFG["seed"])
        np.random.seed(CFG["seed"])
        log = train_one_model(name, model)
        all_logs[name] = log

        ckpt_path = f"{DRIVE_DIR}/{name}_final.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

    # ── Save logs ────────────────────────────────────────────────────────

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

    # ── Quick summary plot ───────────────────────────────────────────────

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Scale {scale_name}: seq_len={CFG['seq_len']}, d={CFG['d_model']}, "
                 f"KV={CFG['num_kv_pairs']}", fontsize=14)

    model_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    ax = axes[0, 0]
    for name, log in scalar_logs.items():
        ax.plot(log["train_loss"], label=name)
    ax.set_title("Train Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for name, log in scalar_logs.items():
        ax.plot(log["val_loss"], label=name)
    ax.set_title("Validation Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    for name, log in scalar_logs.items():
        ent_series = []
        for epoch_data in log["attn_entropy"]:
            val = epoch_data.get("0", epoch_data.get(0, [float("nan")]))
            ent_series.append(val[0] if len(val) > 0 else float("nan"))
        ax.plot(ent_series, label=name)
    ax.set_title("Attn Entropy (L0 H0)"); ax.set_xlabel("Epoch"); ax.set_ylabel("Entropy")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for name, log in scalar_logs.items():
        gn = [d.get("0", d.get(0, float("nan"))) for d in log["mlp_grad_norm"]]
        ax.plot(gn, label=name)
    ax.set_title("MLP Grad Norm (L0)"); ax.set_xlabel("Epoch"); ax.set_ylabel("Norm")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for name, log in scalar_logs.items():
        on = [d.get("0", d.get(0, float("nan"))) for d in log["mlp_output_norm"]]
        ax.plot(on, label=name)
    ax.set_title("MLP Output Norm (L0)"); ax.set_xlabel("Epoch"); ax.set_ylabel("Norm")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    names = list(scalar_logs.keys())
    final_vals = [scalar_logs[n]["val_loss"][-1] for n in names]
    bars = ax.bar(names, final_vals, color=model_colors[:len(names)])
    ax.set_title("Final Validation Loss"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, final_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{DRIVE_DIR}/training_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to {DRIVE_DIR}/training_summary.png")

    # ── Attention pattern grid (final snapshot) ──────────────────────────

    last_snap_epoch = max(CFG["attn_snapshot_epochs"])
    n_models = len(all_logs)
    n_layers = CFG["n_layers"]
    n_heads = CFG["n_heads"]

    fig, axes_grid = plt.subplots(
        n_models, n_layers * n_heads,
        figsize=(5 * n_layers * n_heads, 4 * n_models),
        squeeze=False,
    )
    fig.suptitle(f"Scale {scale_name}: Attention Patterns (Epoch {last_snap_epoch})",
                 fontsize=14)

    for row, (name, log) in enumerate(all_logs.items()):
        snap = log["attn_snapshots"].get(last_snap_epoch, {})
        col = 0
        for layer_idx in range(n_layers):
            arr = snap.get(layer_idx, None)
            for head_idx in range(n_heads):
                ax = axes_grid[row, col]
                if arr is not None:
                    attn_map = arr[0, head_idx]
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

    print(f"\nScale {scale_name} complete. All results saved to {DRIVE_DIR}/")

    # Free GPU memory before next scale
    del models, all_logs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ── 4. Run selected scales ───────────────────────────────────────────────────

for scale in SCALES_TO_RUN:
    run_scale(scale)

print("\n" + "=" * 70)
print("ALL SCALES COMPLETE")
print("=" * 70)
