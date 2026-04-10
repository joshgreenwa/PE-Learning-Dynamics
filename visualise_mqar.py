"""
MQAR Training Visualisation — Colab-friendly
=============================================
Run this after train_mqar.py has completed. Loads saved logs and attention
snapshots from Google Drive and produces detailed plots.

Generates:
  1. Loss curves (train + val, all models)
  2. Attention entropy per head per layer over training
  3. MLP gradient norm per layer over training
  4. MLP output norm per layer over training
  5. Attention pattern snapshots across models and training time
  6. Attention pattern comparison across models at a single epoch
"""

# ── 0. Setup ─────────────────────────────────────────────────────────────────

import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

DRIVE_ROOT = "/content/drive/MyDrive/PE_learning_dynamics"

# ── SELECT SCALE HERE (must match the training run you want to visualise) ──
SCALE = "L"
# ───────────────────────────────────────────────────────────────────────────

DRIVE_DIR = f"{DRIVE_ROOT}/scale_{SCALE}"
SAVE_DIR = f"{DRIVE_DIR}/figures"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Loading results from: {DRIVE_DIR}")

# ── 1. Load data ─────────────────────────────────────────────────────────────

with open(f"{DRIVE_DIR}/training_logs.json") as f:
    logs = json.load(f)

with open(f"{DRIVE_DIR}/config.json") as f:
    CFG = json.load(f)

MODEL_NAMES = list(logs.keys())
N_LAYERS = CFG["n_layers"]
N_HEADS = CFG["n_heads"]
SEQ_LEN = CFG["seq_len"]
SNAP_EPOCHS = CFG["attn_snapshot_epochs"]

# Load attention snapshots
snapshots = {}  # model_name -> {f"epoch{e}_layer{l}": ndarray}
for name in MODEL_NAMES:
    path = f"{DRIVE_DIR}/{name}_attn_snapshots.npz"
    if os.path.exists(path):
        snapshots[name] = dict(np.load(path))

# Consistent colours per model
COLORS = {
    "RoPE":      "#1f77b4",
    "GOAT":      "#ff7f0e",
    "pRoPE_0.5": "#2ca02c",
    "ALiBi":     "#d62728",
    "NoPE":      "#9467bd",
}

def color(name):
    return COLORS.get(name, "gray")


# Helper to pull a scalar series from the entropy / norm dicts
def _extract_series(log_list, layer_key, head_idx=None):
    """Extract a per-epoch scalar series from a list of {layer_key: value} dicts.

    If head_idx is not None, value is expected to be a list and we index into it.
    layer_key is tried as both int and string (JSON keys are strings).
    """
    series = []
    for entry in log_list:
        val = entry.get(str(layer_key), entry.get(layer_key, None))
        if val is None:
            series.append(float("nan"))
        elif head_idx is not None:
            series.append(val[head_idx] if head_idx < len(val) else float("nan"))
        else:
            series.append(val)
    return series


# ── 2. Loss curves ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Train loss
ax = axes[0]
for name in MODEL_NAMES:
    ax.plot(logs[name]["train_loss"], label=name, color=color(name), linewidth=1.5)
ax.set_title("Train Loss", fontsize=13)
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Val loss
ax = axes[1]
for name in MODEL_NAMES:
    ax.plot(logs[name]["val_loss"], label=name, color=color(name), linewidth=1.5)
ax.set_title("Validation Loss", fontsize=13)
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Train vs Val gap
ax = axes[2]
for name in MODEL_NAMES:
    train = np.array(logs[name]["train_loss"])
    val = np.array(logs[name]["val_loss"])
    ax.plot(val - train, label=name, color=color(name), linewidth=1.5)
ax.set_title("Generalisation Gap (Val - Train)", fontsize=13)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss Difference")
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/loss_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {SAVE_DIR}/loss_curves.png")


# ── 3. Attention entropy — per head, per layer ──────────────────────────────

fig, axes = plt.subplots(N_LAYERS, N_HEADS, figsize=(7 * N_HEADS, 5 * N_LAYERS),
                         squeeze=False)
fig.suptitle("Attention Entropy Over Training", fontsize=15, y=1.02)

for layer in range(N_LAYERS):
    for head in range(N_HEADS):
        ax = axes[layer, head]
        for name in MODEL_NAMES:
            series = _extract_series(logs[name]["attn_entropy"], layer, head)
            ax.plot(series, label=name, color=color(name), linewidth=1.5)
        ax.set_title(f"Layer {layer}, Head {head}", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Entropy (nats)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/attn_entropy.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {SAVE_DIR}/attn_entropy.png")


# ── 4. MLP gradient norm — per layer ────────────────────────────────────────

fig, axes = plt.subplots(1, N_LAYERS, figsize=(7 * N_LAYERS, 5), squeeze=False)
fig.suptitle("MLP Gradient Norm Over Training", fontsize=15, y=1.02)

for layer in range(N_LAYERS):
    ax = axes[0, layer]
    for name in MODEL_NAMES:
        series = _extract_series(logs[name]["mlp_grad_norm"], layer)
        ax.plot(series, label=name, color=color(name), linewidth=1.5)
    ax.set_title(f"Layer {layer}", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/mlp_grad_norm.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {SAVE_DIR}/mlp_grad_norm.png")


# ── 5. MLP output norm — per layer ──────────────────────────────────────────

fig, axes = plt.subplots(1, N_LAYERS, figsize=(7 * N_LAYERS, 5), squeeze=False)
fig.suptitle("MLP Output Norm Over Training", fontsize=15, y=1.02)

for layer in range(N_LAYERS):
    ax = axes[0, layer]
    for name in MODEL_NAMES:
        series = _extract_series(logs[name]["mlp_output_norm"], layer)
        ax.plot(series, label=name, color=color(name), linewidth=1.5)
    ax.set_title(f"Layer {layer}", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Output Norm")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/mlp_output_norm.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {SAVE_DIR}/mlp_output_norm.png")


# ── 6. Attention patterns over training (per model) ─────────────────────────
# For each model: rows = snapshot epochs, columns = layer x head
# Shows how attention patterns evolve during training

for name in MODEL_NAMES:
    if name not in snapshots:
        continue

    avail_epochs = sorted(set(
        int(k.split("_")[0].replace("epoch", ""))
        for k in snapshots[name].keys()
    ))
    n_epochs = len(avail_epochs)
    n_cols = N_LAYERS * N_HEADS

    fig, axes = plt.subplots(n_epochs, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_epochs),
                             squeeze=False)
    fig.suptitle(f"Attention Patterns — {name}", fontsize=15, y=1.01)

    for row, epoch in enumerate(avail_epochs):
        col = 0
        for layer in range(N_LAYERS):
            key = f"epoch{epoch}_layer{layer}"
            arr = snapshots[name].get(key, None)
            for head in range(N_HEADS):
                ax = axes[row, col]
                if arr is not None:
                    # Average over the snapshot batch for a cleaner picture
                    attn_map = arr[:, head, :, :].mean(axis=0)  # (T, T)
                    im = ax.imshow(attn_map, aspect="auto", cmap="magma",
                                   vmin=0, vmax=attn_map.max().clip(min=1e-6))
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=12)

                if row == 0:
                    ax.set_title(f"L{layer} H{head}", fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"Epoch {epoch}", fontsize=10)
                ax.set_xlabel("")
                ax.tick_params(labelsize=7)
                col += 1

    plt.tight_layout()
    path = f"{SAVE_DIR}/attn_patterns_{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# ── 7. Cross-model attention comparison at select epochs ─────────────────────
# Pick a few key epochs: first, mid-training, final

comparison_epochs = [SNAP_EPOCHS[0], SNAP_EPOCHS[len(SNAP_EPOCHS)//2], SNAP_EPOCHS[-1]]

for epoch in comparison_epochs:
    n_cols = N_LAYERS * N_HEADS
    n_rows = len(MODEL_NAMES)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    fig.suptitle(f"Attention Patterns — All Models at Epoch {epoch}",
                 fontsize=15, y=1.01)

    for row, name in enumerate(MODEL_NAMES):
        col = 0
        for layer in range(N_LAYERS):
            key = f"epoch{epoch}_layer{layer}"
            arr = snapshots.get(name, {}).get(key, None)
            for head in range(N_HEADS):
                ax = axes[row, col]
                if arr is not None:
                    attn_map = arr[:, head, :, :].mean(axis=0)
                    im = ax.imshow(attn_map, aspect="auto", cmap="magma",
                                   vmin=0, vmax=attn_map.max().clip(min=1e-6))
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=12)

                if row == 0:
                    ax.set_title(f"L{layer} H{head}", fontsize=10)
                if col == 0:
                    ax.set_ylabel(name, fontsize=10, fontweight="bold")
                ax.tick_params(labelsize=7)
                col += 1

    plt.tight_layout()
    path = f"{SAVE_DIR}/attn_compare_epoch{epoch}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# ── 8. Zoomed attention: first 32 positions (KV region + early queries) ──────
# This is the most informative region for MQAR — where the model must
# look back at key-value pairs to answer queries.

ZOOM = 48  # first 48 positions: 16 KV tokens + early query region
final_epoch = SNAP_EPOCHS[-1]

n_cols = N_LAYERS * N_HEADS
n_rows = len(MODEL_NAMES)

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(5 * n_cols, 4 * n_rows),
                         squeeze=False)
fig.suptitle(f"Zoomed Attention (first {ZOOM} positions) — Epoch {final_epoch}",
             fontsize=15, y=1.01)

for row, name in enumerate(MODEL_NAMES):
    col = 0
    for layer in range(N_LAYERS):
        key = f"epoch{final_epoch}_layer{layer}"
        arr = snapshots.get(name, {}).get(key, None)
        for head in range(N_HEADS):
            ax = axes[row, col]
            if arr is not None:
                attn_map = arr[0, head, :ZOOM, :ZOOM]  # single example, zoomed
                im = ax.imshow(attn_map, aspect="auto", cmap="magma",
                               vmin=0, vmax=attn_map.max().clip(min=1e-6))
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Mark the KV boundary
                kv_end = 2 * CFG["num_kv_pairs"]
                ax.axvline(kv_end - 0.5, color="cyan", linewidth=0.8,
                           linestyle="--", alpha=0.7)
                ax.axhline(kv_end - 0.5, color="cyan", linewidth=0.8,
                           linestyle="--", alpha=0.7)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12)

            if row == 0:
                ax.set_title(f"L{layer} H{head}", fontsize=10)
            if col == 0:
                ax.set_ylabel(name, fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=7)
            col += 1

plt.tight_layout()
path = f"{SAVE_DIR}/attn_zoomed_epoch{final_epoch}.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {path}")


# ── 9. Summary table ────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)
header = f"{'Model':<12} {'Train Loss':>12} {'Val Loss':>12} {'Entropy L0H0':>14} {'Entropy L0H1':>14}"
print(header)
print("-" * len(header))
for name in MODEL_NAMES:
    tl = logs[name]["train_loss"][-1]
    vl = logs[name]["val_loss"][-1]
    ent = logs[name]["attn_entropy"][-1]
    e0 = ent.get("0", ent.get(0, [float("nan"), float("nan")]))
    e0h0 = e0[0] if len(e0) > 0 else float("nan")
    e0h1 = e0[1] if len(e0) > 1 else float("nan")
    print(f"{name:<12} {tl:>12.4f} {vl:>12.4f} {e0h0:>14.4f} {e0h1:>14.4f}")

print(f"\nAll figures saved to {SAVE_DIR}/")
print("Done!")
