"""
MQAR Visualisation — L and XL scales
=====================================
Run this after train_mqar.py has completed both L and XL scales.
Loads saved logs and attention snapshots from Google Drive and produces
detailed per-scale plots plus cross-scale comparisons.

Generates per scale (L, XL):
  1. Loss curves (train + val + generalisation gap)
  2. Attention entropy per head per layer over training
  3. MLP gradient norm per layer over training
  4. MLP output norm per layer over training
  5. Attention pattern evolution per model across training
  6. Cross-model attention comparison at early / mid / final epochs
  7. Zoomed attention on the KV region
  8. Summary table

Cross-scale:
  9.  Final val loss comparison (L vs XL, grouped by model)
  10. Val loss trajectories side-by-side
"""

# ── 0. Setup ─────────────────────────────────────────────────────────────────

import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

DRIVE_ROOT = "/content/drive/MyDrive/PE_learning_dynamics"
SCALES = ["L", "XL"]

COLORS = {
    "RoPE":      "#1f77b4",
    "GOAT":      "#ff7f0e",
    "pRoPE_0.5": "#2ca02c",
    "ALiBi":     "#d62728",
    "NoPE":      "#9467bd",
}

def color(name):
    return COLORS.get(name, "gray")


def _extract_series(log_list, layer_key, head_idx=None):
    """Extract a per-epoch scalar series from a list of {layer_key: value} dicts."""
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


# ── 1. Load data for both scales ─────────────────────────────────────────────

all_data = {}  # scale -> {logs, CFG, snapshots, MODEL_NAMES, ...}

for scale in SCALES:
    drive_dir = f"{DRIVE_ROOT}/scale_{scale}"
    save_dir = f"{drive_dir}/figures"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{drive_dir}/training_logs.json") as f:
        logs = json.load(f)
    with open(f"{drive_dir}/config.json") as f:
        cfg = json.load(f)

    model_names = list(logs.keys())

    snapshots = {}
    for name in model_names:
        path = f"{drive_dir}/{name}_attn_snapshots.npz"
        if os.path.exists(path):
            snapshots[name] = dict(np.load(path))

    all_data[scale] = dict(
        logs=logs, CFG=cfg, snapshots=snapshots,
        MODEL_NAMES=model_names, DRIVE_DIR=drive_dir, SAVE_DIR=save_dir,
    )
    print(f"Loaded scale {scale}: seq_len={cfg['seq_len']}, d_model={cfg['d_model']}, "
          f"KV={cfg['num_kv_pairs']}, epochs={cfg['epochs']}")


# ── 2-8. Per-scale visualisation ─────────────────────────────────────────────

def visualise_scale(scale):
    """Produce the full set of plots for one scale."""
    d = all_data[scale]
    logs = d["logs"]
    CFG = d["CFG"]
    snapshots = d["snapshots"]
    MODEL_NAMES = d["MODEL_NAMES"]
    SAVE_DIR = d["SAVE_DIR"]

    N_LAYERS = CFG["n_layers"]
    N_HEADS = CFG["n_heads"]
    SEQ_LEN = CFG["seq_len"]
    SNAP_EPOCHS = CFG["attn_snapshot_epochs"]

    tag = f"Scale {scale} (N={SEQ_LEN}, d={CFG['d_model']}, KV={CFG['num_kv_pairs']})"

    # ── 2. Loss curves ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f"{tag} — Loss Curves", fontsize=14, y=1.03)

    ax = axes[0]
    for name in MODEL_NAMES:
        ax.plot(logs[name]["train_loss"], label=name, color=color(name), lw=1.5)
    ax.set_title("Train Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name in MODEL_NAMES:
        ax.plot(logs[name]["val_loss"], label=name, color=color(name), lw=1.5)
    ax.set_title("Validation Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for name in MODEL_NAMES:
        t = np.array(logs[name]["train_loss"])
        v = np.array(logs[name]["val_loss"])
        ax.plot(v - t, label=name, color=color(name), lw=1.5)
    ax.set_title("Generalisation Gap (Val - Train)"); ax.set_xlabel("Epoch")
    ax.set_ylabel("Difference"); ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/loss_curves.png", dpi=150, bbox_inches="tight")
    plt.show(); print(f"Saved: {SAVE_DIR}/loss_curves.png")

    # ── 3. Attention entropy ─────────────────────────────────────────────
    fig, axes = plt.subplots(N_LAYERS, N_HEADS, figsize=(7*N_HEADS, 5*N_LAYERS),
                             squeeze=False)
    fig.suptitle(f"{tag} — Attention Entropy", fontsize=14, y=1.02)

    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            ax = axes[layer, head]
            for name in MODEL_NAMES:
                s = _extract_series(logs[name]["attn_entropy"], layer, head)
                ax.plot(s, label=name, color=color(name), lw=1.5)
            ax.set_title(f"Layer {layer}, Head {head}"); ax.set_xlabel("Epoch")
            ax.set_ylabel("Entropy (nats)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/attn_entropy.png", dpi=150, bbox_inches="tight")
    plt.show(); print(f"Saved: {SAVE_DIR}/attn_entropy.png")

    # ── 4. MLP gradient norm ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, N_LAYERS, figsize=(7*N_LAYERS, 5), squeeze=False)
    fig.suptitle(f"{tag} — MLP Gradient Norm", fontsize=14, y=1.02)

    for layer in range(N_LAYERS):
        ax = axes[0, layer]
        for name in MODEL_NAMES:
            s = _extract_series(logs[name]["mlp_grad_norm"], layer)
            ax.plot(s, label=name, color=color(name), lw=1.5)
        ax.set_title(f"Layer {layer}"); ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Norm"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/mlp_grad_norm.png", dpi=150, bbox_inches="tight")
    plt.show(); print(f"Saved: {SAVE_DIR}/mlp_grad_norm.png")

    # ── 5. MLP output norm ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, N_LAYERS, figsize=(7*N_LAYERS, 5), squeeze=False)
    fig.suptitle(f"{tag} — MLP Output Norm", fontsize=14, y=1.02)

    for layer in range(N_LAYERS):
        ax = axes[0, layer]
        for name in MODEL_NAMES:
            s = _extract_series(logs[name]["mlp_output_norm"], layer)
            ax.plot(s, label=name, color=color(name), lw=1.5)
        ax.set_title(f"Layer {layer}"); ax.set_xlabel("Epoch")
        ax.set_ylabel("Output Norm"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/mlp_output_norm.png", dpi=150, bbox_inches="tight")
    plt.show(); print(f"Saved: {SAVE_DIR}/mlp_output_norm.png")

    # ── 6. Attention patterns over training (per model) ──────────────────
    for name in MODEL_NAMES:
        if name not in snapshots:
            continue

        avail_epochs = sorted(set(
            int(k.split("_")[0].replace("epoch", ""))
            for k in snapshots[name].keys()
        ))
        n_snap = len(avail_epochs)
        n_cols = N_LAYERS * N_HEADS

        fig, axes_g = plt.subplots(n_snap, n_cols,
                                   figsize=(4.5*n_cols, 3.5*n_snap), squeeze=False)
        fig.suptitle(f"{tag} — {name} Attention Evolution", fontsize=14, y=1.01)

        for row, epoch in enumerate(avail_epochs):
            col = 0
            for layer in range(N_LAYERS):
                key = f"epoch{epoch}_layer{layer}"
                arr = snapshots[name].get(key, None)
                for head in range(N_HEADS):
                    ax = axes_g[row, col]
                    if arr is not None:
                        attn_map = arr[:, head, :, :].mean(axis=0)
                        im = ax.imshow(attn_map, aspect="auto", cmap="magma",
                                       vmin=0, vmax=attn_map.max().clip(min=1e-6))
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    else:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=ax.transAxes)
                    if row == 0:
                        ax.set_title(f"L{layer} H{head}", fontsize=10)
                    if col == 0:
                        ax.set_ylabel(f"Ep {epoch}", fontsize=10)
                    ax.tick_params(labelsize=7)
                    col += 1

        plt.tight_layout()
        p = f"{SAVE_DIR}/attn_patterns_{name}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.show()
        print(f"Saved: {p}")

    # ── 7. Cross-model comparison at select epochs ───────────────────────
    comparison_epochs = [SNAP_EPOCHS[0], SNAP_EPOCHS[len(SNAP_EPOCHS)//2], SNAP_EPOCHS[-1]]

    for epoch in comparison_epochs:
        n_cols = N_LAYERS * N_HEADS
        n_rows = len(MODEL_NAMES)

        fig, axes_g = plt.subplots(n_rows, n_cols,
                                   figsize=(4.5*n_cols, 3.5*n_rows), squeeze=False)
        fig.suptitle(f"{tag} — All Models at Epoch {epoch}", fontsize=14, y=1.01)

        for row, name in enumerate(MODEL_NAMES):
            col = 0
            for layer in range(N_LAYERS):
                key = f"epoch{epoch}_layer{layer}"
                arr = snapshots.get(name, {}).get(key, None)
                for head in range(N_HEADS):
                    ax = axes_g[row, col]
                    if arr is not None:
                        attn_map = arr[:, head, :, :].mean(axis=0)
                        im = ax.imshow(attn_map, aspect="auto", cmap="magma",
                                       vmin=0, vmax=attn_map.max().clip(min=1e-6))
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    else:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=ax.transAxes)
                    if row == 0:
                        ax.set_title(f"L{layer} H{head}", fontsize=10)
                    if col == 0:
                        ax.set_ylabel(name, fontsize=10, fontweight="bold")
                    ax.tick_params(labelsize=7)
                    col += 1

        plt.tight_layout()
        p = f"{SAVE_DIR}/attn_compare_epoch{epoch}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.show()
        print(f"Saved: {p}")

    # ── 8. Zoomed attention (KV region) ──────────────────────────────────
    kv_end = 2 * CFG["num_kv_pairs"]
    ZOOM = min(kv_end * 3, SEQ_LEN)  # show KV region + some queries
    final_epoch = SNAP_EPOCHS[-1]
    n_cols = N_LAYERS * N_HEADS
    n_rows = len(MODEL_NAMES)

    fig, axes_g = plt.subplots(n_rows, n_cols,
                               figsize=(5*n_cols, 4*n_rows), squeeze=False)
    fig.suptitle(f"{tag} — Zoomed Attention (first {ZOOM} pos) — Epoch {final_epoch}",
                 fontsize=14, y=1.01)

    for row, name in enumerate(MODEL_NAMES):
        col = 0
        for layer in range(N_LAYERS):
            key = f"epoch{final_epoch}_layer{layer}"
            arr = snapshots.get(name, {}).get(key, None)
            for head in range(N_HEADS):
                ax = axes_g[row, col]
                if arr is not None:
                    attn_map = arr[0, head, :ZOOM, :ZOOM]
                    im = ax.imshow(attn_map, aspect="auto", cmap="magma",
                                   vmin=0, vmax=attn_map.max().clip(min=1e-6))
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.axvline(kv_end - 0.5, color="cyan", lw=0.8, ls="--", alpha=0.7)
                    ax.axhline(kv_end - 0.5, color="cyan", lw=0.8, ls="--", alpha=0.7)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes)
                if row == 0:
                    ax.set_title(f"L{layer} H{head}", fontsize=10)
                if col == 0:
                    ax.set_ylabel(name, fontsize=10, fontweight="bold")
                ax.tick_params(labelsize=7)
                col += 1

    plt.tight_layout()
    p = f"{SAVE_DIR}/attn_zoomed_epoch{final_epoch}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.show()
    print(f"Saved: {p}")

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"RESULTS — {tag}")
    print(f"{'='*70}")
    header = f"{'Model':<12} {'Train Loss':>12} {'Val Loss':>12} {'Ent L0H0':>10} {'Ent L0H1':>10}"
    print(header); print("-" * len(header))
    for name in MODEL_NAMES:
        tl = logs[name]["train_loss"][-1]
        vl = logs[name]["val_loss"][-1]
        ent = logs[name]["attn_entropy"][-1]
        e0 = ent.get("0", ent.get(0, [float("nan"), float("nan")]))
        e0h0 = e0[0] if len(e0) > 0 else float("nan")
        e0h1 = e0[1] if len(e0) > 1 else float("nan")
        print(f"{name:<12} {tl:>12.4f} {vl:>12.4f} {e0h0:>10.4f} {e0h1:>10.4f}")
    print()


# Run per-scale visualisation
for scale in SCALES:
    visualise_scale(scale)


# ── 9. Cross-scale comparisons ───────────────────────────────────────────────

CROSS_DIR = f"{DRIVE_ROOT}/cross_scale_figures"
os.makedirs(CROSS_DIR, exist_ok=True)

MODEL_NAMES = all_data[SCALES[0]]["MODEL_NAMES"]

# ── 9a. Final val loss: L vs XL grouped bar chart ────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(MODEL_NAMES))
width = 0.35

for i, scale in enumerate(SCALES):
    logs = all_data[scale]["logs"]
    cfg = all_data[scale]["CFG"]
    vals = [logs[n]["val_loss"][-1] for n in MODEL_NAMES]
    offset = (i - 0.5) * width
    bars = ax.bar(x + offset, vals, width, label=f"Scale {scale} (N={cfg['seq_len']}, KV={cfg['num_kv_pairs']})")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES)
ax.set_ylabel("Final Validation Loss")
ax.set_title("Final Validation Loss: L vs XL")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{CROSS_DIR}/final_val_loss_L_vs_XL.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {CROSS_DIR}/final_val_loss_L_vs_XL.png")

# ── 9b. Val loss trajectories: L vs XL per model ────────────────────────────

fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(5*len(MODEL_NAMES), 5),
                         squeeze=False)
fig.suptitle("Validation Loss Trajectories: L vs XL", fontsize=14, y=1.02)

linestyles = {"L": "-", "XL": "--"}

for col, name in enumerate(MODEL_NAMES):
    ax = axes[0, col]
    for scale in SCALES:
        logs = all_data[scale]["logs"]
        cfg = all_data[scale]["CFG"]
        ax.plot(logs[name]["val_loss"], label=f"{scale} (N={cfg['seq_len']})",
                color=color(name), ls=linestyles[scale], lw=1.5)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val Loss")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CROSS_DIR}/val_loss_trajectories_L_vs_XL.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {CROSS_DIR}/val_loss_trajectories_L_vs_XL.png")

# ── 9c. Attention entropy trajectories: L vs XL (layer 0, head 0) ───────────

fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(5*len(MODEL_NAMES), 5),
                         squeeze=False)
fig.suptitle("Attention Entropy (L0 H0): L vs XL", fontsize=14, y=1.02)

for col, name in enumerate(MODEL_NAMES):
    ax = axes[0, col]
    for scale in SCALES:
        logs = all_data[scale]["logs"]
        cfg = all_data[scale]["CFG"]
        s = _extract_series(logs[name]["attn_entropy"], 0, 0)
        ax.plot(s, label=f"{scale} (N={cfg['seq_len']})",
                color=color(name), ls=linestyles[scale], lw=1.5)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Entropy (nats)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CROSS_DIR}/attn_entropy_L_vs_XL.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {CROSS_DIR}/attn_entropy_L_vs_XL.png")

# ── 9d. Cross-scale summary table ───────────────────────────────────────────

print(f"\n{'='*80}")
print("CROSS-SCALE SUMMARY")
print(f"{'='*80}")
header = f"{'Model':<12} {'L Val':>10} {'XL Val':>10} {'Delta':>10} {'L Ent H0':>10} {'XL Ent H0':>10}"
print(header); print("-" * len(header))
for name in MODEL_NAMES:
    vl_L = all_data["L"]["logs"][name]["val_loss"][-1]
    vl_XL = all_data["XL"]["logs"][name]["val_loss"][-1]
    delta = vl_XL - vl_L

    ent_L = all_data["L"]["logs"][name]["attn_entropy"][-1]
    e_L = ent_L.get("0", ent_L.get(0, [float("nan")]))[0]
    ent_XL = all_data["XL"]["logs"][name]["attn_entropy"][-1]
    e_XL = ent_XL.get("0", ent_XL.get(0, [float("nan")]))[0]

    print(f"{name:<12} {vl_L:>10.4f} {vl_XL:>10.4f} {delta:>+10.4f} {e_L:>10.4f} {e_XL:>10.4f}")

print(f"\nAll cross-scale figures saved to {CROSS_DIR}/")
print("Done!")
