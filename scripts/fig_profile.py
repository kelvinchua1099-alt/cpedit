"""fig_profile.py — visualize the Tier-0 profile (outputs/profile/summary.json)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.load(open("outputs/profile/summary.json"))
CONFIGS = ["no_crop", "full_pure", "full"]
PHASES = [
    ("2_source_asset", "source asset"),
    ("3_dinov3_cond", "DINOv3 cond"),
    ("4_stage1_rasi_pmg", "Stage-1 (RASI+PMG)"),
    ("5_tar_stage23", "TAR stage2/3"),
    ("7_save_glb", "save GLB"),
]
colors = ["#b0b7c0", "#f2c14e", "#d1495b", "#6aa84f", "#8e7cc3"]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# (1) stacked bar: time composition per setting
ax = axes[0]
bottom = np.zeros(len(CONFIGS))
for (key, lab), col in zip(PHASES, colors):
    vals = np.array([rows[c]["phases"].get(key, 0) for c in CONFIGS])
    ax.bar(CONFIGS, vals, bottom=bottom, label=lab, color=col)
    bottom += vals
for i, c in enumerate(CONFIGS):
    ax.text(i, rows[c]["total"] + 5, f"{rows[c]['total']:.0f}s", ha="center", fontweight="bold")
ax.set_ylabel("seconds (incl. profiling sync overhead)")
ax.set_title("Wall time composition per setting")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# (2) Stage-1 share + flow-forward count
ax = axes[1]
x = np.arange(len(CONFIGS)); w = 0.38
s1 = [rows[c]["phases"]["4_stage1_rasi_pmg"] for c in CONFIGS]
other = [rows[c]["total"] - s for c, s in zip(CONFIGS, s1)]
ax.bar(x - w/2, s1, w, label="Stage-1", color="#d1495b")
ax.bar(x - w/2, other, w, bottom=s1, label="everything else", color="#b0b7c0")
ax.set_xticks(x); ax.set_xticklabels(CONFIGS)
ax.set_ylabel("seconds")
for i, c in enumerate(CONFIGS):
    pct = 100 * s1[i] / rows[c]["total"]
    ax.text(i - w/2, rows[c]["total"] + 5, f"S1={pct:.0f}%", ha="center", fontsize=9)
ax2 = ax.twinx()
fwd = [rows[c]["fwd_calls"] for c in CONFIGS]
ax2.plot(x, fwd, "o-", color="black", label="flow forwards")
for i, v in enumerate(fwd):
    ax2.text(i + 0.05, v, f"{v}", fontsize=9)
ax2.set_ylabel("flow-model forward calls")
ax.set_title("Stage-1 dominates; forwards scale with crops")
ax.legend(loc="upper left", fontsize=9)

# (3) forwards vs n_crops linearity
ax = axes[2]
nc = np.array([rows[c]["n_crops"] for c in CONFIGS])
fwd = np.array([rows[c]["fwd_calls"] for c in CONFIGS])
order = np.argsort(nc)
ax.plot(nc[order], fwd[order], "o-", color="#d1495b", ms=9)
# fit line through no_crop and full
slope = (fwd[nc == 8][0] - fwd[nc == 0][0]) / 8
ax.plot(nc[order], fwd[nc == 0][0] + slope * nc[order], "--", color="gray",
        label=f"~{slope:.0f} forwards / crop")
for c in CONFIGS:
    ax.annotate(c, (rows[c]["n_crops"], rows[c]["fwd_calls"]),
                textcoords="offset points", xytext=(8, -4), fontsize=9)
ax.set_xlabel("n_crops"); ax.set_ylabel("flow-model forward calls")
ax.set_title("Forwards ≈ base + k·n_crops (linear)")
ax.legend(); ax.grid(alpha=0.3)

fig.suptitle("Tier-0 profile (uid_1) — where wall time goes", fontsize=15)
fig.tight_layout()
fig.savefig("outputs/profile_chart.png", dpi=120)
print("saved outputs/profile_chart.png")
