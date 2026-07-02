"""fig_share.py — visualize the cond-vs-velocity divergence from exp_share.log.

Message: crop conditions look ~0.87 similar to global (DINOv3 global-token bias),
but the velocity difference they produce is nearly orthogonal to the global one
(voxel-cosine ~0.3, relL2 ~1.0).  crop[0] is the exception: it == global exactly.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "outputs/exp_share.log"
txt = open(LOG).read()

# --- cond cosine (tgt) per crop ---
cond = {}
for m in re.finditer(r"crop\[(\d+)\]\s+cos\(tgt\)=([+\-\d.]+)\s+cos\(src\)=([+\-\d.]+)", txt):
    i = int(m.group(1)); cond[i] = (float(m.group(2)), float(m.group(3)))

# --- velocity metrics per step per crop ---
steps = {}
cur = None
for line in txt.splitlines():
    ms = re.search(r"step idx=(\d+)\s+t=([\d.]+)", line)
    if ms:
        cur = float(ms.group(2)); steps[cur] = {}
        continue
    mv = re.match(r"\s+(\d+)\s+([\d.]+)\s+([+\-\d.]+)\s+([\d.]+)\s*$", line)
    if mv and cur is not None:
        i = int(mv.group(1))
        steps[cur][i] = dict(relL2=float(mv.group(2)), voxcos=float(mv.group(3)), mag=float(mv.group(4)))

crops = sorted(cond)
ts = sorted(steps, reverse=True)   # 1.0, 0.8, 0.6
rep_t = ts[1] if len(ts) > 1 else ts[0]     # mid step t=0.8

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

# (1) per-crop: cond cos vs velocity voxcos  (representative step)
ax = axes[0]
x = np.arange(len(crops)); w = 0.38
cond_tgt = [cond[i][0] for i in crops]
vcos = [steps[rep_t][i]["voxcos"] for i in crops]
ax.bar(x - w/2, cond_tgt, w, label="cond cosine (DINOv3)", color="#4c78a8")
ax.bar(x + w/2, vcos, w, label=f"velocity voxel-cosine (t={rep_t})", color="#e45756")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels([f"c{i}" for i in crops])
ax.set_ylabel("cosine similarity to global")
ax.set_title("cond looks similar (~0.87) — but velocity doesn't (~0.3)")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
ax.annotate("crop[0] == global\n(exact duplicate)", xy=(0, 1.0), xytext=(1.2, 1.05),
            fontsize=9, color="#2c7", arrowprops=dict(arrowstyle="->", color="#2c7"))

# (2) scatter: cond cos (x) vs velocity voxcos (y), all crops all steps
ax = axes[1]
markers = {ts[0]: "o", ts[1]: "s", ts[-1]: "^"}
for t in ts:
    xs = [cond[i][0] for i in crops if i in steps[t]]
    ys = [steps[t][i]["voxcos"] for i in crops]
    ax.scatter(xs, ys, s=55, marker=markers.get(t, "o"), label=f"t={t}", alpha=0.8)
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="y=x (if reusable)")
ax.set_xlabel("cond cosine to global"); ax.set_ylabel("velocity voxel-cosine to global")
ax.set_title("decorrelated: high cond ≠ high velocity match")
ax.set_xlim(0, 1.05); ax.set_ylim(-0.2, 1.05)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.annotate("sub-crops:\nindependent signal", xy=(0.87, 0.33), xytext=(0.45, 0.6),
            fontsize=9, arrowprops=dict(arrowstyle="->"))

# (3) relL2 across steps (how large the crop-vs-global difference is)
ax = axes[2]
for t in ts:
    ys = [steps[t][i]["relL2"] for i in crops]
    ax.plot(crops, ys, "o-", label=f"t={t}")
ax.axhline(1.0, color="red", ls=":", lw=1)
ax.text(len(crops)-1, 1.02, "relL2=1 ⇒ diff as big as signal", color="red", ha="right", fontsize=8)
ax.set_xlabel("crop index"); ax.set_ylabel("relL2  ||vΔ_crop − vΔ_global|| / ||vΔ_global||")
ax.set_title("crop velocity differs by ~100% of the signal")
ax.set_xticks(crops); ax.legend(fontsize=9); ax.grid(alpha=0.3)

fig.suptitle("Can crop reuse the global computation?  cond says yes, velocity says no "
             "(except crop[0] ≡ global)", fontsize=13)
fig.tight_layout()
fig.savefig("outputs/share_chart.png", dpi=120)
print("saved outputs/share_chart.png")
