"""
aggregate50.py — aggregate the 3-way batch results (outputs/batch3/uid_*.json)
into summary statistics + a chart, over however many uids have completed.

Metrics per setting (full / full_pure / no_crop):
  • DINO-I v2/v3 ↑ : edit realization (render vs edit image)
  • CD->src     ↓ : identity preservation
  • voxels, n_crops
Plus source-baseline DINO-I ("do nothing") for the edit-gain comparison.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

OUT = "outputs/batch3"
CONFIGS = ["full", "full_pure", "no_crop"]


def load_rows():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "uid_*.json")),
                    key=lambda x: int(os.path.basename(x)[4:-5])):
        rows.append(json.load(open(p)))
    return rows


def _get(row, cfg, *keys):
    d = row.get(cfg) or {}
    if "error" in d:
        return None
    for k in keys:
        d = (d or {}).get(k) if isinstance(d, dict) else None
    return d


def summarize(rows):
    lines = []
    n = len(rows)
    lines.append(f"# 3-way aggregate over {n} uids\n")

    # per-setting central tendency (only completed cases)
    lines.append(f"{'setting':<10} {'ok':>3} {'err':>3} | {'DINO2 mean':>10} {'med':>7} | "
                 f"{'DINO3 mean':>10} {'med':>7} | {'CD_src mean':>11} {'med':>7} | {'vox':>6} {'crops':>5}")
    stats = {}
    for cfg in CONFIGS:
        d2 = [_get(r, cfg, "dino", "dinov2") for r in rows]
        d3 = [_get(r, cfg, "dino", "dinov3") for r in rows]
        cd = [_get(r, cfg, "cd_src") for r in rows]
        vx = [_get(r, cfg, "voxels") for r in rows]
        nc = [_get(r, cfg, "n_crops") for r in rows]
        err = sum(1 for r in rows if "error" in (r.get(cfg) or {}))
        f2 = [x for x in d2 if x is not None]
        f3 = [x for x in d3 if x is not None]
        fc = [x for x in cd if x is not None]
        fv = [x for x in vx if x is not None]
        fn = [x for x in nc if x is not None]
        stats[cfg] = {"d2": d2, "d3": d3, "cd": cd}
        m = lambda a: (np.mean(a) if a else float("nan"))
        md = lambda a: (np.median(a) if a else float("nan"))
        lines.append(f"{cfg:<10} {len(f2):>3} {err:>3} | {m(f2):>10.4f} {md(f2):>7.4f} | "
                     f"{m(f3):>10.4f} {md(f3):>7.4f} | {m(fc):>11.4f} {md(fc):>7.4f} | "
                     f"{m(fv):>6.0f} {m(fn):>5.1f}")

    # source baseline
    sb2 = [ _get(r, "source_dino", "dinov2") if isinstance(r.get("source_dino"), dict) else None for r in rows]
    sb2 = [x for x in sb2 if x is not None]
    lines.append(f"\nsource-baseline DINO2 (do nothing): mean={np.mean(sb2):.4f}  (n={len(sb2)})")

    # win-rate on DINO2 (best edit per uid), only uids where all 3 completed
    lines.append("\n## win-rate (best DINO-I v2 per uid, only fully-completed uids)")
    wins = {c: 0 for c in CONFIGS}
    complete = 0
    for r in rows:
        vals = {c: _get(r, c, "dino", "dinov2") for c in CONFIGS}
        if any(v is None for v in vals.values()):
            continue
        complete += 1
        best = max(vals, key=lambda c: vals[c])
        wins[best] += 1
    for c in CONFIGS:
        pct = 100 * wins[c] / complete if complete else 0
        lines.append(f"  {c:<10} {wins[c]:>3}/{complete}  ({pct:4.1f}%)")

    # paired: crop settings vs no_crop on DINO2 & CD
    lines.append("\n## paired vs no_crop (completed pairs)")
    for cfg in ["full", "full_pure"]:
        pd = [(_get(r, cfg, "dino", "dinov2"), _get(r, "no_crop", "dino", "dinov2")) for r in rows]
        pd = [(a, b) for a, b in pd if a is not None and b is not None]
        if pd:
            diff = [a - b for a, b in pd]
            win = sum(1 for d in diff if d > 0)
            lines.append(f"  {cfg} vs no_crop  DINO2: mean Δ={np.mean(diff):+.4f}  "
                         f"win {win}/{len(pd)} ({100*win/len(pd):.0f}%)")
        pc = [(_get(r, cfg, "cd_src"), _get(r, "no_crop", "cd_src")) for r in rows]
        pc = [(a, b) for a, b in pc if a is not None and b is not None]
        if pc:
            diff = [a - b for a, b in pc]  # lower is better -> negative is win
            win = sum(1 for d in diff if d < 0)
            lines.append(f"  {cfg} vs no_crop  CD_src: mean Δ={np.mean(diff):+.4f}  "
                         f"win {win}/{len(pc)} ({100*win/len(pc):.0f}%)")

    txt = "\n".join(lines)
    return txt, stats


def make_chart(rows, stats, path="outputs/eval50_chart.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"full": "#3b7dd8", "full_pure": "#e0803a", "no_crop": "#6aa84f"}

    # 1) mean DINO-I v2/v3 bars
    ax = axes[0, 0]
    x = np.arange(len(CONFIGS)); w = 0.35
    d2m = [np.nanmean([v for v in stats[c]["d2"] if v is not None] or [np.nan]) for c in CONFIGS]
    d3m = [np.nanmean([v for v in stats[c]["d3"] if v is not None] or [np.nan]) for c in CONFIGS]
    ax.bar(x - w/2, d2m, w, label="DINO-I v2", color="#3b7dd8")
    ax.bar(x + w/2, d3m, w, label="DINO-I v3", color="#8fb8ec")
    ax.set_xticks(x); ax.set_xticklabels(CONFIGS); ax.set_title("Mean DINO-I (edit realization) ↑")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # 2) mean CD_src bars
    ax = axes[0, 1]
    cdm = [np.nanmean([v for v in stats[c]["cd"] if v is not None] or [np.nan]) for c in CONFIGS]
    ax.bar(x, cdm, color=[colors[c] for c in CONFIGS])
    ax.set_xticks(x); ax.set_xticklabels(CONFIGS); ax.set_title("Mean CD→src (identity preservation) ↓")
    ax.grid(axis="y", alpha=0.3)

    # 3) per-uid DINO2 lines (paired)
    ax = axes[1, 0]
    for c in CONFIGS:
        ys = [v for v in stats[c]["d2"]]
        xs = [i for i, v in enumerate(ys) if v is not None]
        yv = [v for v in ys if v is not None]
        ax.plot(xs, yv, "o-", ms=3, label=c, color=colors[c], alpha=0.8)
    ax.set_title("DINO-I v2 per uid"); ax.set_xlabel("uid index"); ax.legend(); ax.grid(alpha=0.3)

    # 4) scatter DINO2 vs CD (tradeoff)
    ax = axes[1, 1]
    for c in CONFIGS:
        xs = [v for v in stats[c]["cd"]]
        ys = [v for v in stats[c]["d2"]]
        px = [a for a, b in zip(xs, ys) if a is not None and b is not None]
        py = [b for a, b in zip(xs, ys) if a is not None and b is not None]
        ax.scatter(px, py, s=18, label=c, color=colors[c], alpha=0.7)
    ax.set_xlabel("CD→src ↓"); ax.set_ylabel("DINO-I v2 ↑")
    ax.set_title("Edit vs identity tradeoff"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(f"3-way crop ablation — {len(rows)} uids", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"[chart] wrote {path}")


if __name__ == "__main__":
    rows = load_rows()
    txt, stats = summarize(rows)
    print(txt)
    open("outputs/eval50_summary.txt", "w").write(txt)
    try:
        make_chart(rows, stats)
    except Exception as e:
        print(f"[chart] skipped: {e}")
