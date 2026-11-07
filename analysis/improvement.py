#!/usr/bin/env python3
"""Before/after comparison of the failure-analysis-driven judge (P5) vs P4, on dev and held-out sets.

Inputs : data/items.jsonl + results/judgments.jsonl                  (dev, 20 scenarios)
         data/holdout_items.jsonl + results/holdout_judgments.jsonl  (held-out, 8 new scenarios)
Outputs: results/improvement.json, analysis/figures/fig6_improvement.png

Metrics (judge = sonnet unless --judge):
  A2 detect        safety == fail on safety-omission items
  A2-critical      same, restricted to items whose deleted sentence held warning signs/urgent action
  T1 detect        grounding == fail on retrieval-failure items (answer correct but ungrounded)
  A6 detect        grounding == fail on citation-mismatch items
  A1 detect        accuracy == fail on hallucination items
  clean FPR        any axis fail on clean items
  padded-clean FPR any axis fail on clean answers + harmless filler (B2)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AXES = ["accuracy", "safety", "grounding", "context"]
B = 2000
RNG = np.random.default_rng(7)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
S1, S2 = "#2a78d6", "#eb6834"


def load(items_path, judg_path, judge):
    items = {json.loads(l)["item_id"]: json.loads(l) for l in open(items_path, encoding="utf-8")}
    J = pd.DataFrame([json.loads(l) for l in open(judg_path, encoding="utf-8")])
    J = J[(J.judge == judge) & (J.protocol.isin(["P4", "P5"]))].copy()
    J["variant"] = J.item_id.map(lambda i: items[i]["variant"])
    J["scenario_id"] = J.item_id.map(lambda i: items[i]["scenario_id"])
    J["gt_critical"] = J.item_id.map(lambda i: items[i]["gt"]["critical"])
    J["anyfail"] = J[AXES].eq("fail").any(axis=1)
    return J


def rate(d, mask_fn, hit_fn):
    sub = d[mask_fn(d)]
    if sub.empty:
        return None
    hits = hit_fn(sub).astype(float)
    g = pd.DataFrame({"s": sub.scenario_id.values, "h": hits.values}).groupby("s").h.agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = RNG.integers(0, len(g), size=(B, len(g)))
    vals = s[idx].sum(1) / c[idx].sum(1)
    return {"rate": float(hits.mean()), "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "k": int(hits.sum()), "n": int(len(sub))}


METRICS = {
    "A2 누락 탐지": (lambda d: d.variant == "A2", lambda s: s.safety == "fail"),
    "A2 중 응급·경고 신호 누락": (lambda d: (d.variant == "A2") & d.gt_critical, lambda s: s.safety == "fail"),
    "T1 검색 실패 탐지": (lambda d: d.variant == "T1", lambda s: s.grounding == "fail"),
    "A6 인용 불일치 탐지": (lambda d: d.variant == "A6", lambda s: s.grounding == "fail"),
    "A1 사실 환각 탐지": (lambda d: d.variant == "A1", lambda s: s.accuracy == "fail"),
    "정상 답변 오탐": (lambda d: d.variant == "clean", lambda s: s.anyfail),
    "정상+군더더기 오탐": (lambda d: d.variant == "B2", lambda s: s.anyfail),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="sonnet")
    a = ap.parse_args()
    sets = {"dev": (os.path.join(ROOT, "data", "items.jsonl"), os.path.join(ROOT, "results", "judgments.jsonl")),
            "holdout": (os.path.join(ROOT, "data", "holdout_items.jsonl"),
                        os.path.join(ROOT, "results", "holdout_judgments.jsonl"))}
    out = {}
    for name, (ip, jp) in sets.items():
        if not (os.path.exists(ip) and os.path.exists(jp)):
            continue
        J = load(ip, jp, a.judge)
        out[name] = {}
        for p in ["P4", "P5"]:
            d = J[J.protocol == p]
            out[name][p] = {m: rate(d, mf, hf) for m, (mf, hf) in METRICS.items()}
    json.dump(out, open(os.path.join(ROOT, "results", f"improvement_{a.judge}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    for name in out:
        print(f"== {name}")
        for m in METRICS:
            r4, r5 = out[name]["P4"][m], out[name]["P5"][m]
            if r4 and r5:
                print(f"  {m:22s} P4 {r4['rate']*100:5.1f}% ({r4['k']}/{r4['n']})  ->  P5 {r5['rate']*100:5.1f}% ({r5['k']}/{r5['n']})")
    figure(out, a.judge)


def figure(out, judge):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Noto Sans CJK KR", "font.size": 9, "axes.edgecolor": GRID,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
                         "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
                         "axes.unicode_minus": False})
    names = [n for n in ["dev", "holdout"] if n in out]
    title = {"dev": "개발셋 (시나리오 20개, P5 설계에 사용)", "holdout": "홀드아웃 (처음 보는 시나리오 8개)"}
    ms = list(METRICS)
    fig, axes = plt.subplots(1, len(names), figsize=(7.4 if len(names) == 2 else 4.2, 3.6), dpi=200, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        y = np.arange(len(ms))[::-1]
        h = 0.36
        for k, (p, col, lab) in enumerate([("P4", S1, "P4 (기존 trace judge)"), ("P5", S2, "P5 (실패 분석 후 개선)")]):
            vals = [(out[n][p][m] or {}).get("rate", np.nan) for m in ms]
            off = h / 2 + 0.01 if k == 0 else -h / 2 - 0.01
            ax.barh(y + off, vals, h, color=col, label=lab)
            for yi, v in zip(y + off, vals):
                if not np.isnan(v):
                    ax.text(v + 0.015, yi, f"{v*100:.0f}", va="center", fontsize=7, color=INK)
        ax.set_yticks(y, ms, fontsize=8)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, .5, 1], ["0", "50", "100%"])
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_title(title[n], loc="left", fontsize=8.8)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "analysis", "figures", "fig6_improvement.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
