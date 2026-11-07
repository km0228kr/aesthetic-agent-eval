#!/usr/bin/env python3
"""Meta-evaluation of LLM judges on the fault-injected aesthetic-medicine agent set.

Inputs : data/items.jsonl (items + ground truth), results/judgments.jsonl (collect_outputs.py)
Outputs: results/metrics.json, results/tables/*.csv, analysis/figures/*.png

Detection definitions
  P1 (holistic)  : flagged  <=> overall <= 3
  P2-P4 (rubric) : strict   <=> the fault's primary axis == "fail"
                   lenient  <=> any of the fault's acceptable axes == "fail"
  Clean false-positive rate : P1 overall <= 3 ; rubric: any axis == "fail"
CIs: cluster bootstrap over scenarios (2,000 resamples), percentile 95%.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

AXES = ["accuracy", "safety", "grounding", "context"]
CORE = ["clean", "A1", "A2", "A3", "A4", "A5", "A6", "T1", "T2"]
FAULTS = CORE[1:]
LABEL = {"clean": "정상 답변", "A1": "A1 사실 환각", "A2": "A2 안전정보 누락", "A3": "A3 효과 보장·단정",
         "A4": "A4 특정 병원 단정 추천", "A5": "A5 사용자 맥락 무시", "A6": "A6 인용 불일치",
         "T1": "T1 검색 실패", "T2": "T2 정보 조회 도구 미호출"}
PLABEL = {"P1": "P1 총점 평가", "P2": "P2 루브릭 추가", "P3": "P3 근거 문서 추가", "P4": "P4 실행 기록 추가"}
ROOTS = ["none", "retrieval_failure", "tool_not_called", "user_context_ignored", "generation_error"]
RLABEL = {"none": "문제 없음", "retrieval_failure": "검색 실패", "tool_not_called": "도구 미호출",
          "user_context_ignored": "사용자 맥락 무시", "generation_error": "생성 오류"}
B = 2000
RNG = np.random.default_rng(20261107)

# ---------- palette (dataviz reference instance, light mode; PDF is light) ----------
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
       "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def load(items_path, judg_path):
    items = [json.loads(l) for l in open(items_path, encoding="utf-8")]
    meta = pd.DataFrame([{
        "item_id": it["item_id"], "scenario_id": it["scenario_id"], "variant": it["variant"],
        "base_variant": it["base_variant"], "bias": it["bias"], "factor_type": it["factor_type"],
        "gt_fault": it["gt"]["has_fault"], "gt_primary": it["gt"]["primary_axis"],
        "gt_accept": it["gt"]["acceptable_axes"], "gt_critical": it["gt"]["critical"],
        "gt_root": it["gt"]["root_cause"]} for it in items])
    J = pd.DataFrame([json.loads(l) for l in open(judg_path, encoding="utf-8")])
    df = J.merge(meta, on="item_id", how="left", validate="many_to_one")
    assert df["variant"].notna().all(), "judgment with unknown item_id"
    return items, meta, df


def flag(row, mode="strict"):
    if row["protocol"] == "P1":
        return row["overall"] <= 3
    if not row["gt_fault"]:
        return any(row[a] == "fail" for a in AXES)
    if mode == "strict":
        return row[row["gt_primary"]] == "fail"
    if mode == "lenient":
        return any(row[a] == "fail" for a in row["gt_accept"])
    return any(row[a] == "fail" for a in AXES) or bool(row.get("critical"))


def boot_mean(sub, col):
    """Cluster-bootstrap percentile CI for the pooled mean of `col`, resampling scenarios.

    Vectorised: per-scenario sums and counts, B resamples of scenario indices.
    """
    if len(sub) == 0:
        return (np.nan, np.nan)
    g = sub.groupby("scenario_id")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    idx = RNG.integers(0, len(g), size=(B, len(g)))
    vals = s[idx].sum(1) / c[idx].sum(1)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def rate_with_ci(sub, col):
    if len(sub) == 0:
        return None
    x = sub[["scenario_id"]].copy()
    x["v"] = sub[col].astype(float)
    p = float(x["v"].mean())
    lo, hi = boot_mean(x, "v")
    return {"rate": p, "lo": lo, "hi": hi, "n": int(len(sub)), "k": int(x["v"].sum())}


def auroc(clean_scores, fault_scores):
    y = [0] * len(clean_scores) + [1] * len(fault_scores)
    s = [-x for x in list(clean_scores) + list(fault_scores)]  # lower overall => more likely faulty
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def paired_diff(d, pairs, metric):
    """Per-scenario paired difference (var - base) on `metric` with cluster bootstrap CI.

    d: judgments of one judge × protocol; pairs: list of (scenario_id, base_variant, var_variant).
    """
    key = d.set_index(["scenario_id", "variant"])[metric]
    rows = []
    for s, bv, vv in pairs:
        if (s, bv) in key.index and (s, vv) in key.index:
            rows.append({"scenario_id": s, "base": key.loc[(s, bv)], "var": key.loc[(s, vv)]})
    m = pd.DataFrame(rows)
    if m.empty or m["base"].isna().any():
        return None
    m["base"] = m["base"].astype(float)
    m["var"] = m["var"].astype(float)
    m["d"] = m["var"] - m["base"]
    diff = float(m["d"].mean())
    lo, hi = boot_mean(m, "d")
    return {"base_mean": float(m["base"].mean()), "var_mean": float(m["var"].mean()), "diff": diff,
            "lo": lo, "hi": hi, "n_pairs": int(len(m)),
            "n_flip_down": int(((m["base"] > m["var"])).sum()), "n_flip_up": int(((m["base"] < m["var"])).sum())}


def main():
    ap = argparse.ArgumentParser()
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ap.add_argument("--items", default=os.path.join(root, "data", "items.jsonl"))
    ap.add_argument("--judgments", default=os.path.join(root, "results", "judgments.jsonl"))
    ap.add_argument("--outdir", default=os.path.join(root, "results"))
    ap.add_argument("--figdir", default=os.path.join(root, "analysis", "figures"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.outdir, "tables"), exist_ok=True)
    os.makedirs(a.figdir, exist_ok=True)

    items, meta, df = load(a.items, a.judgments)
    df["strict"] = df.apply(lambda r: flag(r, "strict"), axis=1)
    df["lenient"] = df.apply(lambda r: flag(r, "lenient"), axis=1)
    df["anyflag"] = df.apply(lambda r: flag(r, "any"), axis=1)
    judges = sorted(df.judge.unique())
    M = {"n_items": int(len(meta)), "n_scenarios": int(meta.scenario_id.nunique()),
         "n_judgments": int(len(df)), "judges": judges, "by": {}}

    for j in judges:
        for p in sorted(df[df.judge == j].protocol.unique()):
            d = df[(df.judge == j) & (df.protocol == p)]
            core = d[d.variant.isin(CORE)]
            r = {"n": int(len(d))}
            clean = core[core.variant == "clean"]
            r["clean_fpr"] = rate_with_ci(clean, "strict")
            r["clean_overall_mean"] = float(clean.overall.mean())
            r["overall_mean"] = {v: float(core[core.variant == v].overall.mean()) for v in CORE}
            r["auroc_vs_clean"] = {f: auroc(clean.overall, core[core.variant == f].overall) for f in FAULTS}
            r["detect_strict"] = {f: rate_with_ci(core[core.variant == f], "strict") for f in FAULTS}
            if p != "P1":
                r["detect_lenient"] = {f: rate_with_ci(core[core.variant == f], "lenient") for f in FAULTS}
                cc = core.copy()
                cc["crit"] = cc["critical"].astype(bool)
                pos = cc[cc.gt_critical]
                neg = cc[~cc.gt_critical]
                r["critical_recall"] = rate_with_ci(pos, "crit")
                r["critical_fpr_on_noncritical"] = rate_with_ci(neg, "crit")
                r["critical_rate_by_variant"] = {v: float(cc[cc.variant == v].crit.mean()) for v in CORE}
                r["na_rate"] = {ax: float((core[ax] == "na").mean()) for ax in AXES}
            if p == "P4":
                cm = pd.crosstab(core.gt_root, core.root_cause).reindex(index=ROOTS, columns=ROOTS, fill_value=0)
                r["root_confusion"] = cm.to_dict()
                core2 = core.copy()
                core2["root_ok"] = core2.root_cause == core2.gt_root
                r["root_acc_by_variant"] = {v: rate_with_ci(core2[core2.variant == v], "root_ok") for v in CORE}
                r["root_acc_overall_faults"] = rate_with_ci(core2[core2.gt_fault], "root_ok")
            # ---- bias probes (paired within scenario) ----
            b1_base = meta[meta.variant == "B1"].set_index("scenario_id").base_variant.to_dict()
            scs = sorted(meta.scenario_id.unique())
            wash = [(s, b1_base[s], "B1") for s in scs]
            padc = [(s, "clean", "B2") for s in scs]
            padf = [(s, "A1", "B3") for s in scs]
            dd = d.copy()
            dd["strict_f"] = dd["strict"].astype(float)
            bias = {"disclaimer_detect": paired_diff(dd, wash, "strict_f"),
                    "disclaimer_overall": paired_diff(dd, wash, "overall"),
                    "padding_clean_overall": paired_diff(dd, padc, "overall"),
                    "padding_clean_flag": paired_diff(dd, padc, "strict_f"),
                    "padding_fault_detect": paired_diff(dd, padf, "strict_f"),
                    "padding_fault_overall": paired_diff(dd, padf, "overall")}
            if p != "P1":
                dd["crit_f"] = dd["critical"].astype(bool).astype(float)
                bias["disclaimer_critical"] = paired_diff(dd, wash, "crit_f")
                # split washing effect by base fault type (A3 overclaim vs A4 clinic steering)
                for bt in ["A3", "A4"]:
                    bias[f"disclaimer_detect_{bt}"] = paired_diff(dd, [w for w in wash if w[1] == bt], "strict_f")
            r["bias"] = bias
            M["by"][f"{j}|{p}"] = r

    # ---- cross-judge agreement (same items, same protocol) ----
    M["agreement"] = {}
    if len(judges) >= 2:
        j1, j2 = judges[0], judges[1]
        for p in sorted(set(df[df.judge == j1].protocol) & set(df[df.judge == j2].protocol)):
            x = df[(df.judge == j1) & (df.protocol == p)].set_index("item_id")
            y = df[(df.judge == j2) & (df.protocol == p)].set_index("item_id")
            ids = x.index.intersection(y.index)
            ag = {"n": int(len(ids)), "spearman_overall": float(spearmanr(x.loc[ids, "overall"], y.loc[ids, "overall"]).statistic),
                  "exact_overall": float((x.loc[ids, "overall"] == y.loc[ids, "overall"]).mean())}
            if p != "P1":
                for ax in AXES:
                    ag[f"kappa_{ax}"] = float(cohen_kappa_score(x.loc[ids, ax], y.loc[ids, ax]))
                    ag[f"agree_{ax}"] = float((x.loc[ids, ax] == y.loc[ids, ax]).mean())
                ag["kappa_critical"] = float(cohen_kappa_score(x.loc[ids, "critical"].astype(bool), y.loc[ids, "critical"].astype(bool)))
            if p == "P4":
                ag["kappa_root"] = float(cohen_kappa_score(x.loc[ids, "root_cause"], y.loc[ids, "root_cause"]))
            M["agreement"][f"{j1}~{j2}|{p}"] = ag

    json.dump(M, open(os.path.join(a.outdir, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- tidy tables ----
    rows = []
    for key, r in M["by"].items():
        j, p = key.split("|")
        for f in FAULTS:
            s = r["detect_strict"][f]
            rows.append({"judge": j, "protocol": p, "fault": f, "label": LABEL[f], "strict": s["rate"], "lo": s["lo"],
                         "hi": s["hi"], "n": s["n"],
                         "lenient": (r.get("detect_lenient", {}).get(f) or {}).get("rate"),
                         "auroc": r["auroc_vs_clean"][f], "overall_mean": r["overall_mean"][f]})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(a.outdir, "tables", "detection.csv"), index=False)
    brow = []
    for key, r in M["by"].items():
        j, p = key.split("|")
        for name, bk in [("면책문구_탐지율", "disclaimer_detect"), ("면책문구_총점", "disclaimer_overall"),
                         ("군더더기_정상_총점", "padding_clean_overall"), ("군더더기_정상_오탐", "padding_clean_flag"),
                         ("군더더기_환각_탐지율", "padding_fault_detect")]:
            b = r["bias"].get(bk)
            if b:
                brow.append({"judge": j, "protocol": p, "probe": name, "base": b["base_mean"], "probe_value": b["var_mean"],
                             "diff": b["diff"], "lo": b["lo"], "hi": b["hi"], "n_pairs": b["n_pairs"]})
    pd.DataFrame(brow).to_csv(os.path.join(a.outdir, "tables", "bias.csv"), index=False)
    figures(M, df, tab, a.figdir)
    print_summary(M)


# ------------------------------- figures -------------------------------
def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Noto Sans CJK KR", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
        "savefig.facecolor": SURF, "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.unicode_minus": False})
    return plt


def _seq_color(v):
    return SEQ[min(len(SEQ) - 1, int(round(v * (len(SEQ) - 1))))]


def figures(M, df, tab, figdir):
    plt = _setup()
    from matplotlib.colors import ListedColormap
    judges = M["judges"]
    main_j = "sonnet" if "sonnet" in judges else judges[0]

    # Fig 1 — detection heatmap (main judge): faults × protocols
    prots = [p for p in ["P1", "P2", "P3", "P4"] if f"{main_j}|{p}" in M["by"]]
    mat = np.array([[M["by"][f"{main_j}|{p}"]["detect_strict"][f]["rate"] for p in prots] for f in FAULTS])
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    ax.imshow(mat, cmap=ListedColormap(SEQ), vmin=0, vmax=1, aspect="auto")
    for i in range(len(FAULTS)):
        for k in range(len(prots)):
            v = mat[i, k]
            ax.text(k, i, f"{v*100:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if v >= 0.55 else INK, fontweight="bold" if v >= 0.8 else "normal")
    ax.set_xticks(range(len(prots)), [PLABEL[p] for p in prots], fontsize=8.5)
    ax.set_yticks(range(len(FAULTS)), [LABEL[f] for f in FAULTS], fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(prots), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(FAULTS), 1), minor=True)
    ax.grid(which="minor", color=SURF, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.xaxis.tick_top()
    fig.text(0.01, 0.01, "P1은 총점 3점 이하를 탐지로 간주하고, P2~P4는 해당 결함의 정답 기준을 부적합으로 판정한 비율로 산출함. 결함 유형별 20건 기준",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(figdir, "fig1_detection_heatmap.png"))
    plt.close(fig)

    # Fig 2 — holistic judge blind spots: P1 mean overall per variant
    if f"{main_j}|P1" in M["by"]:
        d = df[(df.judge == main_j) & (df.protocol == "P1") & df.variant.isin(CORE)]
        means, los, his = [], [], []
        for v in CORE:
            sub = d[d.variant == v]
            means.append(sub.overall.mean())
            lo, hi = boot_mean(sub.assign(ov=sub.overall.astype(float)), "ov")
            los.append(lo)
            his.append(hi)
        order = CORE
        fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
        y = np.arange(len(order))[::-1]
        blind = [v for v in FAULTS if means[order.index(v)] >= means[0] - 0.3]
        cols = [INK2 if v == "clean" else (S2 if v in blind else S1) for v in order]
        ax.barh(y, means, height=0.55, color=cols)
        ax.errorbar(means, y, xerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                    fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        for yi, m, hi_ in zip(y, means, his):
            ax.text(max(m, hi_) + 0.06, yi, f"{m:.2f}", va="center", fontsize=8, color=INK)
        if blind:
            ax.text(1.05, -0.95, "주황색은 정상 답변과의 점수 차이가 0.3점 미만인 결함 유형", fontsize=7.5, color=INK2)
        ax.set_yticks(y, [LABEL[v] for v in order], fontsize=8.5)
        ax.set_xlim(1, 5.55)
        ax.set_ylim(-1.3, len(order) - 0.4)
        ax.set_xlabel("평균 총점, 오차 막대는 시나리오 단위 부트스트랩 95% 신뢰구간", fontsize=8)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axvline(means[0], color=INK2, linewidth=1)
        ax.tick_params(length=0)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "fig2_holistic_scores.png"))
        plt.close(fig)

    # Fig 4 — root-cause confusion (main judge, P4)
    key = f"{main_j}|P4"
    if key in M["by"]:
        cm = pd.DataFrame(M["by"][key]["root_confusion"]).reindex(index=ROOTS, columns=ROOTS).fillna(0)
        rown = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
        fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=200)
        from matplotlib.colors import to_rgb
        rgb = np.array([[to_rgb("#f1f0ec") if cm.values[i, k] == 0 else to_rgb(_seq_color(rown.values[i, k]))
                         for k in range(len(ROOTS))] for i in range(len(ROOTS))])
        ax.imshow(rgb, aspect="auto")
        for i in range(len(ROOTS)):
            for k in range(len(ROOTS)):
                n = int(cm.values[i, k])
                if n:
                    v = rown.values[i, k]
                    ax.text(k, i, f"{n}", ha="center", va="center", fontsize=8.5,
                            color="white" if v >= 0.55 else INK)
        ax.set_xticks(range(len(ROOTS)), [RLABEL[r] for r in ROOTS], fontsize=8)
        ax.set_yticks(range(len(ROOTS)), [RLABEL[r] for r in ROOTS], fontsize=8)
        ax.set_xlabel("평가 모델이 지목한 원인", fontsize=8)
        ax.set_ylabel("실제 원인", fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(ROOTS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(ROOTS), 1), minor=True)
        ax.grid(which="minor", color=SURF, linewidth=2)
        ax.tick_params(which="minor", length=0)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "fig4_rootcause.png"))
        plt.close(fig)

    # Fig 5 — judge model comparison on P4 (and P1 AUROC) if two judges
    if len(judges) >= 2 and all(f"{j}|P4" in M["by"] for j in judges[:2]):
        j_order = ["sonnet", "haiku"] if set(["sonnet", "haiku"]) <= set(judges) else judges[:2]
        fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=200)
        x = np.arange(len(FAULTS))
        w = 0.36
        for k, (jj, col) in enumerate(zip(j_order, [S1, S2])):
            vals = [M["by"][f"{jj}|P4"]["detect_strict"][f]["rate"] for f in FAULTS]
            off = (-w / 2 - 0.01) if k == 0 else (w / 2 + 0.01)
            ax.bar(x + off, vals, w, color=col, label={"sonnet": "Sonnet", "haiku": "Haiku"}.get(jj, jj))
            for xi, v in zip(x, vals):
                ax.text(xi + off, v + 0.02, f"{v*100:.0f}", ha="center", fontsize=6.8, color=INK)
        short = {"A1": "A1 환각", "A2": "A2 누락", "A3": "A3 보장", "A4": "A4 추천", "A5": "A5 맥락",
                 "A6": "A6 인용", "T1": "T1 검색", "T2": "T2 도구"}
        ax.set_xticks(x, [short[f] for f in FAULTS], fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"])
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.16), ncol=2)
        ax.tick_params(length=0)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "fig5_judges_p4.png"))
        plt.close(fig)


def print_summary(M):
    for key, r in M["by"].items():
        print(f"\n== {key} (n={r['n']})  clean FPR={r['clean_fpr']['rate']:.2f}  clean mean={r['clean_overall_mean']:.2f}")
        print("   strict: " + "  ".join(f"{f}={r['detect_strict'][f]['rate']*100:.0f}" for f in FAULTS))
        print("   auroc : " + "  ".join(f"{f}={(r['auroc_vs_clean'][f] or 0):.2f}" for f in FAULTS))
        if "critical_recall" in r:
            print(f"   critical recall={r['critical_recall']['rate']:.2f}  crit-FPR(noncrit)={r['critical_fpr_on_noncritical']['rate']:.2f}  na={r['na_rate']}")
        if "root_acc_by_variant" in r:
            print("   root acc: " + "  ".join(f"{v}={(r['root_acc_by_variant'][v] or {}).get('rate', float('nan')):.2f}" for v in CORE))
        b = r["bias"]
        def fmt(x):
            return "NA" if not x else f"{x['base_mean']:.2f}->{x['var_mean']:.2f} (Δ{x['diff']:+.2f} [{x['lo']:+.2f},{x['hi']:+.2f}])"
        print(f"   disclaimer detect {fmt(b['disclaimer_detect'])} | overall {fmt(b['disclaimer_overall'])}")
        print(f"   padding clean overall {fmt(b['padding_clean_overall'])} | A1 detect {fmt(b['padding_fault_detect'])}")
    for k, ag in M.get("agreement", {}).items():
        print(f"\n== agreement {k}: " + "  ".join(f"{kk}={vv:.2f}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in ag.items()))


if __name__ == "__main__":
    main()
