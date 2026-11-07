#!/usr/bin/env python3
"""Human-vs-ground-truth, human-vs-human and human-vs-judge agreement on the 60-item sample.

Put one CSV per annotator in annotation/ named human_labels_<name>.csv (exported by labeling_tool.html).
Usage: python3 annotation/compute_agreement.py [--judge sonnet --protocol P4]
"""
import argparse
import glob
import itertools
import json
import os

import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
AXES = ["accuracy", "safety", "grounding", "context"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="sonnet")
    ap.add_argument("--protocol", default="P4")
    a = ap.parse_args()
    items = {json.loads(l)["item_id"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "items.jsonl"), encoding="utf-8")}
    files = sorted(glob.glob(os.path.join(HERE, "human_labels_*.csv")))
    if not files:
        raise SystemExit("no annotation/human_labels_*.csv found — label with labeling_tool.html first")
    H = {}
    for f in files:
        name = os.path.basename(f)[len("human_labels_"):-4]
        d = pd.read_csv(f, dtype=str).dropna(subset=["accuracy"])
        H[name] = d.set_index("item_id")
    J = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, "results", "judgments.jsonl"), encoding="utf-8")])
    J = J[(J.judge == a.judge) & (J.protocol == a.protocol)].set_index("item_id")
    out = {"annotators": list(H), "per_annotator": {}, "pairwise": {}, "vs_judge": {}}
    for name, d in H.items():
        rows = []
        for iid, r in d.iterrows():
            g = items[iid]["gt"]
            det = (r[g["primary_axis"]] == "fail") if g["has_fault"] else any(r[ax] == "fail" for ax in AXES)
            rows.append({"variant": items[iid]["variant"], "has_fault": g["has_fault"], "flag": det})
        t = pd.DataFrame(rows)
        out["per_annotator"][name] = {
            "n": int(len(t)), "clean_fpr": float(t[~t.has_fault].flag.mean()),
            "detect_by_variant": t[t.has_fault].groupby("variant").flag.mean().round(3).to_dict()}
        ids = d.index.intersection(J.index)
        out["vs_judge"][name] = {ax: float(cohen_kappa_score(d.loc[ids, ax], J.loc[ids, ax])) for ax in AXES}
    for (n1, d1), (n2, d2) in itertools.combinations(H.items(), 2):
        ids = d1.index.intersection(d2.index)
        out["pairwise"][f"{n1}~{n2}"] = {ax: float(cohen_kappa_score(d1.loc[ids, ax], d2.loc[ids, ax])) for ax in AXES}
    json.dump(out, open(os.path.join(ROOT, "results", "human_agreement.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
