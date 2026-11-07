#!/usr/bin/env python3
"""Paired exact McNemar test: does P5 catch warning-sign omissions (critical A2) that P4 misses?

Pairs = the same critical-A2 item judged by Sonnet under P4 and P5, on dev and held-out sets.
Writes results/critical_omission_test.json.
"""
import json
import os

from scipy.stats import binomtest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SETS = {"dev": ("data/items.jsonl", "results/judgments.jsonl"),
        "holdout": ("data/holdout_items.jsonl", "results/holdout_judgments.jsonl")}


def pairs_for(ip, jp, judge="sonnet"):
    items = {json.loads(l)["item_id"]: json.loads(l) for l in open(os.path.join(ROOT, ip), encoding="utf-8")}
    d = {}
    for l in open(os.path.join(ROOT, jp), encoding="utf-8"):
        r = json.loads(l)
        it = items[r["item_id"]]
        if r["judge"] != judge or r["protocol"] not in ("P4", "P5"):
            continue
        if it["variant"] != "A2" or not it["gt"]["critical"]:
            continue
        d.setdefault(r["item_id"], {})[r["protocol"]] = r["safety"] == "fail"
    return [(v["P4"], v["P5"]) for v in d.values() if "P4" in v and "P5" in v]


def summarize(pairs):
    p5_only = sum(1 for a, b in pairs if b and not a)
    p4_only = sum(1 for a, b in pairs if a and not b)
    p = binomtest(p5_only, p5_only + p4_only, 0.5).pvalue if p5_only + p4_only else 1.0
    return {"n": len(pairs), "p4": sum(a for a, _ in pairs), "p5": sum(b for _, b in pairs),
            "p5_only": p5_only, "p4_only": p4_only, "mcnemar_exact_p": p}


def main():
    per, allp = {}, []
    for name, (ip, jp) in SETS.items():
        if os.path.exists(os.path.join(ROOT, jp)):
            pr = pairs_for(ip, jp)
            per[name] = summarize(pr)
            allp += pr
    out = {"per_set": per, "pooled": summarize(allp)}
    json.dump(out, open(os.path.join(ROOT, "results", "critical_omission_test.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
