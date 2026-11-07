#!/usr/bin/env python3
"""Judge noise floor from identical inputs.

At P1 and P2 the judge sees only the question and the answer. A5 and T2 carry the exact same answer
text (T2 differs only in the hidden trace), so for each scenario the A5 and T2 items are identical
judge inputs placed in different batches. Any disagreement between their verdicts is pure judge noise
(sampling + batch context). Writes results/noise_check.json.
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    items = {json.loads(l)["item_id"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "items.jsonl"), encoding="utf-8")}
    J = [json.loads(l) for l in open(os.path.join(ROOT, "results", "judgments.jsonl"), encoding="utf-8")]
    out = {}
    for judge in ["sonnet", "haiku"]:
        for p in ["P1", "P2"]:
            rows = {}
            for r in J:
                it = items[r["item_id"]]
                if r["judge"] == judge and r["protocol"] == p and it["variant"] in ("A5", "T2"):
                    rows.setdefault(it["scenario_id"], {})[it["variant"]] = r
            pairs = [(v["A5"], v["T2"]) for v in rows.values() if "A5" in v and "T2" in v]
            if not pairs:
                continue
            if p == "P1":
                flag = lambda r: r["overall"] <= 3  # noqa: E731
            else:
                flag = lambda r: r["context"] == "fail"  # noqa: E731
            res = {"n_pairs": len(pairs),
                   "detect_disagree": sum(flag(a) != flag(b) for a, b in pairs),
                   "overall_exact_disagree": sum(a["overall"] != b["overall"] for a, b in pairs)}
            if p == "P2":
                res["any_axis_disagree"] = sum(any(a[x] != b[x] for x in ("accuracy", "safety", "grounding", "context"))
                                               for a, b in pairs)
                res["safety_disagree"] = sum(a["safety"] != b["safety"] for a, b in pairs)
            out[f"{judge}|{p}"] = res
    json.dump(out, open(os.path.join(ROOT, "results", "noise_check.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
