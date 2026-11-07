#!/usr/bin/env python3
"""Collect and validate judge outputs.

Reads runs/outputs/<judge>/<protocol>/batch_XX.jsonl, checks every item in the batch manifest is
present exactly once with valid fields, and writes a tidy merged file.

Usage:
  python3 eval/collect_outputs.py --runs runs --out results/judgments.jsonl
Prints a per-(judge, protocol, batch) status table; exit code 1 if anything is missing/invalid.
"""
import argparse
import glob
import json
import os
import re

AXES = ["accuracy", "safety", "grounding", "context"]
ROOT = {"none", "retrieval_failure", "tool_not_called", "user_context_ignored", "generation_error"}


def parse_line(line):
    line = line.strip()
    if not line or line.startswith("```"):
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", line)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return "BAD"
        return "BAD"


def norm(rec, protocol):
    out = {"item_id": str(rec.get("item_id", "")).strip()}
    try:
        ov = int(rec.get("overall"))
    except (TypeError, ValueError):
        return None, "overall not int"
    if not 1 <= ov <= 5:
        return None, "overall out of range"
    out["overall"] = ov
    if protocol == "P1":
        out["reason"] = rec.get("reason", "")
        return out, None
    for ax in AXES:
        v = str(rec.get(ax, "")).strip().lower()
        if v not in ("pass", "fail", "na", "n/a"):
            return None, f"{ax}={rec.get(ax)!r}"
        out[ax] = "na" if v == "n/a" else v
    c = rec.get("critical")
    if isinstance(c, str):
        c = c.strip().lower() == "true"
    if not isinstance(c, bool):
        return None, "critical not bool"
    out["critical"] = c
    out["issues"] = rec.get("issues", "")
    if protocol == "P5":
        ms = rec.get("missing_safety_info", [])
        out["missing_safety_info"] = ms if isinstance(ms, list) else [str(ms)]
    if protocol in ("P4", "P5"):
        rc = str(rec.get("root_cause", "")).strip()
        if rc not in ROOT:
            return None, f"root_cause={rc!r}"
        out["root_cause"] = rc
        try:
            out["failing_step"] = int(rec.get("failing_step", 0))
        except (TypeError, ValueError):
            out["failing_step"] = None
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/judgments.jsonl")
    a = ap.parse_args()
    manifest = json.load(open(os.path.join(a.runs, "inputs", "manifest.json")))
    merged, problems = [], 0
    for jdir in sorted(glob.glob(os.path.join(a.runs, "outputs", "*"))):
        judge = os.path.basename(jdir)
        for pdir in sorted(glob.glob(os.path.join(jdir, "P*"))):
            protocol = os.path.basename(pdir)
            for batch, ids in sorted(manifest.items()):
                path = os.path.join(pdir, f"{batch}.jsonl")
                if not os.path.exists(path):
                    print(f"MISSING  {judge}/{protocol}/{batch}")
                    problems += 1
                    continue
                got, errs = {}, []
                for line in open(path, encoding="utf-8"):
                    rec = parse_line(line)
                    if rec is None:
                        continue
                    if rec == "BAD":
                        errs.append("unparseable line")
                        continue
                    r, e = norm(rec, protocol)
                    if e:
                        errs.append(f"{rec.get('item_id')}: {e}")
                        continue
                    if r["item_id"] in got:
                        errs.append(f"duplicate {r['item_id']}")
                    got[r["item_id"]] = r
                missing = [i for i in ids if i not in got]
                extra = [i for i in got if i not in ids]
                status = "OK" if not (missing or extra or errs) else "BAD"
                if status == "BAD":
                    problems += 1
                    print(f"BAD      {judge}/{protocol}/{batch}: missing={len(missing)} extra={len(extra)} errs={errs[:3]}")
                for i in ids:
                    if i in got:
                        merged.append({"judge": judge, "protocol": protocol, "batch": batch, **got[i]})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in merged:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"merged {len(merged)} judgments -> {a.out}; problems: {problems}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
