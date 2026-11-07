#!/usr/bin/env python3
"""Assign items to judging batches and write one input file per (protocol, batch).

Batching rule: a batch never contains two variants of the same scenario, so a judge can never
contrast a faulty answer against its clean sibling. Each scenario has 12 variants, so there are
12 batches; each batch holds exactly one variant from every scenario (20 items), order shuffled.
The same assignment is reused for every protocol and judge model (paired comparisons).

Usage:
  python3 eval/make_batches.py --items items.jsonl --outdir runs/inputs --seed 7
"""
import argparse
import json
import os
import random
from collections import defaultdict

from render import render_item

PROTOCOLS = ["P1", "P2", "P3", "P4"]
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--protocols", nargs="+", default=PROTOCOLS)
    ap.add_argument("--variants", nargs="+", default=None, help="keep only these variants (e.g. held-out subset)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    items = [json.loads(line) for line in open(a.items, encoding="utf-8")]
    if a.variants:
        items = [it for it in items if it["variant"] in a.variants]
    by_sc = defaultdict(list)
    for it in items:
        by_sc[it["scenario_id"]].append(it)
    nb = max(len(v) for v in by_sc.values())
    batches = [[] for _ in range(nb)]
    for sc in sorted(by_sc):
        perm = list(range(nb))
        rng.shuffle(perm)
        for it, b in zip(sorted(by_sc[sc], key=lambda x: x["variant"]), perm):
            batches[b].append(it)
    for b in batches:
        rng.shuffle(b)
    manifest = {f"batch_{i:02d}": [it["item_id"] for it in b] for i, b in enumerate(batches)}
    os.makedirs(a.outdir, exist_ok=True)
    json.dump(manifest, open(os.path.join(a.outdir, "manifest.json"), "w"), indent=1)
    for p in a.protocols:
        header = open(os.path.join(HERE, "judge_prompts", f"{p}.md"), encoding="utf-8").read().strip()
        os.makedirs(os.path.join(a.outdir, p), exist_ok=True)
        for i, b in enumerate(batches):
            blocks = [render_item(it, p, k + 1) for k, it in enumerate(b)]
            text = (header + f"\n\n---\n\n# 평가할 항목 ({len(b)}개)\n\n" + "\n\n---\n\n".join(blocks)
                    + f"\n\n---\n\n위 {len(b)}개 항목을 모두, 각각 독립적으로 평가해 항목 순서대로 JSON 한 줄씩 출력하세요.\n")
            with open(os.path.join(a.outdir, p, f"batch_{i:02d}.md"), "w", encoding="utf-8") as fh:
                fh.write(text)
    sizes = [len(b) for b in batches]
    print(f"{len(items)} items -> {nb} batches (sizes {min(sizes)}–{max(sizes)}) × {len(a.protocols)} protocols")


if __name__ == "__main__":
    main()
