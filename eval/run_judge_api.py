#!/usr/bin/env python3
"""Re-run the judges through an API, one independent call per item (stdlib only).

The judgments in this repo were produced with Claude Sonnet / Haiku judges run as sub-agents in an
agent harness (Claude Code), on batches of 20 items (8 for held-out); exact model versions were not
recorded (see README, "한계"). This runner is the replication path for other judges: each item is
judged in its own call, so there is no within-batch context effect. It has been tested with the mock
backend only; real endpoints (OpenAI-compatible servers such as vLLM/Ollama, Anthropic) are untested.

Backends
  openai     any OpenAI-compatible /v1/chat/completions endpoint (OpenAI, vLLM, Ollama, LM Studio, ...)
  anthropic  Anthropic Messages API
  mock       returns a fixed valid answer (pipeline test, no network)

Examples
  export OPENAI_API_KEY=...; python3 eval/run_judge_api.py --backend openai --model gpt-4o-mini \
      --judge-name gpt4omini --protocols P1 P4
  python3 eval/run_judge_api.py --backend openai --base-url http://localhost:11434/v1 \
      --model exaone3.5:7.8b --judge-name exaone --protocols P4
  python3 eval/run_judge_api.py --backend mock --judge-name mock --protocols P1 P2 P3 P4

Outputs land in runs/outputs/<judge-name>/<protocol>/batch_XX.jsonl, the same layout that
collect_outputs.py and analysis/analyze.py read.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import render_item  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def call_openai(base_url, model, system, user, key):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model, "temperature": 0,
                         "messages": [{"role": "system", "content": system},
                                      {"role": "user", "content": user}]}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def call_anthropic(model, system, user, key):
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps({"model": model, "max_tokens": 600, "temperature": 0, "system": system,
                         "messages": [{"role": "user", "content": user}]}).encode(),
        headers={"Content-Type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return "".join(b.get("text", "") for b in json.load(r)["content"])


def call_mock(protocol, item_id):
    rec = {"item_id": item_id, "overall": 4}
    if protocol == "P1":
        rec["reason"] = "mock"
    else:
        rec.update(accuracy="pass", safety="pass", grounding="pass", context="pass",
                   critical=False, issues="없음")
        if protocol in ("P4", "P5"):
            rec.update(root_cause="none", failing_step=0)
        if protocol == "P5":
            rec["missing_safety_info"] = []
    return json.dumps(rec, ensure_ascii=False)


def extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["openai", "anthropic", "mock"], required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--judge-name", required=True)
    ap.add_argument("--items", default=os.path.join(HERE, "..", "data", "items_blind.jsonl"))
    ap.add_argument("--runs", default=os.path.join(HERE, "..", "runs"))
    ap.add_argument("--protocols", nargs="+", default=["P1", "P2", "P3", "P4"])
    ap.add_argument("--sleep", type=float, default=0.0)
    a = ap.parse_args()

    items = {json.loads(l)["item_id"]: json.loads(l) for l in open(a.items, encoding="utf-8")}
    manifest = json.load(open(os.path.join(a.runs, "inputs", "manifest.json")))
    key = os.environ.get("ANTHROPIC_API_KEY" if a.backend == "anthropic" else "OPENAI_API_KEY", "")
    for p in a.protocols:
        system = open(os.path.join(HERE, "judge_prompts", f"{p}.md"), encoding="utf-8").read()
        system += "\n\n이번에는 항목이 1개입니다. JSON 한 줄만 출력하세요."
        outdir = os.path.join(a.runs, "outputs", a.judge_name, p)
        os.makedirs(outdir, exist_ok=True)
        for batch, ids in sorted(manifest.items()):
            path = os.path.join(outdir, f"{batch}.jsonl")
            done = set()
            if os.path.exists(path):
                done = {json.loads(l)["item_id"] for l in open(path, encoding="utf-8") if l.strip()}
            with open(path, "a", encoding="utf-8") as fh:
                for iid in ids:
                    if iid in done:
                        continue
                    user = render_item(items[iid], p)
                    for attempt in range(3):
                        try:
                            if a.backend == "mock":
                                txt = call_mock(p, iid)
                            elif a.backend == "openai":
                                txt = call_openai(a.base_url, a.model, system, user, key)
                            else:
                                txt = call_anthropic(a.model, system, user, key)
                            rec = extract_json(txt)
                            if rec is None:
                                raise ValueError("no JSON in response")
                            rec["item_id"] = iid
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            fh.flush()
                            break
                        except Exception as e:  # noqa: BLE001
                            print(f"[{p} {batch} {iid}] attempt {attempt + 1} failed: {e}", file=sys.stderr)
                            time.sleep(2 * (attempt + 1))
                    time.sleep(a.sleep)
            print(f"{a.judge_name} {p} {batch} done")


if __name__ == "__main__":
    main()
