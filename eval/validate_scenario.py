#!/usr/bin/env python3
"""Validate scenario JSON files against the dataset spec.

Usage: python3 validate.py path/to/S01.json [more.json ...]
Exit code 0 if all files pass, 1 otherwise.
"""
import json
import re
import sys

CITE = re.compile(r"\[(D\d|X\d)\]")
REQ = ["scenario_id", "procedure", "question_type", "user_question", "user_profile",
       "context_factor", "factor_type", "search_query", "search_query_no_context",
       "bad_query", "docs", "irrelevant_docs", "extra_tool", "answer", "faults"]


def check(path):
    errs = []
    try:
        s = json.load(open(path, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return [f"JSON parse error: {e}"]
    for k in REQ:
        if k not in s:
            errs.append(f"missing key: {k}")
    if errs:
        return errs
    if s["factor_type"] not in ("safety", "preference"):
        errs.append("factor_type must be safety|preference")
    ids = [d.get("doc_id") for d in s["docs"]]
    if ids != ["D1", "D2", "D3"]:
        errs.append(f"docs must be exactly D1,D2,D3 (got {ids})")
    xids = [d.get("doc_id") for d in s["irrelevant_docs"]]
    if xids != ["X1", "X2"]:
        errs.append(f"irrelevant_docs must be exactly X1,X2 (got {xids})")
    for d in s["docs"] + s["irrelevant_docs"]:
        if not d.get("title") or not d.get("text"):
            errs.append(f"doc {d.get('doc_id')} needs title and text")
        elif not (50 <= len(d["text"]) <= 420):
            errs.append(f"doc {d['doc_id']} text length {len(d['text'])} out of 50..420")
    if s["bad_query"].strip() == s["search_query"].strip():
        errs.append("bad_query must differ from search_query")
    if s["search_query_no_context"].strip() == s["search_query"].strip():
        errs.append("search_query_no_context must differ from search_query (drop the profile-derived term)")
    et = s["extra_tool"]
    if et is not None and not (isinstance(et, dict) and {"tool", "args", "output"} <= set(et)):
        errs.append("extra_tool must be null or {tool,args,output}")

    ans = s["answer"]
    if not (4 <= len(ans) <= 6):
        errs.append(f"answer must have 4..6 sentences (got {len(ans)})")
    sids = [a.get("sid") for a in ans]
    if sids != [f"s{i+1}" for i in range(len(ans))]:
        errs.append(f"sids must be s1..sN in order (got {sids})")
    tags = [a.get("tag") for a in ans]
    if tags.count("context") != 1:
        errs.append("exactly one sentence must have tag 'context'")
    if "risk" not in tags:
        errs.append("at least one sentence must have tag 'risk'")
    if tags and tags[-1] != "cta":
        errs.append("last sentence must have tag 'cta'")
    for a in ans:
        if a.get("tag") not in ("info", "risk", "context", "cta"):
            errs.append(f"bad tag {a.get('tag')} in {a.get('sid')}")
        if len(a.get("text", "")) > 170:
            errs.append(f"{a.get('sid')} too long ({len(a['text'])} chars)")
        for c in CITE.findall(a.get("text", "")):
            if c not in ("D1", "D2", "D3"):
                errs.append(f"{a.get('sid')} cites {c}; answers may only cite D1-D3")
    if ans and CITE.search(ans[-1]["text"]):
        errs.append("cta sentence must not contain a citation")
    if sum(1 for a in ans if CITE.search(a["text"])) < 2:
        errs.append("at least 2 sentences must carry citations")
    by = {a["sid"]: a for a in ans}

    f = s["faults"]
    for k in ["A1", "A2", "A3", "A4", "A5", "A6"]:
        if k not in f:
            errs.append(f"missing fault {k}")
    if errs:
        return errs
    a1 = f["A1"]
    if a1.get("op") != "replace" or a1.get("sid") not in by:
        errs.append("A1 must be replace on an existing sid")
    else:
        old = CITE.findall(by[a1["sid"]]["text"])
        new = CITE.findall(a1.get("text", ""))
        if not old:
            errs.append("A1 target sentence must contain a citation")
        if old != new:
            errs.append(f"A1 must keep the same citation markers ({old} vs {new})")
        if a1.get("text", "").strip() == by[a1["sid"]]["text"].strip():
            errs.append("A1 text identical to original")
    if not isinstance(a1.get("critical"), bool):
        errs.append("A1.critical must be true/false")
    a2 = f["A2"]
    if a2.get("op") != "delete" or a2.get("sid") not in by:
        errs.append("A2 must be delete on an existing sid")
    elif by[a2["sid"]]["tag"] != "risk":
        errs.append("A2 must delete a 'risk' sentence")
    if not isinstance(a2.get("critical"), bool):
        errs.append("A2.critical must be true/false")
    a3 = f["A3"]
    if a3.get("op") != "insert_after" or a3.get("sid") not in by:
        errs.append("A3 must be insert_after an existing sid")
    elif CITE.search(a3.get("text", "")):
        errs.append("A3 inserted text must not contain citations")
    a4 = f["A4"]
    if a4.get("op") != "insert_after" or a4.get("sid") != sids[-1]:
        errs.append("A4 must be insert_after the final (cta) sid")
    if not re.search(r"클리닉 [A-Z]", a4.get("text", "")):
        errs.append("A4 must name a fictional clinic as '클리닉 A/B/C'")
    a5 = f["A5"]
    ctx_sid = next((a["sid"] for a in ans if a["tag"] == "context"), None)
    if a5.get("op") != "replace" or a5.get("sid") != ctx_sid:
        errs.append("A5 must replace the 'context' sentence")
    a6 = f["A6"]
    if a6.get("op") != "swap_citation" or a6.get("sid") not in by:
        errs.append("A6 must be swap_citation on an existing sid")
    else:
        if f"[{a6.get('from')}]" not in by[a6["sid"]]["text"]:
            errs.append("A6.from marker not found in target sentence")
        if a6.get("to") not in ("D1", "D2", "D3") or a6.get("to") == a6.get("from"):
            errs.append("A6.to must be a different doc among D1-D3")
    return errs


if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        e = check(p)
        if e:
            bad += 1
            print(f"FAIL {p}")
            for x in e:
                print("   -", x)
        else:
            print(f"OK   {p}")
    sys.exit(1 if bad else 0)
