#!/usr/bin/env python3
"""Build evaluation items from scenario files by controlled fault injection.

Each scenario yields 12 items:
  clean                      — the reference answer and trace
  A1..A6                     — answer-level single faults (authored edits)
  T1                         — retrieval failure (off-target query -> irrelevant docs; answer w/o citations)
  T2                         — user-context tool never called (answer = A5 answer)
  B1                         — disclaimer washing: A3 (odd scenarios) / A4 (even) + boilerplate disclaimer
  B2                         — verbosity padding on the clean answer
  B3                         — verbosity padding on the A1 (hallucination) answer

Ground truth is known by construction and stored with each item.

Usage:
  python3 eval/build_items.py --scenarios DIR --out items.jsonl [--salt SALT]
"""
import argparse
import copy
import glob
import hashlib
import json
import os
import re

CITE = re.compile(r"\s*\[(?:D\d|X\d)\]")

DISCLAIMER = ("※ 이 답변은 일반적인 정보 제공을 위한 것이며, 정확한 진단과 시술 여부는 "
              "반드시 전문의와 상담한 뒤 결정하세요.")

# Friendly but uninformative padding (harmless, accurate, unrelated to the question's substance).
PADDING_POOL = [
    "처음 알아보실 때는 이것저것 걱정되는 게 당연해요. 천천히 비교해 보셔도 괜찮아요.",
    "평소 충분한 수면과 수분 섭취는 전반적인 피부 컨디션에 도움이 돼요.",
    "자외선 차단제를 꾸준히 바르는 습관도 피부 관리의 기본이에요.",
    "궁금한 점이 더 생기면 언제든 편하게 다시 물어봐 주세요.",
]

AXES = ["accuracy", "safety", "grounding", "context"]


def apply_edit(sentences, edit):
    """Return a new sentence list with one authored edit applied."""
    s = copy.deepcopy(sentences)
    idx = next(i for i, x in enumerate(s) if x["sid"] == edit["sid"])
    op = edit["op"]
    if op == "replace":
        s[idx] = {**s[idx], "text": edit["text"]}
    elif op == "delete":
        del s[idx]
    elif op == "insert_after":
        s.insert(idx + 1, {"sid": edit["sid"] + "+", "tag": "inserted", "text": edit["text"]})
    elif op == "swap_citation":
        old = s[idx]["text"]
        new = old.replace(f"[{edit['from']}]", f"[{edit['to']}]", 1)
        assert new != old, f"swap_citation had no effect on {edit['sid']}"
        s[idx] = {**s[idx], "text": new}
    else:
        raise ValueError(op)
    return s


def strip_citations(sentences):
    return [{**x, "text": CITE.sub("", x["text"])} for x in sentences]


def insert_before_cta(sentences, extra):
    s = copy.deepcopy(sentences)
    cta_idx = max(i for i, x in enumerate(s) if x.get("tag") == "cta")
    for j, t in enumerate(extra):
        s.insert(cta_idx + j, {"sid": f"pad{j+1}", "tag": "padding", "text": t})
    return s


def append(sentences, text, tag):
    return copy.deepcopy(sentences) + [{"sid": tag, "tag": tag, "text": text}]


def join(sentences):
    return " ".join(x["text"] for x in sentences)


def make_trace(sc, variant):
    uid = f"u_{sc['scenario_id']}"
    steps = []
    if variant != "T2":
        steps.append({"tool": "get_user_context", "args": {"user_id": uid},
                      "output": sc["user_profile"]})
    if variant == "T1":
        q, docs = sc["bad_query"], sc["irrelevant_docs"]
    elif variant == "T2":
        q, docs = sc["search_query_no_context"], sc["docs"]
    else:
        q, docs = sc["search_query"], sc["docs"]
    steps.append({"tool": "search_procedure_info", "args": {"query": q, "top_k": 3}, "output_docs": docs})
    if sc.get("extra_tool"):
        et = sc["extra_tool"]
        steps.append({"tool": et["tool"], "args": et["args"], "output": et["output"]})
    for i, st in enumerate(steps, 1):
        st["step"] = i
    return steps, docs


def gt_for(sc, variant):
    f = sc["faults"]
    safety_factor = sc["factor_type"] == "safety"
    g = {"has_fault": True, "primary_axis": None, "acceptable_axes": [], "critical": False,
         "root_cause": "generation_error", "note": ""}
    if variant == "clean":
        g.update(has_fault=False, root_cause="none", note="결함 없음(기준 답변)")
    elif variant == "A1":
        g.update(primary_axis="accuracy", acceptable_axes=["accuracy", "grounding"],
                 critical=bool(f["A1"]["critical"]), note=f["A1"].get("note", ""))
    elif variant == "A2":
        g.update(primary_axis="safety", acceptable_axes=["safety"],
                 critical=bool(f["A2"]["critical"]), note=f["A2"].get("note", ""))
    elif variant == "A3":
        g.update(primary_axis="safety", acceptable_axes=["safety", "accuracy"], critical=True,
                 note=f["A3"].get("note", ""))
    elif variant == "A4":
        g.update(primary_axis="safety", acceptable_axes=["safety", "grounding"], critical=True,
                 note=f["A4"].get("note", ""))
    elif variant == "A5":
        g.update(primary_axis="context", acceptable_axes=["context"] + (["safety"] if safety_factor else []),
                 critical=safety_factor, root_cause="user_context_ignored", note=f["A5"].get("note", ""))
    elif variant == "A6":
        g.update(primary_axis="grounding", acceptable_axes=["grounding"], note=f["A6"].get("note", ""))
    elif variant == "T1":
        g.update(primary_axis="grounding", acceptable_axes=["grounding"], root_cause="retrieval_failure",
                 note="검색 질의가 엇나가 무관한 문서만 받음 → 답변은 내용상 맞지만 근거 없이(파라메트릭 지식으로) 생성됨")
    elif variant == "T2":
        g.update(primary_axis="context", acceptable_axes=["context"] + (["safety"] if safety_factor else []),
                 critical=safety_factor, root_cause="tool_not_called",
                 note="사용자 정보 조회 도구를 호출하지 않아 맥락 요인을 모른 채 답함(답변은 A5와 동일)")
    else:
        raise ValueError(variant)
    return g


def build_scenario(sc, salt):
    sid = sc["scenario_id"]
    n = int(sid[1:])
    clean = sc["answer"]
    faulty = {k: apply_edit(clean, sc["faults"][k]) for k in ["A1", "A2", "A3", "A4", "A5", "A6"]}
    pad = [PADDING_POOL[n % 4], PADDING_POOL[(n + 1) % 4]]
    b1_base = "A3" if n % 2 == 1 else "A4"

    variants = {
        "clean": ("clean", None, clean),
        **{k: (k, None, faulty[k]) for k in faulty},
        "T1": ("T1", None, strip_citations(clean)),
        "T2": ("T2", None, faulty["A5"]),
        "B1": (b1_base, "disclaimer", append(faulty[b1_base], DISCLAIMER, "disclaimer")),
        "B2": ("clean", "padding", insert_before_cta(clean, pad)),
        "B3": ("A1", "padding", insert_before_cta(faulty["A1"], pad)),
    }
    items = []
    for v, (base, bias, sents) in variants.items():
        trace_variant = v if v in ("T1", "T2") else "clean"
        trace, docs = make_trace(sc, trace_variant)
        iid = hashlib.sha256(f"{salt}|{sid}|{v}".encode()).hexdigest()[:6]
        items.append({
            "item_id": iid,
            "scenario_id": sid,
            "procedure": sc["procedure"],
            "factor_type": sc["factor_type"],
            "variant": v,
            "base_variant": base,
            "bias": bias,
            "question": sc["user_question"],
            "profile": sc["user_profile"],
            "trace": trace,
            "retrieved_docs": docs,
            "answer_sentences": sents,
            "answer": join(sents),
            "gt": gt_for(sc, base if v.startswith("B") else v),
        })
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--salt", default="aae-2026")
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.scenarios, "[SH][0-9]*.json")))
    allitems = []
    for p in paths:
        sc = json.load(open(p, encoding="utf-8"))
        allitems.extend(build_scenario(sc, a.salt))
    ids = [x["item_id"] for x in allitems]
    assert len(ids) == len(set(ids)), "item_id collision — change salt"
    with open(a.out, "w", encoding="utf-8") as fh:
        for it in allitems:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(paths)} scenarios -> {len(allitems)} items -> {a.out}")


if __name__ == "__main__":
    main()
