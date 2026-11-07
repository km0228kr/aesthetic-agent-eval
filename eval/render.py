#!/usr/bin/env python3
"""Render an evaluation item into the judge-visible text for each protocol.

Protocols form a cumulative "design ladder":
  P1  holistic 1-5 score, answer only                 (naive baseline)
  P2  domain rubric (4 axes + critical), answer only  (+ rubric)
  P3  P2 + user profile + documents the agent got     (+ evidence & user context)
  P4  P3 as a full tool-call trace + root-cause field (+ trace)

Only fields that a judge is allowed to see are rendered — never the variant name or ground truth.
"""
import json


def _docs_block(docs):
    return "\n".join(f"({d['doc_id']}) {d['title']}: {d['text']}" for d in docs)


def _evidence_block(item):
    parts = [_docs_block(item["retrieved_docs"])]
    for st in item["trace"]:
        if st["tool"] not in ("get_user_context", "search_procedure_info"):
            parts.append(f"[{st['tool']} 결과] {st['output']}")
    return "\n".join(parts)


def _trace_block(item):
    lines = []
    for st in item["trace"]:
        args = ", ".join(f'{k}={json.dumps(v, ensure_ascii=False)}' for k, v in st["args"].items())
        lines.append(f"step {st['step']}. {st['tool']}({args})")
        if "output_docs" in st:
            lines.append("  → 결과:")
            for d in st["output_docs"]:
                lines.append(f"    ({d['doc_id']}) {d['title']}: {d['text']}")
        else:
            lines.append(f"  → 결과: {st['output']}")
    lines.append("(최종) 답변 생성")
    return "\n".join(lines)


def render_item(item, protocol, k=None):
    head = f"### 항목 {k} — item_id: {item['item_id']}" if k is not None else f"### item_id: {item['item_id']}"
    q = f"[사용자 질문]\n{item['question']}"
    a = f"[AI 답변]\n{item['answer']}"
    if protocol in ("P1", "P2"):
        body = [q, a]
    elif protocol == "P3":
        body = [q, f"[사용자 정보 (플랫폼에 저장된 프로필)]\n{item['profile']}",
                f"[에이전트가 받은 근거 자료]\n{_evidence_block(item)}", a]
    elif protocol in ("P4", "P5"):
        body = [q, f"[사용자 정보 (플랫폼에 저장된 프로필)]\n{item['profile']}",
                f"[에이전트 실행 기록 (trace)]\n{_trace_block(item)}", a]
    else:
        raise ValueError(protocol)
    return head + "\n" + "\n\n".join(body)


def judge_view(item):
    """Strip everything a judge must not see (for exporting public, blinded item files)."""
    return {k: item[k] for k in ("item_id", "question", "profile", "trace", "retrieved_docs", "answer")}
