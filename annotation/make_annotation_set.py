#!/usr/bin/env python3
"""Draw a stratified 60-item sample for human labeling and build a self-contained labeling tool.

Writes
  annotation/sample_items.jsonl      blinded items (no variant / ground truth)
  annotation/labeling_tool.html      open in a browser; label; "CSV 내보내기" -> human_labels_<name>.csv

Usage: python3 annotation/make_annotation_set.py [--items data/items.jsonl] [--seed 11]
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
from render import render_item  # noqa: E402

QUOTA = {"clean": 8, "A1": 5, "A2": 5, "A3": 5, "A4": 5, "A5": 5, "A6": 5, "T1": 5, "T2": 5,
         "B1": 4, "B2": 4, "B3": 4}

HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>답변 라벨링 도구</title>
<style>
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e1;--acc:#2a78d6;--card:#ffffff}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--acc:#3987e5;--card:#232322}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 "Noto Sans KR","Apple SD Gothic Neo",system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
main{max-width:980px;margin:0 auto;padding:16px}pre{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;font:14px/1.65 inherit}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:14px 0}
fieldset{border:1px solid var(--line);border-radius:8px;padding:8px 10px}legend{color:var(--ink2);font-size:13px}
label{margin-right:10px;white-space:nowrap}button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}textarea{width:100%;font:inherit;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px}
.muted{color:var(--ink2);font-size:13px}input[type=text]{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}
</style></head><body>
<header><strong>답변 라벨링</strong><span class="muted" id="pos"></span>
<button id="prev">◀ 이전</button><button id="next">다음 ▶</button>
<span class="muted">평가자 이름</span><input type="text" id="who" size="8" placeholder="예: kkm">
<button class="primary" id="export">CSV 내보내기</button><span class="muted" id="done"></span></header>
<main><p class="muted">라벨링 가이드(labeling_guide.md)를 먼저 읽으세요. 축 판정을 먼저 하고 점수는 마지막에 매깁니다. 입력은 이 브라우저에 자동 저장됩니다.</p>
<pre id="item"></pre>
<div class="grid" id="form"></div>
<textarea id="notes" rows="2" placeholder="메모(헷갈린 이유 등)"></textarea></main>
<script>
const ITEMS = __ITEMS__;
const FIELDS = [["accuracy","정확성",["pass","fail","na"]],["safety","안전성",["pass","fail","na"]],
 ["grounding","근거 적절성",["pass","fail","na"]],["context","맥락 반영",["pass","fail","na"]],
 ["critical","배포 차단",["false","true"]],["overall","종합(1-5)",["1","2","3","4","5"]],
 ["root_cause","원인",["none","retrieval_failure","tool_not_called","user_context_ignored","generation_error"]]];
let i = 0, S = {};
try { S = JSON.parse(localStorage.getItem("aae_labels") || "{}"); } catch (e) { S = {}; }
try { document.getElementById("who").value = localStorage.getItem("aae_who") || ""; } catch (e) {}
function save(){ try { localStorage.setItem("aae_labels", JSON.stringify(S)); } catch (e) {} }
function render(){
  const it = ITEMS[i]; document.getElementById("item").textContent = it.text;
  document.getElementById("pos").textContent = `${i+1} / ${ITEMS.length}`;
  const cur = S[it.item_id] || {}; const f = document.getElementById("form"); f.innerHTML = "";
  for (const [k, name, opts] of FIELDS){
    const fs = document.createElement("fieldset"); fs.innerHTML = `<legend>${name}</legend>`;
    for (const o of opts){ const id = `${k}_${o}`; const l = document.createElement("label");
      l.innerHTML = `<input type="radio" name="${k}" value="${o}" ${cur[k]===o?"checked":""}> ${o}`;
      l.querySelector("input").onchange = e => { (S[it.item_id] ||= {})[k] = e.target.value; save(); count(); };
      fs.appendChild(l); }
    f.appendChild(fs); }
  const n = document.getElementById("notes"); n.value = cur.notes || "";
  n.oninput = e => { (S[it.item_id] ||= {}).notes = e.target.value; save(); };
  count(); }
function count(){ const c = ITEMS.filter(it => S[it.item_id] && FIELDS.every(([k]) => S[it.item_id][k] !== undefined)).length;
  document.getElementById("done").textContent = `완료 ${c}/${ITEMS.length}`; }
document.getElementById("prev").onclick = () => { if (i>0){ i--; render(); } };
document.getElementById("next").onclick = () => { if (i<ITEMS.length-1){ i++; render(); } };
document.getElementById("who").oninput = e => { try { localStorage.setItem("aae_who", e.target.value); } catch (x) {} };
document.getElementById("export").onclick = () => {
  const who = document.getElementById("who").value.trim() || "annotator";
  const cols = ["item_id", ...FIELDS.map(f => f[0]), "notes"];
  const esc = v => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const rows = [cols.join(",")].concat(ITEMS.map(it => cols.map(c => c === "item_id" ? it.item_id : esc((S[it.item_id]||{})[c])).join(",")));
  const blob = new Blob(["﻿" + rows.join("\n")], {type: "text/csv"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `human_labels_${who}.csv`; a.click(); };
render();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=os.path.join(HERE, "..", "data", "items.jsonl"))
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")]
    picked = []
    for v, k in QUOTA.items():
        pool = [it for it in items if it["variant"] == v]
        picked += rng.sample(pool, k)
    rng.shuffle(picked)
    with open(os.path.join(HERE, "sample_items.jsonl"), "w", encoding="utf-8") as fh:
        for it in picked:
            fh.write(json.dumps({"item_id": it["item_id"], "text": render_item(it, "P4")}, ensure_ascii=False) + "\n")
    payload = json.dumps([{"item_id": it["item_id"], "text": render_item(it, "P4")} for it in picked], ensure_ascii=False)
    with open(os.path.join(HERE, "labeling_tool.html"), "w", encoding="utf-8") as fh:
        fh.write(HTML.replace("__ITEMS__", payload.replace("</", "<\\/")))
    print(f"sampled {len(picked)} items -> annotation/sample_items.jsonl, annotation/labeling_tool.html")


if __name__ == "__main__":
    main()
