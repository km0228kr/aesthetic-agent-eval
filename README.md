# aesthetic-agent-eval
### 미용의료 AI 상담 에이전트 평가 체계 + LLM-as-a-Judge 메타 평가 (결함 주입 기반)

미용의료 정보 앱의 AI 상담 에이전트('언니 AI' 유형)를 가정하고,
**① 평가 기준(정확성·안전성·근거의 적절성·사용자 맥락 반영)을 루브릭과 라벨링 가이드로 구체화하고,
② 정답을 아는 결함을 일부러 주입한 평가셋으로 LLM judge 자체가 얼마나 믿을 만한지 측정**한 프로젝트입니다.

> 모든 시나리오·문서·답변은 평가 연구용으로 만든 **합성 데이터**입니다. 실제 서비스 데이터나 실제 병원 정보는 쓰지 않았고, 의학적 조언이 아닙니다.

## 결과 요약 (Sonnet judge, 결함 유형별 n=20)
| 결함 | P1 총점 | P2 +루브릭 | P3 +근거·프로필 | P4 +trace | P5 개선판 | Haiku P4 |
|---|---|---|---|---|---|---|
| A1 사실 환각 | 65% | 85% | 100% | 100% | 95% | 85% |
| A2 안전정보 누락 | 0% | 0% | 5% | 5% | 35% | 25% |
| A3 효과 보장·단정 | 100% | 100% | 100% | 100% | 100% | 100% |
| A4 특정 병원 단정 추천 | 100% | 100% | 100% | 100% | 100% | 100% |
| A5 사용자 맥락 무시 | 85% | 5% | 85% | 85% | 85% | 85% |
| A6 인용 불일치 | 0% | 0% | 80% | 85% | 90% | 60% |
| T1 검색 실패 | 0% | 15% | 75% | 100% | 100% | 100% |
| T2 도구 미호출 | 95% | 25% | 80% | 85% | 80% | 85% |
| 정상 답변 오탐 | 0% | 0% | 0% | 0% | 0% | 0% |

P1은 총점 3점 이하를 탐지로, P2–P5는 해당 결함의 정답 축을 fail로 판정한 비율. 95% CI는 `results/metrics.json`(시나리오 단위 클러스터 부트스트랩).

1. **보여주는 정보가 탐지율을 결정한다** — 검색 실패(T1) 탐지 P1 0% → P4 100%, 인용 불일치(A6) P2 0% → P4 85%.
2. **가장 큰 사각지대는 '누락'** — 안전정보 누락(A2)은 P1–P4에서 0–5%. 응급·경고 신호 누락 7건 중 Sonnet P4 1건, Haiku P4 4건 탐지.
3. **trace가 원인을 알려준다** — 답변이 동일한 T2(도구 미호출)와 A5(맥락 무시)를 P4가 각각 80%, 75% 정확도로 구분.
4. **실패 분석 → judge 개선(P5) → 홀드아웃 재평가** — 응급 신호 누락 탐지 개발셋 1/7→6/7, 홀드아웃 1/8→7/8 (합산 정확 McNemar p=0.001). 근거 결함 탐지·정상 오탐은 악화 없음. 군더더기 오탐은 개발셋 9/20→0/20(홀드아웃은 P4도 0/8이라 비교 불가).
5. **judge 모델 차이** — Sonnet–Haiku κ: 안전성 0.90, 근거 0.70, 정확성 0.68.
6. **편향 탐침** — 면책 문구 한 줄로 탐지율이 떨어지지 않음(노골적 결함에서만 검증). 군더더기는 오히려 총점을 약 1점 낮춤.


## 왜 결함 주입인가
LLM judge를 쓰려면 judge의 판정이 맞는지부터 확인해야 합니다. 사람 라벨은 비싸고 평가자끼리도 갈립니다.
그래서 **깨끗한 기준 답변에 결함을 하나씩 정확히 주입**해, 무엇이 틀렸는지 설계상 알고 있는 평가셋을 만들었습니다.
judge가 그 결함을 잡는지 보면 judge의 탐지율·오탐률·편향을 사람 라벨 없이도 1차로 측정할 수 있습니다.
(실제 트래픽에서 자연 발생한 실패는 사람 라벨링이 필요합니다 → `annotation/`.)

## 구성
| 단계 | 내용 |
|---|---|
| 시나리오 20개 | 시술 정보·금기·회복·시술 후 증상(응급 포함)·가격·병원 선택·비교. 각 시나리오에 사용자 프로필(맥락 요인), 근거 문서 3개, 무관 문서 2개, 기준 답변, 실행 기록 |
| 결함 8종 | A1 사실 환각 · A2 안전정보 누락 · A3 효과 보장·단정 · A4 근거 없는 특정 병원 단정 추천·예약 유도 · A5 맥락 무시 · A6 인용 불일치 · T1 검색 실패 · T2 도구 미호출 |
| 편향 탐침 3종 | B1 면책 문구 세탁(결함 답변 + "전문의와 상담하세요") · B2 정상 답변 + 군더더기 · B3 환각 답변 + 군더더기 |
| 평가 케이스 | 개발 20 × 12 = **240개** + 홀드아웃 8 × 6 = **48개** (모두 설계상 정답 보유) |
| Judge 프로토콜 | P1 총점만 → P2 +도메인 루브릭 → P3 +근거 문서·사용자 정보 → P4 +실행 기록(trace)·원인 귀속 → P5 실패 분석 기반 개선 |
| Judge 모델 | Claude Sonnet (P1–P5), Claude Haiku (P1·P4) |

## 폴더
```
data/items.jsonl            평가 케이스 240개(정답 포함) · data/items_blind.jsonl  judge용(정답 제외)
data/scenarios/S01-S20.json 시나리오 원본(결함 정의 포함) · data/holdout_scenarios/H01-H08.json 홀드아웃
data/holdout_items.jsonl     홀드아웃 48건(clean·A1·A2·A6·T1·B2)
eval/                       결함 주입(build_items) · 렌더링(render) · 배치(make_batches) · 수집(collect_outputs)
eval/judge_prompts/         P1–P5 judge 프롬프트 원문
eval/run_judge_api.py       API로 항목별 독립 호출 재현(OpenAI 호환 · Anthropic · mock)
runs/ · runs_holdout/        judge 입력 배치 · judge 원출력
results/                    judgments.jsonl · metrics.json · tables/
analysis/analyze.py         메타 평가 지표 · 부트스트랩 CI · 그림
analysis/improvement.py     P4→P5 개선 효과(개발셋·홀드아웃)
annotation/                 라벨링 가이드 · 사람 라벨링 도구(HTML) · 일치도 계산
report/                     포트폴리오 PDF 생성
```

## 재현
```bash
git clone https://github.com/km0228kr/aesthetic-agent-eval.git && cd aesthetic-agent-eval
python3 eval/build_items.py --scenarios data/scenarios --out data/items.jsonl
python3 eval/make_batches.py --items data/items.jsonl --outdir runs/inputs --seed 7
# (선택) 다른 judge로 재실행: 항목별 독립 호출
python3 eval/run_judge_api.py --backend openai --model gpt-4o-mini --judge-name gpt4omini --protocols P4 P5
python3 eval/collect_outputs.py --runs runs --out results/judgments.jsonl
python3 analysis/analyze.py
python3 eval/collect_outputs.py --runs runs_holdout --out results/holdout_judgments.jsonl
python3 analysis/improvement.py
python3 analysis/critical_omission_test.py && python3 analysis/noise_check.py
python3 report/build_report.py   # 포트폴리오 PDF
# 사람 라벨링(60건): annotation/labeling_tool.html을 브라우저로 열어 라벨 → CSV를 annotation/에 저장
python3 annotation/compute_agreement.py
```
필요 패키지: numpy, pandas, scipy, scikit-learn, matplotlib (보고서 PDF는 playwright).

## 한계
- 합성 데이터(시나리오·문서·답변 모두 연구용). 실제 트래픽의 실패는 더 미묘할 수 있다.
- 답변 작성 보조와 judge가 모두 Claude 계열 → 다른 계열 judge 추가가 다음 단계.
- 판정은 Claude Code 에이전트 환경에서 Sonnet·Haiku 하위 에이전트로 20건(홀드아웃 8건)씩 배치 수행했고, 정확한 모델 버전은 기록하지 못했다. 같은 시나리오의 변형은 한 배치에 넣지 않아 대조를 막았다. 항목별 독립 호출로 다시 돌리는 러너(`run_judge_api.py`)는 mock으로만 검증했다.
- 같은 입력(P1·P2의 A5·T2 쌍)에서도 판정이 갈렸다: Sonnet 총점 9/20쌍, 탐지 여부 P1 2/20·P2 6/20쌍, 안전성 0/20쌍(`analysis/noise_check.py`). 결함 유형별 1–2건 차이는 노이즈로 봐야 한다.
- 홀드아웃 A2 8건은 설계상 모두 응급·경고 신호형이라, 일반 부작용 누락(개발셋 13건 중 P5도 1건만 탐지)은 홀드아웃으로 검증하지 않았다. P5의 변경별 기여는 분리 실험(ablation)으로 나누지 않았다.
- 표본: 결함 유형별 n=20, 홀드아웃 n=8 → 방향성 확인 수준.
- 사람 라벨 비교는 도구·가이드까지 준비(`annotation/`), 2인 독립 라벨링이 다음 작업.
- 안전성·배포 차단 기준은 평가 설계용 보수적 기준이며 법률 자문이 아니다.


도구: Python(pandas, scikit-learn, matplotlib), Claude(합성 데이터 작성 보조 · judge 모델).
