# C4 사전등록 — 관계 다리(Relational Bridge) 확증 실험

작성: 2026-10-31, Confirm 사이트 실행 전 · 코드 해시는 summary.json의 `code_sha256_16`에 기록

## 배경 (Dev 10사이트, C3.2에서 이미 관찰 = 탐색 결과)
- 정답 노드 텍스트의 BM25 = 0 비율 86%
- 관계에 질문 단어가 있는 문항: Fusion − Shuffled Recall@30 +38.3%p (9/9 사이트)
- 관계에 질문 단어가 없는 문항: −3.7%p
- 게이트(need vs all) 효과 없음, CE의 관계 결합(binding) 증거 없음

## 데이터 분할 (재현 가능)
공식 TRAIN 사이트에서 Dev 10사이트를 제외하고 `sha256("c4-websrc-2026|site")` 순서로 60/20/20 분할 → `split.json`
- train: 관계 추출기 개선과 CE 학습에만 사용
- select: 탐색, 모델 선택
- confirm: **한 번만** 실행 (`CONFIRM_USED.lock`, `CONFIRM_CE_USED.lock`)

## 고정한 설계값 (Dev 결과를 보고 고르지 않음)
λ = 1.0 · 관계 = 행 헤더 + 열 헤더 + 가장 가까운 직전 레이블(12토큰) + 직전 제목, 같은 텍스트 중복 제거, 총 64토큰 · BM25 k1=1.5, b=0.75 · 후보 = 텍스트가 있는 모든 tid 노드 · 사이트당 최대 100문항(문서 순)

> 주의: 이 관계 추출기는 C3 코드와 독립적으로 작성됐어. 따라서 확증하는 대상은 C3의 특정 구현이 아니라 **현상** 자체야.

## 검색 가설 (c4_run.py)
| ID | 비교 | 지표 | 하위집단 | 기대 |
|---|---|---|---|---|
| **H1 (주)** | fusion_all − fusion_shuf | Recall@30 | 전체 | > 0 |
| H2 | fusion_all − fusion_qm | Recall@30 | Q-matched 구성 가능 | > 0 |
| H3a / H3b | fusion_all − fusion_shuf | Recall@30 | bridge / non-bridge | > 0 / ≤ 0 |
| H4 | fusion_all − bm25 | Recall@30 | 전체 | > 0 |
| H5 | fusion_need − fusion_all | Recall@30 | 전체 | ≈ 0 |
| H6a / H6b | tieonly − bm25 | Recall@30 / @100 | 전체 | > 0 |
| H7 | fusion_all − fusion_shuf | Hit@1 | 전체 | 보조 |

**H1 통과 조건:** 사이트 군집 부트스트랩 95% CI 하한 > 0 **그리고** 양(+)의 사이트 수 > 음(−)의 사이트 수.
나머지는 효과 크기와 CI를 그대로 보고하고, 다중 비교이므로 H1 외에는 "지지/비지지"로만 서술.

## CE 가설 (c4_ce.py, train 사이트로 학습 → select에서 선택 → confirm 1회)
| ID | 비교 | 의미 |
|---|---|---|
| T1 | rel/rel − text/text | 학습된 리랭커에 관계가 도움이 되는가 |
| **T2** | rel/rel − rel/shuf | **결합 검정:** 같은 모델에 틀린 관계를 주면 떨어지는가 |
| T3 | rel/rel − shuf/shuf | 같은 토큰 양에서 진짜 관계로 학습한 효과 |
| T4 | text/text − zs/text | 미세조정 자체의 효과 |

T2가 통과해야 "모델이 관계를 올바른 노드에 결합해서 쓴다"고 주장할 수 있음. T1만 통과하고 T2가 실패하면, 관계는 형식이나 길이 신호로만 쓰인 것.
하이퍼파라미터(lr 2e-5, 1 epoch, 음성 7개, max_len 192)는 select에서만 바꿀 수 있고, 바꾸면 기록한 뒤 confirm 실행.

## 탐색 분석 (dev / select 전용, confirm에서는 자동으로 꺼짐)
E1 관계 종류별 기여 · E2 다중 키(엔티티 × 속성) 주소 지정 · E3 관계까지 같은 잔여 모호성 · E4 답변 유형 신호 · E5 다리 경쟁 · E6 사이트 레이아웃 프로필
→ 여기서 나온 패턴은 **새 가설**로만 취급하고, 다음 확증 라운드에서 검증.
