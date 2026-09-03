# Composable KV

**Research Question:**

Can independently prefilled KV caches be composed in cache-space to approximate the full-prefill next-token behavior well enough to yield a quality–latency Pareto improvement on defined regimes?

논문 프로젝트. 4~5개월 목표. 첫 arxiv + 워크숍.

## Project Status

- [x] Research question, hypotheses, controlled evaluation design
- [x] Project scaffold, dependencies, git init
- [x] MVE: extract, serialize, and re-inject KV caches (Qwen-2.5-0.5B, DynamicCache API)
- [x] Raw-concat behavioral baseline (next-token KL, top-k agreement)
- [x] Four-condition controlled benchmark (n=5 per condition, 20 examples total)
- [x] Layer-wise K/V divergence analysis (mid-layer dip identified)
- [x] RoPE-shifted concat baseline (Method 2 -- validated)
- [x] Statistical robustness (n=5 aggregate reveals condition-dependent M2 behavior)
- [x] Method 3: layer-selective RoPE-shift (negative result: M3 ~= M2, mid-layer dip is symptom not cause)
- [x] H3 POC: linear V corrector at L12 (marginal +2% cosine, 5-fold CV)
- [x] H3 end-to-end: L12 linear correction reduces downstream KL by 5-8% (in-sample W)
- [ ] Held-out H3 evaluation (leave-one-out downstream)
- [ ] H3 v2: MLP + V_A context (non-linear + context-conditional)
- [ ] H3 v3: all mid-layers (L10-L15) with corrected V
- [ ] Scale-up to Qwen-2.5-1.5B (baseline reliability)
- [ ] Cost/latency Pareto measurement
- [ ] Public technical report / arXiv submission

## MVE v1 결과 (Qwen-2.5-0.5B, naive concat)

| condition | KL(base‖comp) | top-1 | top-5 | top-10 |
|---|---|---|---|---|
| independent | 0.324 | 0.00 | 0.80 | 0.60 |
| referential | 0.336 | 0.00 | 0.60 | 0.70 |
| conflicting | 0.278 | 0.00 | 0.60 | 0.70 |
| cross_inferential | **0.410** | 0.00 | 0.80 | 0.80 |

**핵심 관찰:**

1. **모든 조건 top-1 = 0.00** — naive concat이 baseline과 같은 첫 토큰을 뱉지 않음 (4/4). 강한 실패 신호.
2. **cross_inferential 최고 KL (0.410)** — 리뷰어 예측 검증: cross-context dependency가 가장 어려움.
3. **Independent가 예상만큼 낮지 않음 (0.324 vs 0.336)** — "무관 컨텍스트는 조합 쉽다" 가설이 empirically 흔들림. full-prefill의 A→B attention coupling이 겉보기 무관 문서에서도 유의미함.
4. **정성적:** conflicting에서 composed가 A와 B를 **literal fusion** ("software engineer at Samsung Medical Center") — 완전한 실패도, 완전한 성공도 아닌 구조적 mixing. independent에서 loop 발생 (위치 혼동 신호).

**함의:** 다음 단계는 (a) RoPE-shifted concat으로 이 KL이 얼마나 줄어드는지, (b) layer-selective composition이 특정 층에서 이득을 주는지.

## Layer-wise 분석 결과 (analyze_layers.py)

B-alone K/V를 full-prefill의 B-portion과 layer별 cosine 유사도로 비교. V는 RoPE 없으므로 순수 cross-context 효과 측정.

**평균 유사도:**

| condition | K mean | V mean |
|---|---|---|
| independent | 0.817 | 0.844 |
| referential | 0.800 | 0.822 |
| conflicting | 0.817 | **0.777** |
| cross_inferential | 0.835 | 0.823 |

**결정적 발견 3개:**

1. **Mid-layer dip (L10-L15).** 모든 조건에서 V 유사도가 중간 레이어에서 급락 (conflicting L12 = 0.56). 초기·후반 레이어는 cross-context 영향이 작음. 이는 해석성 문헌의 "middle layers do semantic integration" 관찰과 일치.

2. **Conflicting 조건에서만 V < K.** 위치 효과보다 cross-context 효과가 강함. → **RoPE-shift만으로 conflicting은 못 고침.**

3. **Method 6 (learned combiner) 설계 방향 결정:** 모든 레이어에 붙일 필요 없음 → **L10-L15에 selective corrector** = 최소 파라미터로 최대 효과. 리뷰어의 "combiner가 크면 prefill이 낫다" 우려를 원천 대응.

## Method 2 결과 (RoPE-shifted concat, rope_test.py + mve.py)

RoPE-shift 구현 검증: L0 shifted K cos = 1.0000 (perfect). 초기 레이어는 pure position offset이었고 shift로 완전 복구.

**Layer-wise K 개선 (independent condition):**
- Raw K: 0.817 → **Shifted K: 0.923** (오차 절반 이상 감소)
- V (unchanged reference): 0.844

**Behavioral (M1 vs M2, next-token distribution):**

| condition | KL M1 → M2 | Top-1 M1 → M2 | 판정 |
|---|---|---|---|
| independent | 0.324 → **0.271** | 0.00 → **1.00** | ✅ M2 압승 |
| conflicting | 0.278 → **0.212** | 0.00 → **1.00** | ✅ M2 압승 |
| referential | 0.336 → 0.524 | 0.00 → 0.00 | ❌ baseline degenerate |
| cross_inferential | 0.410 → 0.451 | 0.00 → 0.00 | ~ baseline degenerate |

**Referential/cross_inferential에서 M2가 나쁘게 보이는 이유:** baseline 자체가 이 조건에서 degenerate — Referential baseline 답이 empty (''), cross_inferential은 산술 오류 ("By 2024" 대신 "2025"가 정답). 실제로는 M1/M2 모두 correct 답 뱉음. 0.5B 모델의 신뢰성 한계.

**리뷰어 예측 검증:** "RoPE-shift는 강한 baseline이지만 정답 KV는 아님. cross-context interaction 복원 불가." → 정확히 empirical하게 확인됨. Learned combiner (H3)의 존재 정당성 확보.

**논문 스토리 뼈대:**
```
M1 naive concat    → universal top-1 = 0
M2 RoPE-shift      → 2/4 top-1 = 1.0 (position 해결, cross-context 잔존)
Layer analysis     → mid-layer V가 병목 (L10-L15)
H3 learned corrector → mid-layer V만 보정 (minimum-parameter form)
```

## n=5 통계적 견고성 (mve.py + conditions.py)

각 조건에 5개 예시로 확장 (총 20 examples). aggregate 결과가 n=1 스토리를 nuanced하게 재작성.

| condition | M1 KL mean+/-std | M2 KL mean+/-std | M1 top1 | M2 top1 |
|---|---|---|---|---|
| independent | 0.338 +/- 0.199 | 0.441 +/- 0.280 | 0.20 | **0.60** |
| **conflicting** | 0.353 +/- 0.164 | **0.270 +/- 0.175** | 0.40 | **0.80** |
| **referential** | **0.295 +/- 0.045** | 0.511 +/- 0.206 | 0.40 | 0.40 |
| cross_inferential | 0.330 +/- 0.077 | 0.399 +/- 0.110 | 0.00 | 0.20 |

**핵심 인사이트 (n=5로 강화):** RoPE-shift는 조건부 우위:
```
낮은 A→B 의존도  → M2 우위      (independent, conflicting)
높은 A→B 의존도  → M2 열세      (referential, cross_inferential)
```

**해석:** Referential에서 B의 "She"는 A의 Alice 참조. Full-prefill B의 hidden state는 이미 A를 "봤음". RoPE-shift는 위치만 맞추지만 **B가 A를 알고 있다는 잘못된 정렬**을 만들어 M1보다 오히려 큰 오류. Naive concat은 misaligned지만 이 잘못된 신호가 없음.

**리뷰어 예측의 정량적 확증** — H3 learned corrector가 겨냥해야 할 정확한 지점.

## H3 POC 결과 (h3_poc.py, layer L12 V, 5-fold cross-val)

가장 단순한 corrector: 64x64 linear map W (ridge regression). 20 examples, 536 samples, leave-4-out CV.

| fold | baseline cos | corrected cos | delta |
|---|---|---|---|
| 0 | 0.6265 | 0.6161 | -0.0104 |
| 1 | 0.5829 | 0.6184 | +0.0355 |
| 2 | 0.5982 | 0.6434 | +0.0452 |
| 3 | 0.6707 | 0.6581 | -0.0125 |
| 4 | 0.5972 | 0.6372 | +0.0400 |
| **mean** | **0.6151** | **0.6346** | **+0.0195** |

**판정: MARGINAL POSITIVE** — 3/5 fold 개선, 평균 +2% cosine.

**해석:** 방향은 맞지만 unconditional 64x64 linear는 너무 단순. **다음 단계 명확한 근거 확보:** (1) non-linear (MLP), (2) context-conditional (V_A 요약 concat), (3) 여러 layer 확장.

## 논문 스토리 아크 (7단, 6단 empirical 완료)

```
1. M1 naive           → broad failure                       [완료]
2. M2 RoPE-shift      → conditional win/loss                [완료]
3. Layer analysis     → mid-layer V dip                     [완료]
4. M3 layer-selective → NEGATIVE (symptom vs cause)         [완료]
5. H3 linear POC      → MARGINAL +2% cosine                 [완료]
6. H3 end-to-end      → -5~8% KL, 4/4 conditions            [완료]
7. H3 v2/v3 + Pareto  → future (MLP, context, multi-layer)  [남음]
```

## H3 End-to-End 결과 (mve_h3.py, in-sample W)

L12 linear corrector를 실제 composition에 삽입. 4 방법 비교, 20 examples:

| method | KL mean | top1 | top5 |
|---|---|---|---|
| M1_naive | 0.329 | 0.25 | 0.77 |
| M2_rope | 0.405 | 0.50 | 0.72 |
| **M4_naive_h3** | **0.311** | 0.30 | **0.79** |
| **M5_rope_h3** | **0.370** | 0.50 | 0.76 |

**M4 vs M1: KL -5.5%. M5 vs M2: KL -8.6%. H3가 모든 4/4 조건에서 KL 감소.**

Conflicting 조건에서 M5 (rope + h3) = **0.228** vs M1 0.353 (**-35%**). **RoPE-shift와 H3가 상보적** — M2는 위치 문제, H3는 cross-context 문제.

**Cosine 개선(+2%)이 downstream KL 개선(-5~8%)으로 실제 translated.** Mechanism 검증 완료.

**Caveat:** In-sample W. Out-of-sample downstream 검증은 다음 세션.

## 핵심 관찰 (프로젝트 동기)

`A+B`를 정직히 prefill하면 B의 K/V에는 **A에 attention한 결과**가 들어있음. 반면 B만 단독으로 prefill한 `KV_B`는 A를 한 번도 못 봄.

→ **어떤 위치 정렬(RoPE-shift)도 이 cross-context interaction을 원리적으로 복원할 수 없음.**
→ 즉 "exact equivalence"는 불가능. **근사만 가능.**

**논문의 진짜 승부처:** 안 되는 이유를 정확히 분해한 뒤, 어느 정도 비용으로 어디까지 복원할 수 있는가.

## 왜 이 문제인가

- Autoregressive decode와 긴 컨텍스트 조건에서 KV cache의 memory capacity 및 memory-bandwidth 비용은 중요한 추론 병목이 될 수 있음
- 긴 prefix를 반복 prefill하는 비용 또한 prefix reuse가 어려운 serving 환경에서 중요한 지연시간·비용 요인
- 기존 시스템은 세션 간 계산을 **재사용하지 못하고** 매번 처음부터 처리
- 저비용 조합이 가능한 regime이 존재한다면 → 세션 간 재사용, 지식 레고 형태의 시스템 가능성

## 검증할 3가지 가설

- **H1 (naive):** raw concat → 실패 예상. 위치 정렬 미비 + cross-context 부재. 실패 정량화가 목적.
- **H2 (structural):** RoPE-shifted concat + 레이어/헤드 선택 → 부분 성공 예상. 문서 의존도별 실패 패턴이 구조화되는지 검증.
- **H3 (learned):** 작은 cache-space corrector가 cross-context interaction 부재에서 오는 행동 차이를 줄이고, 제한된 비용 예산에서 Pareto 개선을 달성할 수 있는지 검증. 결과에 따라 main claim 또는 "learned correction의 한계" 발견이 됨.

## 조합 방법 (우선순위 순)

**Priority 1 (MVE 필수):**
1. **Raw concat** — baseline. 위치 정렬 없이 KV_A ++ KV_B
2. **RoPE-shifted concat** — B의 K에 대해 원래 position의 회전을 역으로 제거한 뒤 `|A|`만큼 이동한 position의 회전을 재적용. 이는 위치 불일치를 완화하지만, B의 hidden state에 결여된 A→B attention history는 복원하지 못함.

**Priority 2 (H2 검증):**
3. Layer-selective composition — 어떤 레이어에서 어떤 segment를 어떻게 처리할지 명확한 규칙 정의 필요
4. Head-selective composition

**Priority 3 (H3 novelty):**
5. Learned combiner / corrector — 파라미터 최소화, 비용 명시

## 4-조건 데이터셋 (핵심 실험 설계)

| 조건 | A와 B의 관계 | 검증할 질문 |
|---|---|---|
| **independent** | 무관한 두 문서 | 무관한 prefix도 조합 가능한가? attention이 완전히 0이 아니어도 실용적 복원이 되는가? |
| **referential** | B가 A의 엔티티 참조 | 참조 관계가 얼마나 조합성을 해치는가? |
| **conflicting** | 같은 엔티티, 다른 사실 | 충돌 정보의 선택 편향은 무엇인가? recency/position bias? |
| **cross-inferential** | 답이 A+B 결합 필수 | correction이 반드시 필요한 regime인가? |

**결과는 MVE가 보여주게 두고, 예측이 틀리면 그것도 발견으로 기록.**

## 평가 3층

| 층 | 메트릭 | 역할 |
|---|---|---|
| Representation | K/V MSE, cosine, CKA | 내부 실패 지점 분석 |
| **Behavioral** | **next-token KL, top-k agreement, perplexity** | **주 지표** |
| Task | QA accuracy | 사용자 가치 |

**주 claim은 behavioral 층에서 승부.** KV MSE가 커도 다음 토큰 분포는 비슷할 수 있고 반대도 성립.

## Learned Combiner 비용 제약 (심사자 대응)

논문에서 반드시 명시:
- Combiner FLOPs / latency / peak memory (vs full prefill)
- 길이 일반화 (학습 길이와 다른 길이)
- 도메인 일반화
- 순서 대칭성 (A+B vs B+A)
- 학습 비용의 amortization

## 스코프

**In scope (첫 논문):**
- 같은 모델
- 두 컨텍스트 조합
- Qwen-2.5-0.5B → Llama-3.2-1B → 3B/7B
- 4-조건 controlled benchmark

**Out of scope (future work):**
- 다른 모델 간 KV 이식
- 조직 간 시스템 (KV Hub)
- 프라이버시 공격/방어

## Related Work (to be expanded)

- KV cache compression, quantization, eviction, offloading (H2O, StreamingLLM, SnapKV, DuoAttention)
- Prefix caching and prefix reuse in LLM serving (RadixAttention, Prompt Cache, vLLM's automatic prefix caching)
- Long-context attention and cache selection
- Representation merging / activation steering / model merging (Task Arithmetic, Model Soups)
- Cache editing or cache-space manipulation (limited prior work — this is our gap)

## 셋업

```bash
git clone <repo-url>
cd composable-kv

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python mve.py
```

첫 실행 시 Qwen-2.5-0.5B 다운로드 (~1GB, HuggingFace 계정 불필요).

## 논문 기여 4-part (계획)

1. **문제 정식화** — 독립 prefix KV의 cache-space 조합 문제 정의
2. **실증 분석** — 위치 정렬로 cross-context interaction 복원 불가, 실패 패턴이 레이어·헤드·문서 의존도에 따라 구조화됨
3. **방법** — 작은 learned corrector가 낮은 추가 비용으로 행동적 근사 개선 (H3 검증 결과에 따라)
4. **결과** — Pareto frontier on specific regimes

**H3가 압도적이 아니어도 논문 됨** (문제 정의 + 실패 지도 + 강한 baseline 자체가 기여).

## 마일스톤

| Month | 시기 | 목표 |
|---|---|---|
| 1 | 2026-10 | 논문 15편, MVE 4조건 실행 완료, 실제 결과 표 |
| 2 | 2026-11 | RoPE-shift 구현, layer-selective, 실패 지도 |
| 3 | 2026-12 | H3 learned corrector 설계·학습, 비용 측정 |
| 4 | 2027-01 | 3B 스케일업, Pareto frontier, arxiv 초안 |
| 5 | 2027-02 | rewriting, arxiv 업로드, 워크숍 제출 |

## Venue 목표

- **Tier A:** arxiv (100%)
- **Tier B:** ICLR/NeurIPS 워크숍 (ES-FoMo, Efficient ML)
- **Tier C:** ACL/EMNLP main
- **Tier D:** ICLR/NeurIPS main
