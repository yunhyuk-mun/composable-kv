# 실험 노트

MVE 실행 결과와 관찰을 여기 기록. 매 실험마다 한 섹션.

---

## Session 1: 2026-09-03 (프로젝트 시작)

**한 것:**
- 프로젝트 스캐폴드 세팅
- MVE v1 코드 작성
- 외부 리뷰 받고 v2로 재구조화

**다음:**
- 가상환경 셋업, requirements 설치
- `python mve.py` 실행
- 4-조건 × 3-메트릭 표 관찰

**MVE v1 실행 결과 (Qwen-2.5-0.5B, naive concat, 4 conditions):**

| condition | KL(base‖comp) | top-1 | top-5 | top-10 |
|---|---|---|---|---|
| independent | 0.324 | 0.00 | 0.80 | 0.60 |
| referential | 0.336 | 0.00 | 0.60 | 0.70 |
| conflicting | 0.278 | 0.00 | 0.60 | 0.70 |
| cross_inferential | **0.410** | 0.00 | 0.80 | 0.80 |

**핵심 발견 5개:**

1. **모든 조건에서 top-1 = 0.00.** Naive concat이 baseline과 동일한 첫 토큰을 뱉는 경우가 4/4 조건에서 단 한 번도 없음. 이건 강한 empirical evidence — 위치 인코딩 충돌만으로도 next-token 선택이 완전히 갈림.

2. **cross_inferential이 가장 높은 KL (0.410).** 리뷰어 예측 검증 — cross-context dependency가 가장 어려움.

3. **Independent가 예상만큼 낮지 않음 (0.324).** referential(0.336)과 거의 차이 없음. "무관한 컨텍스트는 쉽게 조합" 가설이 흔들림. → full-prefill의 A→B attention coupling이 "독립적" 문서에서도 유의미하게 작용한다는 의미.

4. **KL 절대값은 catastrophic 수준 아님 (0.28~0.41).** Top-5/10에서 60~80% 겹침. 즉 composed 분포는 "가능한 답의 이웃"에는 있으나 top pick만 시스템적으로 다름.

5. **정성적 관찰 — composed가 coherent하고 때로 정답을 뱉지만 baseline과 경로가 다름:**
   - **conflicting:** composed가 A와 B를 **literal fusion** — "software engineer at Samsung Medical Center in Seoul" (직업은 B, 회사는 A). Baseline은 B만 채택 (Naver software engineer).
   - **cross_inferential:** composed "2025" (산술적 정답), baseline "By 2024" (틀림). 0.5B 모델이라 reasoning 신뢰도 낮음 — 절대 정확도는 point 아님.
   - **independent:** composed가 attention-sink loop 발생 — "She is a doctor.\nQuestion... She is a doctor..." 반복. 위치 혼동의 신호.

**함의:**

- Naive concat은 top-1 수준에서 empirically incoherent (예상보다 강한 실패)
- 행동 차이가 측정 가능하고 구조화됨 → 논문 첫 그림 재료
- cross-context effect(H2) 조합 방법 필요성 empirical 정당화
- 다음 단계: Method 2 (RoPE-shifted concat) 구현 → 이 KL이 얼마나 줄어드는지 측정

---

## Session 1b: Layer-wise 분석 (analyze_layers.py)

**목적:** 어느 레이어에서 cross-context 손실이 가장 큰지 지도.

**설계:**
- ids_a, ids_b를 tokenize 후 concat해서 kv_full (baseline)
- kv_b만의 K/V vs kv_full의 B-portion K/V를 layer별 cosine sim으로 비교
- V는 RoPE 없음 → 순수 cross-context 효과
- K는 RoPE 있음 → 위치 + cross-context 혼합

**Sanity check:** A-portion cos = 1.0000 (모든 조건). 인과적 attention 확인.

**핵심 발견 3개:**

**1. Mid-layer dip (L10~L15)이 보편적.** V 유사도가 이 구간에서 급락:
- independent: L12=0.69, L13=0.68
- referential: L12=0.63
- conflicting: **L12=0.56, L13=0.60, L14=0.59** (최저)
- cross_inferential: L12=0.61

초기(L0-L2) V cos ≈ 0.95-1.00, 후반(L20-L23) ≈ 0.80-0.90. **중간 레이어가 semantic integration을 담당한다는 해석성 문헌과 정확히 일치.**

**2. Conflicting에서만 V(0.78) < K(0.82).** 다른 조건은 K가 낮음 (RoPE 위치 문제 지배). Conflicting은 **cross-context 효과가 위치 효과보다 강함.** → RoPE-shift만으로는 conflicting 못 고침.

**3. 평균 K/V 요약:**
| condition | K mean | V mean |
|---|---|---|
| independent | 0.817 | 0.844 |
| referential | 0.800 | 0.822 |
| conflicting | 0.817 | **0.777** |
| cross_inferential | 0.835 | 0.823 |

**연구 함의 (Method 6 설계 정보):**

- 모든 레이어에 corrector 붙일 필요 없음 → **L10-L15 selective corrector**가 최소 비용 최대 효과
- V 보정 위주 (K는 RoPE-shift로 대부분 해결)
- 리뷰어의 "combiner가 크면 prefill이 낫다" 우려 → **mid-layer만 corrector**로 파라미터/비용 급감 가능

**다음:**
- Method 2 (RoPE-shifted concat) 구현 → K 유사도가 얼마나 올라가는지 측정
- Method 2 후에도 남는 V 갭 = learned corrector가 채워야 할 몫

---

## Session 1c: Method 2 구현 및 M1 vs M2 (rope_test.py + mve.py)

**RoPE 구현:**
- Qwen2.5-0.5B config: `rope_theta=1000000`, head_dim=64, half-rotation
- 구현: k1,k2 분할 → (k1·cos - k2·sin, k1·sin + k2·cos)
- 검증: L0 shifted K cos = 1.0000 (perfect sanity). 초기 레이어는 cross-context 없어서 K가 위치만 문제였고, shift로 완벽 복구.

**Layer-wise K 개선 (shifted vs raw, independent condition 기준):**
- Raw K mean cos: 0.817
- Shifted K mean cos: **0.923** (오차 절반 이상 감소)
- V mean cos: 0.844 (변하지 않음, reference)
- 흥미: shifted K (0.92) > V (0.84). **K의 cross-context 효과가 V보다 작음.**

**Behavioral 결과 (mve.py, M1 vs M2):**

| condition | KL M1→M2 | Top-1 M1→M2 |
|---|---|---|
| independent | 0.324 → 0.271 | 0.00 → **1.00** |
| conflicting | 0.278 → 0.212 | 0.00 → **1.00** |
| referential | 0.336 → 0.524 | 0.00 → 0.00 |
| cross_inferential | 0.410 → 0.451 | 0.00 → 0.00 |

**해석:**
- ✅ Independent/conflicting: A→B 상호작용 약함 → M2 top-1 완벽 복원
- ❌ Referential/cross_inferential: baseline 자체가 degenerate ('' 또는 "By 2024" 산술오류) → KL 비교 무의미. 실제 답은 M1/M2 모두 correct.
- **0.5B는 baseline 신뢰성 문제. 1.5B+로 스케일업 필요.**

**리뷰어 예측 empirical 검증:**
> "RoPE-shift는 강한 baseline이지만 정답 KV는 아님. cross-context interaction 복원 불가."
- ✅ 정확히 맞음. 상호작용 약한 곳은 잘 되고 강한 곳은 실패.
- ✅ Learned combiner (H3)의 존재 정당성 empirical 확보.

**정성적 하이라이트:**
- Independent M2: `"Alice is a doctor. She lives in Seoul..."` — baseline 정보 완전 복원
- Conflicting M2: `"software engineer at Samsung Medical Center in Seoul"` — A/B literal fusion 지속
- Cross_inferential 둘 다: `"2025"` — 산술적 정답 (baseline "By 2024"는 오히려 틀림)

**논문 스토리 완성:**
```
Naive concat (M1)   → universal top-1 = 0     (완전 실패)
RoPE-shift (M2)     → 2/4 top-1 = 1.0        (위치 문제 해결, cross-context 잔존)
Layer analysis      → mid-layer V가 병목      (L10-L15)
Learned corrector   → mid-layer V만 보정      (H3, 최소 비용)
```

**다음:**
- 더 큰 모델(Qwen-2.5-1.5B) 스케일업으로 referential/cross_inferential baseline 신뢰성 확보
- 각 조건당 예시 확장 (n=1 → n=10) 통계적 견고성
- H3 learned corrector 프로토타입 (L10-L15만, V 위주)

---

## Session 1d: n=5 통계적 견고성 (conditions.py + mve.py refactor)

**설계:** 각 조건 5개 예시로 확장. 총 20개 (A, B, query) 트리플. 조건별로 A-B 관계 구조는 동일하게 유지.

**Aggregate 결과 (n=5, Qwen-2.5-0.5B):**

| condition | M1 KL mean±std | M2 KL mean±std | M1 top1 | M2 top1 | dKL |
|---|---|---|---|---|---|
| independent | 0.338 ± 0.199 | 0.441 ± 0.280 | 0.20 | **0.60** | +0.104 |
| referential | 0.295 ± 0.045 | 0.511 ± 0.206 | 0.40 | 0.40 | +0.217 |
| **conflicting** | 0.353 ± 0.164 | **0.270 ± 0.175** | 0.40 | **0.80** | -0.083 |
| cross_inferential | 0.330 ± 0.077 | 0.399 ± 0.110 | 0.00 | 0.20 | +0.069 |

**핵심 발견 (n=1과 다른 부분):**

1. **M1 top-1이 0.00 아님 (n=1의 결과는 misleading)** — 실제로는 0.20~0.40 대 (조건별). 우연히 특정 프롬프트에서 baseline과 같은 토큰 뽑기도 함.

2. **conflicting은 M2 견고한 승** — n=5 aggregate도 KL↓, top1↑. 우연 아님.

3. **referential은 M2 견고한 패** — top1 = 0.40 (변화 없음), KL 0.295 → 0.511 (심각). 5/5 예시에서 M2 KL이 M1보다 높음.

4. **RoPE-shift의 조건부 우위:**
   ```
   낮은 A→B 의존도  → M2 우위      (independent 부분적, conflicting)
   높은 A→B 의존도  → M2 열세      (referential, cross_inferential)
   ```

**리뷰어 예측 완벽 empirical 확증:**
- Referential에서 B의 "She"는 A의 Alice를 참조
- Full-prefill B는 이미 A의 정보를 attention한 상태
- RoPE-shift는 위치는 맞추지만 **B가 A를 알고 있다는 잘못된 정렬**을 만듦
- 모델이 KV_B를 "A를 본 것처럼" attention하려 함 → naive concat보다 더 나쁠 수 있음
- Naive concat은 misaligned하지만 오히려 이 잘못된 신호가 없음

**이게 논문 introduction의 motivating figure.** M1이 저지르는 실수 vs M2가 저지르는 다른 실수. 두 실수가 상보적일 수 있음 → H3 learned corrector는 이 두 방법 사이의 균형을 학습해야 함.

**Portfolio 강도:**
- n=1 → n=5로 강화. std가 큼 (0.05~0.28) → 진짜 실험임을 반증
- 4 conditions × 5 examples = 20 examples/method × 2 methods = 40 method-runs
- 재현 가능 (`python mve.py` 한 줄)

**다음:**
- Method 3: layer-selective composition (M2를 mid-layer만 적용) — 실패 조건에서 lm-shift 부작용 최소화?
- Method 4: Baseline이 신뢰할 만한 태스크로 확장 (Qwen-1.5B, LongBench 조합)
- H3 프로토타입: L10-L15의 V만 학습된 corrector로 대체

---

## Session 1e: Method 3 (layer-selective RoPE-shift) — negative result

**가설:** analyze_layers.py에서 mid-layer(L10-L15)에서 V dip이 관찰됨. → mid-layer는 shift 안 하고 나머지만 shift하면 M2의 referential 실패를 회피 가능?

**설계:** M3_ex_mid = RoPE-shift on all layers except {L10, L11, L12, L13, L14, L15}. n=5 per condition, 동일 mve.py 파이프라인.

**Aggregate 결과:**
| condition | M1 KL | M2 KL | M3 KL | M2 top1 | M3 top1 |
|---|---|---|---|---|---|
| independent | 0.338 | 0.441 | 0.435 | 0.60 | 0.60 |
| referential | 0.295 | 0.511 | 0.493 | 0.40 | 0.40 |
| conflicting | 0.353 | 0.270 | 0.267 | 0.80 | 0.80 |
| cross_inferential | 0.330 | 0.399 | 0.414 | 0.20 | 0.20 |

**M3 ≈ M2. 모든 지표 1 std 이내 (noise 수준).**

**함의:**
1. **Mid-layer V dip은 증상, 원인 아님.** Layer 선택으론 실패 조건을 못 고침.
2. Cross-context 정보 부재는 layer 조작으로 해결 불가 → **learned mechanism 필요.**
3. H3 learned corrector의 존재 정당성 empirical 확보.

**부수 관찰:** "Best method by KL"에서 M1이 3/4 승. 그러나 top-1은 M2/M3가 우위. 이유: baseline degenerate 조건에서 low KL = "같은 실패 모드". → **top-1이 primary metric으로 더 적합** (리뷰어의 behavioral 층 통찰과 부합).

**다음:**
- 1.5B로 스케일업 (baseline 신뢰성 개선하면 KL이 다시 의미 있어짐)
- H3 proto: 작은 MLP corrector on mid-layer V만
- 데이터 규모 확대 (n=5 → n=20+)

---

## Session 1f: H3 POC (h3_poc.py) — learned linear V corrector

**설계 (최소 스코프):**
- Target: layer L12 V (analyze_layers에서 dip 최저 조건)
- Model: 64x64 linear map W (ridge regression via least-squares)
- Data: 20 examples × ~26 tokens × 2 kv_heads = 536 samples per fold
- Evaluation: 5-fold cross-validation (leave-4-out at example level)
- Metric: cosine similarity (V_pred vs V_full) vs baseline (V_composed vs V_full)

**결과:**
| fold | baseline cos | corrected cos | delta |
|---|---|---|---|
| 0 | 0.6265 | 0.6161 | -0.0104 |
| 1 | 0.5829 | 0.6184 | +0.0355 |
| 2 | 0.5982 | 0.6434 | +0.0452 |
| 3 | 0.6707 | 0.6581 | -0.0125 |
| 4 | 0.5972 | 0.6372 | +0.0400 |
| **평균** | **0.6151** | **0.6346** | **+0.0195** |

**판정: MARGINAL POSITIVE.** 3/5 fold 개선, 2/5 소폭 악화, 평균 +2% cosine.

**함의:**
1. 방향은 맞음. Learned map이 cross-context gap을 어느 정도 닫음.
2. 선형은 너무 단순. Unconditional 64x64 map은 평균적 correction만 학습.
3. **다음 단계 명확한 empirical 근거:**
   - Non-linear (MLP 2~3층)
   - Context-conditional (V_A 요약을 입력에 concat)
   - 여러 layer 확장

**Portfolio 관점:**
- 완결된 5-fold cross-val 실험
- 정직한 marginal 결론 (과대주장 아님)
- 다음 단계 근거 명확
- **논문 스토리 7단 중 5단이 empirical 뒷받침** — 나머지 2단(MLP+context, Pareto)이 남은 연구

**남은 최소 실험:**
- H3 v2: MLP + V_A context — linear보다 얼마나 나은가
- H3 v3: 모든 mid-layer(L10-L15)에 적용 — layer-wise 이득 누적
- End-to-end: corrected KV로 mve.py 재실행 → downstream KL/top-1 개선 확인

---

## 외부 리뷰 요지 (2026-09-03, MVE v1 → v2 재구조화 근거)

### 강한 점 (그대로 유지)
1. 질문이 근본적 — 기존 KV 연구는 압축·eviction·quantization에 집중, 이건 "합성 가능한가"
2. 실패해도 논문 구조가 나옴 — H1 실패 예상 → H2 실패 지도 → H3 개선
3. 스코프 통제가 좋음 — 같은 모델, 작은 모델, 두 문서, 기능적 동등성
4. 평가 태스크가 직관적 — controlled retrieval은 디버깅에 특히 좋음

### 결정적 지적 3가지

**1. RoPE-shifted concat은 강한 baseline이지만 정답 KV는 아님.**
- A+B를 정직히 처리하면 B의 K/V는 A에 attention한 결과를 포함
- 반면 KV_B 단독은 A를 못 봄
- 위치 정렬로는 cross-context interaction 복원 불가
- **함의:** "exact equivalence"는 원리적으로 불가능. "Pareto approximation"으로 재프레임 필요.

**2. "기능적으로 동등"의 정의를 3층으로 나눠야 함.**
- Representation (K/V MSE, cosine, CKA) — 내부 분석
- **Behavioral (next-token KL, top-k, perplexity)** — 주 지표
- Task (accuracy) — 사용자 가치
- 논문 주장: "저비용 composition은 특정 길이·태스크·예산 구간에서 quality-latency Pareto 개선"

**3. Learned combiner 비용 검증이 필요.**
심사자가 물을 질문:
- combiner FLOPs, latency, peak memory는?
- 길이 일반화 되나?
- 도메인 전이 되나?
- 조합 순서 (A+B vs B+A) 대칭인가?
- 너무 크면 그냥 prefill이 낫지 않나?
- **처음부터 이 제약을 명시하며 설계.**

### 데이터셋 4조건 (핵심)
- **independent** — A와 B 무관 → 조합 쉬움
- **referential** — B가 A 엔티티 참조 → 부분 실패
- **conflicting** — 같은 엔티티, 다른 사실 → 선택 문제
- **cross_inferential** — 답이 A+B 결합 필수 → **거의 확실 실패, combiner 정당화**

이 4조건 결과 표가 곧 논문 introduction의 motivating figure.

### 논문 기여 4-part 구조
1. 문제 정식화 — cache-space에서 독립 prefix KV 조합
2. 실증 분석 — 위치 정렬로 cross-context interaction 복원 불가, 실패 패턴이 구조화됨
3. 방법 — 작은 learned corrector가 낮은 비용으로 행동적 근사 개선
4. 결과 — Pareto frontier

**H3가 압도적이지 않아도 논문 됨** — 문제 정의 + 실패 지도 + 강한 baseline 자체가 기여.

### 한 줄 정리
"KV 산술이 가능한가?"는 좋은 질문. **논문의 승부처는 "안 되는 이유를 정확히 분해한 뒤, 어느 정도 비용으로 어디까지 복원할 수 있는가"를 설득력 있게 보이는 것.**

---

## 참고 논문 (읽을 것 — Month 1)

- [ ] Ilharco et al. 2022 — "Editing Models with Task Arithmetic" (arxiv 2212.04089)
- [ ] StreamingLLM (arxiv 2309.17453) — attention sink 관찰
- [ ] H2O (arxiv 2306.14048) — KV eviction
- [ ] SnapKV (arxiv 2404.14469) — prefill 후 KV 예측
- [ ] DuoAttention (arxiv 2410.10819) — head별 KV 차별화
- [ ] Anthropic "Toy Models of Superposition" (2022) — feature 조합 원리
- [ ] YOCO (arxiv 2405.05254) — KV 재사용 구조
- [ ] Prompt Cache (arxiv 2311.04934) — prefix reuse
- [ ] RadixAttention (SGLang 2024) — 실무 prefix caching
- [ ] Cross-attention 관련 최근 서베이

## 다음 마일스톤 (Month 1 세부)

- [ ] Week 1: 프로젝트 세팅 완료, MVE v2 실행, 결과 표
- [ ] Week 2: 결과 분석, RoPE-shifted concat 구현 시작
- [ ] Week 3: RoPE-shift 완성, 4조건에서 재실험
- [ ] Week 4: 논문 15편 정독 완료, layer-selective 아이디어 스케치
