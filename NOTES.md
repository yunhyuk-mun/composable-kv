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
