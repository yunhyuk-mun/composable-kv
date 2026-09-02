# Composable KV

**Research Question (v2, revised per external review):**

Can independently prefilled KV caches be composed in cache-space to **approximate** the full-prefill next-token behavior well enough to yield a **quality–latency Pareto improvement** on defined regimes?

논문 프로젝트. 4~5개월 목표. 첫 arxiv + 워크숍.

## 핵심 관찰 (why the naive framing was wrong)

`A+B`를 정직히 prefill하면 B의 K/V에는 **A에 attention한 결과**가 들어있음. 반면 B만 단독으로 prefill한 `KV_B`는 A를 한 번도 못 봄.

→ **어떤 위치 정렬(RoPE-shift)도 이 cross-context interaction을 복원할 수 없음.**
→ 즉 "exact equivalence"는 원리적으로 불가능. **근사만 가능.**

**논문의 진짜 승부처:** 안 되는 이유를 정확히 분해한 뒤, 어느 정도 비용으로 어디까지 복원할 수 있는가.

## 왜 이 문제인가

- LLM 추론의 99%는 memory-bound (compute 아님)
- KV 캐시는 컨텍스트 길이에 비례해 폭발
- 기존 시스템은 세션 간 계산을 **재사용하지 못하고** 매번 처음부터 처리
- 만약 저비용 조합이 가능하다면 → 세션 간 컨텍스트 재사용, 지식 레고

## 검증할 3가지 가설

- **H1 (naive):** raw concat, simple averaging → 실패 예상. 위치 인코딩 충돌, cross-context 부재
- **H2 (structural):** 레이어·헤드 선택적 조합 + RoPE-shift → 부분 성공. 문서 의존도별 실패 패턴이 구조화됨
- **H3 (learned):** 작은 combiner가 cache-space에서 cross-context interaction을 **근사적으로 주입** → Pareto 개선. 논문 main claim.

## 4-조건 데이터셋 (핵심 실험 설계, per reviewer)

| 조건 | A와 B의 관계 | 예상 결과 |
|---|---|---|
| **independent** | 무관한 두 문서 | 조합 잘 됨 (RoPE-shift만으로도) |
| **referential** | B가 A의 엔티티 참조 | 부분 실패 |
| **conflicting** | 같은 엔티티에 다른 사실 | 어느 쪽 선택되나 |
| **cross-inferential** | 답이 A+B 결합 필수 | 거의 확실히 실패 → **combiner 필요성 증명** |

**이 4조건의 결과 표가 곧 논문의 첫 그림.**

## 조합 방법 스펙트럼

1. Naive concat (baseline, position mismatch)
2. RoPE-shifted concat (강한 baseline, 위치만 정렬)
3. Simple averaging (정량화 위해)
4. Layer-selective averaging (H2)
5. Top-K token selection (H2O 확장)
6. **Learned combiner / corrector** (main novelty — cross-context 주입)
7. Attention-based merger

## 평가 3층 (per reviewer)

| 층 | 메트릭 | 역할 |
|---|---|---|
| Representation | K/V MSE, cosine, CKA | 내부 실패 지점 분석 |
| **Behavioral** | **next-token KL, top-k agreement, perplexity** | **주 지표** |
| Task | QA accuracy | 사용자 가치 |

**주 claim은 behavioral 층에서 승부.** KV MSE는 부차적.

## Learned Combiner 비용 제약 (심사자 대응)

논문에서 반드시 명시:
- Combiner FLOPs / latency / peak memory vs full prefill
- 길이 일반화 (학습 길이와 다른 길이에서도 되나)
- 도메인 일반화
- 순서 대칭성 (A+B vs B+A 비교)
- 학습 비용 (한 번 학습으로 얼마나 재사용 가능한가)

**"작은 모델을 새로 학습해 추론을 대신했다"는 비판을 원천 차단.**

## 스코프

**In scope (첫 논문):**
- 같은 모델
- 두 컨텍스트 조합
- Qwen-2.5-0.5B → Llama-3.2-1B → 3B/7B 순 스케일업
- 4-조건 데이터셋

**Out of scope (future work):**
- 다른 모델 간 KV 이식
- 조직 간 시스템 (KV Hub)
- 프라이버시 공격/방어

## 셋업

```bash
cd C:\Users\User\Desktop\composable-kv
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python mve.py
```

첫 실행 시 Qwen-2.5-0.5B 다운로드 (~1GB, HuggingFace 계정 불필요).

## MVE가 뱉는 것

4-조건 × 3-메트릭 표. 예상 패턴:

```
condition             KL    top1    top5   top10
independent          low    high    high    high    ← RoPE-like 정렬만으로 됨
referential          mid    mid     mid     mid     ← 부분 실패
conflicting          var    var     var     var     ← 선택 불안정
cross_inferential    high   low     low     low     ← cross-context 정보 소실
```

**이 패턴이 확인되면 곧 논문 introduction의 motivating figure.**

## 논문 기여 4-part (per reviewer)

1. **문제 정식화** — 독립 prefix KV들의 cache-space 조합 문제 정의
2. **실증 분석** — 위치 정렬만으로는 cross-context interaction 부재를 해결 못 함. 실패 패턴이 레이어·헤드·문서 의존도에 따라 구조화됨을 보임
3. **방법** — 작은 learned corrector가 낮은 추가 비용으로 행동적 근사를 유의미하게 개선
4. **결과** — 특정 품질 기준에서 prefill 비용/지연시간 Pareto frontier

**중요:** H3(learned combiner)가 압도적이 아니어도 문제 정의 + 실패 지도 + 강한 baseline만으로 논문 됨.

## 마일스톤

| Month | 시기 | 목표 |
|---|---|---|
| 1 | 2026-10 | 관련 논문 15편, MVE 4조건 실행, 결과 표 |
| 2 | 2026-11 | RoPE-shift 구현, layer-selective, 실패 지도 |
| 3 | 2026-12 | H3 learned combiner 설계·학습, 비용 측정 |
| 4 | 2027-01 | 3B 스케일업, Pareto frontier 그림, arxiv 초안 |
| 5 | 2027-02 | rewriting, arxiv 업로드, 워크숍 제출 |

## Venue 목표

- **Tier A:** arxiv (100%)
- **Tier B:** ICLR/NeurIPS 워크숍 (ES-FoMo, Efficient ML)
- **Tier C:** ACL/EMNLP main
- **Tier D:** ICLR/NeurIPS main
