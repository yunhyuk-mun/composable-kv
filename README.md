# Composable KV

**A lightweight linear value-cache correction trained on held-out examples reduces next-token KL relative to the full-prefill oracle on a small controlled benchmark. The improvement is consistent across four context-relation conditions, but has not yet been validated at scale, across model families, or under realistic serving workloads.**

An early-stage empirical study on Qwen-2.5-0.5B. Portfolio / research-in-progress; not yet a paper. Session 1 log lives in [NOTES.md](NOTES.md).

## Research Question

> Independent prefill cache composition errors can be decomposed into (i) position mismatch and (ii) cross-context representation mismatch. Position is fixable by RoPE-shift. Can (ii) be partially closed by a low-cost learned correction on mid-layer V?

## Method Landscape

| Method | Composition | Correction |
|---|---|---|
| M1 naive | K,V from `KV_B` concatenated as-is | none |
| M2 rope | K in `KV_B` rotated by `+|A|` positions before concat | none |
| M3 layer-selective | RoPE-shift except mid-layers (L10-L15) | none |
| M4 naive + H3 | M1 | `W @ V` at L12 |
| M5 rope + H3 | M2 | `W @ V` at L12 |

`W ∈ R^{64×64}` is trained by ridge-regularized least squares to map `V_composed_B → V_full_B` at layer 12.

## Key Results (Qwen-2.5-0.5B, n=5 examples per condition, in-sample W)

**Aggregate downstream next-token distribution vs full-prefill baseline:**

| method | KL mean | top1 | top5 |
|---|---|---|---|
| M1_naive | 0.329 | 0.25 | 0.77 |
| M2_rope | 0.405 | 0.50 | 0.72 |
| **M4_naive_h3** | **0.311** | 0.30 | **0.79** |
| **M5_rope_h3** | 0.370 | 0.50 | 0.76 |

**Main result (held-out, 5-fold example-level split):** A 64×64 linear map on layer-12 V, trained on 16 examples per fold, reduces downstream KL on the 4 held-out examples by 6.4% (M4 vs M1) and 9.4% (M5 vs M2), consistently across all four conditions. Held-out KL matches in-sample KL within 0.003, indicating the corrector generalizes at this sample size rather than memorizing.

**How to read RoPE-shift (M2):**
- By top-1 agreement, M2 improves on M1 in every condition except referential (a tie).
- By KL, M2 improves on M1 only in **conflicting** (0.353 → 0.270); on independent, referential, and cross-inferential, KL actually **worsens** despite top-1 gains.
- These two behavioral metrics can move in opposite directions — hitting the right argmax while shifting probability mass elsewhere.
- Reading this as "RoPE-shift is conditional" (wins on some conditions, loses on others) is only true if you commit to a single metric. Reporting KL and top-1 side by side is more honest.

**M3 (layer-selective RoPE-shift):** Skipping RoPE-shift on mid layers alone did not improve over full M2. Layer selection alone is not sufficient to address the mid-layer V mismatch — this does not disprove the mid-layer dip as a cause, only that this particular intervention doesn't help.

**Standout note:** On the conflicting condition alone, M5 (RoPE + H3) reaches held-out KL 0.228 and top-1 = 1.00 (every held-out example's composed distribution picks the same top token as full prefill). This is the strongest individual condition result, but the honest headline is the aggregate held-out reduction across all four conditions.

## Reproduction

```bash
git clone <this repo>
cd composable-kv

python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate

pip install -r requirements.txt

# Downstream (M1, M2, M3) on 20 controlled examples
python mve.py

# Layer-wise K/V divergence analysis
python analyze_layers.py

# RoPE-shift sanity check on one example
python rope_test.py

# H3 proof-of-concept: 5-fold CV on L12 V linear corrector
python h3_poc.py

# H3 end-to-end: M4, M5 with in-sample W
python mve_h3.py

# H3 held-out: 5-fold, example-level split (leakage-free)
python h3_holdout.py
```

First run downloads `Qwen/Qwen2.5-0.5B` (~1 GB, no HF token needed). CPU-only.

## Repository Structure

```
composable-kv/
├── README.md              # This file
├── NOTES.md               # Session 1 detailed research log
├── LICENSE
├── requirements.txt
├── conditions.py          # 4 conditions × 5 examples
├── mve.py                 # M1 / M2 / M3 pipeline
├── mve_h3.py              # H3 end-to-end (M4 / M5)
├── analyze_layers.py      # Layer-wise K/V cosine
├── h3_poc.py              # H3 5-fold CV (representation-level)
├── h3_holdout.py          # H3 5-fold example-level held-out downstream
├── rope_test.py           # RoPE-shift verification
└── results/               # Raw output logs from all runs
```

## What this does not yet show

- **Scale.** All results are on Qwen-2.5-0.5B; larger models (1.5B, 7B, 70B) untested. Baseline itself is unreliable for referential and reasoning queries at this scale.
- **Model family.** Only Qwen tested. Llama, Mistral, Gemma may respond differently to naive concat, RoPE-shift, and V correction.
- **Real workloads.** Only 20 hand-crafted controlled examples. No RAG, no long-context, no realistic serving pattern.
- **n.** 5 examples per condition. Standard deviations are large; individual example effects visible in the raw logs. `n=50+` per condition is a near-term priority.
- **Cost.** The claim that a 64×64 linear map is "cheap" is not yet backed by FLOP/latency/memory numbers vs full B prefill.
- **Corrector expressivity.** Only one target layer (L12) and only a linear map. Whether MLP + A-context conditioning further reduces KL, or whether multi-layer application accumulates gains, is untested.
- **Composition depth.** Only two contexts combined (A + B). Chains (A + B + C ...) are unexplored.
- **Metric behavior.** KL and top-1 can move in opposite directions (see "How to read RoPE-shift" above). Reporting one alone would be misleading.

None of these blockers is fatal to the direction; each is the subject of a concrete follow-up in the Session 2 priority list at the bottom.

## Related Work (stub, to be expanded)

- KV cache compression / eviction: H2O (arxiv 2306.14048), StreamingLLM (2309.17453), SnapKV (2404.14469), DuoAttention (2410.10819)
- Prefix caching for serving: RadixAttention / SGLang, vLLM automatic prefix caching, Prompt Cache (2311.04934)
- Representation arithmetic: Task Arithmetic (2212.04089), Editing Models with Task Arithmetic
- Activation steering / superposition (Anthropic interpretability): Toy Models of Superposition
- YOCO (2405.05254) for KV reuse structure

## Project Status

**Done in Session 1:**
- [x] MVE with 4 controlled conditions × 5 examples (n=20)
- [x] Layer-wise K/V divergence analysis
- [x] RoPE-shifted baseline (M2) and layer-selective variant (M3)
- [x] Linear H3 corrector: 5-fold CV representation + in-sample end-to-end
- [x] Held-out H3 evaluation (5-fold example-level split): -6.4% / -9.4% KL, matches in-sample within 0.003

**Session 2 priority (before scaling model or corrector expressivity):**
- [ ] Document cache-injection sanity checks (KV_full round-trip KL ≈ 0; attention_mask / cache_position policy)
- [ ] Scale n to 20~30 examples per condition, with entity / length / template variation to reduce leakage
- [ ] Measure H3 cost: FLOPs, latency, peak memory of the 64×64 map vs full B prefill
- [ ] Reproduce main held-out numbers on Qwen-2.5-1.5B where baselines are more reliable

**Then (contingent on the above):**
- [ ] MLP + A-context-conditional corrector (only if scale doesn't already close the gap linearly)
- [ ] Extend corrector to all mid-layers L10-L15 (only after measuring per-layer marginal gain)
- [ ] Pareto measurement (quality vs FLOP / latency, needed for any composition claim)

## License

MIT — see [LICENSE](LICENSE).

## Contact

Yoonh Moon · [moonyoonh91@gmail.com](mailto:moonyoonh91@gmail.com)
