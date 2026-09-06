# Composable KV

**An empirical study of independent LLM prefill KV cache composition on Qwen-2.5-0.5B.** We separate composition error into (i) position mismatch and (ii) cross-context representation mismatch, and evaluate five composition methods on a 20-example controlled benchmark with tokenization-aligned scoring. Preliminary results; single small model, single family, small hand-crafted benchmark.

Session logs in [NOTES.md](NOTES.md). Paper draft in [paper/](paper/).

## Research Question

> Independent prefill cache composition errors can be separated into (i) a position mismatch that RoPE-shift can fix and (ii) a cross-context representation mismatch that no positional operation on isolated caches can recover. On a small controlled benchmark, how much of the remaining gap does a lightweight linear correction on mid-layer values close?

## Method Landscape

| Method | Composition | Correction |
|---|---|---|
| M1 naive | K,V from `KV_B` concatenated as-is | none |
| M2 rope | K in `KV_B` rotated by `+\|A\|` positions before concat | none |
| M3 layer-selective | RoPE-shift on all layers except mid `{L10..L15}` | none |
| M4 naive + H3 | M1 | `W @ V` at L12 |
| M5 rope + H3 | M2 | `W @ V` at L12 |

`W ∈ R^{64×64}` is trained by ridge-regularized least squares to map `V_composed_B → V_full_B` at layer 12.

## Key Results (Qwen-2.5-0.5B, tokenization-aligned)

**Aggregate downstream next-token KL vs full-prefill baseline (all five methods, n=20 examples):**

| method | KL mean | top-1 | top-5 |
|---|---|---|---|
| M1_naive | 0.580 | 0.45 | 0.70 |
| M2_rope | 0.577 | 0.55 | 0.70 |
| **M3_layer_selective** | **0.522** | 0.55 | 0.72 |
| M4_naive_h3 | 0.571 | 0.40 | 0.71 |
| M5_rope_h3 | 0.553 | 0.55 | **0.73** |

**Main results:**

- **M3 (layer-selective RoPE-shift) is the strongest single method** in aggregate on this benchmark (0.522 KL), improving over full RoPE-shift by ~9.5% and over every other method. It improves KL on all four conditions. The mid-layer range `{L10..L15}` was chosen exploratorily from the value cosine dip observed on the full 20-example set (Section 5.1 of the paper), so M3 is not fully held-out at the layer-selection level; a fold-wise re-derivation is deferred to future work.
- **H3 (linear value corrector) adds a small further improvement.** Held-out 5-fold, example-level: M4 reduces aggregate KL by 1.7% over M1; M5 by 3.8% over M2. On the *conflicting* condition, M5 achieves held-out top-1 agreement 1.00.
- **Correction is not low-rank.** SVD of the trained W shows a nearly full-rank spectrum (63/64 singular values above 1% of the largest); rank truncations below k ≈ 22 underperform the un-corrected baseline. Naive rank compression of the corrector is not viable.
- **Cost of composition is small.** M1/M2 composition is under 5% of one `prefill(B)` wall-clock; the H3 correction adds ~0.34 ms and ~0.0021% of one `prefill(B)`'s FLOPs.

**Honest caveat on prior version.** An earlier version of this repo scored the baseline via `tokenize(A + " " + B + Q)` while the composition path used `tokenize(A) ‖ tokenize(B) ‖ tokenize(Q)`; this drift affected 20/20 examples by 1–2 boundary tokens. Aggregate H3 gains shrank from -6.4%/-9.4% (drift-affected) to -1.7%/-3.8% (aligned) once we enforced identical token ids on both paths, and M3 flipped from "no improvement over M2" to "best method." All numbers reported above and in the paper are the aligned ones.

**Pipeline sanity.** Cache-injection round-trip introduces ~3×10⁻⁷ KL noise on identical token ids. The observed method deltas (10⁻² absolute KL) are ~10⁵× larger than this noise floor. See [sanity_check.py](sanity_check.py).

## Reproduction

```bash
git clone https://github.com/yunhyuk-mun/composable-kv.git
cd composable-kv

python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate

pip install -r requirements.txt

# Table 1 aggregate (M1, M2, M3) on 20 controlled examples
python mve.py

# H3 end-to-end on all 20 examples (M4, M5)
python mve_h3.py

# H3 5-fold example-level held-out
python h3_holdout.py

# Layer-wise K/V divergence (Section 5.1)
python analyze_layers.py

# W SVD interpretability (Section 5.5 + Figure 2)
python h3_analysis.py

# Cost measurement
python cost_analysis.py

# Pipeline sanity checks
python sanity_check.py
```

First run downloads Qwen-2.5-0.5B (~1 GB, no HF token needed). CPU-only.

## Repository Structure

```
composable-kv/
├── README.md              # This file
├── NOTES.md               # Session logs (1a–1j)
├── LICENSE
├── requirements.txt
├── conditions.py          # 4 conditions × 5 examples
├── mve.py                 # M1 / M2 / M3 pipeline (aligned)
├── mve_h3.py              # H3 end-to-end (M4 / M5, aligned)
├── h3_holdout.py          # H3 5-fold example-level held-out (aligned)
├── h3_poc.py              # H3 5-fold CV representation study
├── h3_analysis.py         # W SVD interpretability
├── analyze_layers.py      # Per-layer K/V cosine analysis
├── rope_test.py           # RoPE-shift correctness verification
├── cost_analysis.py       # Wall-clock + FLOP estimates
├── sanity_check.py        # Pipeline correctness checks
├── paper/                 # LaTeX paper draft
│   ├── paper.tex
│   ├── paper_overleaf.tex # Single-file variant with embedded bib
│   ├── references.bib
│   ├── W_spectrum.pdf     # Figure 2 (SVD)
│   ├── pareto.pdf         # Figure 1 (quality vs cost)
│   └── README.md
└── results/               # Raw output logs
```

## What this does not yet show

- **Model scale.** Qwen-2.5-0.5B only; larger models untested. Baseline itself is degenerate on the *referential* condition, where the model produces empty completions at this scale.
- **Model family.** Only Qwen tested; Llama/Mistral/Gemma may respond differently.
- **Real workloads.** 20 hand-crafted controlled examples. RAG, long-context QA, multi-turn are unexplored.
- **n and leakage.** n = 5 per condition; example templates share entity and structural patterns. Entity- or template-disjoint splits are a near-term follow-up.
- **Corrector expressivity.** Single target layer (L12), linear map. MLP, A-context-conditional variants, multi-layer application untested.
- **Composition depth.** Only two prefixes. Chains (A + B + C + ...) are unexplored.
- **CacheBlend comparison.** The closest prior work (Yao et al., EuroSys 2025) recovers cross-context via selective recomputation; a direct empirical comparison on the same benchmark is left to future work.

## Related Work (representative)

- KV compression / eviction: H2O, StreamingLLM, SnapKV, DuoAttention
- Prefix caching in serving: vLLM automatic prefix cache, RadixAttention (SGLang), Prompt Cache
- **Cache blending (closest prior work): CacheBlend (Yao et al., EuroSys 2025, arXiv:2405.16444)**
- Representation arithmetic: Task Arithmetic, Activation Addition
- YOCO (You Only Cache Once)
- Long-context / mechanistic interpretability: Toy Models of Superposition

## Project Status

**Session 1 (2026-09-03 ~ 09-04):**
- [x] MVE with 4 controlled conditions × 5 examples (n=20)
- [x] Layer-wise K/V divergence analysis
- [x] RoPE-shifted baseline (M2) and layer-selective variant (M3)
- [x] Linear H3 corrector: 5-fold CV representation + in-sample end-to-end
- [x] Held-out H3 evaluation (5-fold example-level split)
- [x] Cache-injection sanity checks (round-trip KL ≈ 3e-7, plus tokenization boundary check)
- [x] H3 cost analysis (~0.34 ms extra, ~0.0021% FLOPs vs prefill(B))
- [x] LaTeX paper draft

**Session 2 (2026-09-07):**
- [x] W SVD interpretability (Figure 2, Section 5.5)
- [x] Tokenization drift fix: rebuilt all downstream numbers under identical token ids
- [x] CacheBlend added to Related Work with explicit differentiation
- [x] M3 finding reversed from "negative" to positive under aligned scoring
- [x] Paper metadata (title, author, keywords)

**Session 3 priorities (planned):**
- [ ] Bootstrap CI for the M3 vs M1 aggregate KL difference
- [ ] Fold-wise re-derivation of M3's excluded layer range
- [ ] Reproduce main numbers on Qwen-2.5-1.5B
- [ ] Entity- or template-disjoint held-out split
- [ ] Direct comparison against CacheBlend on this benchmark

## License

MIT — see [LICENSE](LICENSE).

## Contact

Mun Yunhyuk · [moonyoonh91@gmail.com](mailto:moonyoonh91@gmail.com)
