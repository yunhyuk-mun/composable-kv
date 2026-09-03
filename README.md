# Composable KV

**Independent prefill cache composition: decomposing failure into position mismatch and cross-context representation mismatch, and testing whether a small learned correction at mid-layer V can partially close the latter.**

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

**Per-condition standout (conflicting):** M5 KL = 0.228, vs M1 0.353 (~35% reduction). RoPE-shift and H3 correction address different error sources and combine additively.

**RoPE-shift is conditional:** wins in low A→B dependency (independent, conflicting), loses in high (referential, cross_inferential). Position alignment cannot recover cross-context information.

**M3 (layer-selective) result:** Skipping RoPE-shift on mid layers alone did not improve over M2. Layer selection is not sufficient to work around the mid-layer V dip observed in layer-wise cosine analysis.

**H3 held-out (5-fold, example-level split):** A 64×64 linear map trained on 16 examples reduces downstream KL on the 4 held-out examples by 6.4% (M4 vs M1) or 9.4% (M5 vs M2), across all 4 conditions. Held-out KL matches in-sample KL within 0.003 — the linear corrector generalizes rather than memorizing. On conflicting condition, M5 achieves top-1 = 1.00 on held-out (every held-out example's composed distribution picks the same top token as full-prefill).

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

## Limitations (be honest)

- **Single model:** Qwen-2.5-0.5B only. Patterns should be re-checked on 1.5B+ where baselines are more reliable.
- **Small n:** 5 examples per condition. Std dev is large; individual example effects visible.
- **In-sample W:** H3 end-to-end trains W on the same examples it evaluates on. Held-out evaluation is required before treating the KL reduction as a main result.
- **KL vs task metric:** When the baseline is degenerate (referential condition's empty output; cross-inferential's wrong arithmetic), low KL rewards matching the failure mode. Top-1 accuracy is a more honest primary metric in those conditions.
- **Composition only tested at n=2:** Only two contexts combined. Longer chains (A + B + C ...) are unexplored.

## Related Work (stub, to be expanded)

- KV cache compression / eviction: H2O (arxiv 2306.14048), StreamingLLM (2309.17453), SnapKV (2404.14469), DuoAttention (2410.10819)
- Prefix caching for serving: RadixAttention / SGLang, vLLM automatic prefix caching, Prompt Cache (2311.04934)
- Representation arithmetic: Task Arithmetic (2212.04089), Editing Models with Task Arithmetic
- Activation steering / superposition (Anthropic interpretability): Toy Models of Superposition
- YOCO (2405.05254) for KV reuse structure

## Project Status

- [x] MVE with 4 controlled conditions × 5 examples
- [x] Layer-wise K/V divergence analysis
- [x] RoPE-shifted baseline (M2) and layer-selective variant (M3)
- [x] Linear H3 corrector: 5-fold CV representation + in-sample end-to-end
- [x] **Held-out H3 evaluation** (5-fold example-level split): -6.4%/-9.4% KL, generalizes cleanly
- [ ] MLP + A-context-conditional corrector
- [ ] Extend corrector to all mid-layers (L10-L15)
- [ ] Qwen-2.5-1.5B scale-up
- [ ] n=50+ examples per condition
- [ ] Pareto measurement (quality vs FLOP / latency)

## License

MIT — see [LICENSE](LICENSE).

## Contact

Yoonh Moon · [moonyoonh91@gmail.com](mailto:moonyoonh91@gmail.com)
