"""
Layer-wise analysis of cross-context divergence.

For each condition, tokenize A and B as ids, then compare:
  - kv_a (B alone)       vs kv_full[:, :, len_a:, :]  (B with A context)
  - kv_b (B alone)       vs kv_full[:, :, len_a:, :]

Split into K vs V:
  - V has no RoPE, so V divergence isolates PURE cross-context effect
  - K has RoPE, so K divergence = position mismatch + cross-context

Higher cosine similarity = closer to full-prefill baseline.
Where similarity drops most reveals the layers that most need cross-context correction.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

CONDITIONS = {
    "independent": {
        "A": "Alice is a doctor. She lives in Seoul and works at Samsung Medical Center.",
        "B": "The capital of France is Paris. It has a population of about 2.1 million.",
    },
    "referential": {
        "A": "Alice is a doctor. She lives in Seoul and works at Samsung Medical Center.",
        "B": "She recently published a paper on cardiac surgery in the New England Journal.",
    },
    "conflicting": {
        "A": "Alice is a doctor working at Samsung Medical Center in Seoul.",
        "B": "Alice is a software engineer at Naver in Bundang.",
    },
    "cross_inferential": {
        "A": "All doctors at Samsung Medical Center must complete residency within 5 years.",
        "B": "Alice is a doctor at Samsung Medical Center who started her residency in 2020.",
    },
}


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    return model, tokenizer


def prefill_ids(model, ids):
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    return out.past_key_values


def cos_sim_mean(x, y):
    """Cosine similarity across last dim, averaged over all remaining dims."""
    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = y.reshape(-1, y.shape[-1])
    return F.cosine_similarity(x_flat, y_flat, dim=-1).mean().item()


def analyze(model, tokenizer, name, cond):
    ids_a = tokenizer(cond["A"], return_tensors="pt").input_ids
    ids_b = tokenizer(cond["B"], return_tensors="pt").input_ids
    ids_full = torch.cat([ids_a, ids_b], dim=1)
    len_a = ids_a.shape[1]
    len_b = ids_b.shape[1]

    kv_a = prefill_ids(model, ids_a)
    kv_b = prefill_ids(model, ids_b)
    kv_full = prefill_ids(model, ids_full)

    n_layers = len(kv_full.layers)
    k_v_sims = []
    v_v_sims = []
    a_check = []

    for L in range(n_layers):
        ka = kv_a.layers[L].keys
        va = kv_a.layers[L].values
        kb = kv_b.layers[L].keys
        vb = kv_b.layers[L].values
        kf = kv_full.layers[L].keys
        vf = kv_full.layers[L].values

        # Sanity: A portion in full == A alone (causal, so this must be ~1.0)
        a_check.append(cos_sim_mean(ka, kf[:, :, :len_a, :]))

        # B-alone K vs B-in-full K (RoPE position mismatch + cross-context)
        k_v_sims.append(cos_sim_mean(kb, kf[:, :, len_a:, :]))

        # B-alone V vs B-in-full V (PURE cross-context, no RoPE on V)
        v_v_sims.append(cos_sim_mean(vb, vf[:, :, len_a:, :]))

    return {"cond": name, "n_layers": n_layers, "k": k_v_sims, "v": v_v_sims, "a_check": a_check}


def print_table(results, key, title):
    print(f"\n{title}")
    n_layers = results[0]["n_layers"]
    header = f"{'condition':<20}" + "".join(f"L{i:<3d}" for i in range(n_layers))
    print(header)
    for r in results:
        row = f"{r['cond']:<20}" + "".join(f"{v:.2f} " for v in r[key])
        print(row)


def print_summary(results):
    print(f"\n{'=' * 60}")
    print("Mean cosine similarity across all layers")
    print(f"{'=' * 60}")
    print(f"{'condition':<20}{'K (B vs full)':>18}{'V (B vs full)':>18}")
    for r in results:
        k_mean = sum(r["k"]) / len(r["k"])
        v_mean = sum(r["v"]) / len(r["v"])
        print(f"{r['cond']:<20}{k_mean:>18.4f}{v_mean:>18.4f}")


def main():
    print(f"Loading {MODEL_NAME} ...")
    model, tokenizer = load_model()
    results = [analyze(model, tokenizer, n, c) for n, c in CONDITIONS.items()]

    # Sanity check first
    print(f"\nSanity check: A-portion in full should ~= A alone (cos ~1.0)")
    for r in results:
        a_mean = sum(r["a_check"]) / len(r["a_check"])
        print(f"  {r['cond']:<20} mean K cos = {a_mean:.4f}")

    print_table(results, "v", "Per-layer V cosine sim -- B alone vs B in full (PURE cross-context)")
    print_table(results, "k", "Per-layer K cosine sim -- B alone vs B in full (position + cross-context)")
    print_summary(results)


if __name__ == "__main__":
    main()
