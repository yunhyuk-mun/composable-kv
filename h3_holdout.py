"""
Held-out H3 evaluation.

Protocol (per reviewer):
  - Split at EXAMPLE level, not token, to prevent leakage
  - 5-fold: each fold holds out 4 examples
  - For each fold: train W on train examples, apply to held-out
  - Compare M1/M2/M4/M5 downstream next-token KL and top-k on held-out set
  - Report per-condition and overall mean +/- std across folds
"""

import statistics
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from conditions import CONDITIONS

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
ROPE_THETA = 1000000.0
TARGET_LAYER = 12
N_FOLDS = 5


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    return model, tokenizer


def prefill_ids(model, ids):
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    return out.past_key_values


def rope_shift(k, delta, rope_theta=ROPE_THETA):
    D = k.shape[-1]
    half = D // 2
    i = torch.arange(half, dtype=torch.float32, device=k.device)
    inv_freq = 1.0 / (rope_theta ** (2 * i / D))
    angles = float(delta) * inv_freq
    cos = torch.cos(angles).to(k.dtype).view(1, 1, 1, half)
    sin = torch.sin(angles).to(k.dtype).view(1, 1, 1, half)
    k1, k2 = k[..., :half], k[..., half:]
    return torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)


def _kv_pair(cache, i):
    return cache.layers[i].keys, cache.layers[i].values


def compose(cache_a, cache_b, len_a, method, W=None, target_layer=TARGET_LAYER):
    """Return a fresh DynamicCache built according to method."""
    new = DynamicCache()
    for i in range(len(cache_a.layers)):
        ka, va = _kv_pair(cache_a, i)
        kb, vb = _kv_pair(cache_b, i)

        # K handling
        if method in ("M2_rope", "M5_rope_h3"):
            kb_use = rope_shift(kb, delta=len_a)
        else:
            kb_use = kb

        # V handling
        if method in ("M4_naive_h3", "M5_rope_h3") and i == target_layer and W is not None:
            orig = vb.shape
            vb_use = (vb.reshape(-1, orig[-1]) @ W).reshape(orig)
        else:
            vb_use = vb

        new.update(torch.cat([ka, kb_use], -2), torch.cat([va, vb_use], -2), i)
    return new


def next_token_dist_with_kv(model, q_ids, cache, past_len):
    q_len = q_ids.shape[1]
    attn = torch.ones(1, past_len + q_len, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=q_ids, attention_mask=attn, past_key_values=cache)
    return F.softmax(out.logits[0, -1, :], dim=-1)


def next_token_dist_from_scratch(model, tokenizer, full_prompt):
    inputs = tokenizer(full_prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    return F.softmax(out.logits[0, -1, :], dim=-1)


def kl_divergence(p_ref, p_test, eps=1e-12):
    p_ref = p_ref.clamp_min(eps)
    p_test = p_test.clamp_min(eps)
    return (p_ref * (p_ref.log() - p_test.log())).sum().item()


def top_k_agreement(p_ref, p_test, k):
    return len(set(torch.topk(p_ref, k).indices.tolist())
               & set(torch.topk(p_test, k).indices.tolist())) / k


def cache_all_examples(model, tokenizer):
    """Prefill A, B, A+B, and compute baseline dist once per example.

    Tokenization-aligned version: both the composition path and the
    from-scratch baseline consume the same concatenated token ids
    tokenize(A) ++ tokenize(B) ++ tokenize(Q), so any KL difference
    reflects composition behavior rather than BPE boundary drift.
    """
    examples = []
    for cond_name, exs in CONDITIONS.items():
        for i, cond in enumerate(exs):
            ids_a = tokenizer(cond["A"], return_tensors="pt").input_ids
            ids_b = tokenizer(cond["B"], return_tensors="pt").input_ids
            ids_q = tokenizer(cond["query"], return_tensors="pt").input_ids
            ids_full = torch.cat([ids_a, ids_b], dim=1)
            ids_all  = torch.cat([ids_a, ids_b, ids_q], dim=1)
            len_a = ids_a.shape[1]

            kv_a = prefill_ids(model, ids_a)
            kv_b = prefill_ids(model, ids_b)
            kv_full = prefill_ids(model, ids_full)

            # Baseline: score the SAME concatenated token ids that the
            # composition path will use (no BPE-boundary drift).
            with torch.no_grad():
                out = model(input_ids=ids_all)
            p_base = F.softmax(out.logits[0, -1, :], dim=-1)

            # For W training
            v_b_l = kv_b.layers[TARGET_LAYER].values.squeeze(0).reshape(-1, 64)
            v_full_b_l = kv_full.layers[TARGET_LAYER].values[:, :, len_a:, :].squeeze(0).reshape(-1, 64)

            examples.append({
                "cond": cond_name,
                "ex_idx": len(examples),
                "kv_a": kv_a,
                "kv_b": kv_b,
                "len_a": len_a,
                "len_b": ids_b.shape[1],
                "ids_q": ids_q,          # <- use exact token ids, no re-tokenize
                "p_base": p_base,
                "v_b_l": v_b_l,          # for training W
                "v_full_b_l": v_full_b_l,
            })
            print(f"  cached ex {len(examples)-1:2d} ({cond_name} #{i})")
    return examples


def fit_W(train_examples, ridge=1e-4):
    Xs = [ex["v_b_l"] for ex in train_examples]
    Ys = [ex["v_full_b_l"] for ex in train_examples]
    X = torch.cat(Xs, 0)
    Y = torch.cat(Ys, 0)
    return torch.linalg.solve(X.T @ X + ridge * torch.eye(64), X.T @ Y)


def evaluate_example(model, tokenizer, ex, method, W=None):
    cache = compose(ex["kv_a"], ex["kv_b"], ex["len_a"], method, W=W)
    p_comp = next_token_dist_with_kv(model, ex["ids_q"], cache, ex["len_a"] + ex["len_b"])
    return {
        "kl":  kl_divergence(ex["p_base"], p_comp),
        "t1":  top_k_agreement(ex["p_base"], p_comp, 1),
        "t5":  top_k_agreement(ex["p_base"], p_comp, 5),
        "t10": top_k_agreement(ex["p_base"], p_comp, 10),
    }


def main():
    print(f"Loading {MODEL_NAME} ...")
    model, tokenizer = load_model()

    print(f"\nPre-caching all 20 examples (prefill A, B, A+B, baseline dist) ...")
    examples = cache_all_examples(model, tokenizer)
    n = len(examples)

    # Deterministic 5-fold at example level (indices 0..19)
    folds = [list(range(k, n, N_FOLDS)) for k in range(N_FOLDS)]

    methods_needing_W = {"M4_naive_h3", "M5_rope_h3"}
    all_methods = ["M1_naive", "M2_rope", "M4_naive_h3", "M5_rope_h3"]

    per_fold_agg = {m: {"kl": [], "t1": [], "t5": [], "t10": []} for m in all_methods}
    per_cond_by_method = {m: {c: {"kl": [], "t1": [], "t5": [], "t10": []}
                              for c in CONDITIONS} for m in all_methods}

    for fold_idx, test_idx in enumerate(folds):
        train_idx = [i for i in range(n) if i not in test_idx]
        train_exs = [examples[i] for i in train_idx]
        test_exs = [examples[i] for i in test_idx]

        W = fit_W(train_exs)
        in_sample_cos = F.cosine_similarity(
            torch.cat([e["v_b_l"] for e in train_exs]) @ W,
            torch.cat([e["v_full_b_l"] for e in train_exs]), dim=-1
        ).mean().item()

        print(f"\n=== Fold {fold_idx}: train={len(train_exs)}, test={len(test_exs)} ===")
        print(f"  In-sample W cos = {in_sample_cos:.4f}")
        print(f"  Test example ids: {test_idx}")

        for ex in test_exs:
            for m in all_methods:
                r = evaluate_example(model, tokenizer, ex, m, W=(W if m in methods_needing_W else None))
                for k in ("kl", "t1", "t5", "t10"):
                    per_fold_agg[m][k].append(r[k])
                    per_cond_by_method[m][ex["cond"]][k].append(r[k])

    print(f"\n{'=' * 70}\nHELD-OUT AGGREGATE (n={n}, 5-fold, example-level split)\n{'=' * 70}")
    print(f"{'method':<15}{'KL mean':>10}{'KL std':>10}{'top1':>8}{'top5':>8}{'top10':>8}")
    for m in all_methods:
        vals = per_fold_agg[m]
        print(f"{m:<15}"
              f"{statistics.mean(vals['kl']):>10.3f}"
              f"{statistics.stdev(vals['kl']):>10.3f}"
              f"{statistics.mean(vals['t1']):>8.2f}"
              f"{statistics.mean(vals['t5']):>8.2f}"
              f"{statistics.mean(vals['t10']):>8.2f}")

    print(f"\n{'=' * 70}\nHELD-OUT PER CONDITION (mean KL / top1)\n{'=' * 70}")
    print(f"{'condition':<20}" + "".join(f"{m:>18}" for m in all_methods))
    for c in CONDITIONS:
        row = f"{c:<20}"
        for m in all_methods:
            vals = per_cond_by_method[m][c]
            kl_m = statistics.mean(vals['kl'])
            t1_m = statistics.mean(vals['t1'])
            row += f"  KL={kl_m:.3f} t1={t1_m:.2f}"
        print(row)

    print(f"\n{'=' * 70}\nDELTA M4-M1 and M5-M2 (held-out, positive dTop = win)\n{'=' * 70}")
    m1_kl = statistics.mean(per_fold_agg["M1_naive"]["kl"])
    m2_kl = statistics.mean(per_fold_agg["M2_rope"]["kl"])
    m4_kl = statistics.mean(per_fold_agg["M4_naive_h3"]["kl"])
    m5_kl = statistics.mean(per_fold_agg["M5_rope_h3"]["kl"])
    m1_t1 = statistics.mean(per_fold_agg["M1_naive"]["t1"])
    m2_t1 = statistics.mean(per_fold_agg["M2_rope"]["t1"])
    m4_t1 = statistics.mean(per_fold_agg["M4_naive_h3"]["t1"])
    m5_t1 = statistics.mean(per_fold_agg["M5_rope_h3"]["t1"])
    print(f"M4-M1  dKL = {m4_kl - m1_kl:+.4f}   dTop1 = {m4_t1 - m1_t1:+.3f}")
    print(f"M5-M2  dKL = {m5_kl - m2_kl:+.4f}   dTop1 = {m5_t1 - m2_t1:+.3f}")


if __name__ == "__main__":
    main()
