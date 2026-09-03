"""
End-to-end H3 integration: does the learned linear V corrector at L12
actually improve downstream KL and top-1?

Warning: for time reasons this uses in-sample training (W fit on all
20 examples, then applied to the same). This tests "mechanism works
in principle"; held-out will be Session 2.
"""

import statistics
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from conditions import CONDITIONS

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
ROPE_THETA = 1000000.0
TARGET_LAYER = 12


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    return model, tokenizer


def prefill(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    return out.past_key_values, inputs.input_ids.shape[1]


def prefill_ids(model, ids):
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    return out.past_key_values


def train_h3_W(model, tokenizer, layer_idx, ridge=1e-4):
    """Fit W in R^{64x64} on all 20 examples: min ||X W - Y||^2 + ridge||W||^2."""
    x_list, y_list = [], []
    for cond_name, exs in CONDITIONS.items():
        for cond in exs:
            ids_a = tokenizer(cond["A"], return_tensors="pt").input_ids
            ids_b = tokenizer(cond["B"], return_tensors="pt").input_ids
            ids_full = torch.cat([ids_a, ids_b], dim=1)
            len_a = ids_a.shape[1]
            kv_b = prefill_ids(model, ids_b)
            kv_full = prefill_ids(model, ids_full)
            v_b = kv_b.layers[layer_idx].values.squeeze(0).reshape(-1, 64)
            v_full_b = kv_full.layers[layer_idx].values[:, :, len_a:, :].squeeze(0).reshape(-1, 64)
            x_list.append(v_b)
            y_list.append(v_full_b)
    X = torch.cat(x_list, 0)
    Y = torch.cat(y_list, 0)
    W = torch.linalg.solve(X.T @ X + ridge * torch.eye(64), X.T @ Y)
    print(f"  Trained W on {X.shape[0]} samples, sanity in-sample cos = "
          f"{F.cosine_similarity(X @ W, Y, dim=-1).mean().item():.4f}")
    return W


# --- composition methods ---

def naive_concat_kv(cache_a, cache_b):
    new_cache = DynamicCache()
    for i in range(len(cache_a.layers)):
        k = torch.cat([cache_a.layers[i].keys, cache_b.layers[i].keys], dim=-2)
        v = torch.cat([cache_a.layers[i].values, cache_b.layers[i].values], dim=-2)
        new_cache.update(k, v, i)
    return new_cache


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


def rope_shifted_concat_kv(cache_a, cache_b, len_a):
    new_cache = DynamicCache()
    for i in range(len(cache_a.layers)):
        ka = cache_a.layers[i].keys
        va = cache_a.layers[i].values
        kb = rope_shift(cache_b.layers[i].keys, delta=len_a)
        vb = cache_b.layers[i].values
        new_cache.update(torch.cat([ka, kb], dim=-2), torch.cat([va, vb], dim=-2), i)
    return new_cache


def h3_naive_corrected_kv(cache_a, cache_b, len_a, W, target_layer):
    """Naive concat, but at target_layer apply W to B's V."""
    new_cache = DynamicCache()
    for i in range(len(cache_a.layers)):
        ka = cache_a.layers[i].keys
        va = cache_a.layers[i].values
        kb = cache_b.layers[i].keys
        vb = cache_b.layers[i].values
        if i == target_layer:
            # apply W per-position: vb shape [1, H, T, 64] -> reshape [1*H*T, 64] @ W
            orig_shape = vb.shape
            vb_flat = vb.reshape(-1, 64)
            vb_corr = (vb_flat @ W).reshape(orig_shape)
            vb = vb_corr
        new_cache.update(torch.cat([ka, kb], dim=-2), torch.cat([va, vb], dim=-2), i)
    return new_cache


def h3_rope_corrected_kv(cache_a, cache_b, len_a, W, target_layer):
    """RoPE-shifted concat with H3 V correction at target_layer."""
    new_cache = DynamicCache()
    for i in range(len(cache_a.layers)):
        ka = cache_a.layers[i].keys
        va = cache_a.layers[i].values
        kb = rope_shift(cache_b.layers[i].keys, delta=len_a)
        vb = cache_b.layers[i].values
        if i == target_layer:
            orig_shape = vb.shape
            vb_flat = vb.reshape(-1, 64)
            vb_corr = (vb_flat @ W).reshape(orig_shape)
            vb = vb_corr
        new_cache.update(torch.cat([ka, kb], dim=-2), torch.cat([va, vb], dim=-2), i)
    return new_cache


# --- evaluation ---

def next_token_dist_with_kv(model, tokenizer, query, cache, past_len):
    q_ids = tokenizer(query, return_tensors="pt").input_ids
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


def evaluate_method(model, tokenizer, cond, compose_fn):
    kv_a, len_a = prefill(model, tokenizer, cond["A"])
    kv_b, len_b = prefill(model, tokenizer, cond["B"])
    composed_len = len_a + len_b
    full_prompt = cond["A"] + " " + cond["B"] + cond["query"]
    p_base = next_token_dist_from_scratch(model, tokenizer, full_prompt)
    kv_c = compose_fn(kv_a, kv_b, len_a)
    p_comp = next_token_dist_with_kv(model, tokenizer, cond["query"], kv_c, composed_len)
    return {
        "kl": kl_divergence(p_base, p_comp),
        "t1": top_k_agreement(p_base, p_comp, 1),
        "t5": top_k_agreement(p_base, p_comp, 5),
        "t10": top_k_agreement(p_base, p_comp, 10),
    }


def main():
    print(f"Loading {MODEL_NAME} ...")
    model, tokenizer = load_model()

    print(f"\nTraining H3 linear W on all 20 examples at layer L{TARGET_LAYER} ...")
    W = train_h3_W(model, tokenizer, TARGET_LAYER)

    methods = {
        "M1_naive":     lambda a, b, la: naive_concat_kv(a, b),
        "M2_rope":      lambda a, b, la: rope_shifted_concat_kv(a, b, la),
        "M4_naive_h3":  lambda a, b, la: h3_naive_corrected_kv(a, b, la, W, TARGET_LAYER),
        "M5_rope_h3":   lambda a, b, la: h3_rope_corrected_kv(a, b, la, W, TARGET_LAYER),
    }

    per_method = {m: {"kl": [], "t1": [], "t5": [], "t10": []} for m in methods}
    for cond_name, exs in CONDITIONS.items():
        print(f"\n{cond_name}:")
        for i, cond in enumerate(exs):
            for m_name, fn in methods.items():
                r = evaluate_method(model, tokenizer, cond, fn)
                for k in r:
                    per_method[m_name][k].append(r[k])
        # per-condition aggregates
        for m in methods:
            vals = per_method[m]
            n_prev = sum(len(CONDITIONS[c]) for c in list(CONDITIONS)[:list(CONDITIONS).index(cond_name)])
            n_cur = len(CONDITIONS[cond_name])
            recent_kl = vals["kl"][-n_cur:]
            recent_t1 = vals["t1"][-n_cur:]
            print(f"  {m:<15} KL={statistics.mean(recent_kl):.3f} top1={statistics.mean(recent_t1):.2f}")

    print(f"\n{'=' * 70}\nOVERALL AGGREGATE (all 20 examples, in-sample W)\n{'=' * 70}")
    print(f"{'method':<15}{'KL mean':>10}{'KL std':>10}{'top1':>8}{'top5':>8}{'top10':>8}")
    for m in methods:
        vals = per_method[m]
        print(f"{m:<15}{statistics.mean(vals['kl']):>10.3f}"
              f"{statistics.stdev(vals['kl']):>10.3f}"
              f"{statistics.mean(vals['t1']):>8.2f}"
              f"{statistics.mean(vals['t5']):>8.2f}"
              f"{statistics.mean(vals['t10']):>8.2f}")


if __name__ == "__main__":
    main()
