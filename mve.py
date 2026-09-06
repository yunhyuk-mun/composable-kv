"""
Minimum Viable Experiment for Composable KV Cache -- v2.

Reframed per external review: we don't test exact equivalence (impossible
because single-context KVs lack cross-context interactions). We measure
how much a cheap composition (here: naive concat) DIVERGES from full
prefill in behavioral terms, across 4 controlled conditions.

Metrics:
  1. KL(p_baseline || p_composed) on next-token distribution
  2. Top-k agreement (k=1, 5, 10)
  3. Greedy answer (qualitative)
"""

import statistics
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from conditions import CONDITIONS

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
ROPE_THETA = 1000000.0


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


def naive_concat_kv(cache_a, cache_b):
    new_cache = DynamicCache()
    for layer_idx in range(len(cache_a.layers)):
        ka = cache_a.layers[layer_idx].keys
        va = cache_a.layers[layer_idx].values
        kb = cache_b.layers[layer_idx].keys
        vb = cache_b.layers[layer_idx].values
        k_cat = torch.cat([ka, kb], dim=-2)
        v_cat = torch.cat([va, vb], dim=-2)
        new_cache.update(k_cat, v_cat, layer_idx)
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
    for layer_idx in range(len(cache_a.layers)):
        ka = cache_a.layers[layer_idx].keys
        va = cache_a.layers[layer_idx].values
        kb = cache_b.layers[layer_idx].keys
        vb = cache_b.layers[layer_idx].values
        kb_shifted = rope_shift(kb, delta=len_a)
        k_cat = torch.cat([ka, kb_shifted], dim=-2)
        v_cat = torch.cat([va, vb], dim=-2)
        new_cache.update(k_cat, v_cat, layer_idx)
    return new_cache


MID_LAYERS = set(range(10, 16))  # L10-L15, per analyze_layers.py mid-layer dip


def layer_selective_shift_kv(cache_a, cache_b, len_a, no_shift_layers):
    """RoPE-shift K in all layers except those in no_shift_layers (naive concat there)."""
    new_cache = DynamicCache()
    for layer_idx in range(len(cache_a.layers)):
        ka = cache_a.layers[layer_idx].keys
        va = cache_a.layers[layer_idx].values
        kb = cache_b.layers[layer_idx].keys
        vb = cache_b.layers[layer_idx].values
        kb_out = kb if layer_idx in no_shift_layers else rope_shift(kb, delta=len_a)
        k_cat = torch.cat([ka, kb_out], dim=-2)
        v_cat = torch.cat([va, vb], dim=-2)
        new_cache.update(k_cat, v_cat, layer_idx)
    return new_cache


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
    top_ref = set(torch.topk(p_ref, k).indices.tolist())
    top_test = set(torch.topk(p_test, k).indices.tolist())
    return len(top_ref & top_test) / k


def greedy_with_kv(model, tokenizer, cache, past_len, query, max_new=25):
    q_ids = tokenizer(query, return_tensors="pt").input_ids
    q_len = q_ids.shape[1]
    attn = torch.ones(1, past_len + q_len, dtype=torch.long)
    with torch.no_grad():
        out = model.generate(
            input_ids=q_ids,
            attention_mask=attn,
            past_key_values=cache,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, q_len:], skip_special_tokens=True)


def greedy_from_scratch(model, tokenizer, full_prompt, max_new=25):
    inputs = tokenizer(full_prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)


# --- Token-id aligned variants (avoid BPE boundary drift) ---

def next_token_dist_with_kv_ids(model, q_ids, cache, past_len):
    q_len = q_ids.shape[1]
    attn = torch.ones(1, past_len + q_len, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=q_ids, attention_mask=attn, past_key_values=cache)
    return F.softmax(out.logits[0, -1, :], dim=-1)


def greedy_with_kv_ids(model, tokenizer, cache, past_len, q_ids, max_new=25):
    q_len = q_ids.shape[1]
    attn = torch.ones(1, past_len + q_len, dtype=torch.long)
    with torch.no_grad():
        out = model.generate(
            input_ids=q_ids, attention_mask=attn, past_key_values=cache,
            max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, q_len:], skip_special_tokens=True)


def greedy_from_ids(model, tokenizer, ids_all, max_new=25):
    with torch.no_grad():
        out = model.generate(
            input_ids=ids_all, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, ids_all.shape[1]:], skip_special_tokens=True)


def evaluate_method(model, tokenizer, cond, method_name, compose_fn):
    # Tokenize A, B, Q separately, then use identical concatenated token
    # ids on both the composition path and the from-scratch baseline so
    # any KL difference reflects composition behavior rather than BPE
    # boundary drift.
    ids_a = tokenizer(cond["A"], return_tensors="pt").input_ids
    ids_b = tokenizer(cond["B"], return_tensors="pt").input_ids
    ids_q = tokenizer(cond["query"], return_tensors="pt").input_ids
    len_a, len_b = ids_a.shape[1], ids_b.shape[1]

    with torch.no_grad():
        kv_a = model(input_ids=ids_a, use_cache=True).past_key_values
        kv_b = model(input_ids=ids_b, use_cache=True).past_key_values

    composed_len = len_a + len_b

    ids_all = torch.cat([ids_a, ids_b, ids_q], dim=1)
    with torch.no_grad():
        p_base = F.softmax(model(input_ids=ids_all).logits[0, -1, :], dim=-1)
    ans_base = greedy_from_ids(model, tokenizer, ids_all)

    kv_dist = compose_fn(kv_a, kv_b, len_a)
    p_comp = next_token_dist_with_kv_ids(model, ids_q, kv_dist, composed_len)

    kv_gen = compose_fn(kv_a, kv_b, len_a)
    ans_comp = greedy_with_kv_ids(model, tokenizer, kv_gen, composed_len, ids_q)

    return {
        "method": method_name,
        "kl": kl_divergence(p_base, p_comp),
        "t1": top_k_agreement(p_base, p_comp, 1),
        "t5": top_k_agreement(p_base, p_comp, 5),
        "t10": top_k_agreement(p_base, p_comp, 10),
        "base": ans_base,
        "comp": ans_comp,
    }


METHODS = {
    "M1_naive":   lambda a, b, la: naive_concat_kv(a, b),
    "M2_rope":    lambda a, b, la: rope_shifted_concat_kv(a, b, la),
    "M3_ex_mid":  lambda a, b, la: layer_selective_shift_kv(a, b, la, no_shift_layers=MID_LAYERS),
}


def run_condition_set(model, tokenizer, name, examples):
    print(f"\n{'=' * 70}\nCONDITION: {name} (n={len(examples)})\n{'=' * 70}")
    per_method = {m: {"kl": [], "t1": [], "t5": [], "t10": []} for m in METHODS}
    per_example = []

    for i, cond in enumerate(examples):
        print(f"\n  Example {i+1}: A={cond['A'][:50]}...")
        row = {"example": i}
        for m_name, fn in METHODS.items():
            r = evaluate_method(model, tokenizer, cond, m_name, fn)
            for k in ("kl", "t1", "t5", "t10"):
                per_method[m_name][k].append(r[k])
            row[m_name] = r
            print(f"    [{m_name}] KL={r['kl']:.3f} top1/5/10={r['t1']:.2f}/{r['t5']:.2f}/{r['t10']:.2f}")
        per_example.append(row)

    print(f"\n  --- {name} aggregated (n={len(examples)}) ---")
    for m in METHODS:
        vals = per_method[m]
        kl_m = statistics.mean(vals["kl"])
        kl_s = statistics.stdev(vals["kl"]) if len(vals["kl"]) > 1 else 0.0
        t1_m = statistics.mean(vals["t1"])
        t5_m = statistics.mean(vals["t5"])
        t10_m = statistics.mean(vals["t10"])
        print(f"    {m}:  KL={kl_m:.3f}+/-{kl_s:.3f}  top1={t1_m:.2f}  top5={t5_m:.2f}  top10={t10_m:.2f}")

    return {"cond": name, "per_method": per_method, "per_example": per_example}


def main():
    print(f"Loading {MODEL_NAME} ...")
    model, tokenizer = load_model()

    results = [run_condition_set(model, tokenizer, n, exs) for n, exs in CONDITIONS.items()]

    print(f"\n{'=' * 70}\nAGGREGATE SUMMARY (n=5 per condition) -- M1 vs M2\n{'=' * 70}")
    print(f"{'condition':<20}{'method':<12}{'KL mean':>10}{'KL std':>10}{'top1':>8}{'top5':>8}{'top10':>8}")
    for r in results:
        for m in METHODS:
            vals = r["per_method"][m]
            kl_m = statistics.mean(vals["kl"])
            kl_s = statistics.stdev(vals["kl"]) if len(vals["kl"]) > 1 else 0.0
            print(f"{r['cond']:<20}{m:<12}{kl_m:>10.3f}{kl_s:>10.3f}"
                  f"{statistics.mean(vals['t1']):>8.2f}"
                  f"{statistics.mean(vals['t5']):>8.2f}"
                  f"{statistics.mean(vals['t10']):>8.2f}")

    print(f"\n{'=' * 70}\nBEST METHOD per condition (by lowest KL mean)\n{'=' * 70}")
    print(f"{'condition':<20}{'best':<12}{'KL':>10}{'top1':>8}")
    for r in results:
        best_m = min(METHODS.keys(), key=lambda m: statistics.mean(r["per_method"][m]["kl"]))
        vals = r["per_method"][best_m]
        print(f"{r['cond']:<20}{best_m:<12}{statistics.mean(vals['kl']):>10.3f}{statistics.mean(vals['t1']):>8.2f}")


if __name__ == "__main__":
    main()
