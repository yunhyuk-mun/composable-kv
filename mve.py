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

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
ROPE_THETA = 1000000.0


CONDITIONS = {
    "independent": {
        "A": "Alice is a doctor. She lives in Seoul and works at Samsung Medical Center.",
        "B": "The capital of France is Paris. It has a population of about 2.1 million.",
        "query": "\nQuestion: What does Alice do?\nAnswer:",
    },
    "referential": {
        "A": "Alice is a doctor. She lives in Seoul and works at Samsung Medical Center.",
        "B": "She recently published a paper on cardiac surgery in the New England Journal.",
        "query": "\nQuestion: Who published a paper on cardiac surgery?\nAnswer:",
    },
    "conflicting": {
        "A": "Alice is a doctor working at Samsung Medical Center in Seoul.",
        "B": "Alice is a software engineer at Naver in Bundang.",
        "query": "\nQuestion: What does Alice do?\nAnswer:",
    },
    "cross_inferential": {
        "A": "All doctors at Samsung Medical Center must complete residency within 5 years.",
        "B": "Alice is a doctor at Samsung Medical Center who started her residency in 2020.",
        "query": "\nQuestion: By what year at the latest must Alice complete her residency?\nAnswer:",
    },
}


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


def evaluate_method(model, tokenizer, cond, method_name, compose_fn):
    kv_a, len_a = prefill(model, tokenizer, cond["A"])
    kv_b, len_b = prefill(model, tokenizer, cond["B"])
    composed_len = len_a + len_b

    full_prompt = cond["A"] + " " + cond["B"] + cond["query"]
    p_base = next_token_dist_from_scratch(model, tokenizer, full_prompt)
    ans_base = greedy_from_scratch(model, tokenizer, full_prompt)

    kv_dist = compose_fn(kv_a, kv_b, len_a)
    p_comp = next_token_dist_with_kv(model, tokenizer, cond["query"], kv_dist, composed_len)

    kv_gen = compose_fn(kv_a, kv_b, len_a)
    ans_comp = greedy_with_kv(model, tokenizer, kv_gen, composed_len, cond["query"])

    return {
        "method": method_name,
        "kl": kl_divergence(p_base, p_comp),
        "t1": top_k_agreement(p_base, p_comp, 1),
        "t5": top_k_agreement(p_base, p_comp, 5),
        "t10": top_k_agreement(p_base, p_comp, 10),
        "base": ans_base,
        "comp": ans_comp,
    }


def run_condition(model, tokenizer, name, cond):
    print(f"\n{'-' * 60}\nCONDITION: {name}\n{'-' * 60}")
    print(f"A: {cond['A']}")
    print(f"B: {cond['B']}")
    print(f"Q: {cond['query'].strip()}")

    methods = {
        "M1_naive":   lambda a, b, la: naive_concat_kv(a, b),
        "M2_rope":    lambda a, b, la: rope_shifted_concat_kv(a, b, la),
    }
    method_results = []
    for name_m, fn in methods.items():
        r = evaluate_method(model, tokenizer, cond, name_m, fn)
        method_results.append(r)
        print(f"  [{name_m}] KL={r['kl']:.4f}  top1/5/10={r['t1']:.2f}/{r['t5']:.2f}/{r['t10']:.2f}")
        print(f"    answer: {r['comp']!r}")
    print(f"  [BASELINE] answer: {method_results[0]['base']!r}")
    return {"cond": name, "methods": method_results}


def main():
    print(f"Loading {MODEL_NAME} ...")
    model, tokenizer = load_model()

    results = [run_condition(model, tokenizer, n, c) for n, c in CONDITIONS.items()]

    print(f"\n{'=' * 60}\nSUMMARY -- M1 (naive concat) vs M2 (RoPE-shifted concat)\n{'=' * 60}")
    print(f"{'condition':<20}{'method':<12}{'KL':>10}{'top1':>8}{'top5':>8}{'top10':>8}")
    for r in results:
        for m in r["methods"]:
            print(f"{r['cond']:<20}{m['method']:<12}{m['kl']:>10.3f}{m['t1']:>8.2f}{m['t5']:>8.2f}{m['t10']:>8.2f}")

    print(f"\n{'=' * 60}\nDELTA (M2 - M1)\n{'=' * 60}")
    print(f"{'condition':<20}{'dKL':>10}{'dTop1':>8}{'dTop5':>8}{'dTop10':>8}")
    for r in results:
        m1, m2 = r["methods"][0], r["methods"][1]
        print(f"{r['cond']:<20}{m2['kl']-m1['kl']:>10.3f}{m2['t1']-m1['t1']:>8.2f}{m2['t5']-m1['t5']:>8.2f}{m2['t10']-m1['t10']:>8.2f}")


if __name__ == "__main__":
    main()
