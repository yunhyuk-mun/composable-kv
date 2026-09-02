"""
Minimum Viable Experiment for Composable KV Cache — v2.

Reframed per external review: we don't test exact equivalence (impossible
because single-context KVs lack cross-context interactions). We measure
how much a cheap composition (here: naive concat) DIVERGES from full
prefill in behavioral terms, across 4 controlled conditions.

Metrics:
  1. KL(p_baseline || p_composed) on next-token distribution
  2. Top-k agreement (k=1, 5, 10)
  3. Greedy answer (qualitative)

Four conditions probe different aspects of cross-context dependency:
  - independent:       A, B unrelated                → composition should be easy
  - referential:       B references entities in A     → partial breakdown
  - conflicting:       same entity, different facts   → which wins?
  - cross_inferential: answer requires A AND B fused  → hardest, motivates combiner
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


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
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    return model, tokenizer


def prefill(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    kv = out.past_key_values
    if hasattr(kv, "to_legacy_cache"):
        kv = kv.to_legacy_cache()
    return kv, inputs.input_ids.shape[1]


def naive_concat_kv(kv_a, kv_b):
    return tuple(
        (torch.cat([ka, kb], dim=2), torch.cat([va, vb], dim=2))
        for (ka, va), (kb, vb) in zip(kv_a, kv_b)
    )


def next_token_dist_with_kv(model, tokenizer, query, past_kv_tuple, past_len):
    q_ids = tokenizer(query, return_tensors="pt").input_ids
    q_len = q_ids.shape[1]
    attn = torch.ones(1, past_len + q_len, dtype=torch.long)
    cache = DynamicCache.from_legacy_cache(past_kv_tuple)
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


def greedy_with_kv(model, tokenizer, past_kv_tuple, past_len, query, max_new=25):
    q_ids = tokenizer(query, return_tensors="pt").input_ids
    q_len = q_ids.shape[1]
    attn = torch.ones(1, past_len + q_len, dtype=torch.long)
    cache = DynamicCache.from_legacy_cache(past_kv_tuple)
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


def run_condition(model, tokenizer, name, cond):
    print(f"\n{'-' * 60}\nCONDITION: {name}\n{'-' * 60}")
    print(f"A: {cond['A']}")
    print(f"B: {cond['B']}")
    print(f"Q: {cond['query'].strip()}")

    kv_a, len_a = prefill(model, tokenizer, cond["A"])
    kv_b, len_b = prefill(model, tokenizer, cond["B"])
    kv_composed = naive_concat_kv(kv_a, kv_b)
    composed_len = len_a + len_b

    full_prompt = cond["A"] + " " + cond["B"] + cond["query"]

    p_base = next_token_dist_from_scratch(model, tokenizer, full_prompt)
    p_comp = next_token_dist_with_kv(model, tokenizer, cond["query"], kv_composed, composed_len)

    kl = kl_divergence(p_base, p_comp)
    top1 = top_k_agreement(p_base, p_comp, 1)
    top5 = top_k_agreement(p_base, p_comp, 5)
    top10 = top_k_agreement(p_base, p_comp, 10)

    ans_base = greedy_from_scratch(model, tokenizer, full_prompt)
    ans_comp = greedy_with_kv(model, tokenizer, kv_composed, composed_len, cond["query"])

    print(f"  KL(base||comp):  {kl:.4f}")
    print(f"  Top-1 / 5 / 10:  {top1:.2f} / {top5:.2f} / {top10:.2f}")
    print(f"  Baseline answer: {ans_base!r}")
    print(f"  Composed answer: {ans_comp!r}")

    return {"cond": name, "kl": kl, "t1": top1, "t5": top5, "t10": top10}


def main():
    print(f"Loading {MODEL_NAME} (first run downloads ~1GB) ...")
    model, tokenizer = load_model()

    results = [run_condition(model, tokenizer, n, c) for n, c in CONDITIONS.items()]

    print(f"\n{'=' * 60}\nSUMMARY — naive concat vs full-prefill baseline\n{'=' * 60}")
    print(f"{'condition':<20}{'KL':>10}{'top1':>8}{'top5':>8}{'top10':>8}")
    for r in results:
        print(f"{r['cond']:<20}{r['kl']:>10.3f}{r['t1']:>8.2f}{r['t5']:>8.2f}{r['t10']:>8.2f}")

    print("\nExpected pattern:")
    print("  independent       — low KL, high top-k")
    print("  referential       — mid")
    print("  conflicting       — variable")
    print("  cross_inferential — high KL, low top-k  (motivates learned combiner)")


if __name__ == "__main__":
    main()
