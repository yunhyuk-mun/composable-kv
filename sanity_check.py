"""
Cache-injection sanity checks.

Confirms the pipeline itself is correct so that observed KL differences
between methods are attributable to the composition, not to injection
bugs.

Checks:
  1. Round-trip: prefill(A+B+Q) vs prefill(A+B) then query with kv_full.
     next-token distributions must match.
  2. RoPE-shift by 0 positions is identity on K.
  3. A-prefill is deterministic: same input twice, same K/V.
  4. A-portion of kv_full == kv_a (causal attention sanity).
"""

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


def prefill_ids(model, ids):
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    return out.past_key_values


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


def kl_divergence(p, q, eps=1e-12):
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum().item()


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


def check_1_roundtrip(model, tokenizer):
    """Same token ids on both paths -- avoids BPE boundary mismatch."""
    print("\n[Check 1] KV round-trip: prefill(prefix) then query, vs prefill(prefix+query)")
    print("  (using identical token ids on both paths)")
    max_kl = 0.0
    for cond_name, exs in CONDITIONS.items():
        for cond in exs:
            ids_a = tokenizer(cond["A"], return_tensors="pt").input_ids
            ids_b = tokenizer(cond["B"], return_tensors="pt").input_ids
            ids_q = tokenizer(cond["query"], return_tensors="pt").input_ids
            ids_prefix = torch.cat([ids_a, ids_b], dim=1)
            ids_all = torch.cat([ids_prefix, ids_q], dim=1)

            # Route 1: prefill prefix, then feed query with past_key_values
            kv_prefix = prefill_ids(model, ids_prefix)
            past_len = ids_prefix.shape[1]
            q_len = ids_q.shape[1]
            attn = torch.ones(1, past_len + q_len, dtype=torch.long)
            with torch.no_grad():
                out_route = model(input_ids=ids_q, attention_mask=attn, past_key_values=kv_prefix)
            p_route = F.softmax(out_route.logits[0, -1, :], dim=-1)

            # Route 2: prefill everything in one shot (same ids)
            with torch.no_grad():
                out_scratch = model(input_ids=ids_all)
            p_scratch = F.softmax(out_scratch.logits[0, -1, :], dim=-1)

            kl = kl_divergence(p_scratch, p_route)
            max_kl = max(max_kl, kl)
    print(f"  Max KL across 20 examples: {max_kl:.2e}")
    print(f"  Verdict: {'PASS (KL ~0)' if max_kl < 1e-3 else 'FAIL (unexpected divergence)'}")
    return max_kl


def check_2_rope_zero(model, tokenizer):
    print("\n[Check 2] RoPE-shift by delta=0 must be identity")
    ids_b = tokenizer(CONDITIONS['independent'][0]['B'], return_tensors="pt").input_ids
    kv_b = prefill_ids(model, ids_b)
    max_diff = 0.0
    for i in range(len(kv_b.layers)):
        k = kv_b.layers[i].keys
        k_shifted = rope_shift(k, delta=0)
        diff = (k - k_shifted).abs().max().item()
        max_diff = max(max_diff, diff)
    print(f"  Max |k - rope_shift(k, 0)| across 24 layers: {max_diff:.2e}")
    print(f"  Verdict: {'PASS' if max_diff < 1e-5 else 'FAIL'}")
    return max_diff


def check_3_determinism(model, tokenizer):
    print("\n[Check 3] Deterministic prefill: two runs of same input give same KV")
    ids_a = tokenizer(CONDITIONS['independent'][0]['A'], return_tensors="pt").input_ids
    kv1 = prefill_ids(model, ids_a)
    kv2 = prefill_ids(model, ids_a)
    max_k_diff = 0.0
    max_v_diff = 0.0
    for i in range(len(kv1.layers)):
        k_diff = (kv1.layers[i].keys - kv2.layers[i].keys).abs().max().item()
        v_diff = (kv1.layers[i].values - kv2.layers[i].values).abs().max().item()
        max_k_diff = max(max_k_diff, k_diff)
        max_v_diff = max(max_v_diff, v_diff)
    print(f"  Max K diff: {max_k_diff:.2e}, Max V diff: {max_v_diff:.2e}")
    print(f"  Verdict: {'PASS' if max(max_k_diff, max_v_diff) < 1e-5 else 'FAIL'}")
    return max(max_k_diff, max_v_diff)


def check_5_tokenization_alignment(model, tokenizer):
    """Independent tokenize(A) + tokenize(B) + tokenize(Q) vs joint tokenize(A+" "+B+Q).

    If they differ, KL diffs between composition and baseline paths may be
    partially explained by token-boundary drift rather than composition error.
    Reports the max token-count difference and, when lengths match, max
    mismatched token positions across all 20 examples.
    """
    print("\n[Check 5] Tokenization: tokenize(A)+tokenize(B)+tokenize(Q) vs tokenize(A+' '+B+Q)")
    n_length_mismatch = 0
    max_len_diff = 0
    max_pos_diffs = 0
    n_examples = 0
    for cond_name, exs in CONDITIONS.items():
        for cond in exs:
            n_examples += 1
            ids_a = tokenizer(cond["A"], return_tensors="pt").input_ids
            ids_b = tokenizer(cond["B"], return_tensors="pt").input_ids
            ids_q = tokenizer(cond["query"], return_tensors="pt").input_ids
            joint_ids = torch.cat([ids_a, ids_b, ids_q], dim=1)
            direct_ids = tokenizer(cond["A"] + " " + cond["B"] + cond["query"], return_tensors="pt").input_ids

            if joint_ids.shape[1] != direct_ids.shape[1]:
                n_length_mismatch += 1
                max_len_diff = max(max_len_diff, abs(joint_ids.shape[1] - direct_ids.shape[1]))
            else:
                pos_diffs = (joint_ids != direct_ids).sum().item()
                max_pos_diffs = max(max_pos_diffs, pos_diffs)
    print(f"  examples with length mismatch: {n_length_mismatch}/{n_examples}")
    print(f"  max length difference (tokens): {max_len_diff}")
    print(f"  max positional token diffs (when lengths match): {max_pos_diffs}")
    ok = (n_length_mismatch == 0) and (max_pos_diffs == 0)
    print(f"  Verdict: {'PASS (both routes identical)' if ok else 'INFORMATIONAL (boundary drift present)'}")
    return (n_length_mismatch, max_len_diff, max_pos_diffs)


def check_4_a_portion(model, tokenizer):
    print("\n[Check 4] A-portion of kv_full equals kv_a (causal attention)")
    ids_a = tokenizer(CONDITIONS['independent'][0]['A'], return_tensors="pt").input_ids
    ids_b = tokenizer(CONDITIONS['independent'][0]['B'], return_tensors="pt").input_ids
    ids_full = torch.cat([ids_a, ids_b], dim=1)
    len_a = ids_a.shape[1]
    kv_a = prefill_ids(model, ids_a)
    kv_full = prefill_ids(model, ids_full)
    max_k_diff = 0.0
    max_v_diff = 0.0
    for i in range(len(kv_a.layers)):
        k_full_a = kv_full.layers[i].keys[:, :, :len_a, :]
        v_full_a = kv_full.layers[i].values[:, :, :len_a, :]
        k_diff = (kv_a.layers[i].keys - k_full_a).abs().max().item()
        v_diff = (kv_a.layers[i].values - v_full_a).abs().max().item()
        max_k_diff = max(max_k_diff, k_diff)
        max_v_diff = max(max_v_diff, v_diff)
    print(f"  Max K diff: {max_k_diff:.2e}, Max V diff: {max_v_diff:.2e}")
    print(f"  Verdict: {'PASS' if max(max_k_diff, max_v_diff) < 1e-4 else 'FAIL'} (float32 precision ~1e-5)")
    return max(max_k_diff, max_v_diff)


def main():
    print(f"Loading {MODEL_NAME} ...")
    model, tokenizer = load_model()

    r1 = check_1_roundtrip(model, tokenizer)
    r2 = check_2_rope_zero(model, tokenizer)
    r3 = check_3_determinism(model, tokenizer)
    r4 = check_4_a_portion(model, tokenizer)
    r5 = check_5_tokenization_alignment(model, tokenizer)

    print(f"\n{'=' * 60}\nSANITY SUMMARY\n{'=' * 60}")
    print(f"  round-trip max KL:      {r1:.2e}   (< 1e-3 expected)")
    print(f"  rope-shift(0) max diff: {r2:.2e}   (< 1e-5 expected)")
    print(f"  determinism max diff:   {r3:.2e}   (< 1e-5 expected)")
    print(f"  A-portion max diff:     {r4:.2e}   (< 1e-4 expected, float32)")
    print(f"  tokenize alignment:     {r5[0]} length-mismatched examples, max len_diff={r5[1]}, max pos_diffs={r5[2]}")

    ok = (r1 < 1e-3) and (r2 < 1e-5) and (r3 < 1e-5) and (r4 < 1e-4)
    print(f"\n  Pipeline sanity (checks 1-4): {'ALL PASS' if ok else 'FAILURE - do not trust downstream numbers'}")
    print(f"  Check 5 is informational: if boundary drift exists, downstream comparisons\n"
          f"  should account for it. Both mve.py and h3_holdout.py use identical token ids\n"
          f"  on both routes to avoid this confound in the main results.")


if __name__ == "__main__":
    main()
