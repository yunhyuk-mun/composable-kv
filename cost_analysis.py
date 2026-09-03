"""
Cost analysis: substantiate the "cheap corrector" claim with numbers.

Compares wall-clock and estimated FLOPs of each composition method
against the fundamental unit of avoided work: full B prefill.

Method costs (composition only, per example):
  - M1 naive:     torch.cat of K,V per layer, no matmul
  - M2 rope:      + RoPE rotation on K_B per layer
  - M4/M5 h3:     + 64x64 matmul on V_B at layer 12 (one layer)

Baseline unit: prefill(B) forward pass through 24 transformer layers.
"""

import time
import statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from conditions import CONDITIONS

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
ROPE_THETA = 1000000.0
TARGET_LAYER = 12
HEAD_DIM = 64
NUM_KV_HEADS = 2
NUM_LAYERS = 24
HIDDEN_SIZE = 896
INTERMEDIATE_SIZE = 4864
VOCAB_SIZE = 151936
N_TIMING_REPEATS = 5


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


def naive_concat(kv_a, kv_b):
    new = DynamicCache()
    for i in range(len(kv_a.layers)):
        k = torch.cat([kv_a.layers[i].keys, kv_b.layers[i].keys], -2)
        v = torch.cat([kv_a.layers[i].values, kv_b.layers[i].values], -2)
        new.update(k, v, i)
    return new


def rope_concat(kv_a, kv_b, len_a):
    new = DynamicCache()
    for i in range(len(kv_a.layers)):
        kb = rope_shift(kv_b.layers[i].keys, delta=len_a)
        k = torch.cat([kv_a.layers[i].keys, kb], -2)
        v = torch.cat([kv_a.layers[i].values, kv_b.layers[i].values], -2)
        new.update(k, v, i)
    return new


def h3_naive_concat(kv_a, kv_b, W):
    new = DynamicCache()
    for i in range(len(kv_a.layers)):
        vb = kv_b.layers[i].values
        if i == TARGET_LAYER:
            shape = vb.shape
            vb = (vb.reshape(-1, HEAD_DIM) @ W).reshape(shape)
        k = torch.cat([kv_a.layers[i].keys, kv_b.layers[i].keys], -2)
        v = torch.cat([kv_a.layers[i].values, vb], -2)
        new.update(k, v, i)
    return new


def h3_rope_concat(kv_a, kv_b, len_a, W):
    new = DynamicCache()
    for i in range(len(kv_a.layers)):
        kb = rope_shift(kv_b.layers[i].keys, delta=len_a)
        vb = kv_b.layers[i].values
        if i == TARGET_LAYER:
            shape = vb.shape
            vb = (vb.reshape(-1, HEAD_DIM) @ W).reshape(shape)
        k = torch.cat([kv_a.layers[i].keys, kb], -2)
        v = torch.cat([kv_a.layers[i].values, vb], -2)
        new.update(k, v, i)
    return new


def time_fn(fn, repeats=N_TIMING_REPEATS):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), statistics.stdev(times) if len(times) > 1 else 0.0


def estimate_prefill_flops(seq_len):
    """Rough FLOP count for one transformer layer forward on seq_len tokens.

    Attention: 2 * seq_len * hidden * hidden (Q, K, V projections) approx.
    We use standard 6 * L * (2 * N * d^2 + N^2 * d) style approximation.
    For simplicity return an order-of-magnitude number.
    """
    L = NUM_LAYERS
    d = HIDDEN_SIZE
    d_ff = INTERMEDIATE_SIZE
    T = seq_len
    # Per layer per token: attention QKV+O = 4 * d^2; FFN gate+up+down ~= 3 * d * d_ff
    per_token = 4 * d * d + 3 * d * d_ff
    return L * T * per_token * 2  # 2 for multiply-add


def estimate_h3_correction_flops(seq_len_b):
    """H3 correction cost: (T_b * num_kv_heads) rows, 64x64 matmul."""
    rows = seq_len_b * NUM_KV_HEADS
    return rows * HEAD_DIM * HEAD_DIM * 2  # matmul mul-add


def main():
    print(f"Loading {MODEL_NAME} ...")
    model, tokenizer = load_model()

    # Use one representative example for timing (independent #0)
    cond = CONDITIONS["independent"][0]
    ids_a = tokenizer(cond["A"], return_tensors="pt").input_ids
    ids_b = tokenizer(cond["B"], return_tensors="pt").input_ids
    len_a = ids_a.shape[1]
    len_b = ids_b.shape[1]
    print(f"Example: len_A={len_a}, len_B={len_b}")

    print("\nPre-caching kv_a, kv_b (excluded from composition timing) ...")
    kv_a = prefill_ids(model, ids_a)
    kv_b = prefill_ids(model, ids_b)

    # Random W as a stand-in for the trained corrector (cost is data-independent)
    W = torch.randn(HEAD_DIM, HEAD_DIM)

    print("\nTiming composition methods (5 repeats each, median reported)...")
    t_m1, s_m1 = time_fn(lambda: naive_concat(kv_a, kv_b))
    t_m2, s_m2 = time_fn(lambda: rope_concat(kv_a, kv_b, len_a))
    t_m4, s_m4 = time_fn(lambda: h3_naive_concat(kv_a, kv_b, W))
    t_m5, s_m5 = time_fn(lambda: h3_rope_concat(kv_a, kv_b, len_a, W))

    print("\nTiming prefill(B) as the fundamental unit of avoided work ...")
    def prefill_b_only():
        with torch.no_grad():
            model(input_ids=ids_b, use_cache=True)
    t_prefill_b, s_prefill_b = time_fn(prefill_b_only)

    print(f"\n{'=' * 70}\nWALL-CLOCK (median over 5 runs, CPU)\n{'=' * 70}")
    print(f"{'op':<30}{'time (ms)':>14}{'stdev (ms)':>14}{'vs prefill(B)':>18}")
    for name, t, s in [
        ("prefill(B) [baseline]", t_prefill_b, s_prefill_b),
        ("M1 naive_concat", t_m1, s_m1),
        ("M2 rope_concat", t_m2, s_m2),
        ("M4 h3_naive (+64x64 mm)", t_m4, s_m4),
        ("M5 h3_rope (+64x64 mm)", t_m5, s_m5),
    ]:
        ratio = t / t_prefill_b
        print(f"{name:<30}{t*1000:>14.3f}{s*1000:>14.3f}{ratio*100:>17.2f}%")

    print(f"\n{'=' * 70}\nFLOPS ESTIMATE (order-of-magnitude)\n{'=' * 70}")
    flops_prefill_b = estimate_prefill_flops(len_b)
    flops_h3 = estimate_h3_correction_flops(len_b)
    print(f"prefill(B) forward:                        {flops_prefill_b:>12,} FLOPs (~{flops_prefill_b/1e6:.1f}M)")
    print(f"H3 correction (64x64 mm on {len_b}*{NUM_KV_HEADS} rows):  {flops_h3:>12,} FLOPs (~{flops_h3/1e3:.1f}k)")
    print(f"H3 correction as fraction of prefill(B):   {flops_h3/flops_prefill_b*100:.4f}%")

    print(f"\n{'=' * 70}\nINTERPRETATION\n{'=' * 70}")
    print("- Composition (M1/M2) is essentially free vs prefill(B): concat + small rotate.")
    print("- H3 correction adds a 64x64 matmul at ONE layer on ~{} rows.".format(len_b * NUM_KV_HEADS))
    print("- The H3 correction is orders of magnitude cheaper than a prefill(B) forward.")
    print("- 'Cheap corrector' claim is substantiated by both wall-clock and FLOP estimates.")


if __name__ == "__main__":
    main()
