"""
Sanity test for RoPE-shift implementation on Qwen-2.5-0.5B.

Method: Take K from B-alone prefill, shift by len_A, compare against
full-prefill's B-portion K. If shift is correct, cosine similarity
should increase substantially from the naive (unshifted) baseline.

Any remaining gap after shift = pure cross-context effect on K
representation (analogous to V's cross-context effect).
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def rope_shift(k, delta, rope_theta=ROPE_THETA):
    """Rotate stored K forward by `delta` positions using half-rotation RoPE."""
    D = k.shape[-1]
    half = D // 2
    device, dtype = k.device, k.dtype

    i = torch.arange(half, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (rope_theta ** (2 * i / D))
    angles = float(delta) * inv_freq
    cos = torch.cos(angles).to(dtype).view(1, 1, 1, half)
    sin = torch.sin(angles).to(dtype).view(1, 1, 1, half)

    k1 = k[..., :half]
    k2 = k[..., half:]
    k1_new = k1 * cos - k2 * sin
    k2_new = k1 * sin + k2 * cos
    return torch.cat([k1_new, k2_new], dim=-1)


def cos_sim_mean(x, y):
    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = y.reshape(-1, y.shape[-1])
    return F.cosine_similarity(x_flat, y_flat, dim=-1).mean().item()


def test():
    print(f"Loading {MODEL_NAME} ...")
    model, tokenizer = load_model()

    A = "Alice is a doctor. She lives in Seoul and works at Samsung Medical Center."
    B = "The capital of France is Paris. It has a population of about 2.1 million."

    ids_a = tokenizer(A, return_tensors="pt").input_ids
    ids_b = tokenizer(B, return_tensors="pt").input_ids
    ids_full = torch.cat([ids_a, ids_b], dim=1)
    len_a = ids_a.shape[1]
    len_b = ids_b.shape[1]
    print(f"len_a={len_a}, len_b={len_b}")

    kv_b = prefill_ids(model, ids_b)
    kv_full = prefill_ids(model, ids_full)

    print(f"\n{'layer':<8}{'raw K cos':>14}{'shifted K cos':>18}{'V cos (ref)':>14}")
    total_raw = 0.0
    total_shifted = 0.0
    total_v = 0.0
    n = len(kv_full.layers)

    for L in range(n):
        kb = kv_b.layers[L].keys
        vb = kv_b.layers[L].values
        kf_b = kv_full.layers[L].keys[:, :, len_a:, :]
        vf_b = kv_full.layers[L].values[:, :, len_a:, :]

        raw_cos = cos_sim_mean(kb, kf_b)
        kb_shifted = rope_shift(kb, delta=len_a)
        shifted_cos = cos_sim_mean(kb_shifted, kf_b)
        v_cos = cos_sim_mean(vb, vf_b)

        total_raw += raw_cos
        total_shifted += shifted_cos
        total_v += v_cos

        print(f"L{L:<7d}{raw_cos:>14.4f}{shifted_cos:>18.4f}{v_cos:>14.4f}")

    print(f"\n{'MEAN':<8}{total_raw/n:>14.4f}{total_shifted/n:>18.4f}{total_v/n:>14.4f}")
    print(f"\nInterpretation:")
    print(f"  raw K cos    = position mismatch + cross-context effect")
    print(f"  shifted K cos = ONLY cross-context effect (position fixed)")
    print(f"  V cos         = ONLY cross-context effect (V has no RoPE, reference)")
    print(f"\nIf RoPE-shift is correct, shifted K cos should be much closer to V cos.")


if __name__ == "__main__":
    test()
