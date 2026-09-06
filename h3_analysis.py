"""
H3 corrector interpretability: what does the learned W actually do?

Trains W on all 20 examples (same fit as h3_holdout), then analyzes:
  - Singular value spectrum: is the correction low-rank?
  - How far is W from identity? (does it just pass-through, or transform?)
  - Which V_composed directions get amplified vs suppressed?
  - Cosine improvement per singular component (rank-k approximations).

Saves:
  paper/W_spectrum.pdf/png  -- singular value + rank-k cosine curve
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from transformers import AutoModelForCausalLM, AutoTokenizer

from conditions import CONDITIONS

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
TARGET_LAYER = 12
HEAD_DIM = 64

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    return model, tokenizer


def prefill_ids(model, ids):
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    return out.past_key_values


def collect_pairs(model, tokenizer, layer_idx):
    X_list, Y_list = [], []
    for cond_name, exs in CONDITIONS.items():
        for cond in exs:
            ids_a = tokenizer(cond["A"], return_tensors="pt").input_ids
            ids_b = tokenizer(cond["B"], return_tensors="pt").input_ids
            ids_full = torch.cat([ids_a, ids_b], dim=1)
            len_a = ids_a.shape[1]

            kv_b = prefill_ids(model, ids_b)
            kv_full = prefill_ids(model, ids_full)

            v_b = kv_b.layers[layer_idx].values.squeeze(0).reshape(-1, HEAD_DIM)
            v_full_b = kv_full.layers[layer_idx].values[:, :, len_a:, :].squeeze(0).reshape(-1, HEAD_DIM)
            X_list.append(v_b)
            Y_list.append(v_full_b)
    return torch.cat(X_list, 0), torch.cat(Y_list, 0)


def fit_W(X, Y, ridge=1e-4):
    return torch.linalg.solve(X.T @ X + ridge * torch.eye(HEAD_DIM), X.T @ Y)


def cos_sim(a, b):
    return F.cosine_similarity(a, b, dim=-1).mean().item()


def main():
    print(f"Loading {MODEL_NAME} ...")
    model, tokenizer = load_model()

    print(f"Collecting V pairs at layer L{TARGET_LAYER} ...")
    X, Y = collect_pairs(model, tokenizer, TARGET_LAYER)
    print(f"Samples: {X.shape[0]}, dim = {HEAD_DIM}")

    W = fit_W(X, Y)

    # --- SVD of W ---
    U, S, Vh = torch.linalg.svd(W)
    print(f"\nSingular values (first 10 of 64):")
    for i, s in enumerate(S[:10].tolist()):
        print(f"  s{i:2d} = {s:.4f}")
    print(f"  ...  s63 = {S[-1].item():.4f}")

    # Ratio to identity: if W ~= identity, singular values all ~1
    identity_dist = (W - torch.eye(HEAD_DIM)).norm().item()
    frobenius_W = W.norm().item()
    print(f"\n||W - I||_F = {identity_dist:.4f}")
    print(f"||W||_F     = {frobenius_W:.4f}")
    print(f"||W - I|| / ||W|| = {identity_dist/frobenius_W:.4f}   "
          f"(0 = W is identity, ~1 = W very different from I)")

    # Effective rank: how many singular values above threshold
    thresh = 0.01 * S[0].item()
    eff_rank = int((S > thresh).sum().item())
    print(f"\nEffective rank (s > 1% of s0): {eff_rank}/64")

    # --- Rank-k truncated W: how much of the improvement comes from top k? ---
    baseline_cos = cos_sim(X, Y)  # V_b vs V_full_b (no correction)
    full_cos = cos_sim(X @ W, Y)  # V_b @ W vs V_full_b
    print(f"\nBaseline cos (no correction):  {baseline_cos:.4f}")
    print(f"Full-rank W corrected cos:     {full_cos:.4f}")
    print(f"Gain from full W: {full_cos - baseline_cos:+.4f}")

    ranks = list(range(1, 33)) + [40, 48, 56, 64]
    rank_cos = []
    for k in ranks:
        Wk = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
        rank_cos.append(cos_sim(X @ Wk, Y))

    print(f"\nRank-k truncated W cosine (V_b @ W_k vs V_full_b):")
    for k, c in zip(ranks, rank_cos):
        gain_pct = 100 * (c - baseline_cos) / max(1e-6, full_cos - baseline_cos)
        print(f"  k={k:3d}  cos={c:.4f}  ({gain_pct:5.1f}% of full gain)")

    # --- Plot ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # Left: singular value spectrum
    axL.plot(range(1, HEAD_DIM + 1), S.tolist(), "o-",
             color="#1b9e77", markersize=4, linewidth=1.2)
    axL.set_yscale("log")
    axL.set_xlabel("Singular value index")
    axL.set_ylabel("Singular value (log scale)")
    axL.set_title(f"Singular value spectrum of learned W (L{TARGET_LAYER})")
    axL.grid(True, alpha=0.3, which="both")
    axL.axhline(thresh, color="grey", linestyle=":", linewidth=1,
                label=f"1% of s0 ({eff_rank}/64 dims above)")
    axL.legend(loc="upper right")

    # Right: rank-k cosine curve
    axR.plot(ranks, rank_cos, "o-", color="#7570b3", markersize=5, linewidth=1.4,
             label="V @ W_k vs V_full")
    axR.axhline(baseline_cos, color="#4d4d4d", linestyle="--", linewidth=1,
                label=f"baseline (no correction) = {baseline_cos:.3f}")
    axR.axhline(full_cos, color="#1b9e77", linestyle="--", linewidth=1,
                label=f"full-rank W = {full_cos:.3f}")
    axR.set_xlabel("Rank k of truncated W")
    axR.set_ylabel("Mean V cosine similarity (in-sample)")
    axR.set_title("Rank-k correction efficiency")
    axR.set_xlim(0, 66)
    axR.grid(True, alpha=0.3)
    axR.legend(loc="lower right", frameon=True)

    plt.suptitle("What does the H3 corrector learn?  (layer 12, 20-example fit)",
                 y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig("paper/W_spectrum.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("paper/W_spectrum.png", format="png", dpi=200, bbox_inches="tight")
    print("\nWrote paper/W_spectrum.pdf and paper/W_spectrum.png")

    # --- Additional analysis: which V directions matter? ---
    # V_composed_B has some natural covariance. Project onto W's singular directions.
    V_normalized = X - X.mean(dim=0)
    var_per_dir = ((V_normalized @ Vh.T) ** 2).mean(dim=0)  # variance along each right-singular-vec
    print(f"\nTop 5 right-singular-vectors of W ranked by their explained variance in V_composed:")
    order = torch.argsort(var_per_dir, descending=True)
    for idx in order[:5].tolist():
        print(f"  s-index {idx:2d}: singular value = {S[idx].item():.4f}, "
              f"V-var along this dir = {var_per_dir[idx].item():.4f}")


if __name__ == "__main__":
    main()
