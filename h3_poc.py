"""
H3 POC: can a learned linear map close the V cross-context gap?

Minimal proof-of-concept. Single layer (L12, where V cosine was lowest
in analyze_layers.py). Simplest possible corrector: 64x64 linear map W.

Setup:
  - Collect (V_composed_B, V_full_B) pairs at layer L12 across all 20
    controlled examples
  - Flatten to per-token per-head samples
  - Fit W (via least-squares) on training portion
  - Report cosine improvement on held-out portion

Fails-informatively: if even the linear map doesn't help, cross-context
correction may need non-linear or context-conditional models.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from conditions import CONDITIONS

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
TARGET_LAYER = 12
TRAIN_FRAC = 0.8


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
    """Return (V_composed_B, V_full_B) tensors flattened over all examples.

    Shapes returned: [N, 64] and [N, 64], where N = sum over examples of
    (num_kv_heads * len_B).
    """
    x_list = []
    y_list = []
    example_ids = []
    all_examples = []
    for cond_name, exs in CONDITIONS.items():
        for i, cond in enumerate(exs):
            all_examples.append((cond_name, i, cond))

    for ex_idx, (cond_name, i, cond) in enumerate(all_examples):
        ids_a = tokenizer(cond["A"], return_tensors="pt").input_ids
        ids_b = tokenizer(cond["B"], return_tensors="pt").input_ids
        ids_full = torch.cat([ids_a, ids_b], dim=1)
        len_a = ids_a.shape[1]

        kv_b = prefill_ids(model, ids_b)
        kv_full = prefill_ids(model, ids_full)

        v_b = kv_b.layers[layer_idx].values                # [1, H, T_b, 64]
        v_full_b = kv_full.layers[layer_idx].values[:, :, len_a:, :]  # [1, H, T_b, 64]

        # Flatten to [H * T_b, 64]
        v_b_flat = v_b.squeeze(0).reshape(-1, v_b.shape[-1])
        v_full_flat = v_full_b.squeeze(0).reshape(-1, v_full_b.shape[-1])

        x_list.append(v_b_flat)
        y_list.append(v_full_flat)
        example_ids.extend([ex_idx] * v_b_flat.shape[0])

        print(f"  ex {ex_idx:2d} ({cond_name:<18} #{i}): {v_b_flat.shape[0]} samples")

    X = torch.cat(x_list, dim=0)  # [N, 64]
    Y = torch.cat(y_list, dim=0)  # [N, 64]
    ex_ids = torch.tensor(example_ids)
    return X, Y, ex_ids


def leave_out_split(X, Y, ex_ids, holdout_examples):
    mask_test = torch.zeros_like(ex_ids, dtype=torch.bool)
    for eid in holdout_examples:
        mask_test |= (ex_ids == eid)
    mask_train = ~mask_test
    return X[mask_train], Y[mask_train], X[mask_test], Y[mask_test]


def fit_linear(X_train, Y_train, ridge=1e-4):
    """Solve W = argmin ||X W - Y||^2 + ridge * ||W||^2, W in R^{64x64}."""
    XtX = X_train.T @ X_train
    XtY = X_train.T @ Y_train
    reg = ridge * torch.eye(XtX.shape[0])
    W = torch.linalg.solve(XtX + reg, XtY)
    return W


def cosine(a, b):
    return F.cosine_similarity(a, b, dim=-1).mean().item()


def evaluate(W, X_test, Y_test):
    Y_pred = X_test @ W
    baseline_cos = cosine(X_test, Y_test)  # composed vs full (no correction)
    corrected_cos = cosine(Y_pred, Y_test)  # corrected vs full
    return baseline_cos, corrected_cos


def main():
    print(f"Loading {MODEL_NAME} ...")
    model, tokenizer = load_model()

    print(f"\nCollecting (V_composed_B, V_full_B) pairs at layer L{TARGET_LAYER} ...")
    X, Y, ex_ids = collect_pairs(model, tokenizer, TARGET_LAYER)
    print(f"\nTotal samples: N = {X.shape[0]}, dim = {X.shape[1]}")

    n_examples = int(ex_ids.max().item()) + 1
    print(f"Examples: {n_examples}, train/test split = leave-N-out at example level")

    # Leave-4-out cross-validation (4/20 examples per fold, 5 folds)
    folds = [list(range(k, n_examples, 5)) for k in range(5)]

    print(f"\n{'fold':<6}{'baseline cos':>15}{'corrected cos':>18}{'delta':>10}")
    baselines = []
    correcteds = []
    for fold_idx, holdout in enumerate(folds):
        X_tr, Y_tr, X_te, Y_te = leave_out_split(X, Y, ex_ids, holdout)
        W = fit_linear(X_tr, Y_tr)
        base, corr = evaluate(W, X_te, Y_te)
        baselines.append(base)
        correcteds.append(corr)
        print(f"{fold_idx:<6}{base:>15.4f}{corr:>18.4f}{corr - base:>+10.4f}")

    mean_base = sum(baselines) / len(baselines)
    mean_corr = sum(correcteds) / len(correcteds)
    print(f"\n{'MEAN':<6}{mean_base:>15.4f}{mean_corr:>18.4f}{mean_corr - mean_base:>+10.4f}")

    print(f"\n{'=' * 60}")
    print(f"H3 POC verdict (layer L{TARGET_LAYER}):")
    if mean_corr > mean_base + 0.02:
        print(f"  POSITIVE: learned linear map improves V cosine by "
              f"{mean_corr - mean_base:.4f}. Extend to more layers / non-linear.")
    elif mean_corr > mean_base:
        print(f"  MARGINAL: tiny improvement {mean_corr - mean_base:.4f}. "
              f"Linear too simple, but direction correct.")
    else:
        print(f"  NEGATIVE: no improvement. Linear map cannot close gap. "
              f"Try context-conditional (also feed V_A).")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
