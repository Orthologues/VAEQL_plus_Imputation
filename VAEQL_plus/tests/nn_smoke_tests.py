from __future__ import annotations

import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from VAEQL_plus.beta_DVAE.torch_nn import BetaGausMixedDVAE
from VAEQL_plus.tests import NN_SMOKE_TEST_SEED

# This deterministic smoke suite checks gradient connectivity. Larger repeated
# sweeps can be added later for empirical temperature selection.

## TODO: run 1000 randomized tests to choose between Vanilla Softmax K-logit activation / $\tau$ value for Gumbel-Softmax K-logit activation
## summarize the mean loss/gradient sum and their STD

FEAT_TYPE_DICT = {
    "all_feats": [
        "real_a",
        "real_b",
        "pos_real_a",
        "pos_real_b",
        "count_a",
        "count_b",
        "ord_a-ge_1",
        "ord_a-ge_2",
        "ord_a-ge_3",
        "ord_a-ge_4",
        "ord_b-ge_1",
        "ord_b-ge_2",
        "ord_b-ge_3",
        "ord_b-ge_4",
        "ord_b-ge_5",
        "ord_b-ge_6",
        "ord_b-ge_7",
        "ord_b-ge_8",
        "ord_b-ge_9",
        "bin_a",
        "bin_b",
        "cat_a-is_a",
        "cat_a-is_b",
        "cat_a-is_c",
        "cat_b-is_w",
        "cat_b-is_x",
        "cat_b-is_y",
        "cat_b-is_z",
    ],
    "real_val_feats": {"real_a", "real_b"},
    "pos_real_val_feats": {"pos_real_a", "pos_real_b"},
    "count_feats": {"count_a", "count_b"},
    "ord_feats": {
        "ord_a": 5,
        "ord_b": 10,
    },
    "bi_feats": {
        "bin_a": {"no", "yes"},
        "bin_b": {"false", "true"},
    },
    "cat_feats": {
        "cat_a": {"a", "b", "c"},
        "cat_b": {"w", "x", "y", "z"},
    },
}


def _print_smoke_header() -> None:
    print(
        "=============================================\n"
        "SMOKE TESTS for VAEQL_plus's torch.nn MODULE\n"
        "============================================="
    )


def _build_toy_batch() -> torch.Tensor:
    return torch.tensor(
        [
            [
                -0.75, 0.40, 0.10, 0.20, 0.00, 0.30,
                1.0, 0.0, 0.0, 0.0,
                1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0,
                1.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0,
            ],
            [
                -0.25, 0.15, 0.35, 0.45, 0.69, 0.85,
                1.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
            ],
            [
                0.25, -0.15, 0.70, 0.80, 1.10, 1.25,
                1.0, 1.0, 1.0, 0.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                1.0, 1.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, 1.0, 0.0,
            ],
            [
                0.75, -0.40, 1.15, 1.30, 1.39, 1.61,
                1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 0.0,
                1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        ],
        dtype=torch.float32,
    )


def _build_Gumbel_Softmax_model(tau_start: float) -> BetaGausMixedDVAE:
    torch.manual_seed(NN_SMOKE_TEST_SEED)
    tau_end = max(0.05, float(tau_start) / 2.0)
    return BetaGausMixedDVAE(
        input_dim=28,
        latent_dim=3,
        hidden_dim1=23,
        hidden_dim2=11,
        n_gmm_components=2,
        batch_size=4,
        tau_start=tau_start,
        tau_end=tau_end,
        device="cpu",
    )


def _reconstruction_loss_and_grad(
    model: BetaGausMixedDVAE,
    x: torch.Tensor,
    recon_logits: torch.Tensor,
    posterior_z_component_mean: torch.Tensor,
    posterior_z_component_logvar: torch.Tensor,
    posterior_k_probs: torch.Tensor,
) -> tuple[float, float, tuple[int, ...]]:
    loss = model.beta_capacity_loss(
        recon_logits=recon_logits,
        x_obs_processed=x,
        posterior_z_component_mean=posterior_z_component_mean,
        posterior_z_component_logvar=posterior_z_component_logvar,
        posterior_k_probs=posterior_k_probs,
        beta=0.0,
        capacity_C=0.0,
        gmm_prior=model.get_gmm_prior_params(),
        feat_type_dict=FEAT_TYPE_DICT,
        obs_mask=torch.zeros_like(x, dtype=torch.int8),
        num_feat_loss_metric="RMSE",
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()

    grad = model.encoder["gmm_encoder"]["k_logits_head"].weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    grad_sum = float(grad.abs().sum())
    assert grad_sum > 0.0
    return float(loss.total.detach()), grad_sum, tuple(grad.shape)


def _forward_with_vanilla_softmax(
    model: BetaGausMixedDVAE,
    x: torch.Tensor,
) -> tuple[torch.Tensor, BetaGausMixedDVAE.EncodingOutput]:
    encoded = model.encode(x)
    component_z = model.reparameterize_gaussian(
        encoded.posterior_z_component_mean,
        encoded.posterior_z_component_logvar,
    )
    k_weights = torch.softmax(encoded.posterior_k_logits, dim=1)
    z = torch.sum(k_weights.unsqueeze(-1) * component_z, dim=1)
    return model.decode(z), encoded


def test_k_logits_backprop_with_vanilla_softmax() -> None:
    _print_smoke_header()
    x = _build_toy_batch()
    # the input `tau_start=0.5` is only a placeholder
    model = _build_Gumbel_Softmax_model(tau_start=0.5)
    recon_logits, encoded = _forward_with_vanilla_softmax(model, x)

    loss, grad_sum, grad_shape = _reconstruction_loss_and_grad(
        model=model,
        x=x,
        recon_logits=recon_logits,
        posterior_z_component_mean=encoded.posterior_z_component_mean,
        posterior_z_component_logvar=encoded.posterior_z_component_logvar,
        posterior_k_probs=encoded.posterior_k_probs,
    )
    print(
        "vanilla_softmax "
        f"loss={loss:.4f} "
        f"k_logits_head_grad_sum={grad_sum:.6f} "
        f"k_logits_head_grad_shape={grad_shape}"
    )


def test_k_logits_backprop_with_gumbel_softmax_tau_grid() -> None:
    x = _build_toy_batch()
    for tau in (1.0, 0.5, 0.25, 0.1, 2.0, 4.0, 10.0):
        model = _build_Gumbel_Softmax_model(tau_start=tau)
        torch.manual_seed(NN_SMOKE_TEST_SEED)
        out = model(x)

        loss, grad_sum, grad_shape = _reconstruction_loss_and_grad(
            model=model,
            x=x,
            recon_logits=out["recon_logits"],
            posterior_z_component_mean=out["posterior_z_component_mean"],
            posterior_z_component_logvar=out["posterior_z_component_logvar"],
            posterior_k_probs=out["posterior_k_probs"],
        )
        print(
            f"gumbel_softmax_tau={tau:g} "
            f"loss={loss:.4f} "
            f"k_logits_head_grad_sum={grad_sum:.6f} "
            f"k_logits_head_grad_shape={grad_shape}"
        )
