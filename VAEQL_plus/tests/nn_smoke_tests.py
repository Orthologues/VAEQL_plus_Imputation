#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-08-01
# Description: Deterministic neural-network smoke tests for the beta-DVAE.
# Development: Mainly written with GPT-5.5 Medium/GPT-5.6 Luna-XHigh on Codex, with Jiawei Zhao's human
# review and revisions.
#########################################################

from __future__ import annotations

import math
import torch
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from VAEQL_plus.beta_DVAE.torch_nn import BetaGausMixedDVAE
from VAEQL_plus.tests import NN_SMOKE_TEST_SEED

# This deterministic smoke suite checks gradient connectivity. Larger repeated
# sweeps can be added later for empirical temperature selection.

RANDOMIZED_ACTIVATION_TRIALS_PER_CANDIDATE = 100
GUMBEL_SOFTMAX_TAU_CANDIDATES = (1.0, 0.5, 0.25, 0.1, 2.0, 4.0, 10.0)

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


def _build_random_toy_batch(generator: torch.Generator, batch_size: int = 4) -> torch.Tensor:
    numeric = torch.randn(batch_size, 6, generator=generator)
    numeric[:, 2:4] = torch.rand(batch_size, 2, generator=generator)
    numeric[:, 4:6] = torch.poisson(
        torch.rand(batch_size, 2, generator=generator) * 2.0
    )

    ord_a_level = torch.randint(0, 5, (batch_size, 1), generator=generator)
    ord_a = (ord_a_level >= torch.arange(1, 5)).to(torch.float32)
    ord_b_level = torch.randint(0, 10, (batch_size, 1), generator=generator)
    ord_b = (ord_b_level >= torch.arange(1, 10)).to(torch.float32)

    binary = torch.randint(0, 2, (batch_size, 2), generator=generator).to(torch.float32)

    cat_a_idx = torch.randint(0, 3, (batch_size,), generator=generator)
    cat_a = torch.nn.functional.one_hot(cat_a_idx, num_classes=3).to(torch.float32)
    cat_b_idx = torch.randint(0, 4, (batch_size,), generator=generator)
    cat_b = torch.nn.functional.one_hot(cat_b_idx, num_classes=4).to(torch.float32)

    return torch.cat((numeric, ord_a, ord_b, binary, cat_a, cat_b), dim=1)


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


def test_component_wise_posterior_sampling_keeps_gmm_axis() -> None:
    x = _build_toy_batch()
    model = _build_Gumbel_Softmax_model(tau_start=0.5)
    encoded = model.encode(x)

    expected_component_shape = (x.shape[0], model.k_gmm, model.z_dim)
    assert encoded.posterior_z_component_mean.shape == expected_component_shape
    assert encoded.posterior_z_component_logvar.shape == expected_component_shape
    assert encoded.posterior_k_logits.shape == (x.shape[0], model.k_gmm)
    assert encoded.posterior_k_probs.shape == (x.shape[0], model.k_gmm)

    torch.manual_seed(NN_SMOKE_TEST_SEED)
    z = model.reparameterize_mixture(
        encoded.posterior_z_component_mean,
        encoded.posterior_z_component_logvar,
        encoded.posterior_k_logits,
    )
    assert z.shape == (x.shape[0], model.z_dim)

    # Calling ``.mean(dim=1)` averages over the GMM component axis K, collapsing it into: `(B, Z)`
    aggregated_mean = encoded.posterior_z_component_mean.mean(dim=1)
    aggregated_logvar = encoded.posterior_z_component_logvar.mean(dim=1)
    
    # this call is expected to fail since `aggregated_mean` and `aggregated_logvar` 
    # both do not possess the correct shape `(B, K, Z)`
    with pytest.raises(ValueError, match="must have shape"):
        model.reparameterize_mixture(
            aggregated_mean,
            aggregated_logvar,
            encoded.posterior_k_logits,
        )


def test_randomized_k_logits_activation_selection_summary() -> None:
    input_generator = torch.Generator().manual_seed(NN_SMOKE_TEST_SEED + 10_000)
    candidates = [("vanilla_softmax", _build_Gumbel_Softmax_model(tau_start=0.5))]
    candidates.extend(
        (f"gumbel_softmax_tau={tau:g}", _build_Gumbel_Softmax_model(tau_start=tau))
        for tau in GUMBEL_SOFTMAX_TAU_CANDIDATES
    )
    stats = {
        name: {"loss": [], "grad_sum": []}
        for name, _model in candidates
    }

    trials_per_candidate = RANDOMIZED_ACTIVATION_TRIALS_PER_CANDIDATE
    for candidate_idx, (name, model) in enumerate(candidates):
        for trial_idx in range(trials_per_candidate):
            x = _build_random_toy_batch(input_generator)
            sample_seed = (
                NN_SMOKE_TEST_SEED
                + 20_000
                + candidate_idx * trials_per_candidate
                + trial_idx
            )
            model.zero_grad(set_to_none=True)
            torch.manual_seed(sample_seed)
            if name == "vanilla_softmax":
                recon_logits, encoded = _forward_with_vanilla_softmax(model, x)
                loss, grad_sum, _grad_shape = _reconstruction_loss_and_grad(
                    model=model,
                    x=x,
                    recon_logits=recon_logits,
                    posterior_z_component_mean=encoded.posterior_z_component_mean,
                    posterior_z_component_logvar=encoded.posterior_z_component_logvar,
                    posterior_k_probs=encoded.posterior_k_probs,
                )
            else:
                out = model(x)
                loss, grad_sum, _grad_shape = _reconstruction_loss_and_grad(
                    model=model,
                    x=x,
                    recon_logits=out["recon_logits"],
                    posterior_z_component_mean=out["posterior_z_component_mean"],
                    posterior_z_component_logvar=out["posterior_z_component_logvar"],
                    posterior_k_probs=out["posterior_k_probs"],
                )
            stats[name]["loss"].append(loss)
            stats[name]["grad_sum"].append(grad_sum)

    summaries = []
    for name, values in stats.items():
        loss_tensor = torch.tensor(values["loss"])
        grad_tensor = torch.tensor(values["grad_sum"])
        summaries.append(
            (
                float(loss_tensor.mean()),
                name,
                float(loss_tensor.std(unbiased=False)),
                float(grad_tensor.mean()),
                float(grad_tensor.std(unbiased=False)),
            )
        )

    summaries.sort()
    best_loss = min(summaries, key=lambda row: row[0])
    best_loss_std = min(summaries, key=lambda row: row[2])
    
    _print_smoke_header()
    print("loss_metric=entire_type_aware_grouped_recon_loss_of_disentangled_beta_VAE")
    print(f"activation_selection_summary trials_per_candidate={trials_per_candidate}")
    loss_mean, name, loss_std, grad_mean, grad_std = best_loss
    print("############################################################")
    print(
        f"winner_loss_candidate={name} "
        f"loss_mean={loss_mean:.6f} "
        f"loss_std={loss_std:.6f} "
        f"k_logits_head_grad_sum_mean={grad_mean:.6f} "
        f"k_logits_head_grad_sum_std={grad_std:.6f}"
    )
    print("############################################################")
    loss_mean, name, loss_std, grad_mean, grad_std = best_loss_std
    print(
        f"winner_loss_std_candidate={name} "
        f"loss_mean={loss_mean:.6f} "
        f"loss_std={loss_std:.6f} "
        f"k_logits_head_grad_sum_mean={grad_mean:.6f} "
        f"k_logits_head_grad_sum_std={grad_std:.6f}"
    )
    print("############################################################")
    for loss_mean, name, loss_std, grad_mean, grad_std in summaries:
        print(
            f"{name} "
            f"loss_mean={loss_mean:.6f} "
            f"loss_std={loss_std:.6f} "
            f"k_logits_head_grad_sum_mean={grad_mean:.6f} "
            f"k_logits_head_grad_sum_std={grad_std:.6f}"
        )


def test_k_logits_backprop_with_vanilla_softmax() -> None:
    x = _build_toy_batch()
    # the input `tau_start=0.5` is only a placeholder
    model = _build_Gumbel_Softmax_model(tau_start=0.5)
    recon_logits, encoded = _forward_with_vanilla_softmax(model, x)

    _reconstruction_loss_and_grad(
        model=model,
        x=x,
        recon_logits=recon_logits,
        posterior_z_component_mean=encoded.posterior_z_component_mean,
        posterior_z_component_logvar=encoded.posterior_z_component_logvar,
        posterior_k_probs=encoded.posterior_k_probs,
    )


def test_k_logits_backprop_with_gumbel_softmax_tau_grid() -> None:
    x = _build_toy_batch()
    for tau in (1.0, 0.5, 0.25, 0.1, 2.0, 4.0, 10.0):
        model = _build_Gumbel_Softmax_model(tau_start=tau)
        torch.manual_seed(NN_SMOKE_TEST_SEED)
        out = model(x)

        _reconstruction_loss_and_grad(
            model=model,
            x=x,
            recon_logits=out["recon_logits"],
            posterior_z_component_mean=out["posterior_z_component_mean"],
            posterior_z_component_logvar=out["posterior_z_component_logvar"],
            posterior_k_probs=out["posterior_k_probs"],
        )


def test_reconstruction_loss_counts_encoded_groups_once() -> None:
    x = _build_toy_batch()
    model = _build_Gumbel_Softmax_model(tau_start=0.5)
    encoded = model.encode(x)
    recon_logits = torch.zeros_like(x)

    loss = model.beta_capacity_loss(
        recon_logits=recon_logits,
        x_obs_processed=x,
        posterior_z_component_mean=encoded.posterior_z_component_mean,
        posterior_z_component_logvar=encoded.posterior_z_component_logvar,
        posterior_k_probs=encoded.posterior_k_probs,
        beta=0.0,
        capacity_C=0.0,
        gmm_prior=model.get_gmm_prior_params(),
        feat_type_dict=FEAT_TYPE_DICT,
        obs_mask=torch.zeros_like(x, dtype=torch.int8),
    )

    # Feature-level loss units:
    # 6 numeric columns, 2 ordinal groups, 2 binary columns, 2 categorical groups.
    # Zero binary logits give BCE=log(2), cat_a CE=log(3), cat_b CE=log(4),
    # ordinal raw outputs are converted through positive softplus gaps, and
    # numeric features use the same z-score-discounted RMSE rule as the model.
    numeric_expected_terms = []
    for col_idx in range(6):
        target_col = x[:, col_idx]
        z_score_disc_coef = torch.exp(-0.5 * target_col.pow(2))
        numeric_expected_terms.append(
            torch.sqrt((target_col.pow(2) * z_score_disc_coef).mean() + 1e-12)
        )
    ordinal_expected_terms = []
    name_to_index = {
        feat_name: idx for idx, feat_name in enumerate(FEAT_TYPE_DICT["all_feats"])
    }
    for feat, n_orders in sorted(FEAT_TYPE_DICT["ord_feats"].items()):
        grp_idx = [
            name_to_index[f"{feat}-ge_{order}"]
            for order in range(1, int(n_orders))
        ]
        ordered_logits = model._ordered_ordinal_logits(recon_logits[:, grp_idx])
        ordinal_target = x[:, grp_idx].clamp(0.0, 1.0)
        ordinal_bce_raw = torch.nn.functional.binary_cross_entropy_with_logits(
            ordered_logits,
            ordinal_target,
            reduction="none",
        )
        ordinal_expected_terms.append(ordinal_bce_raw.mean())
    expected = (
        torch.stack(numeric_expected_terms + ordinal_expected_terms).sum()
        + 2 * math.log(2.0)  # two binary features
        + math.log(3.0)  # the first categorical feature with three categories
        + math.log(4.0)  # the second categorical feature with four categories
    ) / 12
    abs_diff = torch.abs(loss.recon_loss - expected.to(dtype=loss.recon_loss.dtype))
    
    _print_smoke_header()
    print(
        "feature_grouped_recon_loss "
        f"actual={float(loss.recon_loss):.6f} "
        f"expected={float(expected):.6f} "
        f"abs_diff={float(abs_diff):.6f}"
    )
    
    assert torch.isclose(
        loss.recon_loss,
        expected.to(dtype=loss.recon_loss.dtype),
        atol=1e-6,
    )


def test_ordinal_activation_enforces_monotone_cumulative_probs() -> None:
    r"""Ordinal activation should enforce cumulative ordering:
        $P(Y \ge r) \ge P(Y \ge r+1)$.
    """
    model = _build_Gumbel_Softmax_model(tau_start=0.5)
    x = _build_toy_batch()
    assert x.shape[1] == len(FEAT_TYPE_DICT["all_feats"])
    recon_raw = torch.zeros_like(x)
    name_to_index = {
        feat_name: idx for idx, feat_name in enumerate(FEAT_TYPE_DICT["all_feats"])
    }

    expected_probs_by_feat = {}
    for feat, n_orders in FEAT_TYPE_DICT["ord_feats"].items():
        grp_idx = [
            name_to_index[f"{feat}-ge_{order}"]
            for order in range(1, int(n_orders))
        ]
        eta_smoke = torch.linspace(-0.5, 0.5, x.shape[0], dtype=torch.float32)
        recon_raw[:, grp_idx[0]] = eta_smoke
        assert len(grp_idx) > 1
        # These remaining raw decoder columns parameterize positive logit gaps
        # via softplus, so each later `feat-ge_*` logit is lower than the
        # previous cumulative-threshold logit.
        raw_gap_logits_smoke = torch.randn(
            x.shape[0],
            len(grp_idx) - 1,
            generator=torch.Generator().manual_seed(NN_SMOKE_TEST_SEED),
        )
        recon_raw[:, grp_idx[1:]] = raw_gap_logits_smoke
        positive_gaps = torch.nn.functional.softplus(raw_gap_logits_smoke)
        assert torch.all(positive_gaps > 0)
        expected_logits = torch.cat(
            (
                eta_smoke.unsqueeze(1),
                eta_smoke.unsqueeze(1) - torch.cumsum(positive_gaps, dim=1),
            ),
            dim=1,
        )
        expected_probs_by_feat[feat] = (
            grp_idx,
            torch.sigmoid(expected_logits),
        )
    
    recon = model.activate_reconstruction(recon_raw, FEAT_TYPE_DICT)

    for grp_idx, expected_probs in expected_probs_by_feat.values():
        activated_probs = recon[:, grp_idx]
        assert torch.all(activated_probs[:, :-1] >= activated_probs[:, 1:])
        # `rtol` is the allowed relative error scaled by the expected value magnitude. Default: 1e-5
        # `atol` is the allowed absolute error floor for values close to zero. Default: 1e-8
        assert torch.allclose(activated_probs, expected_probs)
