import numpy as np
import torch

from VAEQL_plus.step1_preprocessing.feat_preprocessor import OrderedFeature
from VAEQL_plus.beta_DVAE.torch_nn import BetaGausMixedDVAE
from VAEQL_plus.step3_training_and_imputation.prototype import (
    Step3Handoff,
    _feature_aware_MI_aggregate,
    generate_feature_aware_MI_aggregate,
)


def _scalar_spec(name: str, feat_type: str) -> OrderedFeature:
    return OrderedFeature(name=name, feat_type=feat_type, base_feat=name)


def _ordinal_specs(base_feature: str, n_levels: int) -> tuple[OrderedFeature, ...]:
    return tuple(
        OrderedFeature(
            name=f"{base_feature}-ge_{threshold}",
            feat_type="ordinal",
            base_feat=base_feature,
        )
        for threshold in range(1, n_levels)
    )


def _categorical_specs(
    base_feature: str, levels: tuple[str, ...]
) -> tuple[OrderedFeature, ...]:
    return tuple(
        OrderedFeature(
            name=f"{base_feature}-is_{level}",
            feat_type="categorical",
            base_feat=base_feature,
        )
        for level in levels
    )


#COMMENTLOG: this test file is a prototype, it will need constant updates due to the modifications for the module `VAEQL_plus.step3_training_and_imputation`
def test_feature_aware_aggregate_preserves_valid_feature_encodings():
    specs = (
        _scalar_spec("real", "real_val"),
        _scalar_spec("count", "count_val"),
        _scalar_spec("binary", "bi_val"),
        *_ordinal_specs("ordinal", n_levels=3),
        *_categorical_specs("category", ("a", "b", "c")),
        *_categorical_specs("two_level", ("no", "yes")),
    )
    X0 = np.asarray(
        [
            [1.25, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    source_mask = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 1, 1, 1, 2, 2, 2, 1, 1],
        ],
        dtype=np.int8,
    )
    imputation_draws = [
        np.asarray(
            [
                # Predictions for observed row 0 must not overwrite its source values.
                [-4.0, 4.0, 1.0, 0.1, 0.9, 0.0, 1.0, 0.0, 0.0, 1.0],
                [12.0, -6.0, 0.6, 0.8, 0.3, 0.3, 0.45, 0.25, 0.8, 0.2],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [-3.0, 3.0, 0.0, 0.9, 0.1, 1.0, 0.0, 0.0, 1.0, 0.0],
                [8.0, -4.0, 0.8, 0.6, 0.5, 0.5, 0.2, 0.3, 0.6, 0.4],
            ],
            dtype=np.float64,
        ),
    ]

    aggregate = _feature_aware_MI_aggregate(
        imputation_draws, X0, source_mask, specs
    )

    np.testing.assert_array_equal(aggregate[0].detach().numpy(), X0[0])
    assert aggregate[1, 0].item() == 5.0  # numeric transformed features are clipped
    assert aggregate[1, 1].item() == -5.0  # standardized log-count is clipped to [-5, 5]
    torch.testing.assert_close(aggregate[1, 2], torch.tensor(0.7))
    torch.testing.assert_close(aggregate[1, 3:5], torch.tensor([0.7, 0.4]))
    assert aggregate[1, 3] >= aggregate[1, 4]  # mean cumulative probabilities remain monotone
    torch.testing.assert_close(aggregate[1, 5:8], torch.tensor([0.4, 0.325, 0.275]))
    assert torch.isclose(aggregate[1, 5:8].sum(), torch.tensor(1.0))
    torch.testing.assert_close(aggregate[1, 8:10], torch.tensor([0.7, 0.3]))
    assert torch.isclose(aggregate[1, 8:10].sum(), torch.tensor(1.0))


def test_feature_aware_aggregate_averages_soft_discrete_draws() -> None:
    specs = (
        _scalar_spec("bi_demo", "binary"),
        *_categorical_specs("cat_demo", ("a", "b")),
    )
    X0 = np.zeros((1, 3), dtype=np.float32)
    source_mask = np.asarray([[1, 2, 2]], dtype=np.int8)
    imputation_draws = [
        np.asarray([[0.51, 0.99, 0.01]], dtype=np.float32),
        np.asarray([[0.51, 0.40, 0.60]], dtype=np.float32),
        np.asarray([[0.00, 0.40, 0.60]], dtype=np.float32),
    ]

    aggregate = _feature_aware_MI_aggregate(
        imputation_draws, X0, source_mask, specs
    )

    torch.testing.assert_close(
        aggregate,
        torch.tensor([[0.34, 1.79 / 3, 1.21 / 3]]),
    )
    assert torch.isclose(aggregate[0, 1:].sum(), torch.tensor(1.0))
    assert torch.all((aggregate[0, 1:] > 0.0) & (aggregate[0, 1:] < 1.0))


def test_batch_handoff_returns_soft_aggregate_with_vae_gradients():
    # A small differentiable stand-in makes the handoff's draw and gradient behavior explicit.
    class DrawModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Shared trainable scalar used by each draw to test gradient flow into the model.
            self.anchor = torch.nn.Parameter(torch.tensor(0.25))
            self.next_draw = 0  # produces a different output on each imputation call

        def impute_multiple(
            self, model_input, max_iter, obs_mask, feat_type_dict, update_mask, differentiable=False
        ):
            assert max_iter == 2  # handoff forwards the configured imputation-cycle count
            # The observed first column is 1; both source-missing and amputed cells are 0.
            torch.testing.assert_close(
                obs_mask,
                torch.tensor([[1, 0, 0, 0]], dtype=obs_mask.dtype, device=obs_mask.device),
            )
            torch.testing.assert_close(
                update_mask,
                torch.tensor([[False, True, True, True]], device=update_mask.device),
            )
            assert differentiable  # handoff requests a graph-connected imputation
            offset = self.next_draw  # draw-specific offset, from 0 through 4
            self.next_draw += 1  # advance so the next draw differs
            numeric = (self.anchor + offset).reshape(1, 1)  # differentiable scalar feature
            # A soft two-class probability vector, also connected to `anchor`.
            # For the first draw, the stacked logits are [0.25, 0.0]; softmax gives
            # approximately [0.562, 0.438], then reshape makes the category shape (1, 2).
            zero_logit = torch.zeros_like(self.anchor)
            category = torch.softmax(
                torch.stack((self.anchor + offset / 10, zero_logit)),
                dim=0,
            ).reshape(1, 2)
            # Give the count prediction the draw offset only, independent of `anchor`.
            count = torch.full_like(numeric, float(offset))
            # Match feature-spec order: count, real value, then the two categorical columns.
            return torch.cat((count, numeric, category), dim=1)

    # The selected row has an observed count, an amputed real value, and a source-missing category.
    specs = (
        _scalar_spec("count", "count"),
        _scalar_spec("real", "real_val"),
        *_categorical_specs("category", ("a", "b")),
    )
    X0 = torch.tensor([[0.4, 0.0, 0.0, 0.0], [0.2, 0.1, 1.0, 0.0]])
    source_mask = torch.tensor([[0, 2, 1, 1], [0, 0, 0, 0]], dtype=torch.int8)
    model = DrawModel()  # its `anchor` parameter lets the test inspect gradient flow

    # Configure five stochastic draws and two iterative imputation cycles per draw.
    handoff = Step3Handoff(
        model=model,
        X0=X0,
        source_mask=source_mask,
        feature_specs=specs,
        feature_dict={},
        training_metrics={},
        num_imputations=5,
        imputation_cycles=2,
    )
    # Even under no-grad at the caller, the handoff explicitly enables gradients internally.
    with torch.no_grad():
        aggregate = handoff.impute_batch([0])

    # The observed count is restored from X0 despite deliberately different model predictions.
    torch.testing.assert_close(aggregate[0, 0], X0[0, 0])
    # Real-feature draws are 0.25, 1.25, 2.25, 3.25, 4.25; their mean is 2.25.
    torch.testing.assert_close(aggregate[0, 1], torch.tensor(2.25))
    # The averaged categorical probabilities still sum to one.
    assert torch.isclose(aggregate[0, 2:].sum(), torch.tensor(1.0))
    # The returned aggregate retains its autograd graph despite the outer no-grad context.
    assert aggregate.requires_grad
    # Backpropagating through the numeric aggregate reaches the model parameter.
    aggregate[0, 1].backward()
    # The test’s fake model defines each real-feature draw as `numeric = anchor + offset`, so each draw has `gradient == 1` with respect to anchor
    torch.testing.assert_close(model.anchor.grad, torch.tensor(1.0))


def test_step3_imputation_backpropagates_through_type1_and_type2_cells():
    model = BetaGausMixedDVAE(
        input_dim=3,
        latent_dim=2,
        hidden_dim1=8,
        hidden_dim2=4,
        n_gmm_components=2,
    )
    X0 = torch.tensor([[0.2, 0.5, 0.7], [0.4, 0.6, 0.9]])
    # Columns represent previously amputed (type 2), observed (type 0), and
    # source-missing (type 1) values, respectively.
    source_mask = torch.tensor([[2, 0, 1], [2, 0, 1]], dtype=torch.int8)
    feature_specs = (
        _scalar_spec("type2_value", "real_val"),
        _scalar_spec("observed_value", "real_val"),
        _scalar_spec("type1_value", "real_val"),
    )

    # Step 3 recodes both missingness types to update-mask value 2 before imputation.
    with torch.no_grad():
        imputed = generate_feature_aware_MI_aggregate(
            model=model,
            X0=X0,
            source_mask=source_mask,
            feature_dict=None,
            feature_specs=feature_specs,
            num_imputations=2,
            imputation_cycles=2,
        )

    # Step 3 enables autograd internally and preserves observed values unchanged.
    assert imputed.requires_grad
    torch.testing.assert_close(imputed[:, 1], X0[:, 1].to(imputed.device))

    decoder_parameters = tuple(
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("decoder")
    )
    # Check each imputed column separately so type-1 and type-2 gradient paths are both tested.
    for column in (0, 2):
        gradients = torch.autograd.grad(
            imputed[:, column].sum(),
            decoder_parameters,
            retain_graph=(column == 0),
            #allow_unused=True,
        )
        assert any(
            gradient is not None and torch.count_nonzero(gradient).item() > 0
            for gradient in gradients
        )
