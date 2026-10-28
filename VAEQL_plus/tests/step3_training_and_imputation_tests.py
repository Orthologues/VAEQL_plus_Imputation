import numpy as np
import torch

from VAEQL_plus.step1_preprocessing.feat_preprocessor import OrderedFeature
from VAEQL_plus.step3_training_and_imputation.prototype import (
    _feature_aware_aggregate,
    generate_feature_aware_aggregate,
)


def test_feature_aware_aggregate_preserves_valid_feature_encodings():
    specs = (
        OrderedFeature("real", "real_val", "real"),
        OrderedFeature("count", "count", "count"),
        OrderedFeature("binary", "binary", "binary"),
        OrderedFeature("ordinal-ge_1", "ordinal", "ordinal"),
        OrderedFeature("ordinal-ge_2", "ordinal", "ordinal"),
        OrderedFeature("category-is_a", "categorical", "category"),
        OrderedFeature("category-is_b", "categorical", "category"),
        OrderedFeature("category-is_c", "categorical", "category"),
        OrderedFeature("two_level-is_no", "categorical", "two_level"),
        OrderedFeature("two_level-is_yes", "categorical", "two_level"),
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
    mean_predictions = np.asarray(
        [
            # Predictions for observed row 0 must not overwrite its source values.
            [-4.0, 4.0, 1.0, 0.1, 0.9, 0.0, 1.0, 0.0, 0.0, 1.0],
            [12.0, -6.0, 0.6, 0.8, 0.3, 0.3, 0.45, 0.25, 0.8, 0.2],
        ],
        dtype=np.float64,
    )

    aggregate = _feature_aware_aggregate(mean_predictions, X0, source_mask, specs)

    np.testing.assert_array_equal(aggregate[0], X0[0])
    assert aggregate[1, 0] == 5.0  # numeric transformed features are clipped
    assert aggregate[1, 1] == -5.0  # standardized log-count is clipped to [-5, 5]
    assert aggregate[1, 2] == 1.0  # binary output is restored to {0, 1}
    np.testing.assert_array_equal(aggregate[1, 3:5], [1.0, 0.0])
    assert aggregate[1, 3] >= aggregate[1, 4]  # unary ordinal thresholds stay monotone
    np.testing.assert_array_equal(aggregate[1, 5:8], [0.0, 1.0, 0.0])
    assert aggregate[1, 5:8].sum() == 1.0
    np.testing.assert_array_equal(aggregate[1, 8:10], [1.0, 0.0])
    assert aggregate[1, 8:10].sum() == 1.0


def test_generation_aggregates_draws_without_returning_the_draw_stack():
    class DrawModel(torch.nn.Module):
        def __init__(self, draws):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.draws = iter(draws)

        def impute_multiple(self, model_input, max_iter, X_mask, feat_type_dict):
            assert max_iter == 2
            assert torch.all(X_mask == 2)
            return torch.as_tensor(next(self.draws), dtype=torch.float32, device=model_input.device)

    specs = (
        OrderedFeature("count", "count", "count"),
        OrderedFeature("category-is_a", "categorical", "category"),
        OrderedFeature("category-is_b", "categorical", "category"),
    )
    X0 = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    source_mask = np.asarray([[1, 2, 2]], dtype=np.int8)
    draws = [
        np.asarray([[value, *category]], dtype=np.float32)
        for value, category in zip(
            (1.0, 2.0, 3.0, 4.0, 5.0),
            ([1, 0], [0, 1], [1, 0], [0, 1], [0, 1]),
        )
    ]
    model = DrawModel(draws)

    aggregate = generate_feature_aware_aggregate(
        model=model,
        X0=X0,
        source_mask=source_mask,
        feature_dict={},
        feature_specs=specs,
        num_imputations=5,
        imputation_cycles=2,
    )

    np.testing.assert_array_equal(aggregate, [[3.0, 0.0, 1.0]])
