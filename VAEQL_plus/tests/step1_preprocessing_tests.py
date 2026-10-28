#########################################################
# Feature-aware placeholder tests for Step 1 preprocessing.
#########################################################

from __future__ import annotations

import numpy as np
import pandas as pd

from VAEQL_plus.step1_preprocessing import feat_preprocessor as preprocessing_module
from VAEQL_plus.step1_preprocessing.feat_preprocessor import FeaturePreprocessor


ORD_GROUPS = {"ordinal": ["ordinal-ge_1", "ordinal-ge_2"]}
BI_GROUPS = {"binary": [0, 1]}
CAT_GROUPS = {
    "category": ["category-is_a", "category-is_b", "category-is_c"]
}


def _make_preprocessor() -> FeaturePreprocessor:
    feat_dict = {
        "all_feats": {"real", "positive", "count", "ordinal", "binary", "category"},
        "real_val_feats": {"real"},
        "pos_real_val_feats": {"positive"},
        "count_feats": {"count"},
        "ord_feats": {"ordinal": 3},
        "bi_feats": {"binary": {"no", "yes"}},
        "cat_feats": {"category": {"a", "b", "c"}},
    }
    raw_data = pd.DataFrame(
        {
            "real": [1.0, 3.0, 5.0],
            "positive": [1.0, 2.0, 3.0],
            "count": [0.0, 1.0, 2.0],
            "ordinal": [1, 2, 3],
            "binary": ["yes", "no", "yes"],
            "category": ["a", "b", "c"],
        }
    )
    return FeaturePreprocessor(
        feat_dict=feat_dict,
        missing_mechanism="MAR",
        missing_rate=0.2,
        input_df=raw_data,
    )


def _transformed_example() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "real": [1.0, 3.0, np.nan, 5.0],
            "positive": [-2.0, 1.0, np.nan, 3.0],
            "count": [0.0, 4.0, np.nan, 9.0],
            "ordinal-ge_1": [1.0, 1.0, np.nan, 0.0],
            "ordinal-ge_2": [0.0, 1.0, np.nan, 0.0],
            "binary": [1.0, 0.0, np.nan, 1.0],
            "category-is_a": [1.0, 0.0, np.nan, 0.0],
            "category-is_b": [0.0, 1.0, np.nan, 0.0],
            "category-is_c": [0.0, 0.0, np.nan, 1.0],
        }
    )


def test_source_prefill_uses_type_aware_marginals(monkeypatch) -> None:
    preprocessor = _make_preprocessor()
    pre_df = _transformed_example()
    captured_inputs: list[pd.DataFrame] = []

    class IdentityAmputation:
        def __init__(self, **kwargs) -> None:
            pass

        def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
            captured_inputs.append(data.copy())
            return data.to_numpy(dtype=np.float64)

    monkeypatch.setattr(
        preprocessing_module,
        "MultivariateAmputation",
        IdentityAmputation,
    )
    amputed_df, mask = preprocessor._apply_prefill_and_pyampute(
        pre_df,
        ORD_GROUPS,
        BI_GROUPS,
        CAT_GROUPS,
    )

    assert captured_inputs
    assert all(not frame.isna().any().any() for frame in captured_inputs)
    expected = np.asarray(
        [3.0, 1.0, 4.0, 2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    )
    assert np.allclose(captured_inputs[0].iloc[2].to_numpy(dtype=float), expected)
    assert np.all(mask[2] == 1)
    assert np.allclose(amputed_df.iloc[2].to_numpy(dtype=float), expected)


def test_post_amputation_fills_only_type2_from_type0_marginals() -> None:
    preprocessor = _make_preprocessor()
    data = pd.DataFrame(
        [
            [1.0, -2.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
            [np.nan] * 9,
            [5.0, 3.0, 9.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ],
        columns=_transformed_example().columns,
    )
    mask = np.asarray(
        [
            [0] * 9,
            [1] * 9,
            [2] * 9,
            [0] * 9,
        ],
        dtype=np.int8,
    )

    output = preprocessor._fill_amputed_placeholders(
        data,
        mask,
        ORD_GROUPS,
        BI_GROUPS,
        CAT_GROUPS,
    )

    expected_type2 = np.asarray([3.0, 0.5, 4.5, 0.5, 0.0, 1.0, 0.5, 0.0, 0.5])
    assert np.allclose(output.iloc[2].to_numpy(dtype=float), expected_type2)
    assert np.allclose(output.iloc[1].to_numpy(dtype=float), np.full(9, 7.0))
    assert np.allclose(output.iloc[0].to_numpy(dtype=float), data.iloc[0].to_numpy(dtype=float))
    assert np.isclose(output.iloc[2][CAT_GROUPS["category"]].sum(), 1.0)
    assert output.iloc[2][ORD_GROUPS["ordinal"][0]] >= output.iloc[2][ORD_GROUPS["ordinal"][1]]


def test_placeholder_fallbacks_are_finite_and_group_valid() -> None:
    empty_data = pd.DataFrame(
        np.nan,
        index=range(2),
        columns=_transformed_example().columns,
    )

    placeholders = FeaturePreprocessor._compute_placeholder_values(
        empty_data,
        np.zeros(empty_data.shape, dtype=bool),
        ORD_GROUPS,
        BI_GROUPS,
        CAT_GROUPS,
    )

    assert np.isfinite(placeholders).all()
    assert np.allclose(placeholders[:3], [0.0, 0.0, 0.0])
    assert np.allclose(placeholders[3:5], [2.0 / 3.0, 1.0 / 3.0])
    assert placeholders[5] == 0.5
    assert np.allclose(placeholders[6:], [1.0 / 3.0] * 3)


def test_string_binary_categories_are_mapped_to_numeric_bernoulli_values() -> None:
    preprocessor = _make_preprocessor()

    binary_df, groups = preprocessor._transform_binary(
        {"binary": {"no", "yes"}}
    )

    assert binary_df["binary"].tolist() == [1.0, 0.0, 1.0]
    assert groups["binary"] == ["no", "yes"]


def test_count_features_are_log1p_z_scored_and_preserve_missing_values() -> None:
    preprocessor = _make_preprocessor()
    preprocessor.input_df = preprocessor.input_df.reindex(range(4))
    preprocessor.input_df.loc[:, "count"] = [0.0, 1.0, 1_000_000.0, np.nan]
    transformed, columns = preprocessor._transform_count(["count"])

    log_values = np.log1p(np.asarray([0.0, 1.0, 1_000_000.0], dtype=np.float32))
    expected = (log_values - log_values.mean()) / log_values.std()

    assert columns == ["count"]
    assert np.allclose(transformed.loc[:2, "count"].to_numpy(), expected)
    assert np.isnan(transformed.loc[3, "count"])
    assert expected[2] < 5.0  # a large raw count is compressed by log1p, not raw-capped


def test_constant_count_feature_is_z_scored_without_division_by_zero() -> None:
    preprocessor = _make_preprocessor()
    preprocessor.input_df.loc[:, "count"] = [2.0, 2.0, np.nan]
    transformed, _ = preprocessor._transform_count(["count"])

    assert np.allclose(transformed.loc[:1, "count"].to_numpy(), [0.0, 0.0])
    assert np.isnan(transformed.loc[2, "count"])
