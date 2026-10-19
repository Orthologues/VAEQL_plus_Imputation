#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-02-19
# Description1: Feature preprocessor utilities for the VAE-Q learning imputation baseline based on the provided features in "VAEQL_plus.conf.FeaturesTypeDict".
# Description2: This module includes the class(es) with functions for preprocessing features according to their types (real-valued, positive real-valued, count, ordinal, binary, categorical) as defined in the FeaturesTypeDict.
# Description3: 
# Feature transformation assumptions before the hybrid type-aware reconstruction objective are:
# - Real-valued features: Gaussian-style transformed-space reconstruction (Z-score scaling mean and std estimated from observed data)
# - Positive real-valued features: Yeo-Johnson transform, then Z-score scaling
#   this is not an explicit log-normal decoder likelihood.
# - Count features: nonnegative count-valued variables transformed with log1p and clipping
#   this is not an explicit Poisson decoder likelihood.
#   Clinical count variables can be overdispersed, zero-inflated, outlier-prone,
#   bounded by study design, or administratively coded, so we do not assume a
#   Poisson distribution for count values here.
# - Ordinal features: Ordinal logit-model (unary encoding, then monotone cumulative logits in $\beta$-DVAE reconstruction)
#   reconstructed with grouped threshold-wise binary losses after the softplus-gap ordered-logit transform;
#   this enforces monotone cumulative probabilities for feat-ge_* columns.
# - Binary features: Bernoulli distribution (numeric-coded values stay in {0, 1}; string categories map to canonical {0, 1} codes)
# - Categorical features: Categorical distribution (one-hot encoding, pre-activated transform as logits, then applied with Gumbel-Softmax activation to add more stochasticity compared to Vanilla-Softmax)
#########################################################


from __future__ import annotations


from collections import namedtuple
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from pyampute.ampute import MultivariateAmputation
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from torch import Tensor

from ..conf import FeaturesTypeDict

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame


OrderedFeature = namedtuple("OrderedFeature", ["name", "feat_type", "base_feat"])


class FeaturePreprocessor:

    def __init__(
        self,
        feat_dict: FeaturesTypeDict,
        missing_mechanism: str,
        missing_rate: float,
        input_df: Union[DataFrame, SparkDataFrame] = None,
        use_spark: bool = False,
    ):
        self.feat_dict = feat_dict
        self.use_spark = use_spark
        self.MECHANISMS = {"MAR", "MNAR", "MCAR"}
        self.MAX_MISSING_RATE = 0.5

        if input_df is None:
            raise ValueError("input_df cannot be None, a DataFrame must be provided for preprocessing")
        if use_spark:
            try:
                from pyspark.sql import DataFrame as SparkDataFrame
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "pyspark is required only when use_spark=True; install it in "
                    "the preprocessing runtime or use the pandas backend."
                ) from exc
            if not isinstance(input_df, SparkDataFrame):
                raise TypeError("Expected a Spark DataFrame when use_spark is True")
        if not use_spark and not isinstance(input_df, DataFrame):
            raise TypeError("Expected a pandas DataFrame when use_spark is False")
        self.input_df = input_df

        if "pos_real_val_feats" not in self.feat_dict:
            raise KeyError("feat_dict must include mandatory key 'pos_real_val_feats'")

        if not (0.0 < missing_rate <= self.MAX_MISSING_RATE):
            raise ValueError(f"missing_rate must satisfy 0 < missing_rate <= {self.MAX_MISSING_RATE!r}, got {missing_rate!r}")
        self.missing_rate = float(missing_rate)

        mechanism = missing_mechanism.upper()
        if mechanism not in self.MECHANISMS:
            raise ValueError(f"missing_mechanism must be one of {self.MECHANISMS!r}, got {missing_mechanism!r}")
        self.missing_mechanism = mechanism
        
        self.ordered_feats_info_tuple: Tuple[OrderedFeature, ...] = tuple()
        self.missingness_mask: np.ndarray | None = None

    # =========================================================================
    # Main entry point: transforms features, applies synthetic amputation,
    # fills masked entries with provisional placeholders, and returns the tensor,
    # missingness mask, and ordered feature specifications.
    # =========================================================================
    def preprocess(self) -> Tuple[Tensor, Tuple[OrderedFeature, ...]]:
        self._validate_columns()

        # Reorder by feature families and encode special feature types.
        #   Ordinal/categorical groups are tracked for grouped amputation.
        pre_df, real_cols, pos_cols, count_cols, ord_groups, bi_groups, cat_groups = self._build_preprocessed_dataframe()

        # ampute with pyampute and generate missingness mask
        amputed_df, mask = self._apply_prefill_and_pyampute(
            pre_df,
            ord_groups,
            bi_groups,
            cat_groups,
        )
        # Fill only newly amputed cells with fast, feature-aware marginals.
        model_input_df = self._fill_amputed_placeholders(
            amputed_df,
            mask,
            ord_groups,
            bi_groups,
            cat_groups,
        )
        x = model_input_df.to_numpy(dtype=np.float32, copy=True)
        self.ordered_feats_info_tuple = self._build_ordered_feature_ordered_feats(
            ordered_cols=list(model_input_df.columns),
            real_cols=real_cols,
            pos_cols=pos_cols,
            count_cols=count_cols,
            ord_groups=ord_groups,
            bi_groups=bi_groups,
            cat_groups=cat_groups,
        )
        self.missingness_mask = mask.astype(np.int8)

        return torch.from_numpy(x), self.ordered_feats_info_tuple


    # =========================================================================
    # Wrappers (dispatch only): one method per feature family
    # Returns to:
    ## EITHER an unchanged DataFrame based on the input DF & and an empty list, 
    ## OR a preprocessed DataFrame and the list of column names (after encoding if applicable) for the feature family.
    # =========================================================================
    def preprocess_real_valued_features(self) -> Tuple[DataFrame, List[str]]:
        cols = sorted(self.feat_dict.get("real_val_feats", set()))
        if not cols:
            return pd.DataFrame(index=self._row_index()), []
        return self._transform_real(cols, spark=self.use_spark)

    def preprocess_positive_real_valued_features(self) -> Tuple[DataFrame, List[str]]:
        cols = sorted(self.feat_dict["pos_real_val_feats"])
        if not cols:
            return pd.DataFrame(index=self._row_index()), []
        return self._transform_pos_real(cols, spark=self.use_spark)

    def preprocess_count_features(self) -> Tuple[DataFrame, List[str]]:
        cols = sorted(self.feat_dict.get("count_feats", set()))
        if not cols:
            return pd.DataFrame(index=self._row_index()), []
        return self._transform_count(cols, spark=self.use_spark)

    def preprocess_ordinal_features(self) -> Tuple[DataFrame, Dict[str, List[str]]]:
        ord_feats = self.feat_dict.get("ord_feats", {})
        if not ord_feats:
            return pd.DataFrame(index=self._row_index()), {}
        return self._transform_ordinal(ord_feats, spark=self.use_spark)

    def preprocess_binary_features(self) -> Tuple[DataFrame, Dict[str, List[str]]]:
        bi_feats = self.feat_dict.get("bi_feats", {})
        if not bi_feats:
            return pd.DataFrame(index=self._row_index()), {}
        return self._transform_binary(bi_feats, spark=self.use_spark)

    def preprocess_categorical_features(self) -> Tuple[DataFrame, Dict[str, List[str]]]:
        cat_feats = self.feat_dict.get("cat_feats", {})
        if not cat_feats:
            return pd.DataFrame(index=self._row_index()), {}
        return self._transform_categorical(cat_feats, spark=self.use_spark)
    
    
    # =========================================================================
    # Actual preprocessing implementations (backend-specific)
    # =========================================================================
    # Distribution: Gaussian (real-valued features)
    # Transform stage here: numeric coercion + per-column z-score scaling.
    # Post-scale clipping to [-5, 5] reduces extreme outliers before model training.
    def _transform_real(self, cols: Sequence[str], spark: bool = False) -> Tuple[DataFrame, List[str]]:
        if spark:
            from pyspark.sql import functions as F

            pre_df = self.input_df.select(
                *[F.col(col).cast("double").alias(col) for col in cols]
            ).toPandas()
        else:
            pre_df = self.input_df.loc[:, list(cols)]
        pre_df = pre_df.apply(pd.to_numeric, errors="coerce")
        for col in cols:
            series = pre_df[col].astype(np.float32)
            observed_mask = series.notna()
            if observed_mask.any():
                scaler = StandardScaler(with_mean=True, with_std=True)
                scaled_values = scaler.fit_transform(
                    series.loc[observed_mask].to_numpy().reshape(-1, 1)
                ).reshape(-1).astype(np.float32)
                scaled_values = np.clip(scaled_values, -5.0, 5.0).astype(np.float32)
                series.loc[observed_mask] = scaled_values
            pre_df[col] = series
        return pre_df, list(cols)

    # Distribution: Log-normal (positive real-valued features)
    # Transform stage here: Yeo-Johnson transform with standardization.
    # Note: Yeo-Johnson handles the stabilizing shift/shape internally; values are clipped at zero first.
    # Post-transform clipping to [-5, 5] reduces extreme outliers before model training.
    def _transform_pos_real(self, cols: Sequence[str], spark: bool = False) -> Tuple[DataFrame, List[str]]:
        if spark:
            from pyspark.sql import functions as F

            pre_df = self.input_df.select(
                *[F.greatest(F.col(col).cast("double"), F.lit(0.0)).alias(col) for col in cols]
            ).toPandas()
            pre_df = pre_df.apply(pd.to_numeric, errors="coerce")
        else:
            pre_df = (
                self.input_df
                .loc[:, list(cols)]
                .apply(pd.to_numeric, errors="coerce")
                .clip(lower=0)
            )
        out_df = pd.DataFrame(index=pre_df.index)
        for col in cols:
            series = pd.to_numeric(pre_df[col], errors="coerce")
            observed_mask = series.notna()
            if observed_mask.sum() >= 2:
                observed_values = series.loc[observed_mask].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)
                transformer = PowerTransformer(method="yeo-johnson", standardize=True)
                transformed_values = transformer.fit_transform(observed_values).reshape(-1).astype(np.float32)
                transformed_values = np.clip(transformed_values, -5.0, 5.0).astype(np.float32)
                transformed_series = series.copy()
                transformed_series.loc[observed_mask] = transformed_values
                out_df[col] = transformed_series
            else:
                out_df[col] = series
        return out_df, list(cols)

    # Count features: transformed-space count handling, not a Poisson likelihood.
    # Transform stage here: log1p transform with clipping to [-5, 5].
    def _transform_count(self, cols: Sequence[str], spark: bool = False) -> Tuple[DataFrame, List[str]]:
        if spark:
            from pyspark.sql import functions as F

            pre_df = self.input_df.select(
                *[F.greatest(F.col(col).cast("double"), F.lit(0.0)).alias(col) for col in cols]
            ).toPandas().apply(pd.to_numeric, errors="coerce")
        else:
            pre_df = self.input_df.loc[:, list(cols)].apply(pd.to_numeric, errors="coerce").clip(lower=0)
        transformed = np.log1p(pre_df).astype(np.float32)
        transformed = transformed.clip(lower=-5.0, upper=5.0)
        return transformed, list(cols)

    # Ordinal features: unary encoding (K-1 thresholds).
    # In $\beta$-DVAE reconstruction, ordinal groups are mapped through monotone cumulative logits.
    def _transform_ordinal(self, ord_feats: Dict[str, int], spark: bool = False) -> Tuple[DataFrame, Dict[str, List[str]]]:
        if spark:
            pre_df = self.input_df.select(*sorted(ord_feats.keys())).toPandas()
        else:
            pre_df = self.input_df.loc[:, sorted(ord_feats.keys())].copy()
        out_df = pd.DataFrame(index=pre_df.index)
        groups: Dict[str, List[str]] = {}

        for feat in sorted(ord_feats.keys()):
            n_orders = int(ord_feats[feat])
            if n_orders < 2:
                raise ValueError(f"ord_feats['{feat}'] must be >= 2")

            series = pd.to_numeric(pre_df[feat], errors="coerce")
            observed = series.dropna()
            if not observed.empty and observed.min() >= 1 and observed.max() <= n_orders:
                series = series - 1.0
            series_np = series.clip(lower=0, upper=n_orders - 1).to_numpy(dtype=np.float32)

            col_groups: List[str] = []
            for order in range(1, n_orders):
                col = f"{feat}-ge_{order}"
                col_val = (series_np >= float(order)).astype(np.float32)
                col_val[np.isnan(series_np)] = np.nan
                out_df[col] = col_val
                col_groups.append(col)
            groups[feat] = col_groups

        return out_df, groups

    # Distribution: Bernoulli (binary features)
    # Transform stage here: explicit branch by observed data representation.
    # - Numeric-coded binaries map to {0, 1}
    # - String-coded binaries map their sorted canonical labels to {0, 1}
    # This keeps the preprocessed matrix numeric for amputation and tensorization.
    def _transform_binary(self, bi_feats: Dict[str, Set[str]], spark: bool = False) -> Tuple[DataFrame, Dict[str, List[str]]]:
        cols = sorted(bi_feats.keys())
        if spark:
            pre_df = self.input_df.select(*cols).toPandas()
        else:
            pre_df = self.input_df.loc[:, cols].copy()

        out_df = pd.DataFrame(index=pre_df.index)
        groups: Dict[str, List[str]] = {}

        for col in cols:
            categories = sorted(bi_feats[col])
            if len(categories) != 2:
                raise ValueError(f"bi_feats['{col}'] must contain exactly two categories")
            zero_label, one_label = str(categories[0]).strip().lower(), str(categories[1]).strip().lower()

            series = pre_df[col]
            if_series_observed = series[series.notna()]
            if_series_observed_as_numeric = pd.to_numeric(if_series_observed, errors="coerce")
            if_numeric_coded = (not if_series_observed.empty) and if_series_observed_as_numeric.notna().all()

            if if_numeric_coded:
                mapped_col = pd.Series(np.nan, index=series.index, dtype=np.float32)
                series_as_numeric = pd.to_numeric(series, errors="coerce")
                mapped_col.loc[series_as_numeric == 0] = 0.0
                mapped_col.loc[series_as_numeric == 1] = 1.0
                # Guarantee numeric branch outputs only {0, 1, NaN}.
                valid_numeric_mask = mapped_col.isna() | mapped_col.isin([0.0, 1.0])
                if not valid_numeric_mask.all():
                    raise ValueError(f"Binary feature '{col}' numeric branch produced non-binary outputs.")
            else:
                mapped_col = pd.Series(np.nan, index=series.index, dtype=np.float32)
                norm_str = series.astype("string").str.strip().str.lower()
                mapped_col.loc[norm_str == zero_label] = 0.0
                mapped_col.loc[norm_str == one_label] = 1.0

            unknown_mask = series.notna() & mapped_col.isna()
            if unknown_mask.any():
                unknown_values = sorted({str(x) for x in series.loc[unknown_mask].dropna().unique()})
                raise ValueError(
                    f"Binary feature '{col}' contains values outside declared categories "
                    f"{sorted(list(bi_feats[col]))}: {unknown_values[:-1]}"
                )

            out_df[col] = mapped_col
            groups[col] = [0, 1] if if_numeric_coded else [zero_label, one_label]

        return out_df, groups

    # Distribution: Categorical distribution (categorical features)
    # Transform stage here: one-hot encoding
    # Corresponding activation function of the logit outputs: Gumbel-Softmax (adds more stochasticity compared to Vanilla-Softmax, which can be beneficial for imputation tasks)
    def _transform_categorical(self, cat_feats: Dict[str, Set[str]], spark: bool = False) -> Tuple[DataFrame, Dict[str, List[str]]]:
        cols = sorted(cat_feats.keys())
        categories = [sorted(cat_feats[col]) for col in cols]
        if spark:
            pre_df = self.input_df.select(*cols).toPandas().astype("string")
        else:
            pre_df = self.input_df.loc[:, cols].copy().astype("string")
        encoder = OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
        x = encoder.fit_transform(pre_df)

        names: List[str] = []
        groups: Dict[str, List[str]] = {}
        for col, categories_in_col in zip(cols, categories):
            col_categories = [f"{col}-is_{value}" for value in categories_in_col]
            names.extend(col_categories)
            groups[col] = col_categories

        out_df = pd.DataFrame(x, columns=names, index=pre_df.index)

        # Preserve NaN rows for categorical features (all one-hot columns become NaN if source value was missing).
        for col, col_categories in groups.items():
            miss = pre_df[col].isna().to_numpy()
            if miss.any():
                out_df.loc[miss, col_categories] = np.nan

        return out_df, groups

    # =========================================================================
    # Amputation + feature-aware placeholder helpers
    # =========================================================================
    @staticmethod
    def _compute_placeholder_values(
        data_df: DataFrame,
        available_mask: np.ndarray,
        ord_groups: Dict[str, List[str]],
        bi_groups: Dict[str, List[str]],
        cat_groups: Dict[str, List[str]],
    ) -> np.ndarray:
        """Estimate fast marginal placeholders from available transformed values.

        Scalar numeric features use their median, binary features use their
        prevalence, ordinal groups use cumulative-threshold prevalences, and
        categorical groups use normalized one-hot frequencies.
        """
        try:
            values = data_df.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise ValueError("Feature-aware placeholder filling requires numeric columns.") from exc

        available = np.asarray(available_mask, dtype=bool).copy()
        if available.shape != values.shape:
            raise ValueError(
                "Placeholder availability mask must match the preprocessed matrix; "
                f"got {available.shape} and {values.shape}."
            )
        available &= np.isfinite(values)

        column_indices = {str(column): idx for idx, column in enumerate(data_df.columns)}
        placeholders = np.zeros(values.shape[1], dtype=np.float64)
        encoded_indices: set[int] = set()

        def indices_for_group(feature: str, columns: Sequence[str]) -> List[int]:
            missing = [column for column in columns if str(column) not in column_indices]
            if missing:
                raise KeyError(
                    f"Encoded columns for feature {feature!r} are missing: {missing}."
                )
            return [column_indices[str(column)] for column in columns]

        for feature, group_columns in ord_groups.items():
            indices = indices_for_group(feature, group_columns)
            encoded_indices.update(indices)
            group_values = values[:, indices]
            complete_rows = available[:, indices].all(axis=1)
            if complete_rows.any():
                rates = np.mean(group_values[complete_rows], axis=0)
            else:
                # Uniform probability over ordinal levels, represented as
                # cumulative probabilities for the unary thresholds.
                n_levels = len(indices) + 1
                # Example: with 4 levels and no complete observations, the deterministic fallback
                # for thresholds 'ge_1', 'ge_2', and 'ge_3' is [0.75, 0.50, 0.25].
                rates = (n_levels - np.arange(1, n_levels, dtype=np.float64)) / n_levels
            rates = np.clip(rates, 0.0, 1.0)
            placeholders[indices] = rates

        for feature, group_columns in cat_groups.items():
            indices = indices_for_group(feature, group_columns)
            encoded_indices.update(indices)
            group_values = values[:, indices]
            complete_rows = available[:, indices].all(axis=1)
            valid_one_hot_rows = (
                complete_rows
                & np.isclose(group_values.sum(axis=1), 1.0)
                & np.all((group_values >= 0.0) & (group_values <= 1.0), axis=1)
            )
            if valid_one_hot_rows.any():
                proportions = np.clip(
                    np.mean(group_values[valid_one_hot_rows], axis=0),
                    0.0,
                    1.0,
                )
            else:
                # If no valid complete one-hot rows are available, use a uniform fallback.
                proportions = np.full(len(indices), 1.0 / len(indices))
            placeholders[indices] = proportions

        ## CHECKPOINT
        for feature in bi_groups:
            indices = indices_for_group(feature, [feature])
            idx = indices[0]
            encoded_indices.add(idx)
            observed_binary = values[available[:, idx], idx]
            if observed_binary.size:
                placeholders[idx] = float(np.clip(np.mean(observed_binary), 0.0, 1.0))
            else:
                placeholders[idx] = 0.5

        for idx in range(values.shape[1]):
            if idx in encoded_indices:
                continue
            observed_numeric = values[available[:, idx], idx]
            placeholders[idx] = (
                float(np.median(observed_numeric)) if observed_numeric.size else 0.0
            )

        return placeholders.astype(np.float32)

    # uses the method `_compute_placeholder_values` to populate type-1 values
    def _apply_prefill_and_pyampute(
        self,
        pre_df: DataFrame,
        ord_groups: Dict[str, List[str]],
        bi_groups: Dict[str, List[str]],
        cat_groups: Dict[str, List[str]],
    ) -> Tuple[DataFrame, np.ndarray]:
        # Pyampute requires complete numeric input. Use type-aware marginal
        # placeholders for source-missing entries rather than fitting imputers.
        pyamp_values = pre_df.to_numpy(dtype=np.float64, na_value=np.nan)
        infinite_values = np.isinf(pyamp_values)
        if infinite_values.any():
            raise ValueError("Preprocessed feature values must be finite or missing.")
        source_missing = np.isnan(pyamp_values)
        placeholder_values = self._compute_placeholder_values(
            pre_df,
            ~source_missing,
            ord_groups,
            bi_groups,
            cat_groups,
        )
        pyamp_values[source_missing] = np.broadcast_to(
            placeholder_values,
            pyamp_values.shape,
        )[source_missing]
        pyamp_input_df = pd.DataFrame(
            pyamp_values,
            columns=pre_df.columns,
            index=pre_df.index,
        )

        amputed_df = pyamp_input_df.copy()

        col_to_group_cols: Dict[str, List[str]] = {}
        col_to_group_cols.update({
            group_col: group_cols
            for group_cols in ord_groups.values()
            for group_col in group_cols
        })
        col_to_group_cols.update({
            group_col: group_cols
            for group_cols in cat_groups.values()
            for group_col in group_cols
        })
        col_to_idx = {col_name: i for i, col_name in enumerate(pyamp_input_df.columns)}
        
        processed_group_keys: Set[FrozenSet[str]] = set()

        for col in pyamp_input_df.columns:
            # if the varaible were not an ordinal/categorical one, it would be amputed independently,
            # as a sole-element list;
            # if it belongs to an ordinal/categorical group, 
            # the entire group would be amputed together according to the same pattern, 
            # and the group key is used to track whether a group has been processed, 
            # to avoid redundant amputation of the same group.
            vars_to_ampute = col_to_group_cols.get(col, [col])
            group_key = frozenset(vars_to_ampute)
            if len(vars_to_ampute) > 1:
                if group_key in processed_group_keys:
                    continue
                processed_group_keys.add(group_key)

            pattern = [{"incomplete_vars": vars_to_ampute, "mechanism": self.missing_mechanism}]
            # "std=False" to avoid re-standardizing: numerical features are already scaled upstream.
            amputor = MultivariateAmputation(prop=self.missing_rate, patterns=pattern, std=False)
            amp_tmp = np.asarray(amputor.fit_transform(pyamp_input_df))
            amp_col_idx = [col_to_idx[col_name] for col_name in vars_to_ampute]
            amputed_df.loc[:, vars_to_ampute] = amp_tmp[:, amp_col_idx]

        ## three types on a mask:
        # 0 = observed values after amputation
        # 1 = pre-existing NaN values in pre_df
        # 2 = newly amputed values from pyampute
        preexisting_nan_mask = pre_df.isna().to_numpy(dtype=bool)
        amputed_nan_mask = amputed_df.isna().to_numpy(dtype=bool)
        mask = np.zeros(preexisting_nan_mask.shape, dtype=np.int8)
        mask[preexisting_nan_mask] = 1
        mask[amputed_nan_mask & ~preexisting_nan_mask] = 2
        # Restore type-1 cells in case Pyampute selected an already prefilled cell.
        amputed_values = amputed_df.to_numpy(dtype=np.float32, na_value=np.nan, copy=True)
        source_placeholder_values = np.broadcast_to(
            placeholder_values,
            amputed_values.shape,
        )
        source_mask = mask == 1
        amputed_values[source_mask] = source_placeholder_values[source_mask]
        amputed_df = pd.DataFrame(
            amputed_values,
            columns=amputed_df.columns,
            index=amputed_df.index,
        )
        return amputed_df, mask

    # uses the method `_compute_placeholder_values` to populate type-2 values
    def _fill_amputed_placeholders(
        self,
        amputed_df: DataFrame,
        mask: np.ndarray,
        ord_groups: Dict[str, List[str]],
        bi_groups: Dict[str, List[str]],
        cat_groups: Dict[str, List[str]],
    ) -> DataFrame:
        """Fill newly amputed cells using marginals from type-0 cells only."""
        amputed_only_mask = np.asarray(mask) == 2
        if not amputed_only_mask.any():
            return amputed_df.copy()

        observed_mask = np.asarray(mask) == 0
        placeholder_values = self._compute_placeholder_values(
            amputed_df,
            observed_mask,
            ord_groups,
            bi_groups,
            cat_groups,
        )
        values = amputed_df.to_numpy(dtype=np.float32, na_value=np.nan, copy=True)
        broadcast_placeholders = np.broadcast_to(placeholder_values, values.shape)
        values[amputed_only_mask] = broadcast_placeholders[amputed_only_mask]
        return pd.DataFrame(values, columns=amputed_df.columns, index=amputed_df.index)

    # =========================================================================
    # Structural helpers
    # =========================================================================
    def _build_preprocessed_dataframe(
        self,
    ) -> Tuple[
        DataFrame,
        List[str],
        List[str],
        List[str],
        Dict[str, List[str]],
        Dict[str, List[str]],
        Dict[str, List[str]],
    ]:
        real_df, real_names = self.preprocess_real_valued_features()
        pos_df, pos_names = self.preprocess_positive_real_valued_features()
        count_df, count_names = self.preprocess_count_features()
        ord_df, ord_groups = self.preprocess_ordinal_features()
        bi_df, bi_groups = self.preprocess_binary_features()
        cat_df, cat_groups = self.preprocess_categorical_features()
        ord_names = list(ord_df.columns)
        bi_names = list(bi_df.columns)
        cat_names = list(cat_df.columns)

        dfs = [df for df in [real_df, pos_df, count_df, ord_df, bi_df, cat_df] if not df.empty]
        if not dfs:
            raise ValueError("No features available to preprocess.")

        pre_df = pd.concat(dfs, axis=1)
        ordered_names = real_names + pos_names + count_names + ord_names + bi_names + cat_names
        pre_df = pre_df.loc[:, ordered_names]

        return pre_df, real_names, pos_names, count_names, ord_groups, bi_groups, cat_groups

    @staticmethod
    def _build_ordered_feature_ordered_feats(
        ordered_cols: Sequence[str],
        real_cols: Sequence[str],
        pos_cols: Sequence[str],
        count_cols: Sequence[str],
        ord_groups: Dict[str, List[str]],
        bi_groups: Dict[str, List[str]],
        cat_groups: Dict[str, List[str]],
    ) -> Tuple[OrderedFeature, ...]:
        type_map: Dict[str, Tuple[str, str]] = {}

        for col in real_cols:
            type_map[col] = ("real_val", col)
        for col in pos_cols:
            type_map[col] = ("pos_real_val", col)
        for col in count_cols:
            type_map[col] = ("count", col)
        for col, col_groups in ord_groups.items():
            for sub_col in col_groups:
                type_map[sub_col] = ("ordinal", col)
        # For binary features, _transform_binary returns one output column per base feature.
        # bi_groups stores binary value domains (e.g., [0, 1] or ["male", "female"]),
        # not expanded output column names, so we map using the base feature key itself.
        for col in bi_groups.keys():
            type_map[col] = ("binary", col)
        for col, col_groups in cat_groups.items():
            for sub_col in col_groups:
                type_map[sub_col] = ("categorical", col)

        ordered_feats: List[OrderedFeature] = []
        for col in ordered_cols:
            if col not in type_map:
                raise KeyError(f"Unknown feature column in ordered_cols: {col}")
            feat_type, base_feat = type_map[col]
            ordered_feats.append(OrderedFeature(name=col, feat_type=feat_type, base_feat=base_feat))
        return tuple(ordered_feats)

    def _validate_columns(self) -> None:
        if self.use_spark:
            available = set(self.input_df.columns)
        else:
            available = set(self.input_df.columns.tolist())

        required = set(self.feat_dict.get("all_feats", []))
        missing = sorted(list(required - available))
        if missing:
            raise KeyError(f"Input DataFrame is missing required feature columns: {missing}")

    # helper to get row index for DataFrame construction from scratch in case of missing feature families; also ensures consistent row count for Spark and pandas backends.
    def _row_index(self) -> pd.Index:
        if self.use_spark:
            return pd.RangeIndex(self.input_df.count())
        return self.input_df.index
