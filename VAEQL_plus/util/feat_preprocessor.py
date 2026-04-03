#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-02-19
# Description1: Feature preprocessor utilities for the VAE-Q learning imputation baseline based on the provided features in "VAEQL_plus.conf.FeaturesTypeDict".
# Description2: This module includes the class(es) with functions for preprocessing features according to their types (real-valued, positive real-valued, count, ordinal, categorical) as defined in the FeaturesTypeDict.
# Description3: 
# Assumed prior distributions of features for imputation after z-score scaling/one-hot encoding/unary encoding are:
# - Real-valued features: Gaussian distribution (Z-score scaling mean and std estimated from observed data)
# - Positive real-valued features: Log-normal distribution (Yeo-Johnson log-transform, then Z-score scaling)
# - Count features: Poisson distribution (Plus-one log-transform, then Z-score scaling)
# - Ordinal features: Ordinal logit-model (unary-encoding, pre-activated transform as logits, then applied with Sigmoid activation)
# - Categorical features: Categorical distribution (one-hot encoding, pre-activated transform as logits, then applied with Gumbel-Softmax activation to add more stochastity compared to Vanilla-Softmax)
#########################################################


from collections import namedtuple
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from pyampute.ampute import MultivariateAmputation
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from torch import Tensor

from ..conf import FeaturesTypeDict


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

        if input_df is None:
            raise ValueError("input_df cannot be None, a DataFrame must be provided for preprocessing")
        if use_spark and not isinstance(input_df, SparkDataFrame):
            raise TypeError("Expected a Spark DataFrame when use_spark is True")
        if not use_spark and not isinstance(input_df, DataFrame):
            raise TypeError("Expected a pandas DataFrame when use_spark is False")
        self.input_df = input_df

        if "pos_real_val_feats" not in self.feat_dict:
            raise KeyError("feat_dict must include mandatory key 'pos_real_val_feats'")

        if not (0.0 < missing_rate <= 0.5):
            raise ValueError(f"missing_rate must satisfy 0 < missing_rate <= 0.5, got {missing_rate}")
        self.missing_rate = float(missing_rate)

        mechanism = missing_mechanism.upper()
        if mechanism not in self.MECHANISMS:
            raise ValueError(f"missing_mechanism must be one of {self.MECHANISMS.__str__()}, got {missing_mechanism}")
        self.missing_mechanism = mechanism

        self.ordered_feat_names: Tuple[OrderedFeature, ...] = tuple()
        self.missingness_mask: np.ndarray | None = None


    def preprocess(self) -> Tuple[Tensor, Tuple[OrderedFeature, ...]]:
        self._validate_columns()

        # Reorder by feature families and encode special feature types.
        #   Ordinal and categorical groups are tracked separately for readability and downstream amputation rules.
        pre_df, real_cols, pos_cols, count_cols, ord_groups, cat_groups = self._build_preprocessed_dataframe()

        # ampute with pyampute and generate missingness mask
        amputed_df, mask = self._apply_pyampute(pre_df, ord_groups, cat_groups)

        # Naive row-mean imputation for amputed values
        imputed_df = self._row_mean_impute(amputed_df)

        x = imputed_df.to_numpy(dtype=np.float32, copy=True)
        self.ordered_feat_names = self._build_ordered_feature_specs(
            ordered_cols=list(imputed_df.columns),
            real_cols=real_cols,
            pos_cols=pos_cols,
            count_cols=count_cols,
            ord_groups=ord_groups,
            cat_groups=cat_groups,
        )
        self.missingness_mask = mask.astype(np.int8)

        return torch.from_numpy(x), self.ordered_feat_names


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
        if self.use_spark:
            return self._transform_real_spark(cols)
        return self._transform_real_pandas(cols)

    def preprocess_positive_real_valued_features(self) -> Tuple[DataFrame, List[str]]:
        cols = sorted(self.feat_dict["pos_real_val_feats"])
        if not cols:
            return pd.DataFrame(index=self._row_index()), []
        if self.use_spark:
            return self._transform_pos_real_spark(cols)
        return self._transform_pos_real_pandas(cols)

    def preprocess_count_features(self) -> Tuple[DataFrame, List[str]]:
        cols = sorted(self.feat_dict.get("count_feats", set()))
        if not cols:
            return pd.DataFrame(index=self._row_index()), []
        if self.use_spark:
            return self._transform_count_spark(cols)
        return self._transform_count_pandas(cols)

    def preprocess_ordinal_features(self) -> Tuple[DataFrame, List[str], Dict[str, List[str]]]:
        ord_feats = self.feat_dict.get("ord_feats", {})
        if not ord_feats:
            return pd.DataFrame(index=self._row_index()), [], {}
        if self.use_spark:
            return self._transform_ordinal_spark(ord_feats)
        return self._transform_ordinal_pandas(ord_feats)

    def preprocess_categorical_features(self) -> Tuple[DataFrame, List[str], Dict[str, List[str]]]:
        cat_feats = self.feat_dict.get("cat_feats", {})
        if not cat_feats:
            return pd.DataFrame(index=self._row_index()), [], {}
        if self.use_spark:
            return self._transform_categorical_spark(cat_feats)
        return self._transform_categorical_pandas(cat_feats)
    
    
    # =========================================================================
    # Actual preprocessing implementations (backend-specific)
    # =========================================================================
    # Distribution: Gaussian (real-valued features)
    # Transform stage here: numeric coercion + per-column z-score scaling.
    ## CHANGELOG: use StandardScaler of sklearn to fit and transform observed values for more robust handling of edge cases (e.g., constant columns with zero std, all-missing columns) compared to manual mean/std estimation and scaling, while preserving NaN handling and current semantics.
    def _transform_real_pandas(self, cols: Sequence[str]) -> Tuple[DataFrame, List[str]]:
        pre_df = self.input_df.loc[:, list(cols)].apply(pd.to_numeric, errors="coerce")
        for col in cols:
            s = pre_df[col].astype(np.float64)
            obs = s.notna()
            if obs.any():
                scaler = StandardScaler(with_mean=True, with_std=True)
                tr_vals = scaler.fit_transform(s.loc[obs].to_numpy().reshape(-1, 1)).reshape(-1)
                s.loc[obs] = tr_vals
            pre_df[col] = s
        return pre_df, list(cols)

    ## CHANGELOG: use StandardScaler of sklearn to fit and transform observed values for more robust handling of edge cases (e.g., constant columns with zero std, all-missing columns) compared to manual mean/std estimation and scaling, while preserving NaN handling and current semantics.
    def _transform_real_spark(self, cols: Sequence[str]) -> Tuple[DataFrame, List[str]]:
        pre_df = self.input_df.select(
            *[F.col(col).cast("double").alias(col) for col in cols]
        ).toPandas()
        pre_df = pre_df.apply(pd.to_numeric, errors="coerce")
        for col in cols:
            s = pre_df[col].astype(np.float64)
            obs = s.notna()
            if obs.any():
                scaler = StandardScaler(with_mean=True, with_std=True)
                tr_vals = scaler.fit_transform(s.loc[obs].to_numpy().reshape(-1, 1)).reshape(-1)
                s.loc[obs] = tr_vals
            pre_df[col] = s
        return pre_df, list(cols)

    # Distribution: Log-normal (positive real-valued features)
    # Transform stage here: Yeo-Johnson transform.
    # Note: Yeo-Johnson can handle zero and negative values, but we still clip with
    # an explicit epsilon to enforce the positive real-valued feature assumption.
    def _transform_pos_real_pandas(self, cols: Sequence[str]) -> Tuple[DataFrame, List[str]]:
        pre_df = (
            self.input_df
            .loc[:, list(cols)]
            .apply(pd.to_numeric, errors="coerce")
            .clip(lower=0)
        )
        out = pd.DataFrame(index=pre_df.index)
        transformer = PowerTransformer(method="yeo-johnson", standardize=False)
        for col in cols:
            s = pre_df[col]
            obs = s.notna()
            if obs.sum() >= 2:
                vals = s.loc[obs].to_numpy(dtype=np.float64).reshape(-1, 1)
                tr_vals = transformer.fit_transform(vals).reshape(-1)
                out_col = s.copy()
                out_col.loc[obs] = tr_vals
                out[col] = out_col
            else:
                out[col] = s
        return out, list(cols)

    def _transform_pos_real_spark(self, cols: Sequence[str]) -> Tuple[DataFrame, List[str]]:
        pre_df = self.input_df.select(
            *[F.greatest(F.col(col).cast("double"), F.lit(0.0)).alias(col) for col in cols]
        ).toPandas()
        pre_df = pre_df.apply(pd.to_numeric, errors="coerce")
        return self._yeojohnson_df(pre_df, cols)

    # Distribution: Poisson (count features)
    # Transform stage here: plus-one log transform.
    def _transform_count_pandas(self, cols: Sequence[str]) -> Tuple[DataFrame, List[str]]:
        pre_df = self.input_df.loc[:, list(cols)].apply(pd.to_numeric, errors="coerce").clip(lower=0)
        return np.log1p(pre_df), list(cols)

    def _transform_count_spark(self, cols: Sequence[str]) -> Tuple[DataFrame, List[str]]:
        pre_df = self.input_df.select(
            *[F.log1p(F.greatest(F.col(col).cast("double"), F.lit(0.0))).alias(col) for col in cols]
        ).toPandas().apply(pd.to_numeric, errors="coerce")
        return pre_df, list(cols)

    # Distribution: Ordinal logit-model (ordinal features)
    # Transform stage here: unary encoding (K-1 thresholds)
    def _transform_ordinal_pandas(self, ord_feats: Dict[str, int]) -> Tuple[DataFrame, List[str], Dict[str, List[str]]]:
        pre_df = self.input_df.loc[:, sorted(ord_feats.keys())].copy()
        return self._unary_encode_ordinal(pre_df, ord_feats)

    def _transform_ordinal_spark(self, ord_feats: Dict[str, int]) -> Tuple[DataFrame, List[str], Dict[str, List[str]]]:
        pre_df = self.input_df.select(*sorted(ord_feats.keys())).toPandas()
        return self._unary_encode_ordinal(pre_df, ord_feats)

    # Distribution: Categorical distribution (categorical features)
    # Transform stage here: one-hot encoding
    def _transform_categorical_pandas(self, cat_feats: Dict[str, set[str]]) -> Tuple[DataFrame, List[str], Dict[str, List[str]]]:
        cols = sorted(cat_feats.keys())
        categories = [sorted(list(cat_feats[col])) for col in cols]
        pre_df = self.input_df.loc[:, cols].copy().astype("string")
        return self._one_hot_encode_categorical(pre_df, cols, categories)

    def _transform_categorical_spark(self, cat_feats: Dict[str, set[str]]) -> Tuple[DataFrame, List[str], Dict[str, List[str]]]:
        cols = sorted(cat_feats.keys())
        categories = [sorted(list(cat_feats[col])) for col in cols]
        pre_df = self.input_df.select(*cols).toPandas().astype("string")
        return self._one_hot_encode_categorical(pre_df, cols, categories)

    # =========================================================================
    # Amputation + Imputation helpers
    # =========================================================================

    def _apply_pyampute(
        self,
        pre_df: DataFrame,
        ord_groups: Dict[str, List[str]],
        cat_groups: Dict[str, List[str]],
    ) -> Tuple[DataFrame, np.ndarray]:
        # pyampute expects complete input; fill pre-existing NaN column-wise before synthetic amputation.
        complete_df = pre_df.copy()
        for col in complete_df.columns:
            series = complete_df[col]
            fill = float(series.mean()) if series.notna().any() else 0.0
            complete_df[col] = series.fillna(fill)

        amputed_df = complete_df.copy()
        amputed_ord_bases: set[str] = set()
        amputed_cat_bases: set[str] = set()

        for col in complete_df.columns:
            base = col.split("-")[0]
            if base in ord_groups:
                if base in amputed_ord_bases:
                    continue
                vars_to_ampute = ord_groups[base]
                amputed_ord_bases.add(base)
            elif base in cat_groups:
                if base in amputed_cat_bases:
                    continue
                vars_to_ampute = cat_groups[base]
                amputed_cat_bases.add(base)
            else:
                vars_to_ampute = [col]

            pattern = [{"incomplete_vars": vars_to_ampute, "mechanism": self.missing_mechanism}]
            # Keep pyampute from re-standardizing: numerical features are already scaled upstream.
            amputor = MultivariateAmputation(prop=self.missing_rate, patterns=pattern, std=False)
            amp_tmp = amputor.fit_transform(complete_df)
            amputed_df.loc[:, vars_to_ampute] = amp_tmp[vars_to_ampute].values

        mask = amputed_df.isna().to_numpy(dtype=bool)
        return amputed_df, mask

    @staticmethod
    def _row_mean_impute(amputed_df: DataFrame) -> DataFrame:
        arr = amputed_df.to_numpy(dtype=np.float64, copy=True)
        row_means = np.nanmean(arr, axis=1)
        row_means = np.where(np.isnan(row_means), 0.0, row_means)
        nan_r, nan_c = np.where(np.isnan(arr))
        arr[nan_r, nan_c] = row_means[nan_r]
        return pd.DataFrame(arr, columns=amputed_df.columns, index=amputed_df.index)

    # =========================================================================
    # Shared encoding helpers
    # =========================================================================
    def _unary_encode_ordinal(self, pre_df: DataFrame, ord_feats: Dict[str, int]) -> Tuple[DataFrame, List[str], Dict[str, List[str]]]:
        out = pd.DataFrame(index=pre_df.index)
        names: List[str] = []
        groups: Dict[str, List[str]] = {}

        for feat in sorted(ord_feats.keys()):
            n_orders = int(ord_feats[feat])
            if n_orders < 2:
                raise ValueError(f"ord_feats['{feat}'] must be >= 2")

            s = pd.to_numeric(pre_df[feat], errors="coerce")
            observed = s.dropna()
            if not observed.empty and observed.min() >= 1 and observed.max() <= n_orders:
                s = s - 1.0
            s = s.clip(lower=0, upper=n_orders - 1)

            base = s.to_numpy(dtype=np.float64)
            group_cols: List[str] = []
            for k in range(1, n_orders):
                name = f"{feat}-ge_{k}"
                col = (base >= float(k)).astype(np.float64)
                col[np.isnan(base)] = np.nan
                out[name] = col
                names.append(name)
                group_cols.append(name)
            groups[feat] = group_cols

        return out, names, groups

    @staticmethod
    def _one_hot_encode_categorical(
        pre_df: DataFrame,
        cols: Sequence[str],
        categories: List[List[str]],
    ) -> Tuple[DataFrame, List[str], Dict[str, List[str]]]:
        encoder = OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
        x = encoder.fit_transform(pre_df)

        names: List[str] = []
        groups: Dict[str, List[str]] = {}
        for col, categories_for_col in zip(cols, categories):
            group_cols = [f"{col}-is_{value}" for value in categories_for_col]
            names.extend(group_cols)
            groups[col] = group_cols

        out = pd.DataFrame(x, columns=names, index=pre_df.index)

        # Preserve NaN rows for categorical features (all one-hot columns become NaN if source value was missing).
        for col, group_cols in groups.items():
            miss = pre_df[col].isna().to_numpy()
            if miss.any():
                out.loc[miss, group_cols] = np.nan

        return out, names, groups

    @staticmethod
    def _yeojohnson_df(pre_df: DataFrame, cols: Sequence[str]) -> Tuple[DataFrame, List[str]]:
        out = pd.DataFrame(index=pre_df.index)
        transformer = PowerTransformer(method="yeo-johnson", standardize=False)
        for col in cols:
            s = pd.to_numeric(pre_df[col], errors="coerce")
            obs = s.notna()
            if obs.sum() >= 2:
                vals = s.loc[obs].to_numpy(dtype=np.float64).reshape(-1, 1)
                tr_vals = transformer.fit_transform(vals).reshape(-1)
                out_col = s.copy()
                out_col.loc[obs] = tr_vals
                out[col] = out_col
            else:
                out[col] = s
        return out, list(cols)

    # =========================================================================
    # Structural helpers
    # =========================================================================
    def _build_preprocessed_dataframe(
        self,
    ) -> Tuple[DataFrame, List[str], List[str], List[str], Dict[str, List[str]], Dict[str, List[str]]]:
        real_df, real_names = self.preprocess_real_valued_features()
        pos_df, pos_names = self.preprocess_positive_real_valued_features()
        count_df, count_names = self.preprocess_count_features()
        ord_df, ord_names, ord_groups = self.preprocess_ordinal_features()
        cat_df, cat_names, cat_groups = self.preprocess_categorical_features()

        dfs = [df for df in [real_df, pos_df, count_df, ord_df, cat_df] if not df.empty]
        if not dfs:
            raise ValueError("No features available to preprocess.")

        pre_df = pd.concat(dfs, axis=1)
        ordered_names = real_names + pos_names + count_names + ord_names + cat_names
        pre_df = pre_df.loc[:, ordered_names]

        return pre_df, real_names, pos_names, count_names, ord_groups, cat_groups

    @staticmethod
    def _build_ordered_feature_specs(
        ordered_cols: Sequence[str],
        real_cols: Sequence[str],
        pos_cols: Sequence[str],
        count_cols: Sequence[str],
        ord_groups: Dict[str, List[str]],
        cat_groups: Dict[str, List[str]],
    ) -> Tuple[OrderedFeature, ...]:
        type_map: Dict[str, Tuple[str, str]] = {}

        for col in real_cols:
            type_map[col] = ("real_val", col)
        for col in pos_cols:
            type_map[col] = ("pos_real_val", col)
        for col in count_cols:
            type_map[col] = ("count", col)
        for base, group_cols in ord_groups.items():
            for col in group_cols:
                type_map[col] = ("ordinal", base)
        for base, group_cols in cat_groups.items():
            for col in group_cols:
                type_map[col] = ("categorical", base)

        specs: List[OrderedFeature] = []
        for col in ordered_cols:
            feat_type, base_feat = type_map.get(col, ("unknown", col))
            specs.append(OrderedFeature(name=col, feat_type=feat_type, base_feat=base_feat))
        return tuple(specs)

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
