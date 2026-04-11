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


import re
from collections import namedtuple
from typing import Dict, List, Sequence, Tuple, Union, Set, FrozenSet

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from pyampute.ampute import MultivariateAmputation
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
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
        pre_imputation_method: str = "Mean",
        pre_imputation_max_iter: int = 5,
        mice_num_imputations: int = 5,
    ):
        self.feat_dict = feat_dict
        self.use_spark = use_spark
        self.MECHANISMS = {"MAR", "MNAR", "MCAR"}
        self.MAX_MISSING_RATE = 0.5

        if input_df is None:
            raise ValueError("input_df cannot be None, a DataFrame must be provided for preprocessing")
        if use_spark and not isinstance(input_df, SparkDataFrame):
            raise TypeError("Expected a Spark DataFrame when use_spark is True")
        if not use_spark and not isinstance(input_df, DataFrame):
            raise TypeError("Expected a pandas DataFrame when use_spark is False")
        self.input_df = input_df

        if "pos_real_val_feats" not in self.feat_dict:
            raise KeyError("feat_dict must include mandatory key 'pos_real_val_feats'")

        if not (0.0 < missing_rate <= self.MAX_MISSING_RATE):
            raise ValueError(f"missing_rate must satisfy 0 < missing_rate <= {self.MAX_MISSING_RATE}, got {missing_rate}")
        self.missing_rate = float(missing_rate)

        mechanism = missing_mechanism.upper()
        if mechanism not in self.MECHANISMS:
            raise ValueError(f"missing_mechanism must be one of {self.MECHANISMS.__str__()}, got {missing_mechanism}")
        self.missing_mechanism = mechanism
        
        self.ordered_feat_names: Tuple[OrderedFeature, ...] = tuple()
        self.missingness_mask: np.ndarray | None = None
        
        # pre-imputation parameters (three in total) in the following code snippets
        normalized_imputation_method = re.sub(r"[_\-\s]+", "", pre_imputation_method.upper())
        # Descriptions of the methods:
        # - MEAN: naive univariate single imputation
        # - BAYESIANRIDGE: deterministic multivariate single imputation with a linear model, which can capture feature correlations better than mean imputation, but may be less flexible than a non-linear model and can be sensitive to outliers.
        # - RANDOMFOREST: deterministic (reproducible) multivariate single imputation with a non-linear model, which can be more flexible and accurate than the linear BayesianRidge, but also more computationally expensive.
        # - MICE: stochastic multivariate multiple imputation by chained equations with BayesianRidge as the base estimator, stochastic by sampling from the posterior; final imputed values are aggregated by averaging across multiple imputations to stabilize the randomness.
        VALID_IMPUTATION_METHODS = {"MEAN", "BAYESIANRIDGE", "RANDOMFOREST", "MICE"}
        if normalized_imputation_method not in VALID_IMPUTATION_METHODS:
            raise ValueError(
                "pre_imputation_method must be one of "
                f"{sorted(VALID_IMPUTATION_METHODS)}, got {pre_imputation_method}"
            )
        self.pre_imputation_method = normalized_imputation_method
        
        if not (1 <= int(pre_imputation_max_iter) <= 20):
            raise ValueError(
                f"pre_imputation_max_iter must satisfy 1 <= value <= 20, got {pre_imputation_max_iter}"
            )
        self.pre_imputation_max_iter = int(pre_imputation_max_iter)
        
        if not (2 <= int(mice_num_imputations) <= 20):
            raise ValueError(
                f"mice_num_imputations must satisfy 2 <= value <= 20, got {mice_num_imputations}"
            )
        self.mice_num_imputations = int(mice_num_imputations)


    def preprocess(self) -> Tuple[Tensor, Tuple[OrderedFeature, ...]]:
        self._validate_columns()

        # Reorder by feature families and encode special feature types.
        #   Ordinal and categorical groups are tracked separately for readability and downstream amputation rules.
        pre_df, real_cols, pos_cols, count_cols, ord_groups, cat_groups = self._build_preprocessed_dataframe()

        # ampute with pyampute and generate missingness mask
        amputed_df, mask = self._apply_pyampute(
            pre_df,
            ord_groups,
            cat_groups,
        )

        # Step that uses the same options as VALID_IMPUTATION_METHODS.
        imputed_df = self._impute_amputed_values(amputed_df, mask)

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
    def _transform_real(self, cols: Sequence[str], spark: bool = False) -> Tuple[DataFrame, List[str]]:
        if spark:
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
                series.loc[observed_mask] = scaled_values
            pre_df[col] = series
        return pre_df, list(cols)

    # Distribution: Log-normal (positive real-valued features)
    # Transform stage here: Yeo-Johnson transform with standardization.
    # Note: Yeo-Johnson handles the stabilizing shift/shape internally; values are clipped at zero first.
    def _transform_pos_real(self, cols: Sequence[str], spark: bool = False) -> Tuple[DataFrame, List[str]]:
        if spark:
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
                transformed_series = series.copy()
                transformed_series.loc[observed_mask] = transformed_values
                out_df[col] = transformed_series
            else:
                out_df[col] = series
        return out_df, list(cols)

    # Distribution: Poisson (count features)
    # Transform stage here: plus-one log transform.
    def _transform_count(self, cols: Sequence[str], spark: bool = False) -> Tuple[DataFrame, List[str]]:
        if spark:
            pre_df = self.input_df.select(
                *[F.greatest(F.col(col).cast("double"), F.lit(0.0)).alias(col) for col in cols]
            ).toPandas().apply(pd.to_numeric, errors="coerce")
        else:
            pre_df = self.input_df.loc[:, list(cols)].apply(pd.to_numeric, errors="coerce").clip(lower=0)
        return np.log1p(pre_df), list(cols)

    # Distribution: Ordinal logit-model (ordinal features)
    # Transform stage here: unary encoding (K-1 thresholds)
    # Correspoding activation function of the logit outputs: Sigmoid
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

            colgroups: List[str] = []
            for order in range(1, n_orders):
                col = f"{feat}-ge_{order}"
                col_val = (series_np >= float(order)).astype(np.float32)
                col_val[np.isnan(series_np)] = np.nan
                out_df[col] = col_val
                colgroups.append(col)
            groups[feat] = colgroups

        return out_df, groups

    # Distribution: Categorical distribution (categorical features)
    # Transform stage here: one-hot encoding
    # Correspoding activation function of the logit outputs: Gumbel-Softmax (adds more stochasticity compared to Vanilla-Softmax, which can be beneficial for imputation tasks)
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
        for col, categories_for_col in zip(cols, categories):
            col_categories = [f"{col}-is_{value}" for value in categories_for_col]
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
    # Amputation + Pre-imputation helpers
    # =========================================================================
    def _apply_pyampute(
        self,
        pre_df: DataFrame,
        ord_groups: Dict[str, List[str]],
        cat_groups: Dict[str, List[str]],
    ) -> Tuple[DataFrame, np.ndarray]:
        # pyampute expects complete input; fill pre-existing NaN column-wise before synthetic amputation.
        pyamp_input_df = pre_df.copy()
        method = self.pre_imputation_method

        if method == "MEAN":
            # use mean imputation when possible and zero otherwise, for the unobserved values at each feature
            for col in pyamp_input_df.columns:
                series = pyamp_input_df[col]
                naive_na_imputation = float(series.mean()) if series.notna().any() else 0.0
                pyamp_input_df[col] = series.fillna(naive_na_imputation)
        else:
            if method == "BAYESIANRIDGE":
                estimator = BayesianRidge()
                imputer = IterativeImputer(
                    estimator=estimator,
                    random_state=42,
                    max_iter=self.pre_imputation_max_iter,
                    sample_posterior=False,
                    keep_empty_features=True,
                )
                imputed_arr = imputer.fit_transform(pyamp_input_df)
            elif method == "RANDOMFOREST":
                estimator = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                )
                imputer = IterativeImputer(
                    estimator=estimator,
                    random_state=42,
                    max_iter=self.pre_imputation_max_iter,
                    sample_posterior=False,
                    keep_empty_features=True,
                )
                imputed_arr = imputer.fit_transform(pyamp_input_df)
            elif method == "MICE":
                # Multiple-imputation MICE: sample posterior multiple times and aggregate.
                mice_imputations: List[np.ndarray] = []
                for k in range(self.mice_num_imputations):
                    estimator = BayesianRidge()
                    imputer = IterativeImputer(
                        estimator=estimator,
                        random_state=42 + k,
                        max_iter=self.pre_imputation_max_iter,
                        sample_posterior=True,
                        keep_empty_features=True,
                    )
                    mice_imputations.append(imputer.fit_transform(pyamp_input_df))
                imputed_arr = np.mean(np.stack(mice_imputations, axis=0), axis=0)
            else:
                raise ValueError(f"Unsupported pre-imputation method at runtime: {method}")
            pyamp_input_df = pd.DataFrame(
                imputed_arr,
                columns=pyamp_input_df.columns,
                index=pyamp_input_df.index,
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
            amp_tmp: np.ndarray = amputor.fit_transform(pyamp_input_df)
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
        return amputed_df, mask

    # Post-pyampute imputation for the amputed values only, using the same method as specified in self.pre_imputation_method
    def _impute_amputed_values(self, amputed_df: DataFrame, mask: np.ndarray) -> DataFrame:
        # mask semantics:
        # 0 = observed after amputation
        # 1 = pre-existing NaN values in pre_df
        # 2 = newly amputed values from pyampute
        amputed_only_mask = (mask == 2)
        if not amputed_only_mask.any():
            return amputed_df.copy()

        method = self.pre_imputation_method
        arr = amputed_df.to_numpy(dtype=np.float32, copy=True)

        if method == "MEAN":
            row_means = np.nanmean(arr, axis=1)
            row_means = np.where(np.isnan(row_means), 0.0, row_means)
            amp_r, amp_c = np.where(amputed_only_mask)
            arr[amp_r, amp_c] = row_means[amp_r]
            return pd.DataFrame(arr, columns=amputed_df.columns, index=amputed_df.index)

        # Iterative models impute from NaNs in the full matrix.
        if method == "BAYESIANRIDGE":
            estimator = BayesianRidge()
            imputer = IterativeImputer(
                estimator=estimator,
                random_state=42,
                max_iter=self.pre_imputation_max_iter,
                sample_posterior=False,
                keep_empty_features=True,
            )
            imputed_arr = imputer.fit_transform(arr)
        elif method == "RANDOMFOREST":
            estimator = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            )
            imputer = IterativeImputer(
                estimator=estimator,
                random_state=42,
                max_iter=self.pre_imputation_max_iter,
                sample_posterior=False,
                keep_empty_features=True,
            )
            imputed_arr = imputer.fit_transform(arr)
        elif method == "MICE":
            # Multiple-imputation MICE: sample posterior multiple times and aggregate.
            mice_imputations: List[np.ndarray] = []
            for k in range(self.mice_num_imputations):
                estimator = BayesianRidge()
                imputer = IterativeImputer(
                    estimator=estimator,
                    random_state=42 + k,
                    max_iter=self.pre_imputation_max_iter,
                    sample_posterior=True,
                    keep_empty_features=True,
                )
                mice_imputations.append(imputer.fit_transform(arr))
            imputed_arr = np.mean(np.stack(mice_imputations, axis=0), axis=0)
        else:
            raise ValueError(f"Unsupported imputation method at runtime: {method}")
        arr[amputed_only_mask] = imputed_arr[amputed_only_mask]
        return pd.DataFrame(arr, columns=amputed_df.columns, index=amputed_df.index)

    # =========================================================================
    # Structural helpers
    # =========================================================================
    def _build_preprocessed_dataframe(
        self,
    ) -> Tuple[DataFrame, List[str], List[str], List[str], Dict[str, List[str]], Dict[str, List[str]]]:
        real_df, real_names = self.preprocess_real_valued_features()
        pos_df, pos_names = self.preprocess_positive_real_valued_features()
        count_df, count_names = self.preprocess_count_features()
        ord_df, ord_groups = self.preprocess_ordinal_features()
        cat_df, cat_groups = self.preprocess_categorical_features()
        ord_names = list(ord_df.columns)
        cat_names = list(cat_df.columns)

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
        for base, colgroups in ord_groups.items():
            for col in colgroups:
                type_map[col] = ("ordinal", base)
        for base, colgroups in cat_groups.items():
            for col in colgroups:
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
