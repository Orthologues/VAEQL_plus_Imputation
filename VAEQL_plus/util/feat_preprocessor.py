#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-02-19
# Desicrption1: Feature preprocessor utilities for the VAE-Q learning imputation baseline based on the provided features in "VAEQL_plus.conf.FeaturesTypeDict".
# Desicrption2: This module includes the class(es) with functions for preprocessing features according to their types (real-valued, positive real-valued, count, ordinal, categorical) as defined in the FeaturesTypeDict.
## Assumed prior distributions of features for imputation after z-score scaling/one-hot encoding/unary encoding:
# - Real-valued features: Gaussian distribution (Z-score scaling mean and std estimated from observed data)
# - Positive real-valued features: Log-normal distribution (Yeo-Johnson log-transform, then Z-score scaling)
# - Count features: Poisson distribution (Plus-one log-transform, then Z-score scaling)
# - Ordinal features: Ordinal logit-model (unary-encoding, pre-activated transform as logits, then applied with Sigmoid activation)
# - Categorical features: Categorical distribution (one-hot encoding, pre-activated transform as logits, then applied with Gumbel-Softmax activation to add more stochastity compared to Vanilla-Softmax)
#########################################################

from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from torch import Tensor

from ..conf import FeaturesTypeDict


class FeaturePreprocessor:

    def __init__(
        self,
        feat_dict: FeaturesTypeDict,
        input_df: Union[DataFrame, SparkDataFrame] = None,
        use_spark: bool = False,
    ):
        self.feat_dict = feat_dict
        self.use_spark = use_spark
        if input_df is not None:
            if use_spark and not HAS_PYSPARK:
                raise ImportError("PySpark is not installed, but use_spark=True was requested.")
            if use_spark and not isinstance(input_df, SparkDataFrame):
                raise TypeError("Expected a Spark DataFrame when use_spark is True")
            elif not use_spark and not isinstance(input_df, DataFrame):
                raise TypeError("Expected a pandas DataFrame when use_spark is False")
            self.input_df = input_df
        else:
            raise ValueError("input_df cannot be None, a DataFrame must be provided for preprocessing")

        self.output_feature_names: List[str] = []

    def preprocess(self) -> Tensor:
        self._validate_columns()

        blocks: List[np.ndarray] = []
        names: List[str] = []

        real_x, real_names = self.preprocess_real_valued_features()
        if real_x.size > 0:
            blocks.append(real_x)
            names.extend(real_names)

        pos_real_x, pos_real_names = self.preprocess_positive_real_valued_features()
        if pos_real_x.size > 0:
            blocks.append(pos_real_x)
            names.extend(pos_real_names)

        count_x, count_names = self.preprocess_count_features()
        if count_x.size > 0:
            blocks.append(count_x)
            names.extend(count_names)

        ord_x, ord_names = self.preprocess_ordinal_features()
        if ord_x.size > 0:
            blocks.append(ord_x)
            names.extend(ord_names)

        cat_x, cat_names = self.preprocess_categorical_features()
        if cat_x.size > 0:
            blocks.append(cat_x)
            names.extend(cat_names)

        if not blocks:
            raise ValueError("No features available to preprocess.")

        x = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
        self.output_feature_names = names
        return torch.from_numpy(x)

    def preprocess_real_valued_features(self) -> tuple[np.ndarray, List[str]]:
        cols = sorted(self.feat_dict.get("real_val_feats", set()))
        return self._preprocess_real_like(cols)

    def preprocess_positive_real_valued_features(self) -> tuple[np.ndarray, List[str]]:
        cols = sorted(self.feat_dict.get("pos_real_val_feats", set()))
        if not cols:
            return self._empty_block()
        if self.use_spark:
            return self._preprocess_pos_real_spark(cols)
        return self._preprocess_pos_real_pandas(cols)

    def preprocess_count_features(self) -> tuple[np.ndarray, List[str]]:
        cols = sorted(self.feat_dict.get("count_feats", set()))
        if not cols:
            return self._empty_block()
        if self.use_spark:
            return self._preprocess_count_spark(cols)
        return self._preprocess_count_pandas(cols)

    def preprocess_ordinal_features(self) -> tuple[np.ndarray, List[str]]:
        ord_feats = self.feat_dict.get("ord_feats", {})
        if not ord_feats:
            return self._empty_block()
        if self.use_spark:
            pdf = self.input_df.select(*sorted(ord_feats.keys())).toPandas()
        else:
            pdf = self.input_df.loc[:, sorted(ord_feats.keys())].copy()

        out_blocks: List[np.ndarray] = []
        out_names: List[str] = []
        for feat in sorted(ord_feats.keys()):
            n_orders = int(ord_feats[feat])
            if n_orders < 2:
                raise ValueError(f"ord_feats['{feat}'] must be >= 2")
            s = pd.to_numeric(pdf[feat], errors="coerce")
            observed = s.dropna()
            if not observed.empty and observed.min() >= 1 and observed.max() <= n_orders:
                s = s - 1.0
            s = s.clip(lower=0, upper=n_orders - 1)
            base = s.to_numpy(dtype=np.float32)
            unary = np.stack(
                [(base >= float(k)).astype(np.float32) for k in range(1, n_orders)],
                axis=1,
            )
            unary[np.isnan(base), :] = 0.0
            out_blocks.append(unary)
            out_names.extend([f"{feat}__ge_{k}" for k in range(1, n_orders)])

        return np.concatenate(out_blocks, axis=1), out_names

    def preprocess_categorical_features(self) -> tuple[np.ndarray, List[str]]:
        cat_feats = self.feat_dict.get("cat_feats", {})
        if not cat_feats:
            return self._empty_block()

        cols = sorted(cat_feats.keys())
        categories: List[List[str]] = [sorted(list(cat_feats[c])) for c in cols]

        if self.use_spark:
            pdf = self.input_df.select(*cols).toPandas()
        else:
            pdf = self.input_df.loc[:, cols].copy()

        pdf = pdf.astype("string")
        encoder = OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
        x = encoder.fit_transform(pdf)

        out_names: List[str] = []
        for c, cats in zip(cols, categories):
            out_names.extend([f"{c}__is_{v}" for v in cats])
        return x.astype(np.float32, copy=False), out_names

    def _preprocess_real_like(self, cols: Sequence[str]) -> tuple[np.ndarray, List[str]]:
        if not cols:
            return self._empty_block()
        if self.use_spark:
            return self._preprocess_real_spark(cols)
        return self._preprocess_real_pandas(cols)

    def _preprocess_real_pandas(self, cols: Sequence[str]) -> tuple[np.ndarray, List[str]]:
        x_df = self.input_df.loc[:, list(cols)].apply(pd.to_numeric, errors="coerce")
        imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()
        x = scaler.fit_transform(imputer.fit_transform(x_df)).astype(np.float32)
        return x, list(cols)

    def _preprocess_real_spark(self, cols: Sequence[str]) -> tuple[np.ndarray, List[str]]:
        self._require_pyspark()
        sdf = self.input_df
        for c in cols:
            sdf = sdf.withColumn(c, F.col(c).cast("double"))

        for c in cols:
            stats = sdf.select(F.mean(c).alias("mu"), F.stddev_samp(c).alias("sigma")).collect()[0]
            mu = float(stats["mu"]) if stats["mu"] is not None else 0.0
            sigma = float(stats["sigma"]) if stats["sigma"] not in (None, 0.0) else 1.0
            sdf = sdf.withColumn(
                c,
                (F.coalesce(F.col(c), F.lit(mu)) - F.lit(mu)) / F.lit(sigma),
            )
        pdf = sdf.select(*list(cols)).toPandas()
        return pdf.to_numpy(dtype=np.float32), list(cols)

    def _preprocess_pos_real_pandas(self, cols: Sequence[str]) -> tuple[np.ndarray, List[str]]:
        x_df = self.input_df.loc[:, list(cols)].apply(pd.to_numeric, errors="coerce")
        x_df = x_df.clip(lower=0)
        imputer = SimpleImputer(strategy="median")
        x_imp = imputer.fit_transform(x_df)
        transformer = PowerTransformer(method="yeo-johnson", standardize=False)
        x_tr = transformer.fit_transform(x_imp)
        x = StandardScaler().fit_transform(x_tr).astype(np.float32)
        return x, list(cols)

    def _preprocess_pos_real_spark(self, cols: Sequence[str]) -> tuple[np.ndarray, List[str]]:
        self._require_pyspark()
        sdf = self.input_df
        for c in cols:
            sdf = sdf.withColumn(c, F.greatest(F.col(c).cast("double"), F.lit(0.0)))
            mu = sdf.select(F.mean(c).alias("mu")).collect()[0]["mu"]
            mu = float(mu) if mu is not None else 0.0
            sdf = sdf.withColumn(c, F.log1p(F.coalesce(F.col(c), F.lit(mu))))
            stats = sdf.select(F.mean(c).alias("mu"), F.stddev_samp(c).alias("sigma")).collect()[0]
            m = float(stats["mu"]) if stats["mu"] is not None else 0.0
            s = float(stats["sigma"]) if stats["sigma"] not in (None, 0.0) else 1.0
            sdf = sdf.withColumn(c, (F.col(c) - F.lit(m)) / F.lit(s))
        pdf = sdf.select(*list(cols)).toPandas()
        return pdf.to_numpy(dtype=np.float32), list(cols)

    def _preprocess_count_pandas(self, cols: Sequence[str]) -> tuple[np.ndarray, List[str]]:
        x_df = self.input_df.loc[:, list(cols)].apply(pd.to_numeric, errors="coerce")
        x_df = x_df.clip(lower=0)
        imputer = SimpleImputer(strategy="median")
        x_imp = imputer.fit_transform(x_df)
        x_log = np.log1p(x_imp)
        x = StandardScaler().fit_transform(x_log).astype(np.float32)
        return x, list(cols)

    def _preprocess_count_spark(self, cols: Sequence[str]) -> tuple[np.ndarray, List[str]]:
        self._require_pyspark()
        sdf = self.input_df
        for c in cols:
            sdf = sdf.withColumn(c, F.greatest(F.col(c).cast("double"), F.lit(0.0)))
            mu = sdf.select(F.mean(c).alias("mu")).collect()[0]["mu"]
            mu = float(mu) if mu is not None else 0.0
            sdf = sdf.withColumn(c, F.log1p(F.coalesce(F.col(c), F.lit(mu))))
            stats = sdf.select(F.mean(c).alias("mu"), F.stddev_samp(c).alias("sigma")).collect()[0]
            m = float(stats["mu"]) if stats["mu"] is not None else 0.0
            s = float(stats["sigma"]) if stats["sigma"] not in (None, 0.0) else 1.0
            sdf = sdf.withColumn(c, (F.col(c) - F.lit(m)) / F.lit(s))
        pdf = sdf.select(*list(cols)).toPandas()
        return pdf.to_numpy(dtype=np.float32), list(cols)

    def _validate_columns(self) -> None:
        if self.use_spark:
            available = set(self.input_df.columns)
        else:
            available = set(self.input_df.columns.tolist())

        required = set(self.feat_dict.get("all_feats", set()))
        missing = sorted(list(required - available))
        if missing:
            raise KeyError(f"Input DataFrame is missing required feature columns: {missing}")

    @staticmethod
    def _empty_block() -> tuple[np.ndarray, List[str]]:
        return np.empty((0, 0), dtype=np.float32), []

    @staticmethod
    def _require_pyspark() -> None:
        if not HAS_PYSPARK:
            raise ImportError("PySpark is required for Spark preprocessing but is not installed.")
