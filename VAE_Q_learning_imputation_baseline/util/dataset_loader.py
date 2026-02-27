#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-02-19
# Desicrption: Dataset loader utilities for the VAE-Q learning imputation baseline.
#########################################################

from pathlib import Path
from typing import Iterable, Literal, TYPE_CHECKING
import pandas as pd

from ..conf import FeaturesTypeDict


if TYPE_CHECKING:  # avoid hard dependency on pyspark unless the pyspark loader is used
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame  # pragma: no cover


def _resolve_path(dataset_name: str, root: str | Path | None) -> Path:
    """Resolve dataset name to a CSV path."""
    base = Path(root) if root is not None else (
        Path(__file__).resolve().parents[1] / "datasets_preprocessing" / "preprocessed_datasets"
    )
    # Allow callers to pass an explicit file path or bare name.
    candidate = Path(dataset_name)
    if candidate.suffix and candidate.exists():
        return candidate
    csv_path = base / (dataset_name if dataset_name.endswith(".csv") else f"{dataset_name}.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    return csv_path


def _ordered_feature_columns(df_columns: Iterable[str], features: FeaturesTypeDict) -> list[str]:
    """Return columns ordered as in the file but filtered to required features; raise on missing."""
    required = set(features.get("all_feats", set()))
    missing = required - set(df_columns)
    if missing:
        raise ValueError(f"Missing required features in dataset: {sorted(missing)}")
    return [col for col in df_columns if col in required]


def load_pandas(
    dataset_name: str,
    features: FeaturesTypeDict,
    *,
    data_root: str | Path | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load dataset as pandas DataFrame with columns validated against provided features."""
    path = _resolve_path(dataset_name, data_root)
    df = pd.read_csv(path, **read_csv_kwargs)
    cols = _ordered_feature_columns(df.columns, features)
    return df.loc[:, cols]


def load_pyspark(
    dataset_name: str,
    features: FeaturesTypeDict,
    *,
    spark: "SparkSession | None" = None,
    data_root: str | Path | None = None,
    **read_csv_kwargs,
) -> "SparkDataFrame":
    """Load dataset as Spark DataFrame with columns validated against provided features."""
    from pyspark.sql import SparkSession  # local import to keep dependency optional

    path = _resolve_path(dataset_name, data_root)
    spark = spark or SparkSession.builder.getOrCreate()
    df = spark.read.csv(str(path), header=True, inferSchema=True, **read_csv_kwargs)
    cols = _ordered_feature_columns(df.columns, features)
    return df.select(cols)


def load_dataset(
    dataset_name: str,
    features: FeaturesTypeDict,
    *,
    engine: Literal["pandas", "pyspark"] = "pandas",
    data_root: str | Path | None = None,
    **kwargs,
):
    """Convenience wrapper choosing pandas or pyspark backend."""
    if engine == "pandas":
        return load_pandas(dataset_name, features, data_root=data_root, **kwargs)
    if engine == "pyspark":
        return load_pyspark(dataset_name, features, data_root=data_root, **kwargs)
    raise ValueError(f"Unsupported engine '{engine}'. Use 'pandas' or 'pyspark'.")
