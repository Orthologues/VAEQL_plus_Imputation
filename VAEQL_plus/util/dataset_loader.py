#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-02-19
# Desicrption: Dataset loader utilities for the VAE-Q learning imputation baseline.
#########################################################

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal

import pandas as pd

from ..conf import FeaturesTypeDict


if TYPE_CHECKING:  # avoid hard dependency on pyspark unless the pyspark loader is used
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame  # pragma: no cover


PANDAS_BASE_ROW_LIMIT = 1_000_000
PANDAS_BASE_MEMORY_BYTES = 16 * 1024**3
DatasetEngine = Literal["auto", "pandas", "pyspark"]


def _machine_memory_bytes() -> int:
    """Return the smaller of host RAM and an applicable Linux cgroup limit."""
    candidates: list[int] = []
    if hasattr(os, "sysconf"):
        try:
            candidates.append(
                int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
            )
        except (OSError, TypeError, ValueError):
            pass

    for limit_path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            raw_limit = limit_path.read_text(encoding="ascii").strip()
        except (FileNotFoundError, OSError):
            continue
        if raw_limit == "max":
            continue
        try:
            limit = int(raw_limit)
        except ValueError:
            continue
        if 0 < limit < 2**60:
            candidates.append(limit)

    if not candidates:
        raise RuntimeError("Unable to determine machine or container memory")
    return min(candidates)


def pandas_row_limit(*, total_memory_bytes: int | None = None) -> int:
    """Scale one million CSV rows at 16 GiB to the available runtime memory."""
    memory_bytes = _machine_memory_bytes() if total_memory_bytes is None else int(total_memory_bytes)
    if memory_bytes <= 0:
        raise ValueError("total_memory_bytes must be greater than zero")
    return max(
        1,
        int(PANDAS_BASE_ROW_LIMIT * memory_bytes / PANDAS_BASE_MEMORY_BYTES),
    )


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
        raise FileNotFoundError(f"Dataset file not found: {csv_path!r}")
    return csv_path


def _ordered_feature_columns(df_columns: Iterable[str], features: FeaturesTypeDict) -> list[str]:
    """Return columns ordered as in the file but filtered to required features; raise on missing."""
    required = set(features.get("all_feats", set()))
    missing = required - set(df_columns)
    if missing:
        raise ValueError(f"Missing required features in dataset: {sorted(missing)!r}")
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
    try:
        from pyspark.sql import SparkSession
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyspark is an optional backend required for datasets above the "
            "RAM-scaled pandas row limit; install pyspark or force engine='pandas'."
        ) from exc

    path = _resolve_path(dataset_name, data_root)
    spark = spark or SparkSession.builder.getOrCreate()
    df = spark.read.csv(str(path), header=True, inferSchema=True, **read_csv_kwargs)
    cols = _ordered_feature_columns(df.columns, features)
    return df.select(cols)


def select_dataset_engine(
    dataset_name: str,
    *,
    data_root: str | Path | None = None,
    total_memory_bytes: int | None = None,
    pandas_max_rows: int | None = None,
) -> Literal["pandas", "pyspark"]:
    """Select pandas or PySpark from the CSV row count and available RAM."""
    path = _resolve_path(dataset_name, data_root)
    row_limit = (
        pandas_row_limit(total_memory_bytes=total_memory_bytes)
        if pandas_max_rows is None
        else int(pandas_max_rows)
    )
    if row_limit < 1:
        raise ValueError("pandas_max_rows must be at least 1")

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        if next(reader, None) is None:
            raise ValueError(f"Dataset CSV is empty: {path!r}")
        data_rows = 0
        for row in reader:
            if not row:
                continue
            data_rows += 1
            if data_rows > row_limit:
                return "pyspark"
    return "pandas"


def load_dataset(
    dataset_name: str,
    features: FeaturesTypeDict,
    *,
    engine: DatasetEngine = "auto",
    data_root: str | Path | None = None,
    total_memory_bytes: int | None = None,
    pandas_max_rows: int | None = None,
    **kwargs,
):
    """Load with pandas by default and PySpark above the RAM-scaled row limit."""
    if engine == "auto":
        engine = select_dataset_engine(
            dataset_name,
            data_root=data_root,
            total_memory_bytes=total_memory_bytes,
            pandas_max_rows=pandas_max_rows,
        )
    if engine == "pandas":
        return load_pandas(dataset_name, features, data_root=data_root, **kwargs)
    if engine == "pyspark":
        return load_pyspark(dataset_name, features, data_root=data_root, **kwargs)
    raise ValueError(f"Unsupported engine {engine!r}. Use 'auto', 'pandas', or 'pyspark'.")
