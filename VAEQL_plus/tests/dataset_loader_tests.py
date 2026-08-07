#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-08-07
# Description: Dataset backend selection and pandas loading tests.
# Development: Mainly written with GPT-5.5 Medium/GPT-5.6 Luna-XHigh on Codex, with Jiawei Zhao's human
# review and revisions.
#########################################################

from __future__ import annotations

from VAEQL_plus.util.dataset_loader import (
    PANDAS_BASE_MEMORY_BYTES,
    load_dataset,
    pandas_row_limit,
    select_dataset_engine,
)


def test_pandas_row_limit_scales_with_runtime_memory() -> None:
    assert pandas_row_limit(total_memory_bytes=PANDAS_BASE_MEMORY_BYTES) == 1_000_000
    assert pandas_row_limit(total_memory_bytes=2 * PANDAS_BASE_MEMORY_BYTES) == 2_000_000


def test_auto_engine_uses_ram_scaled_row_boundary(tmp_path) -> None:
    dataset_path = tmp_path / "trial.csv"
    dataset_path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")

    assert select_dataset_engine(str(dataset_path), pandas_max_rows=3) == "pandas"
    assert select_dataset_engine(str(dataset_path), pandas_max_rows=2) == "pyspark"


def test_auto_engine_loads_small_dataset_with_pandas(tmp_path) -> None:
    dataset_path = tmp_path / "trial.csv"
    dataset_path.write_text("a,b,unused\n1,2,x\n3,4,y\n", encoding="utf-8")
    features = {"all_feats": {"a", "b"}}

    frame = load_dataset(
        str(dataset_path),
        features,
        engine="auto",
        pandas_max_rows=2,
    )

    assert frame.columns.tolist() == ["a", "b"]
    assert frame.to_dict(orient="records") == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
