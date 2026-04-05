from __future__ import annotations

import hashlib

import pandas as pd
from datasets import DatasetDict, load_dataset

from banking77_project.config import ProjectConfig
from banking77_project.io_utils import ensure_dir, read_json, write_json
from banking77_project.preprocessing import preprocess_many


def _with_record_id(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["record_id"] = output["text"].map(
        lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest()
    )
    return output


def _prepare_split(frame: pd.DataFrame, label_names: list[str]) -> pd.DataFrame:
    prepared = frame.rename(columns={"label": "label_id"})
    if "label_text" not in prepared.columns:
        prepared["label_text"] = prepared["label_id"].map(lambda idx: label_names[idx])
    selected = prepared[["text", "label_id", "label_text"]]
    return _with_record_id(selected)


def load_or_create_dataset(
    config: ProjectConfig,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    raw_dir = ensure_dir(config.resolve_path(config.paths.raw_data_dir))
    processed_dir = ensure_dir(config.resolve_path(config.paths.processed_data_dir))

    train_path = processed_dir / "banking77_train.parquet"
    test_path = processed_dir / "banking77_test.parquet"
    labels_path = processed_dir / "label_names.json"

    if train_path.exists() and test_path.exists() and labels_path.exists() and not force_refresh:
        return pd.read_parquet(train_path), pd.read_parquet(test_path), read_json(labels_path)

    dataset: DatasetDict = load_dataset(config.dataset.name)
    train_raw = dataset[config.dataset.train_split].to_pandas()
    if "label_text" in train_raw.columns:
        label_mapping = (
            train_raw[["label", "label_text"]]
            .drop_duplicates()
            .sort_values("label")
        )
        label_names = label_mapping["label_text"].tolist()
    else:
        label_names = dataset[config.dataset.train_split].features["label"].names

    train_df = _prepare_split(train_raw, label_names)
    test_df = _prepare_split(dataset[config.dataset.test_split].to_pandas(), label_names)

    train_df["clean_text"] = preprocess_many(train_df["text"].tolist(), config.preprocessing)
    test_df["clean_text"] = preprocess_many(test_df["text"].tolist(), config.preprocessing)

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    write_json(labels_path, label_names)

    dataset[config.dataset.train_split].to_parquet(str(raw_dir / "banking77_train_raw.parquet"))
    dataset[config.dataset.test_split].to_parquet(str(raw_dir / "banking77_test_raw.parquet"))

    return train_df, test_df, label_names
