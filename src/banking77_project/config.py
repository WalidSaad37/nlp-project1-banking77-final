from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PathsConfig:
    raw_data_dir: str
    processed_data_dir: str
    cache_dir: str
    output_dir: str


@dataclass(slots=True)
class DatasetConfig:
    name: str
    train_split: str
    test_split: str


@dataclass(slots=True)
class PreprocessingConfig:
    lowercase: bool
    expand_contractions: bool
    remove_punctuation: bool
    remove_digits: bool
    remove_stopwords: bool
    min_token_length: int


@dataclass(slots=True)
class ClassicalConfig:
    embedding_model: str
    embedding_source: str
    embedding_local_path: str | None
    classifier_name: str
    svm_c: float
    benchmark_sample_size: int


@dataclass(slots=True)
class LLMConfig:
    generation_model: str
    embedding_model: str
    retrieval_top_k: int
    temperature: float
    max_output_tokens: int
    benchmark_sample_size: int
    max_train_examples_for_retrieval: int | None
    max_test_examples: int | None
    batch_size: int


@dataclass(slots=True)
class ProjectConfig:
    project_name: str
    random_state: int
    paths: PathsConfig
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    classical: ClassicalConfig
    llm: LLMConfig
    root_dir: Path

    def resolve_path(self, relative_path: str) -> Path:
        return (self.root_dir / relative_path).resolve()


def _coerce_config(data: dict[str, Any], root_dir: Path) -> ProjectConfig:
    return ProjectConfig(
        project_name=data["project_name"],
        random_state=int(data["random_state"]),
        paths=PathsConfig(**data["paths"]),
        dataset=DatasetConfig(**data["dataset"]),
        preprocessing=PreprocessingConfig(**data["preprocessing"]),
        classical=ClassicalConfig(**data["classical"]),
        llm=LLMConfig(**data["llm"]),
        root_dir=root_dir.resolve(),
    )


def load_config(config_path: str | Path) -> ProjectConfig:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return _coerce_config(raw, config_path.parent.parent)
