from __future__ import annotations

import time
from typing import Any

import gensim.downloader as gensim_api
import joblib
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

from banking77_project.benchmark import benchmark_predictor
from banking77_project.config import ProjectConfig
from banking77_project.io_utils import ensure_dir, write_json
from banking77_project.metrics import compute_classification_metrics
from banking77_project.preprocessing import tokenize


def load_embedding_model(config: ProjectConfig) -> KeyedVectors:
    if config.classical.embedding_local_path:
        return KeyedVectors.load_word2vec_format(
            config.classical.embedding_local_path,
            binary=config.classical.embedding_local_path.endswith(".bin"),
        )

    if config.classical.embedding_source != "gensim":
        raise ValueError(f"Unsupported embedding source: {config.classical.embedding_source}")

    return gensim_api.load(config.classical.embedding_model)


def document_vector(text: str, vectors: KeyedVectors, config: ProjectConfig) -> np.ndarray:
    tokens = tokenize(text, config.preprocessing)
    known_vectors = [vectors[token] for token in tokens if token in vectors]
    if not known_vectors:
        return np.zeros(vectors.vector_size, dtype=np.float32)
    return np.mean(known_vectors, axis=0).astype(np.float32)


def build_feature_matrix(
    texts: list[str],
    vectors: KeyedVectors,
    config: ProjectConfig,
    desc: str | None = None,
) -> np.ndarray:
    iterator = tqdm(texts, desc=desc, unit="doc", leave=False) if desc else texts
    return np.vstack([document_vector(text, vectors, config) for text in iterator])


class ClassicalIntentClassifier:
    def __init__(self, pipeline: Pipeline, vectors: KeyedVectors, config: ProjectConfig) -> None:
        self.pipeline = pipeline
        self.vectors = vectors
        self.config = config

    def predict(self, texts: list[str]) -> list[str]:
        features = build_feature_matrix(texts, self.vectors, self.config)
        return self.pipeline.predict(features).tolist()

    def predict_one(self, text: str) -> str:
        return self.predict([text])[0]


def run_classical_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ProjectConfig,
) -> dict[str, Any]:
    output_dir = ensure_dir(config.resolve_path(config.paths.output_dir) / "classical")
    print("[classical] Loading static embeddings...", flush=True)
    vectors = load_embedding_model(config)

    train_start = time.perf_counter()
    print("[classical] Building training feature matrix...", flush=True)
    x_train = build_feature_matrix(train_df["text"].tolist(), vectors, config, desc="Classical train vectors")
    print("[classical] Building test feature matrix...", flush=True)
    x_test = build_feature_matrix(test_df["text"].tolist(), vectors, config, desc="Classical test vectors")

    print("[classical] Training LinearSVC classifier...", flush=True)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(C=config.classical.svm_c, random_state=config.random_state)),
        ]
    )
    pipeline.fit(x_train, train_df["label_text"])
    training_seconds = time.perf_counter() - train_start

    inference_start = time.perf_counter()
    print("[classical] Running test-set inference...", flush=True)
    predictions = pipeline.predict(x_test).tolist()
    test_inference_seconds = time.perf_counter() - inference_start

    metrics = compute_classification_metrics(test_df["label_text"].tolist(), predictions)
    runtime = {
        "training_seconds": training_seconds,
        "test_inference_seconds": test_inference_seconds,
    }

    print("[classical] Running latency benchmark...", flush=True)
    benchmark = benchmark_predictor(
        test_df,
        ClassicalIntentClassifier(pipeline, vectors, config).predict_one,
        sample_size=config.classical.benchmark_sample_size,
        random_state=config.random_state,
    )

    print("[classical] Writing artifacts...", flush=True)
    prediction_frame = test_df[["record_id", "text", "label_text"]].copy()
    prediction_frame["predicted_label"] = predictions
    prediction_frame.to_parquet(output_dir / "test_predictions.parquet", index=False)

    joblib.dump(pipeline, output_dir / "classifier.joblib")
    write_json(
        output_dir / "metrics.json",
        {
            "embedding_model": config.classical.embedding_model,
            "classifier_name": config.classical.classifier_name,
            "runtime": runtime,
            "metrics": metrics,
            "benchmark": benchmark,
        },
    )
    return {
        "embedding_model": config.classical.embedding_model,
        "classifier_name": config.classical.classifier_name,
        "runtime": runtime,
        "metrics": metrics,
        "benchmark": benchmark,
    }
