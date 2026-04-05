from __future__ import annotations

import json
import hashlib
import os
import runpy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from google.genai.errors import APIError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

from banking77_project.benchmark import benchmark_predictor
from banking77_project.config import ProjectConfig
from banking77_project.io_utils import ensure_dir, write_json
from banking77_project.metrics import compute_classification_metrics


def _is_retryable(exception: BaseException) -> bool:
    return isinstance(exception, APIError) and exception.code in {429, 500, 502, 503, 504}


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return a_norm @ b_norm.T


def _response_text(response: Any) -> str:
    if getattr(response, "text", None):
        return response.text.strip()

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    chunks: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            chunks.append(text)
    return "".join(chunks).strip()


def _load_local_api_key(config: ProjectConfig) -> str | None:
    settings_path = config.resolve_path("configs/local_settings.py")
    if not settings_path.exists():
        return None

    values = runpy.run_path(str(settings_path))
    candidate = values.get("GEMINI_API_KEY")
    if isinstance(candidate, str):
        cleaned = candidate.strip()
        if cleaned and cleaned != "PASTE_YOUR_NEW_GEMINI_API_KEY_HERE":
            return cleaned
    return None


class GeminiIntentClassifier:
    def __init__(self, train_df: pd.DataFrame, label_names: list[str], config: ProjectConfig) -> None:
        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or _load_local_api_key(config)
        )
        if not api_key:
            raise RuntimeError(
                "Gemini API key not found. Set GEMINI_API_KEY, GOOGLE_API_KEY, or put it in configs/local_settings.py."
            )

        self.client = genai.Client(api_key=api_key)
        self.config = config
        self.label_names = label_names
        self.train_df = train_df.reset_index(drop=True).copy()
        self.cache_dir = ensure_dir(config.resolve_path(config.paths.cache_dir) / "gemini")
        self.embedding_cache_path = self.cache_dir / "embeddings.parquet"

        if config.llm.max_train_examples_for_retrieval:
            self.train_df = self.train_df.sample(
                n=min(config.llm.max_train_examples_for_retrieval, len(self.train_df)),
                random_state=config.random_state,
            ).sort_values("record_id")

        self.train_embeddings = self.embed_texts(
            self.train_df["text"].tolist(),
            task_type="RETRIEVAL_DOCUMENT",
            namespace="train_documents",
        )

    def _load_embedding_cache(self) -> pd.DataFrame:
        if self.embedding_cache_path.exists():
            return pd.read_parquet(self.embedding_cache_path)
        return pd.DataFrame(columns=["cache_key", "namespace", "text", "embedding"])

    def _save_embedding_cache(self, cache_df: pd.DataFrame) -> None:
        cache_df.to_parquet(self.embedding_cache_path, index=False)

    def _cache_key(self, text: str, namespace: str) -> str:
        raw = f"{self.config.llm.embedding_model}::{namespace}::{text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        response = self.client.models.embed_content(
            model=self.config.llm.embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(taskType=task_type),
        )
        return [item.values for item in response.embeddings]

    def embed_texts(self, texts: list[str], task_type: str, namespace: str) -> np.ndarray:
        cache_df = self._load_embedding_cache()
        cache_lookup = cache_df.set_index("cache_key").to_dict("index")

        ordered_vectors: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str, str]] = []
        for index, text in enumerate(texts):
            cache_key = self._cache_key(text, namespace)
            cached = cache_lookup.get(cache_key)
            if cached:
                ordered_vectors[index] = cached["embedding"]
            else:
                missing.append((index, cache_key, text))

        cached_count = len(texts) - len(missing)
        print(
            f"[llm] Embeddings for '{namespace}': {cached_count}/{len(texts)} cached, {len(missing)} to request.",
            flush=True,
        )

        if missing:
            new_rows: list[dict[str, Any]] = []
            batch_starts = range(0, len(missing), self.config.llm.batch_size)
            for start in tqdm(
                batch_starts,
                total=(len(missing) + self.config.llm.batch_size - 1) // self.config.llm.batch_size,
                desc=f"Embedding {namespace}",
                unit="batch",
            ):
                batch = missing[start : start + self.config.llm.batch_size]
                batch_texts = [item[2] for item in batch]
                embeddings = self._embed_batch(batch_texts, task_type=task_type)
                for (index, cache_key, text), embedding in zip(batch, embeddings, strict=True):
                    ordered_vectors[index] = embedding
                    new_rows.append(
                        {
                            "cache_key": cache_key,
                            "namespace": namespace,
                            "text": text,
                            "embedding": embedding,
                        }
                    )

            updated_cache = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
            updated_cache = updated_cache.drop_duplicates(subset=["cache_key"], keep="last")
            self._save_embedding_cache(updated_cache)

        return np.asarray(ordered_vectors, dtype=np.float32)

    def retrieve_examples(self, query: str) -> list[dict[str, str]]:
        query_embedding = self.embed_texts([query], task_type="RETRIEVAL_QUERY", namespace="query_text")[0]
        return self.retrieve_examples_from_embedding(query_embedding)

    def retrieve_examples_from_embedding(self, query_embedding: np.ndarray) -> list[dict[str, str]]:
        scores = _cosine_similarity_matrix(np.asarray([query_embedding], dtype=np.float32), self.train_embeddings)[0]
        top_indices = np.argsort(scores)[::-1][: self.config.llm.retrieval_top_k]
        examples: list[dict[str, str]] = []
        for idx in top_indices:
            row = self.train_df.iloc[int(idx)]
            examples.append(
                {
                    "text": row["text"],
                    "label": row["label_text"],
                    "score": f"{float(scores[idx]):.4f}",
                }
            )
        return examples

    def _system_prompt(self) -> str:
        label_block = "\n".join(f"- {label}" for label in self.label_names)
        return (
            "You are an expert banking support ticket classifier.\n"
            "Choose exactly one label from the approved Banking77 taxonomy.\n"
            "Do not invent labels.\n"
            "Use the retrieved examples only as guidance, not as labels to copy blindly.\n"
            "Return only the label text and nothing else.\n\n"
            f"Approved labels:\n{label_block}"
        )

    def _user_prompt(self, query: str, examples: list[dict[str, str]]) -> str:
        example_block = "\n\n".join(
            f"Example {idx + 1}\nLabel: {item['label']}\nSimilarity: {item['score']}\nText: {item['text']}"
            for idx, item in enumerate(examples)
        )
        return (
            "Return only one exact Banking77 label from the approved list.\n"
            "Do not return JSON.\n"
            "Do not add explanation.\n\n"
            f"Customer query:\n{query}\n\n"
            f"Nearest labeled examples:\n{example_block}\n\n"
            "Predict the most appropriate Banking77 label."
        )

    def _extract_label_from_text(self, text: str) -> str | None:
        candidate = text.strip()
        if not candidate:
            return None

        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                label = payload.get("predicted_label")
                if isinstance(label, str) and label in self.label_names:
                    return label
        except Exception:
            pass

        if candidate in self.label_names:
            return candidate

        lowered = candidate.lower()
        for label in self.label_names:
            if label.lower() == lowered:
                return label

        for label in sorted(self.label_names, key=len, reverse=True):
            if label.lower() in lowered:
                return label

        return None

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def classify_one(self, query: str, query_embedding: np.ndarray | None = None) -> str:
        examples = (
            self.retrieve_examples_from_embedding(query_embedding)
            if query_embedding is not None
            else self.retrieve_examples(query)
        )
        response = self.client.models.generate_content(
            model=self.config.llm.generation_model,
            contents=self._user_prompt(query, examples),
            config=types.GenerateContentConfig(
                systemInstruction=self._system_prompt(),
                temperature=self.config.llm.temperature,
                maxOutputTokens=self.config.llm.max_output_tokens,
                seed=self.config.random_state,
            ),
        )
        extracted = self._extract_label_from_text(_response_text(response))
        if extracted:
            return extracted

        return examples[0]["label"]


def run_llm_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_names: list[str],
    config: ProjectConfig,
) -> dict[str, Any]:
    output_dir = ensure_dir(config.resolve_path(config.paths.output_dir) / "llm")
    print("[llm] Initializing Gemini classifier and retrieval store...", flush=True)
    classifier = GeminiIntentClassifier(train_df=train_df, label_names=label_names, config=config)

    eval_df = test_df.copy()
    if config.llm.max_test_examples:
        eval_df = eval_df.sample(
            n=min(config.llm.max_test_examples, len(eval_df)),
            random_state=config.random_state,
        ).sort_values("record_id")

    print(f"[llm] Evaluation rows: {len(eval_df)}", flush=True)
    eval_texts = eval_df["text"].tolist()
    print("[llm] Embedding evaluation queries...", flush=True)
    eval_query_embeddings = classifier.embed_texts(
        eval_texts,
        task_type="RETRIEVAL_QUERY",
        namespace="query_text",
    )

    predictions: list[str | None] = [None] * len(eval_texts)
    latencies_ms: list[float] = [0.0] * len(eval_texts)

    def _predict_row(index: int) -> tuple[int, str, float]:
        text = eval_texts[index]
        start = time.perf_counter()
        prediction = classifier.classify_one(text, query_embedding=eval_query_embeddings[index])
        latency = (time.perf_counter() - start) * 1000
        return index, prediction, latency

    max_workers = min(4, max(1, len(eval_texts)))
    print(f"[llm] Classifying queries with {max_workers} worker(s)...", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_predict_row, index): index for index in range(len(eval_texts))}
        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM classification", unit="query"):
            index, prediction, latency = future.result()
            predictions[index] = prediction
            latencies_ms[index] = latency

    if any(prediction is None for prediction in predictions):
        raise RuntimeError("LLM evaluation did not produce predictions for every test row.")
    resolved_predictions = [str(prediction) for prediction in predictions]

    print("[llm] Computing evaluation metrics...", flush=True)
    metrics = compute_classification_metrics(eval_df["label_text"].tolist(), resolved_predictions)
    print("[llm] Running latency benchmark...", flush=True)
    benchmark = benchmark_predictor(
        eval_df,
        classifier.classify_one,
        sample_size=config.llm.benchmark_sample_size,
        random_state=config.random_state,
    )

    print("[llm] Writing artifacts...", flush=True)
    prediction_frame = eval_df[["record_id", "text", "label_text"]].copy()
    prediction_frame["predicted_label"] = resolved_predictions
    prediction_frame["latency_ms"] = latencies_ms
    prediction_frame.to_parquet(output_dir / "test_predictions.parquet", index=False)

    result = {
        "embedding_model": config.llm.embedding_model,
        "generation_model": config.llm.generation_model,
        "evaluated_rows": len(eval_df),
        "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "median_latency_ms": float(np.median(latencies_ms)) if latencies_ms else 0.0,
        "metrics": metrics,
        "benchmark": benchmark,
    }
    write_json(output_dir / "metrics.json", result)
    return result
