from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import pandas as pd


def benchmark_predictor(
    frame: pd.DataFrame,
    predict_one: Callable[[str], str],
    sample_size: int,
    random_state: int,
) -> dict[str, object]:
    sample = frame.sample(n=min(sample_size, len(frame)), random_state=random_state).copy()

    rows: list[dict[str, object]] = []
    latencies_ms: list[float] = []
    for _, row in sample.iterrows():
        start = time.perf_counter()
        prediction = predict_one(row["text"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)
        rows.append(
            {
                "record_id": row["record_id"],
                "text": row["text"],
                "expected_label": row["label_text"],
                "predicted_label": prediction,
                "latency_ms": elapsed_ms,
            }
        )

    return {
        "sample_size": len(rows),
        "mean_latency_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
        "median_latency_ms": statistics.median(latencies_ms) if latencies_ms else 0.0,
        "min_latency_ms": min(latencies_ms) if latencies_ms else 0.0,
        "max_latency_ms": max(latencies_ms) if latencies_ms else 0.0,
        "samples": rows,
    }
