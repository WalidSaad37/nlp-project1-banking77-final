from __future__ import annotations

from typing import Any

import pandas as pd

from banking77_project.config import ProjectConfig
from banking77_project.io_utils import ensure_dir, write_json
from banking77_project.llm import GeminiIntentClassifier

HOURS_PER_MONTH = 730
C4_STANDARD_8_HOURLY_USD = 0.39534
C4_STANDARD_2_HOURLY_USD = 0.096866
GEMINI_FLASH_INPUT_PER_M_TOKENS_USD = 0.30
GEMINI_FLASH_OUTPUT_PER_M_TOKENS_USD = 2.50


def _estimate_tokens_from_chars(char_count: float) -> float:
    return char_count / 4.0


def generate_business_case(
    train_df: pd.DataFrame,
    label_names: list[str],
    classical_results: dict[str, Any] | None,
    llm_results: dict[str, Any] | None,
    config: ProjectConfig,
) -> dict[str, Any]:
    class DummyClassifier:
        def __init__(self) -> None:
            self.label_names = label_names
            self.config = config

    dummy = DummyClassifier()
    system_prompt = GeminiIntentClassifier._system_prompt(dummy)

    avg_query_chars = float(train_df["text"].str.len().mean())
    avg_example_chars = avg_query_chars
    example_block_chars = config.llm.retrieval_top_k * (avg_example_chars + 45.0)
    user_prompt_chars = len("Customer query:\n\n\nNearest labeled examples:\n\nPredict the most appropriate Banking77 label.")
    total_input_chars = len(system_prompt) + user_prompt_chars + avg_query_chars + example_block_chars

    label_block_chars = sum(len(label) + 6 for label in label_names)
    vectorization_prompt_chars = len(
        "Create an intent-affinity vector for the following customer banking query.\n\nLabel order:\n\n\nCustomer query:\n"
    )
    total_vectorization_input_chars = vectorization_prompt_chars + label_block_chars + avg_query_chars
    estimated_vectorization_input_tokens = _estimate_tokens_from_chars(total_vectorization_input_chars)
    estimated_vectorization_output_tokens = float(len(label_names) * 6)
    estimated_generation_input_tokens = _estimate_tokens_from_chars(total_input_chars)
    estimated_generation_output_tokens = 8.0

    monthly_classical_hosting_usd = C4_STANDARD_8_HOURLY_USD * HOURS_PER_MONTH
    monthly_llm_hosting_usd = C4_STANDARD_2_HOURLY_USD * HOURS_PER_MONTH

    monthly_llm_vectorization_input_cost_usd = (
        estimated_vectorization_input_tokens * GEMINI_FLASH_INPUT_PER_M_TOKENS_USD
    )
    monthly_llm_vectorization_output_cost_usd = (
        estimated_vectorization_output_tokens * GEMINI_FLASH_OUTPUT_PER_M_TOKENS_USD
    )
    monthly_llm_generation_input_cost_usd = (
        estimated_generation_input_tokens * GEMINI_FLASH_INPUT_PER_M_TOKENS_USD
    )
    monthly_llm_generation_output_cost_usd = (
        estimated_generation_output_tokens * GEMINI_FLASH_OUTPUT_PER_M_TOKENS_USD
    )

    monthly_request_volume_millions = 1.0
    monthly_llm_api_cost_usd = (
        monthly_request_volume_millions
        * (
            monthly_llm_vectorization_input_cost_usd
            + monthly_llm_vectorization_output_cost_usd
            + monthly_llm_generation_input_cost_usd
            + monthly_llm_generation_output_cost_usd
        )
    )

    initial_train_vectorization_input_tokens = _estimate_tokens_from_chars(
        float(train_df["text"].str.len().sum()) + len(train_df) * (vectorization_prompt_chars + label_block_chars)
    )
    initial_train_vectorization_output_tokens = float(len(train_df) * len(label_names) * 6)
    one_time_train_vectorization_cost_usd = (
        (initial_train_vectorization_input_tokens / 1_000_000) * GEMINI_FLASH_INPUT_PER_M_TOKENS_USD
        + (initial_train_vectorization_output_tokens / 1_000_000) * GEMINI_FLASH_OUTPUT_PER_M_TOKENS_USD
    )

    recommendation = (
        "Choose the classical model for a production-first deployment unless the Gemini pipeline "
        "delivers a materially higher held-out F1 after evaluation with a real API key. "
        "The classical system already offers strong accuracy, sub-millisecond single-query latency, "
        "and lower operational dependency risk."
    )

    if classical_results and llm_results:
        classical_f1 = classical_results["metrics"]["macro_f1"]
        llm_f1 = llm_results["metrics"]["macro_f1"]
        if llm_f1 - classical_f1 >= 0.03:
            recommendation = (
                "Choose the Gemini retrieval-plus-LLM classifier for production if accuracy is the primary KPI. "
                "In this project it delivers a materially higher held-out Macro F1 than the classical baseline, "
                "while the estimated monthly cost remains in the same range. The tradeoff is higher latency and "
                "greater operational complexity."
            )

    result = {
        "assumptions": {
            "monthly_requests": 1_000_000,
            "classical_host_machine": "c4-standard-8",
            "classical_host_hourly_usd": C4_STANDARD_8_HOURLY_USD,
            "llm_host_machine": "c4-standard-2",
            "llm_host_hourly_usd": C4_STANDARD_2_HOURLY_USD,
            "gemini_generation_model": config.llm.generation_model,
            "gemini_vectorization_model": config.llm.vectorization_model,
            "gemini_flash_input_price_per_m_tokens_usd": GEMINI_FLASH_INPUT_PER_M_TOKENS_USD,
            "gemini_flash_output_price_per_m_tokens_usd": GEMINI_FLASH_OUTPUT_PER_M_TOKENS_USD,
            "estimated_vectorization_input_tokens_per_request": estimated_vectorization_input_tokens,
            "estimated_vectorization_output_tokens_per_request": estimated_vectorization_output_tokens,
            "estimated_generation_input_tokens_per_request": estimated_generation_input_tokens,
            "estimated_generation_output_tokens_per_request": estimated_generation_output_tokens,
        },
        "classical_monthly_cost_usd": monthly_classical_hosting_usd,
        "llm_monthly_cost_usd": monthly_llm_hosting_usd + monthly_llm_api_cost_usd,
        "llm_cost_breakdown_usd": {
            "host": monthly_llm_hosting_usd,
            "query_vectorization_input_api": monthly_llm_vectorization_input_cost_usd,
            "query_vectorization_output_api": monthly_llm_vectorization_output_cost_usd,
            "generation_input_api": monthly_llm_generation_input_cost_usd,
            "generation_output_api": monthly_llm_generation_output_cost_usd,
            "one_time_train_vectorization_bootstrap": one_time_train_vectorization_cost_usd,
        },
        "current_classical_baseline": classical_results,
        "current_llm_baseline": llm_results,
        "recommendation": recommendation,
    }

    output_path = ensure_dir(config.resolve_path(config.paths.output_dir)) / "business_case.json"
    write_json(output_path, result)
    return result
