from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from banking77_project.classical import run_classical_experiment
from banking77_project.business_case import generate_business_case
from banking77_project.config import load_config
from banking77_project.data import load_or_create_dataset
from banking77_project.io_utils import ensure_dir, write_json
from banking77_project.llm import _load_local_api_key, run_llm_experiment


def _log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _format_seconds(value: float) -> str:
    if value < 60:
        return f"{value:.1f}s"
    minutes, seconds = divmod(value, 60)
    return f"{int(minutes)}m {seconds:.1f}s"


def _gemini_key_status(config) -> str:
    try:
        import os

        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or _load_local_api_key(config):
            return "detected"
    except Exception:
        pass
    return "missing"


def _print_banner(command: str, train_rows: int, test_rows: int, label_count: int, gemini_status: str) -> None:
    line = "=" * 56
    print(line, flush=True)
    print("Banking77 Final Project Runner", flush=True)
    print(f"Command: {command}", flush=True)
    print(f"Train rows: {train_rows} | Test rows: {test_rows} | Labels: {label_count}", flush=True)
    print(f"Gemini key: {gemini_status}", flush=True)
    print(line, flush=True)


def _print_plan(command: str) -> None:
    stages = ["Prepare dataset"]
    if command in {"run-classical", "run-all"}:
        stages.append("Run classical pipeline")
    if command in {"run-llm", "run-all"}:
        stages.append("Run Gemini pipeline")
    if command == "business-case":
        stages.append("Generate business case")

    print("Execution plan:", flush=True)
    for index, stage in enumerate(stages, start=1):
        print(f"  {index}. {stage}", flush=True)
    print("", flush=True)


def _print_stage_header(index: int, total: int, label: str) -> None:
    _log(f"[{index}/{total}] {label}")


def _print_result_summary(results: dict[str, object], summary_path: Path) -> None:
    print("=" * 56, flush=True)
    print("Final Results", flush=True)

    classical = results.get("classical")
    if classical:
        classical_metrics = classical["metrics"]
        classical_runtime = classical["runtime"]
        classical_benchmark = classical["benchmark"]
        print(
            "Classical | "
            f"Macro F1 {classical_metrics['macro_f1']:.4f} | "
            f"Accuracy {classical_metrics['accuracy']:.4f} | "
            f"Train {classical_runtime['training_seconds']:.1f}s | "
            f"Mean latency {classical_benchmark['mean_latency_ms']:.2f} ms",
            flush=True,
        )

    llm = results.get("llm")
    if llm:
        llm_metrics = llm["metrics"]
        llm_benchmark = llm["benchmark"]
        print(
            "Gemini    | "
            f"Macro F1 {llm_metrics['macro_f1']:.4f} | "
            f"Accuracy {llm_metrics['accuracy']:.4f} | "
            f"Rows {llm['evaluated_rows']} | "
            f"Mean latency {llm_benchmark['mean_latency_ms']:.2f} ms",
            flush=True,
        )

    if classical and llm:
        winner = "Gemini" if llm["metrics"]["macro_f1"] > classical["metrics"]["macro_f1"] else "Classical"
        print(f"Winner on accuracy: {winner}", flush=True)

    print("-" * 56, flush=True)
    print(f"Summary file: {summary_path}", flush=True)
    print("=" * 56, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Banking77 project pipelines.")
    parser.add_argument(
        "command",
        choices=["prepare-data", "run-classical", "run-llm", "run-all", "business-case"],
        help="Pipeline command to execute.",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "configs" / "project.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Rebuild processed dataset artifacts from source.",
    )
    return parser.parse_args()


def main() -> None:
    overall_start = time.perf_counter()
    args = parse_args()
    _log(f"Starting command: {args.command}")

    config = load_config(args.config)
    _log(f"Loaded config from {args.config}")

    dataset_start = time.perf_counter()
    _log("Preparing dataset artifacts...")
    train_df, test_df, label_names = load_or_create_dataset(config, force_refresh=args.force_refresh)
    _log(
        "Dataset ready: "
        f"train={len(train_df)} rows, test={len(test_df)} rows, labels={len(label_names)} "
        f"in {_format_seconds(time.perf_counter() - dataset_start)}"
    )
    _print_banner(
        command=args.command,
        train_rows=len(train_df),
        test_rows=len(test_df),
        label_count=len(label_names),
        gemini_status=_gemini_key_status(config),
    )
    _print_plan(args.command)

    if args.command == "prepare-data":
        print(
            json.dumps(
                {
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                    "label_count": len(label_names),
                },
                indent=2,
            )
        )
        return

    if args.command == "business-case":
        _log("Generating business-case analysis...")
        summary_path = config.resolve_path(config.paths.output_dir) / "run_summary.json"
        classical_results = None
        llm_results = None
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            classical_results = summary.get("classical")
            llm_results = summary.get("llm")
        result = generate_business_case(train_df, label_names, classical_results, llm_results, config)
        _log(f"Business-case file written in {_format_seconds(time.perf_counter() - overall_start)}")
        print(json.dumps({"business_case_path": str(config.resolve_path(config.paths.output_dir) / "business_case.json")}, indent=2))
        if result:
            return

    results: dict[str, object] = {}
    total_stages = 0
    if args.command in {"run-classical", "run-all"}:
        total_stages += 1
    if args.command in {"run-llm", "run-all"}:
        total_stages += 1
    stage_index = 0

    if args.command in {"run-classical", "run-all"}:
        stage_index += 1
        _print_stage_header(stage_index, total_stages, "Running classical pipeline")
        stage_start = time.perf_counter()
        results["classical"] = run_classical_experiment(train_df, test_df, config)
        _log(
            "Classical pipeline completed in "
            f"{_format_seconds(time.perf_counter() - stage_start)} | "
            f"Macro F1 {results['classical']['metrics']['macro_f1']:.4f}"
        )

    if args.command in {"run-llm", "run-all"}:
        try:
            stage_index += 1
            _print_stage_header(stage_index, total_stages, "Running Gemini LLM pipeline")
            stage_start = time.perf_counter()
            results["llm"] = run_llm_experiment(train_df, test_df, label_names, config)
            _log(
                "Gemini LLM pipeline completed in "
                f"{_format_seconds(time.perf_counter() - stage_start)} | "
                f"Macro F1 {results['llm']['metrics']['macro_f1']:.4f}"
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

    summary_path = ensure_dir(config.resolve_path(config.paths.output_dir)) / "run_summary.json"
    write_json(summary_path, results)
    _log(f"All requested work completed in {_format_seconds(time.perf_counter() - overall_start)}")
    if results:
        _print_result_summary(results, summary_path)
    print(json.dumps({"summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
