# Final Project 1: Comparative Analysis of NLP Architectures for Automated Intent Classification

This repository contains a full implementation of Project 1 from the assignment brief. It compares a classical NLP intent-classification pipeline against a Gemini-based LLM classifier on the `Banking77` dataset, with reproducible scripts, cached artifacts, and report-ready outputs.

## Objective

The goal is to compare two approaches for classifying banking support queries into the 77 Banking77 intent labels:

1. A classical pipeline based on static pretrained word embeddings and a supervised classifier.
2. An LLM-based pipeline based on Gemini embeddings, retrieval, and few-shot prompting.

The project evaluates not just predictive quality, but also inference speed, implementation complexity, and production tradeoffs.

## Dataset

- Dataset: `mteb/banking77`
- Number of classes: 77
- Split strategy: predefined `train` and `test` splits from Hugging Face

The loader persists both raw and processed artifacts locally:

- raw parquet snapshots in [`data/raw`](data/raw)
- processed train/test datasets in [`data/processed`](data/processed)

## Implementation Summary

### Classical pipeline

The classical branch follows this sequence:

1. Normalize and tokenize the customer query.
2. Build a dense document vector by averaging pretrained static word vectors.
3. Train a `LinearSVC` classifier on the resulting vectors.
4. Evaluate on the held-out test set.
5. Benchmark single-query latency on 10 random tickets.

Default configuration:

- embedding model: `glove-wiki-gigaword-100`
- classifier: `LinearSVC`

The assignment specifically mentions pretrained Word2Vec. To stay reproducible on a standard laptop, the default config uses a smaller static embedding model from Gensim, but the code also supports swapping in a local Word2Vec-format file through `embedding_local_path` in [`configs/project.yaml`](configs/project.yaml).

### Gemini pipeline

The LLM branch is built around the assignment's architectural constraints:

1. Embed training questions with Gemini and cache the vectors locally in Parquet.
2. Reuse cached embeddings when a question has already been processed.
3. Embed the incoming query.
4. Retrieve the nearest labeled training examples.
5. Prompt `gemini-2.5-flash` to return exactly one valid Banking77 label.
6. Retry transient API failures with exponential backoff.

The core implementation lives in [`src/banking77_project/llm.py`](src/banking77_project/llm.py).

## Current API interpretation

The assignment text asks for `gemini-2.5-flash` to vectorize questions. The current Gemini API separates embedding generation from text generation, so this implementation uses:

- `gemini-embedding-001` for vector embeddings
- `gemini-2.5-flash` for final label prediction

This keeps the project aligned with the current official Gemini API while preserving the intended design of a retrieval-plus-prompting classifier.

## Project Structure

```text
.
|-- README.md
|-- requirements.txt
|-- configs/
|   `-- project.yaml
|-- data/
|   |-- cache/
|   |-- processed/
|   `-- raw/
|-- outputs/
|-- scripts/
|   `-- run_project.py
`-- src/
    `-- banking77_project/
        |-- benchmark.py
        |-- classical.py
        |-- config.py
        |-- data.py
        |-- io_utils.py
        |-- llm.py
        |-- metrics.py
        `-- preprocessing.py
```

## Installation

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

To enable the Gemini pipeline, set an API key:

```bash
set GEMINI_API_KEY=your_key_here
```

You can also use [`.env.example`](.env.example) as a reference.

If you prefer storing the key in the project instead of the terminal, copy [`configs/local_settings.example.py`](configs/local_settings.example.py) to `configs/local_settings.py` and put your key there:

```python
GEMINI_API_KEY = "your_key_here"
```

## Commands

Prepare the dataset:

```bash
python scripts/run_project.py prepare-data
```

Run the classical pipeline:

```bash
python scripts/run_project.py run-classical
```

Run the Gemini pipeline:

```bash
python scripts/run_project.py run-llm
```

Run the full comparison:

```bash
python scripts/run_project.py run-all
```

## Results

Both pipelines were executed locally on the full Banking77 test split.

Classical baseline from [`outputs/classical/metrics.json`](outputs/classical/metrics.json):

- Accuracy: `0.8241`
- Macro Precision: `0.8269`
- Macro Recall: `0.8240`
- Macro F1: `0.8234`
- Training time: `9.45s`
- Full test-set inference time: `0.0101s`
- Mean sampled latency: `0.45 ms`

Gemini pipeline from [`outputs/llm/metrics.json`](outputs/llm/metrics.json):

- Accuracy: `0.9395`
- Macro Precision: `0.9432`
- Macro Recall: `0.9395`
- Macro F1: `0.9391`
- Evaluated rows: `3076`
- Mean end-to-end per-query latency during the test run: `888.74 ms`
- Mean sampled latency over 10 benchmark tickets: `2197.66 ms`

Interpretation:

- The Gemini system clearly outperformed the classical baseline on held-out predictive quality.
- The classical system remained dramatically faster and simpler operationally.
- The practical tradeoff is accuracy versus latency and maintenance burden.

## Business Case

The repository generates an explicit business estimate in [`outputs/business_case.json`](outputs/business_case.json).

Current estimate for 1,000,000 monthly requests:

- Classical hosting on `c4-standard-8`: about `$288.60/month`
- Gemini orchestration on `c4-standard-2` plus API usage: about `$299.27/month`

Estimated Gemini monthly breakdown:

- Host VM: `$70.71`
- Query embeddings: `$2.23`
- Generation input tokens: `$206.33`
- Generation output tokens: `$20.00`
- One-time initial train-set embedding bootstrap: about `$0.02`

Important note:

- The Gemini estimate is based on the current prompt template in this repository and a token-count approximation derived from the actual Banking77 query lengths.
- The number should be treated as a planning estimate, not a billing invoice.

## Executive Recommendation

Based on the measured results in this project, the recommended production choice is the Gemini retrieval-plus-LLM classifier when accuracy is the primary business KPI.

Why:

- It improves Macro F1 from `0.8234` to `0.9391`, which is a large performance gain.
- The monthly serving-cost estimate remains in the same overall range as the classical deployment.
- The main penalty is latency: Gemini is materially slower than the local classical model.
- The secondary penalty is complexity: caching, API quota handling, and external service dependence all increase maintenance burden.

If the target application values speed, low infrastructure dependency, and simple deployment over maximum predictive quality, the classical model is still a valid production fallback.

## Evaluation Outputs

After execution, the pipeline writes:

- [`outputs/classical/metrics.json`](outputs/classical/metrics.json)
- [`outputs/classical/test_predictions.parquet`](outputs/classical/test_predictions.parquet)
- [`outputs/llm/metrics.json`](outputs/llm/metrics.json)
- [`outputs/llm/test_predictions.parquet`](outputs/llm/test_predictions.parquet)
- [`outputs/run_summary.json`](outputs/run_summary.json)
- [`outputs/business_case.json`](outputs/business_case.json)

## Complexity Audit

The classical model is operationally simpler:

- no external inference API
- low-latency prediction
- compact dependency surface once trained

The Gemini system is more flexible but more complex:

- external quota and service dependency
- retrieval index and cache management
- retry handling and structured prompting

For a business recommendation, the final choice should be based on Macro F1, latency, and monthly serving cost for large request volumes.

## Status

This repository now includes the full implementation, real measured results for both systems, and a generated business-case file. The project is complete locally.
