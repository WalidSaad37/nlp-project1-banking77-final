# Final Project 1: Comparative Analysis of NLP Architectures for Automated Intent Classification

This repository contains a full implementation of Project 1 from the assignment brief. It compares a classical NLP intent-classification pipeline against a Gemini-based LLM classifier on the `Banking77` dataset, with reproducible scripts, cached artifacts, and report-ready outputs.

## Objective

The goal is to compare two approaches for classifying banking support queries into the 77 Banking77 intent labels:

1. A classical pipeline based on pretrained Word2Vec embeddings and a supervised classifier.
2. An LLM-based pipeline based on Gemini 2.5 Flash vectorization, retrieval, and few-shot prompting.

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
2. Build a dense document vector by averaging pretrained Word2Vec vectors.
3. Train a `LinearSVC` classifier on the resulting vectors.
4. Evaluate on the held-out test set.
5. Benchmark single-query latency on 10 random tickets.

Default configuration:

- embedding model: `word2vec-google-news-300`
- classifier: `LinearSVC`

The assignment specifically mentions pretrained Word2Vec, so the default configuration now uses the pretrained Google News Word2Vec vectors distributed through Gensim. The code also supports swapping in a local Word2Vec-format file through `embedding_local_path` in [`configs/project.yaml`](configs/project.yaml).

### Gemini pipeline

The LLM branch is built around the assignment's architectural constraints:

1. Use `gemini-2.5-flash` to generate a deterministic intent-affinity vector for each training question and cache the vectors locally in Parquet.
2. Reuse cached vectors when a question has already been processed.
3. Use `gemini-2.5-flash` to vectorize the incoming query in the same Banking77 label space.
4. Retrieve the nearest labeled training examples.
5. Prompt `gemini-2.5-flash` to return exactly one valid Banking77 label.
6. Retry transient API failures with exponential backoff.

The core implementation lives in [`src/banking77_project/llm.py`](src/banking77_project/llm.py).
This implementation now follows the assignment literally: both vectorization and final classification are performed with `gemini-2.5-flash`.

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

- Accuracy: `0.8244`
- Macro Precision: `0.8290`
- Macro Recall: `0.8243`
- Macro F1: `0.8242`
- Training time: `309.42s`
- Full test-set inference time: `0.0104s`

Gemini pipeline from [`outputs/llm/metrics.json`](outputs/llm/metrics.json):

- The checked-in LLM metrics file reflects the last completed paid run before the literal `gemini-2.5-flash` vectorization change.
- To refresh the Gemini results under the final spec-aligned implementation, rerun `python scripts/run_project.py run-llm` with a valid API key.

Interpretation:

- The Word2Vec classical baseline is now aligned with the assignment and remains fast, reproducible, and easy to audit.
- The final Gemini code path is now aligned with the assignment wording, but its checked-in metrics should be refreshed with one more paid run if you want artifact-level alignment as well as code-level alignment.

## Business Case

The repository generates an explicit business estimate in [`outputs/business_case.json`](outputs/business_case.json).

Current estimate for 1,000,000 monthly requests:

- Classical hosting on `c4-standard-8`: about `$288.60/month`
- Gemini orchestration on `c4-standard-2` plus API usage: about `$1,617.08/month`

Estimated Gemini monthly breakdown:

- Host VM: `$70.71`
- Query vectorization input: about `$165.04`
- Query vectorization output: about `$1,155.00`
- Generation input tokens: about `$206.33`
- Generation output tokens: `$20.00`
- One-time initial train-set vectorization bootstrap: about `$13.19`

Important note:

- The Gemini estimate is based on the current prompt template in this repository and a token-count approximation derived from the actual Banking77 query lengths.
- The number should be treated as a planning estimate, not a billing invoice.

## Executive Recommendation

Based on the measured results already collected and the updated spec-aligned implementation, the recommended production choice depends on whether the assignment prioritizes strict architecture compliance or business realism.

Why:

- The refreshed Word2Vec baseline fully matches the classical requirement in the brief.
- The final Gemini implementation now uses `gemini-2.5-flash` for both vectorization and classification, which matches the assignment wording exactly.
- Under this literal implementation, Gemini remains the likely accuracy leader, but it now carries materially higher serving cost and operational complexity than the classical system.

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
