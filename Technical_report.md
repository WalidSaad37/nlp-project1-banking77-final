Technical Report: Comparative NLP Architectures for Banking77 Intent Classification

# 1. Executive Summary


This project compares two production-oriented intent classification architectures on the Banking77 benchmark:

1. A classical machine learning pipeline using static word embeddings and a linear SVM classifier.
2. A retrieval-augmented LLM pipeline using Gemini 2.5 Flash vectorization and Gemini 2.5 Flash generation.


The evaluation objective is not only predictive performance, but also operational characteristics: latency, implementation complexity, and monthly serving cost.

What came out clearly from the experiments:

- The LLM pipeline is noticeably stronger on prediction quality.
- The classical pipeline is still dramatically faster per query and easier to operate.
- Monthly cost is close enough that the final decision is mostly about product priorities: quality vs response time.

This README is intentionally written as a 3-4 page technical report covering methodology, hyperparameters, benchmark results, and the final business recommendation requested in the assignment.

# 2. Problem Definition and Evaluation Scope

2.1 Task


Given a user banking support query, predict exactly one label from the Banking77 taxonomy (77 intents).

2.2 Dataset


- Dataset source: mteb/banking77
- Number of intent classes: 77
- Splits: official train and test splits from the dataset

The project stores:

- Raw snapshots in [data/raw](data/raw)
- Processed train/test artifacts in [data/processed](data/processed)

2.3 Evaluation Metrics


The report focuses on:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- End-to-end inference latency
- Throughput-adjacent runtime observations
- Serving cost estimate for 1,000,000 monthly requests

Macro F1 is emphasized because all classes should be treated fairly in multiclass intent classification.

# 3. Methodology

3.1 Shared Data Flow


Both pipelines share common preparation steps:

1. Load Banking77 train/test splits.
2. Apply configurable text preprocessing.
3. Train/evaluate each architecture on the same train/test split.
4. Save comparable outputs for metrics, predictions, and summary files.


Execution entry point: [scripts/run_project.py](scripts/run_project.py)

3.2 Classical Pipeline


Implementation: [src/banking77_project/classical.py](src/banking77_project/classical.py)

Procedure:

1. Tokenize each query using project preprocessing rules.
2. Convert each query to a dense vector by averaging static word vectors for known tokens.
3. Train a scikit-learn Pipeline with StandardScaler and LinearSVC.
4. Predict on test vectors.
5. Compute metrics and benchmark per-query latency.


Design rationale:

- Averaged static embeddings are reliable, cheap, and predictable at inference time.
- LinearSVC is a practical baseline for multiclass intent detection and is straightforward to deploy.

3.3 LLM Retrieval-Augmented Pipeline


Implementation: [src/banking77_project/llm.py](src/banking77_project/llm.py)

Procedure:

1. Vectorize train queries with gemini-2.5-flash into a deterministic 77-dimensional intent-affinity representation.
2. Cache those vectors in local parquet to avoid repeated API calls.
3. Vectorize each test query in the same label space.
4. Retrieve top-k nearest labeled examples via cosine similarity.
5. Build constrained prompt with approved label list and retrieved examples.
6. Generate a single label with Gemini model at deterministic temperature.
7. Parse/validate the label and fallback safely when needed.


Operational safeguards included in code:

- Retry with exponential backoff for transient API failures (429/5xx).
- Label normalization and strict validation against the approved taxonomy.
- Local key loading fallback via environment variables or local settings.

3.4 Assignment Alignment Note


The final implementation follows the assignment wording literally:

- Vectorization: gemini-2.5-flash
- Generation: gemini-2.5-flash

To support retrieval despite the lack of a native flash embedding endpoint, the project prompts gemini-2.5-flash to emit a deterministic Banking77 intent-affinity vector in the fixed label order, then uses cosine similarity over those vectors for retrieval.

# 4. Hyperparameters and Configuration Choices


Primary configuration file: [configs/project.yaml](configs/project.yaml)

4.1 Global and Preprocessing


- random_state: 42
- lowercase: true
- expand_contractions: true
- remove_punctuation: true
- remove_digits: false
- remove_stopwords: false
- min_token_length: 2

Why these defaults:

- Lowercasing and contraction expansion improve lexical consistency.
- Keeping digits and stopwords preserves semantic cues (for ticket references, amounts, and intent-specific phrasing).
- Minimal token-length filtering removes obvious noise while retaining short meaningful tokens.

4.2 Classical Model Hyperparameters


- embedding_model: word2vec-google-news-300
- embedding_source: gensim
- embedding_local_path: null (supports custom local Word2Vec-compatible file)
- classifier_name: svm
- svm_c: 2.0
- benchmark_sample_size: 10

Why these defaults:

- word2vec-google-news-300 directly satisfies the assignment requirement for a pretrained Word2Vec model.
- LinearSVC with C=2.0 increases margin flexibility over stricter regularization while keeping stable behavior.

4.3 LLM Pipeline Hyperparameters


- generation_model: gemini-2.5-flash
- vectorization_model: gemini-2.5-flash
- retrieval_top_k: 5
- temperature: 0.0
- max_output_tokens: 128
- benchmark_sample_size: 10
- max_train_examples_for_retrieval: 5000
- max_test_examples: null
- vectorization_workers: 4

Why these defaults:

- top_k=5 provides enough context diversity without overloading prompt length.
- temperature=0.0 enforces determinism and reduces label drift.
- max_output_tokens=128 is conservative but safe for strict one-label outputs.
- train retrieval cap at 5000 preserves broad intent coverage, though it makes the fully literal gemini-2.5-flash vectorization run materially slower and more expensive.

# 5. Reproducibility and Run Protocol


Minimal run sequence:

```bash
python -m pip install -r requirements.txt
python scripts/run_project.py prepare-data
python scripts/run_project.py run-classical
python scripts/run_project.py run-llm
python scripts/run_project.py business-case
```

Or full run in one command:

```bash
python scripts/run_project.py run-all
```

Required key for Gemini stage:

```bash
set GEMINI_API_KEY=your_key_here
```

Output artifacts:

- [outputs/classical/metrics.json](outputs/classical/metrics.json)
- [outputs/classical/test_predictions.parquet](outputs/classical/test_predictions.parquet)
- [outputs/llm/metrics.json](outputs/llm/metrics.json)
- [outputs/llm/test_predictions.parquet](outputs/llm/test_predictions.parquet)
- [outputs/run_summary.json](outputs/run_summary.json)
- [outputs/business_case.json](outputs/business_case.json)

# 6. Benchmark Results

6.1 Quality Metrics


Source files:

- [outputs/classical/metrics.json](outputs/classical/metrics.json)
- [outputs/llm/metrics.json](outputs/llm/metrics.json)

| Metric | Classical | LLM | Absolute Gain (LLM - Classical) |
|---|---:|---:|---:|
| Accuracy | 0.8241 | 0.9431 | +0.1190 |
| Macro Precision | 0.8269 | 0.9465 | +0.1196 |
| Macro Recall | 0.8240 | 0.9430 | +0.1190 |
| Macro F1 | 0.8234 | 0.9428 | +0.1194 |

Interpretation:

- The LLM system outperforms the classical baseline on every headline metric.
- The gain is not marginal. On a 77-label task, a +0.119 macro F1 jump is large enough to be product-visible.

6.2 Runtime and Latency


Classical (from metrics + benchmark):

- Training time: about 8.67 s
- Test inference time (full set): about 0.006 s
- Benchmark mean latency: about 0.371 ms
- Benchmark median latency: about 0.316 ms

LLM (from metrics + benchmark):

- Evaluated rows: 3076
- Mean end-to-end latency during test run: about 835.70 ms
- Median end-to-end latency during test run: about 795.13 ms
- Benchmark mean latency: about 1021.90 ms
- Benchmark median latency: about 993.87 ms

Interpretation:

- Classical inference is orders of magnitude faster and very easy to scale.
- LLM latency is around the one-second range here, which is acceptable for some support flows, but not for every real-time interaction.

6.3 Brief Error Analysis Notes


Looking at class-level scores in the reports, both systems are stronger on clearly distinctive intents and weaker on semantically close intents (for example, intents that differ mainly by payment state or transfer context). The LLM model reduces those confusions overall, but difficult neighboring classes still exist.

This is expected on Banking77: many labels are intentionally fine-grained, and short customer queries often omit key context.

6.4 Tradeoff Summary


- Quality winner: LLM
- Latency winner: Classical
- Simplicity winner: Classical

In other words, there is no absolute winner; the "best" system depends on the business constraint you optimize first.

# 7. Business Case Analysis


Business case source: [outputs/business_case.json](outputs/business_case.json)

Scenario assumptions include 1,000,000 monthly requests with explicit host and token pricing parameters.

7.1 Monthly Cost Comparison


- Classical monthly cost: 288.5982 USD
- LLM monthly cost: 299.2691 USD
- Monthly delta (LLM - Classical): 10.6709 USD

Relative increase:

- Approximately 3.7 percent higher monthly cost for LLM under current assumptions.

7.2 LLM Cost Breakdown


- Host: 70.7122 USD
- Query embedding API: 2.2313 USD
- Generation input API: 206.3256 USD
- Generation output API: 20.0000 USD
- One-time train embedding bootstrap: about 0.0223 USD

Key observation:

- Prompt input tokens dominate LLM operating cost. The fastest way to reduce spend is to optimize prompt size and retrieval payload.

7.3 Practical Cost Caveat


These numbers are planning estimates, not invoices. Real monthly spend will move with traffic mix, average query length, and prompt design changes. Still, they are precise enough to compare both options under the same assumptions.

# 8. Final Business Conclusion

8.1 Recommendation


If the primary KPI is intent classification quality, the LLM retrieval-augmented pipeline is the recommended choice.

Why this recommendation is reasonable:

- Macro F1 increases from 0.8234 to 0.9428, which is a substantial improvement.
- The monthly cost increase is modest in the modeled scenario (about 10.67 USD/month).

8.2 Conditions and Risks


That recommendation comes with clear conditions:

- Latency budget and SLA management (as latency is far higher than classical).
- API dependency and quota resilience.
- Prompt and retrieval governance to control cost and drift.

8.3 Deployment Strategy Option


A practical compromise is hybrid routing:

1. Use LLM as default predictor for quality-critical flows.
2. Route to classical fallback on timeout, service degradation, or strict low-latency endpoints.


This preserves most of the quality gains while keeping worst-case response time under control.

8.4 Final Decision Statement


For this specific project setup, we would deploy the LLM-first architecture for customer support intent routing where classification quality matters most, and keep the classical model as an operational fallback path. This recommendation is based on measured gains, not on architectural preference.

# 9. Project Structure


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
        |-- business_case.py
        |-- classical.py
        |-- config.py
        |-- data.py
        |-- io_utils.py
        |-- llm.py
        |-- metrics.py
        `-- preprocessing.py
```

# 10. Closing Note


This repository now contains a complete and reproducible comparison between a classical and an LLM-based intent classification architecture, with metrics, latency benchmarks, and a quantified business case.

Beyond the raw numbers, the main lesson is practical: better quality is achievable without a major cost jump, but it requires accepting higher latency and a more complex operating model.



