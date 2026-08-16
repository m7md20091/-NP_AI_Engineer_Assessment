# NP Employee Assistant — Design Report

## Executive summary

This prototype implements a retrieval-augmented employee-information assistant over the
provided Excel workbook. It focuses on reproducibility, grounded answers, and a small
operational footprint. The same application runs without credentials in extractive mode,
with a cloud OpenAI-compatible model, or with a local Ollama model.

## System design and storage

The ingestion pipeline validates the workbook schema, removes duplicate employee IDs,
normalizes whitespace and missing values, converts numeric fields, and converts hire dates
to ISO `YYYY-MM-DD`. Each row becomes a self-contained document and is written to JSONL.

The retrieval layer uses TF-IDF word and bigram embeddings. It stores the fitted vectorizer,
sparse document matrix, original document text, and source metadata in a local Joblib index.
This choice is transparent, fast for 10,018 records, works offline, and needs no managed
infrastructure. A production system could replace it with dense embeddings and pgvector
without changing the API contract.

Questions enter through FastAPI. The service embeds the question with the same vectorizer,
calculates cosine similarity, selects top-k documents, and removes results below a relevance
threshold. Retrieved text is passed to the configured generator. The prompt tells an LLM to
use only that context and admit when evidence is insufficient. Responses include sources and
scores so that users can inspect the evidence.

## Data processing decisions

One employee row is one chunk because each row is already a coherent unit and splitting it
would separate related fields. Duplicate employee IDs are removed. Required columns are
checked before processing; invalid numeric records are excluded; dates are parsed whether
Excel supplies serial values or datetime objects. Generated JSONL makes preprocessing easy
to inspect and decouples ingestion from indexing.

## AI/ML implementation and evaluation

TF-IDF produces sparse local embeddings, and cosine similarity provides deterministic
retrieval. It is a sensible baseline for names, departments, job titles, IDs, and exact
business terms. The included evaluation script creates labeled queries from department and
job-title fields and reports Precision@5 and Mean Reciprocal Rank. API and unit tests cover
document construction, relevant retrieval, successful questions, health status, and invalid
input.

The default response composer quotes retrieved facts and therefore cannot hallucinate new
facts. OpenAI-compatible and Ollama adapters provide full generative RAG when configured.
Temperature is zero and the system prompt prohibits unsupported answers.

## Limitations

- TF-IDF is lexical rather than a dense semantic embedding, so paraphrases and synonyms may
  retrieve less effectively.
- Top-k row retrieval is not an analytics engine. Questions requiring exact counts, sums, or
  organization-wide averages should be routed to validated SQL/pandas aggregation tools.
- Spreadsheet refreshes currently rebuild the complete index at startup.
- Prompt constraints reduce but cannot eliminate LLM hallucinations.
- The prototype has no authentication, authorization, audit log, or field-level access rules.
- Evaluation uses synthetic labels derived from the source data rather than human judgments.

## Improving accuracy and reducing hallucinations

Replace TF-IDF with a domain-tested dense embedding model and add hybrid lexical/vector
search with a cross-encoder reranker. Create a curated question-and-answer evaluation set.
Add metadata filters for department, status, and role. Route analytical questions to a safe
structured-query component. Keep citations mandatory, reject answers without sufficient
retrieval confidence, and evaluate faithfulness with both deterministic checks and periodic
human review. Sensitive salary fields should be filtered using role-based access control.

## Production deployment

Build the supplied Docker image in CI, scan dependencies, and deploy behind an authenticated
API gateway to a managed container platform. Store normalized employee data in PostgreSQL and
vectors in pgvector or a managed vector database. Run ingestion as a separate scheduled job,
version indexes, and switch indexes atomically after validation. Keep model credentials in a
secret manager. Add TLS, request limits, structured logs, tracing, and least-privilege access.

## Monitoring over time

Track request volume, latency percentiles, failures, model cost, empty-result rate, retrieval
scores, selected sources, and user feedback. Run the evaluation set on every data, prompt,
embedding, or model change. Monitor source freshness, schema drift, missing values, duplicates,
and distribution changes. Alert on declining retrieval metrics, unusual token use, elevated
provider errors, stale indexes, and attempts to access restricted information.

