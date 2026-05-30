# K-12 AI Public School Pipeline

Portfolio version of a Python/GCP pipeline for collecting, classifying, and structuring public evidence about AI use and adoption in U.S. K-12 education.

This repository contains a sanitized implementation of the engineering pipeline. It excludes private datasets, API keys, annotation files, meeting notes, funder reports, and project-specific cloud credentials.

## What This Project Does

The pipeline turns unstructured public articles into a structured research dataset:

1. Collect candidate articles from NewsAPI and education-specific sources.
2. Scrape full article text when source APIs return only snippets.
3. Use a two-stage Gemini classifier:
   - Stage 1: triage articles as `use_case`, `adoption`, or `drop`.
   - Stage 2: extract structured schema fields only for retained articles.
4. Store raw, enriched, and classified records in BigQuery.
5. Export samples for human validation and downstream analysis.

## Core Design

The key optimization is a two-stage LLM workflow:

| Stage | Purpose | Token Strategy |
| --- | --- | --- |
| Triage | Decide whether an article should be retained and assign a relevance score | Lightweight prompt without the location database or extraction schema |
| Extraction | Fill structured fields for retained `use_case` and `adoption` articles | Heavier prompt with schema rules and U.S. location reference data |

This avoids sending expensive extraction context to articles that will be dropped.

## Retained Article Types

`use_case` means the article describes a concrete AI-enabled workflow carried out by an educator, administrator, counselor, psychologist, staff member, or bounded school staff group in a public K-12 setting.

`adoption` means the article describes institutional uptake by a named public K-12 school, district, county education office, public-charter system, or state education agency. Examples include purchases, procurements, contracts, licenses, pilots, rollouts, deployments, MOUs, or partnerships.

`drop` means the article does not provide enough evidence for either retained category. Examples include generic commentary, vendor promotion without a named public-school implementation, student-only use, weak public-school evidence, higher education, private-school cases, or future speculation.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `classifier/` | Flask/Gunicorn Cloud Run service for Gemini triage and extraction |
| `classifier/prompts.py` | Prompt templates and prompt builders |
| `collector-newsapi/` | NewsAPI collector service |
| `collector-v2/` | Alternative-source collector for RSS, education sources, district pages, and search APIs |
| `Dockerfile` | Classifier service container |
| `requirements.txt` | Classifier dependencies |
| `us_location_mapping.json` | Small U.S. location reference used during extraction |
| `.env.example` | Required environment variables without secrets |
| `docs/ARCHITECTURE.md` | Technical architecture notes |
| `docs/SCHEMA.md` | Structured output schema |

## Environment

Copy `.env.example` and fill in your own project and API keys:

```bash
cp .env.example .env
```

Required cloud variables:

```bash
GCP_PROJECT=your-gcp-project-id
GCP_REGION=us-central1
BIGQUERY_DATASET=k12_ai_dataset
BIGQUERY_TABLE=classified_articles
```

Optional collector keys:

```bash
NEWS_API_KEY=your-newsapi-key
SERPAPI_KEY=your-serpapi-key
GNEWS_KEY=your-gnews-key
```

Do not commit `.env`, service-account JSON files, raw data, annotation files, or generated reports.

## Local Development

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the classifier service locally:

```bash
export GCP_PROJECT=your-gcp-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
gunicorn --bind :8080 classifier.main:app
```

Health check:

```bash
curl http://localhost:8080/health
```

## Cloud Run Deployment

The root `Dockerfile` builds the classifier service:

```bash
gcloud run deploy k12-classifier \
  --source . \
  --region us-central1 \
  --project your-gcp-project-id \
  --set-env-vars GCP_PROJECT=your-gcp-project-id,BIGQUERY_DATASET=k12_ai_dataset
```

Collectors are deployed separately from their own directories:

```bash
gcloud run deploy k12-collector-newsapi \
  --source collector-newsapi \
  --region us-central1 \
  --project your-gcp-project-id
```

```bash
gcloud run deploy k12-collector-v2 \
  --source collector-v2 \
  --region us-central1 \
  --project your-gcp-project-id
```

## Notes

This is a portfolio-safe code release. The original research workspace included private analysis outputs, human annotation files, meeting notes, and cloud resources that are intentionally not included here.
