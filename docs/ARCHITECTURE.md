# Architecture

## Pipeline Overview

```text
Article sources
  -> raw_articles
  -> full-text enrichment
  -> Gemini triage
  -> Gemini extraction for retained rows
  -> classified_articles
  -> human validation and analysis
```

## Services

| Service | Path | Responsibility |
| --- | --- | --- |
| Classifier | `classifier/main.py` | Runs triage and extraction with Gemini, writes retained rows to BigQuery |
| NewsAPI collector | `collector-newsapi/main.py` | Collects broad candidate articles from NewsAPI |
| Alternative collector | `collector-v2/main_v2.py` | Collects from RSS, district, education, search, and other public-source streams |

## Two-Stage Classification

Stage 1 triage is intentionally lightweight. It assigns:

- `article_type`
- `relevance_score`
- `reason`

Articles classified as `drop` stop at this stage.

Stage 2 extraction runs only for retained articles. It receives the predetermined article type, longer article text, U.S. location reference data, and detailed schema rules.

## Operational Rule

Rows with `relevance_score < 8` are flagged for manual review. Rows with scores of 8 or higher are treated as high-confidence retained rows, subject to downstream audit or sampling.

## BigQuery Tables

| Table | Purpose |
| --- | --- |
| `raw_articles` | Candidate article metadata and snippets |
| `raw_articles_content` | Full text collected after initial article discovery |
| `classified_articles` | Retained classified rows with extracted fields |
| `classified_articles_latest` | Deduplicated view for analysis |
| `discovered_sources` | Non-primary sources that produced at least one retained article |
