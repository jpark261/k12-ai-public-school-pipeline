# Structured Output Schema

The classifier returns a merged record containing triage fields and extraction fields.

## Triage Fields

| Field | Type | Description |
| --- | --- | --- |
| `article_type` | string | `use_case`, `adoption`, or `drop` |
| `relevance_score` | integer | 0-10 confidence/relevance score |
| `reason` | string | One-sentence rationale for the classification |
| `manual_review` | boolean | `true` when `relevance_score < 8` |

## Common Extraction Fields

| Field | Type | Description |
| --- | --- | --- |
| `source` | string | Article source |
| `published_date` | string | Publication date when available |
| `state` | string | U.S. state |
| `county` | string | County when available |
| `school` | string | School, district, or agency name |
| `level_of_school` | string | Elementary, middle, high, district, state, or other level |
| `AI_product` | string | Named AI product or platform |
| `AI_type` | string | Type of AI system |
| `application_type` | string | Application area |
| `adoption_use_date` | string | Date of adoption, rollout, or use when available |

## Use-Case Fields

| Field | Type | Description |
| --- | --- | --- |
| `subject` | string | Subject area |
| `user_type` | string | Teacher, counselor, administrator, staff, or other user role |
| `purpose_of_AI` | string | Purpose of AI use |
| `use_case_type` | string | Normalized use-case category |
| `use_case_description` | string | Short description of the AI-enabled workflow |
| `outcome` | string | Reported outcome when available |
| `impact` | string | Reported impact dimension |

## Notes

| Field | Type | Description |
| --- | --- | --- |
| `notes_code` | string | Comma-separated note codes for ambiguity or extraction issues |
| `notes` | string | Free-text notes, including adoption details that do not fit use-case fields |
