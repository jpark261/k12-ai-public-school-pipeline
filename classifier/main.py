"""
K-12 AI Use Cases Classifier
Reads from raw_articles (BigQuery) and classifies with Gemini Flash.
Collection is handled separately by k12-collector (NewsAPI) and k12-collector-v2 (alternative sources).

Endpoints:
  POST/GET /classify-raw    — classify unclassified articles from raw_articles
  POST     /classify-sample — classify specific article_ids (for manual review)
  GET      /cleanup         — deduplicate + drop low-score rows
  GET      /stats           — summary statistics
  GET      /health          — health check
"""

import os
import json
import logging
import time
import re
import hashlib

from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from flask import Flask, request, jsonify
from google.cloud import bigquery
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from classifier.prompts import build_triage_prompt, build_extraction_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize Vertex AI
PROJECT_ID = os.getenv('GCP_PROJECT', 'your-gcp-project-id')
LOCATION = os.getenv('GCP_REGION', 'us-central1')
vertexai.init(project=PROJECT_ID, location=LOCATION)

BIGQUERY_DATASET = os.getenv('BIGQUERY_DATASET', 'k12_ai_dataset')
BIGQUERY_TABLE = os.getenv('BIGQUERY_TABLE', 'classified_articles')
TRACKING_TABLE = "discovered_sources"
LOCAL_TRIAGE_AUDIT_PATH = os.getenv('LOCAL_TRIAGE_AUDIT_PATH', '/tmp/k12_triage_audit.jsonl')
ENABLE_LOCAL_TRIAGE_AUDIT = os.getenv('ENABLE_LOCAL_TRIAGE_AUDIT', 'true').strip().lower() not in (
    '0', 'false', 'no', 'off'
)


def _load_manual_review_threshold() -> int:
    raw = os.getenv('MANUAL_REVIEW_THRESHOLD', '8')
    try:
        value = int(raw)
    except ValueError:
        logger.warning(f"Invalid MANUAL_REVIEW_THRESHOLD={raw!r}; falling back to 8")
        return 8
    return max(0, min(10, value))


MANUAL_REVIEW_THRESHOLD = _load_manual_review_threshold()
TRIAGE_PROMPT_CHAR_LIMIT = int(os.getenv('TRIAGE_PROMPT_CHAR_LIMIT', '8000'))
EXTRACTION_PROMPT_CHAR_LIMIT = int(os.getenv('EXTRACTION_PROMPT_CHAR_LIMIT', '8000'))

# Load US location mapping once
with open('us_location_mapping.json', 'r') as f:
    US_LOCATION_DATA = json.load(f)
US_LOCATION_DATA_JSON = json.dumps(US_LOCATION_DATA, indent=2)

EXTRACTION_SCHEMA_FIELDS = (
    'source',
    'published_date',
    'state',
    'county',
    'school',
    'level_of_school',
    'AI_product',
    'AI_type',
    'application_type',
    'unit_of_AI_use',
    'adoption_use_date',
    'notes_code',
    'notes',
    'subject',
    'user_type',
    'purpose_of_AI',
    'use_case_type',
    'use_case_description',
    'outcome',
    'impact',
)

MULTI_VALUE_FIELDS = (
    'notes_code',
    'impact',
    'use_case_type',
    'user_type',
    'AI_type',
)

TRIAGE_FLAG_FIELDS = (
    'has_named_public_k12_institution',
    'has_named_educator_or_staff_group',
    'has_current_work_task',
    'has_named_ai_tool',
    'has_explicit_deal_language',
    'has_single_focal_use_case',
)


def make_article_id(url: str) -> str:
    return hashlib.sha1((url or '').strip().encode('utf-8')).hexdigest()


def _bq_url_hash_expr(column_sql: str) -> str:
    return f"LOWER(TO_HEX(SHA1(CAST(TRIM({column_sql}) AS BYTES))))"


class ClassificationPipelineError(Exception):
    def __init__(self, message: str, audit_row: Dict[str, Any]):
        super().__init__(message)
        self.audit_row = audit_row


def _source_name_from_article(article: Dict[str, Any]) -> str:
    source = article.get('source', {})
    if isinstance(source, dict):
        return str(source.get('name', '') or '').strip()
    return str(source or '').strip()


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def _build_triage_audit_row(article: Dict[str, Any], *, status: str,
                            article_type: str = "", relevance_score: Optional[int] = None,
                            manual_review: Optional[bool] = None, reason: str = "",
                            error_message: str = "", classified_at: Optional[str] = None,
                            triage_flags: Optional[Dict[str, Optional[bool]]] = None) -> Dict[str, Any]:
    url = str(article.get('url', '') or '').strip()
    article_id = make_article_id(url)
    classified_at = classified_at or _utc_now_iso()
    triage_flags = triage_flags or {field: None for field in TRIAGE_FLAG_FIELDS}
    payload = "|".join([
        article_id,
        classified_at,
        status,
        str(article_type or ""),
        "" if relevance_score is None else str(relevance_score),
        reason or "",
        error_message or "",
    ])
    audit_id = hashlib.sha1(payload.encode('utf-8')).hexdigest()
    content = article.get('content') or ""

    return {
        "audit_id": audit_id,
        "article_id": article_id,
        "classified_at": classified_at,
        "status": status,
        "article_type": article_type or "",
        "relevance_score": relevance_score,
        "manual_review": manual_review,
        "reason": reason or "",
        "error_message": error_message or "",
        "title": str(article.get('title', '') or ''),
        "source": _source_name_from_article(article),
        "published_date": str(article.get('publishedAt', '') or ''),
        "url": url,
        "collection_source": str(article.get('collection_source', '') or ''),
        "description": str(article.get('description', '') or ''),
        "content_char_count": len(content),
        **{field: triage_flags.get(field) for field in TRIAGE_FLAG_FIELDS},
    }


def write_local_triage_audit_rows(rows: List[Dict[str, Any]]) -> int:
    """Append triage audit rows to a local JSONL file for analysis only."""
    if not ENABLE_LOCAL_TRIAGE_AUDIT or not rows:
        return 0

    try:
        parent = os.path.dirname(LOCAL_TRIAGE_AUDIT_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(LOCAL_TRIAGE_AUDIT_PATH, 'a', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        return len(rows)
    except Exception as e:
        logger.warning(f"Could not write local triage audit rows: {e}")
        return 0


def _summarize_audit_rows(rows: List[Dict[str, Any]], *, statuses: tuple, limit: int = 10) -> List[Dict[str, Any]]:
    summaries = []
    for row in rows:
        if row.get('status') not in statuses:
            continue
        summaries.append({
            "article_id": row.get("article_id"),
            "title": row.get("title"),
            "status": row.get("status"),
            "article_type": row.get("article_type"),
            "relevance_score": row.get("relevance_score"),
            "reason": row.get("reason"),
            "error_message": row.get("error_message"),
            "gate_flags": {
                field: row.get(field) for field in TRIAGE_FLAG_FIELDS
            },
        })
        if len(summaries) >= limit:
            break
    return summaries


def parse_gemini_json(response_text: str) -> dict:
    """Robust JSON extraction from Gemini responses."""
    text = response_text.strip()

    if text.startswith('```json'):
        text = text[7:]
    if text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Direct JSON parse failed: {e}")

    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    for match in re.finditer(r'\{.*?\}', text, re.DOTALL):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except:
            continue

    raise ValueError(f"Cannot extract valid JSON from response. Text: {text[:300]}")


def _is_429_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("429" in msg) or ("resource exhausted" in msg) or ("quota" in msg)


def gemini_generate_with_retry(model: GenerativeModel, prompt: str, generation_config: dict,
                               max_retries: int = 6, base_sleep: float = 1.0) -> str:
    """Retry Gemini calls on 429 / quota exhaustion with exponential backoff."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = model.generate_content(prompt, generation_config=generation_config)
            return (resp.text or "").strip()
        except Exception as e:
            last_err = e
            if _is_429_error(e) and attempt < max_retries:
                sleep_s = min(base_sleep * (2 ** (attempt - 1)), 60)
                logger.warning(f"Gemini 429/Resource exhausted. Retry in {sleep_s:.0f}s (attempt {attempt}/{max_retries})")
                time.sleep(sleep_s)
                continue
            raise
    raise last_err


def get_classified_article_ids() -> set:
    """Load already-classified article_ids from BigQuery to prevent re-classification."""
    try:
        client = bigquery.Client(project=PROJECT_ID)
        table_id = f"{PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"
        query = f"SELECT DISTINCT article_id FROM `{table_id}`"
        result = client.query(query).result()
        ids = {row.article_id for row in result}
        logger.info(f"Loaded {len(ids)} existing classified article_ids")
        return ids
    except Exception as e:
        logger.warning(f"Could not load classified article_ids: {e}")
        return set()



def _normalize_article_type(article_type: Any) -> str:
    normalized = str(article_type or 'drop').strip().lower()
    if normalized in ('adoption_opinion', 'other'):
        return 'drop'
    return normalized


def _coerce_relevance_score(score: Any) -> int:
    try:
        return max(0, min(10, int(score)))
    except (TypeError, ValueError):
        return 0


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in ('true', '1', 'yes', 'y'):
        return True
    if normalized in ('false', '0', 'no', 'n', '', 'null', 'none'):
        return False
    return None


def _extract_triage_flags(result: Dict[str, Any]) -> Dict[str, Optional[bool]]:
    return {
        field: _coerce_optional_bool(result.get(field))
        for field in TRIAGE_FLAG_FIELDS
    }


def _combined_article_text(article: Dict[str, Any]) -> str:
    return " ".join([
        str(article.get('title', '') or ''),
        str(article.get('description', '') or ''),
        str(article.get('content', '') or ''),
    ]).lower()


def _is_nonpublic_micro_school_case(article: Dict[str, Any],
                                    triage_flags: Dict[str, Optional[bool]]) -> bool:
    """Detect obvious nonpublic or founder-run school cases that should not survive as use_case."""
    if triage_flags.get('has_named_public_k12_institution') is not False:
        return False
    if triage_flags.get('has_single_focal_use_case') is not True:
        return False

    combined = _combined_article_text(article)

    has_micro_school = ('micro school' in combined) or ('microschool' in combined)
    has_private_school = 'private school' in combined
    has_independent_school = (
        'independent school' in combined and 'independent school district' not in combined
    )
    has_founder_run_school = any(token in combined for token in (
        'founder-run school',
        'founder of mysa',
        'founder of mysa schools',
        'founder of a school',
        'founder of the school',
    ))
    mentions_public_charter = ('public charter' in combined) or ('charter school' in combined)
    return (
        has_micro_school
        or has_private_school
        or has_independent_school
        or has_founder_run_school
    ) and not mentions_public_charter


def _is_vague_student_benefit_case(article: Dict[str, Any],
                                   triage_flags: Dict[str, Optional[bool]]) -> bool:
    """Drop vague student-benefit articles with no concrete teacher-side AI action."""
    if triage_flags.get('has_named_public_k12_institution') is not True:
        return False
    if triage_flags.get('has_named_educator_or_staff_group') is not True:
        return False
    if triage_flags.get('has_current_work_task') is not True:
        return False
    if triage_flags.get('has_named_ai_tool') is True:
        return False

    combined = _combined_article_text(article)
    has_student_benefit_language = any(token in combined for token in (
        'help students',
        'helps students',
        'students who struggle',
        'struggle with vocabulary',
        'vocabulary or concepts',
        'keep students engaged',
        'engaged in learning',
        'support for students',
        'students say they like',
        'interactive teaching style',
    ))
    has_concrete_teacher_action = any(token in combined for token in (
        'lesson plan',
        'lesson plans',
        'worksheet',
        'worksheets',
        'rubric',
        'rubrics',
        'quiz',
        'quizzes',
        'feedback',
        'grade papers',
        'grading',
        'study guide',
        'study guides',
        'unit plan',
        'unit plans',
        'reading passage',
        'reading passages',
        'email parents',
        'report writing',
        'generate recommendations',
        'iep',
        'treatment plan',
        'counseling materials',
        'advising materials',
    ))
    return has_student_benefit_language and not has_concrete_teacher_action


def _is_noninstitutional_library_demo_case(article: Dict[str, Any],
                                           triage_flags: Dict[str, Optional[bool]]) -> bool:
    """Drop noninstitutional library demo/blog cases that lack a public-school anchor."""
    if triage_flags.get('has_named_public_k12_institution') is not False:
        return False
    if triage_flags.get('has_named_educator_or_staff_group') is not True:
        return False
    if triage_flags.get('has_current_work_task') is not True:
        return False

    combined = _combined_article_text(article)
    title = str(article.get('title', '') or '').lower()
    has_library_context = ('library' in combined) or ('librarian' in combined)
    has_demo_context = any(token in combined for token in (
        '#tlchat',
        '#schoolai',
        'conference',
        'shared these tools',
        'custom spaces',
        'what book should i read next',
    ))
    return has_library_context and (has_demo_context or '#' in title)


def _is_student_use_primary_case(article: Dict[str, Any],
                                 triage_flags: Dict[str, Optional[bool]]) -> bool:
    """Cap first-person student-use essays when educator workflow is secondary."""
    if triage_flags.get('has_named_public_k12_institution') is not True:
        return False
    if triage_flags.get('has_named_educator_or_staff_group') is not True:
        return False
    if triage_flags.get('has_current_work_task') is not True:
        return False
    if triage_flags.get('has_named_ai_tool') is not True:
        return False

    title = str(article.get('title', '') or '').lower()
    combined = _combined_article_text(article)
    has_student_use_title = any(token in title for token in (
        'let my students use',
        'students use it to write essays',
        'students use ai',
        'student use ai',
    ))
    has_student_essay_frame = any(token in combined for token in (
        'students use it to write essays',
        'students write essays',
        'let my students use',
        'student essays',
    ))
    return has_student_use_title or (
        "i've been a teacher" in title and has_student_essay_frame
    )


def _is_staff_alert_workflow_case(article: Dict[str, Any],
                                  triage_flags: Dict[str, Optional[bool]]) -> bool:
    """Detect student-support AI alert workflows that imply a concrete staff use_case."""
    if triage_flags.get('has_named_public_k12_institution') is not True:
        return False
    if triage_flags.get('has_named_educator_or_staff_group') is not True:
        return False
    if triage_flags.get('has_current_work_task') is True:
        return False

    combined = _combined_article_text(article)

    has_student_support_context = any(
        token in combined for token in (
            'mental health', 'self-harm', 'suicid', 'therapy', 'well-being', 'wellbeing'
        )
    )
    has_alert_language = 'alert' in combined
    has_staff_role = any(token in combined for token in ('counselor', 'principal', 'social worker'))
    has_ai_tool_context = any(token in combined for token in ('chatbot', 'app', 'platform'))
    has_student_context = 'student' in combined

    return all((
        has_student_support_context,
        has_alert_language,
        has_staff_role,
        has_ai_tool_context,
        has_student_context,
    ))


def _apply_triage_score_caps(article_type: str, relevance_score: int, reason: str,
                             triage_flags: Dict[str, Optional[bool]]) -> tuple[int, str]:
    """Apply deterministic score caps after parsing triage JSON."""
    updated_score = relevance_score
    updated_reason = reason

    if article_type == 'use_case':
        if triage_flags.get('has_named_public_k12_institution') is False and updated_score > 7:
            updated_score = 7
            if updated_reason:
                updated_reason += " Capped at 7 because the article does not name a public K-12 institution."
        if triage_flags.get('has_single_focal_use_case') is False and updated_score > 7:
            updated_score = 7
            if updated_reason:
                updated_reason += " Capped at 7 because no single focal educator use case clearly dominates the article."

    return updated_score, updated_reason


def _normalize_multi_value_fields(row: Dict[str, Any]) -> None:
    for field in MULTI_VALUE_FIELDS:
        value = row.get(field)
        if isinstance(value, (list, tuple, set)):
            items = [str(item).strip() for item in value if item not in (None, "")]
            row[field] = ', '.join(items) if items else None


def _ensure_extraction_schema_fields(row: Dict[str, Any]) -> None:
    for field in EXTRACTION_SCHEMA_FIELDS:
        row.setdefault(field, None)


def sync_discovered_sources(rows: List[Dict]) -> int:
    """Track validated non-NewsAPI sources only after articles survive classification."""
    if not rows:
        return 0

    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{BIGQUERY_DATASET}.{TRACKING_TABLE}"

    try:
        existing_q = f"""
            SELECT LOWER(source_name) AS src
            FROM `{table_id}`
            WHERE source_name IS NOT NULL
        """
        existing_sources = {
            r.src for r in client.query(existing_q).result()
            if r.src
        }
    except Exception as e:
        logger.warning(f"Could not load existing discovered_sources: {e}")
        existing_sources = set()

    new_source_rows = []
    seen_sources = set()
    discovered_at = datetime.utcnow().isoformat()

    for row in rows:
        try:
            raw_article = json.loads(row.get('raw_article') or '{}')
        except json.JSONDecodeError:
            continue

        collection_source = str(raw_article.get('collection_source') or '').strip()
        if not collection_source or collection_source == 'NewsAPI':
            continue

        source = raw_article.get('source') or {}
        if isinstance(source, dict):
            source_name = (source.get('name') or '').strip()
        else:
            source_name = str(source or '').strip()

        if not source_name:
            continue

        normalized_source = source_name.lower()
        if normalized_source in existing_sources or normalized_source in seen_sources:
            continue

        url = str(raw_article.get('url') or '').strip()
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else ""

        new_source_rows.append({
            "source_name": source_name,
            "source_url": base_url,
            "example_url": url,
            "discovery_tool": collection_source,
            "discovered_at": discovered_at,
        })
        seen_sources.add(normalized_source)

    if not new_source_rows:
        return 0

    errors = client.insert_rows_json(table_id, new_source_rows)
    if errors:
        logger.warning(f"discovered_sources insert errors: {errors}")
        return 0

    return len(new_source_rows)


def classify_article_with_audit(article: Dict) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    2-stage Gemini pipeline.
    Stage 1: lightweight triage for article_type + relevance_score + reason.
    Stage 2: detailed schema extraction only for retained articles.
    Returns (full schema dict for use_case/adoption articles, triage audit row).
    """
    title = article.get('title', 'Untitled')
    content = article.get('content', '')
    logger.info(f"Classifying: {title[:60]}...")

    source = _source_name_from_article(article)
    classified_at = _utc_now_iso()
    article_type = ""
    relevance_score = None
    reason = ""
    triage_flags = {field: None for field in TRIAGE_FLAG_FIELDS}

    try:
        model = GenerativeModel("gemini-2.5-flash")
        triage_prompt = build_triage_prompt(
            title=title,
            description=article.get('description', ''),
            content=content,
            source=source,
            published_date=str(article.get('publishedAt', '')),
            char_limit=TRIAGE_PROMPT_CHAR_LIMIT,
        )
        triage_config = {
            "temperature": 0,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",
            "thinking_config": {"thinking_budget": 0},
        }
        extraction_config = {
            "temperature": 0,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
            "thinking_config": {"thinking_budget": 0},
        }

        triage_text = gemini_generate_with_retry(
            model=model, prompt=triage_prompt, generation_config=triage_config,
            max_retries=6, base_sleep=1.0,
        )
        triage_result = parse_gemini_json(triage_text)
        if not isinstance(triage_result, dict):
            raise ValueError("Triage response is not a dict")

        article_type = _normalize_article_type(triage_result.get('article_type'))
        relevance_score = _coerce_relevance_score(triage_result.get('relevance_score'))
        reason = str(triage_result.get('reason', '') or '').strip()
        triage_flags = _extract_triage_flags(triage_result)
        relevance_score, reason = _apply_triage_score_caps(
            article_type=article_type,
            relevance_score=relevance_score,
            reason=reason,
            triage_flags=triage_flags,
        )

        if article_type == 'use_case' and _is_nonpublic_micro_school_case(article, triage_flags):
            article_type = 'drop'
            reason = (
                reason + " Dropped because the article describes a microschool/nonpublic context "
                "without evidence that it is a public charter school."
            ).strip()
            logger.info(f"✗ drop (score={relevance_score}): {reason[:80]}")
            return None, _build_triage_audit_row(
                article,
                status='drop',
                article_type='drop',
                relevance_score=relevance_score,
                reason=reason,
                classified_at=classified_at,
                triage_flags=triage_flags,
            )

        if article_type == 'use_case' and _is_vague_student_benefit_case(article, triage_flags):
            article_type = 'drop'
            reason = (
                reason + " Dropped because the article only describes AI helping students in a vague "
                "way and does not state a concrete teacher-side AI workflow."
            ).strip()
            logger.info(f"✗ drop (score={relevance_score}): {reason[:80]}")
            return None, _build_triage_audit_row(
                article,
                status='drop',
                article_type='drop',
                relevance_score=relevance_score,
                reason=reason,
                classified_at=classified_at,
                triage_flags=triage_flags,
            )

        if article_type == 'use_case' and _is_noninstitutional_library_demo_case(article, triage_flags):
            article_type = 'drop'
            reason = (
                reason + " Dropped because the article is a noninstitutional library demo/blog case "
                "without a named public K-12 school or district."
            ).strip()
            logger.info(f"✗ drop (score={relevance_score}): {reason[:80]}")
            return None, _build_triage_audit_row(
                article,
                status='drop',
                article_type='drop',
                relevance_score=relevance_score,
                reason=reason,
                classified_at=classified_at,
                triage_flags=triage_flags,
            )

        if article_type == 'use_case' and _is_student_use_primary_case(article, triage_flags) and relevance_score > 7:
            relevance_score = 7
            reason = (
                reason + " Capped at 7 because the article mainly centers on students using AI, "
                "while the educator's own workflow is secondary."
            ).strip()

        if article_type in ('drop', 'adoption') and _is_staff_alert_workflow_case(article, triage_flags):
            article_type = 'use_case'
            relevance_score = max(relevance_score, 7)
            triage_flags['has_current_work_task'] = True
            triage_flags['has_single_focal_use_case'] = True
            reason = (
                reason + " Reclassified as use_case because the article describes a concrete staff "
                "workflow where school personnel receive AI-generated student risk alerts and intervene."
            ).strip()

        if article_type == 'drop':
            logger.info(f"✗ drop (score={relevance_score}): {reason[:80]}")
            return None, _build_triage_audit_row(
                article,
                status='drop',
                article_type='drop',
                relevance_score=relevance_score,
                reason=reason,
                classified_at=classified_at,
                triage_flags=triage_flags,
            )
        if article_type not in ('use_case', 'adoption'):
            reason = reason or f"Unexpected triage article_type={article_type!r}"
            logger.warning(f"Unexpected triage article_type={article_type!r}; dropping article")
            return None, _build_triage_audit_row(
                article,
                status='drop',
                article_type=article_type,
                relevance_score=relevance_score,
                reason=reason,
                classified_at=classified_at,
                triage_flags=triage_flags,
            )

        extraction_prompt = build_extraction_prompt(
            title=title,
            description=article.get('description', ''),
            content=content,
            article_type=article_type,
            source=source,
            published_date=str(article.get('publishedAt', '')),
            location_data=US_LOCATION_DATA_JSON,
            char_limit=EXTRACTION_PROMPT_CHAR_LIMIT,
        )
        extraction_text = gemini_generate_with_retry(
            model=model, prompt=extraction_prompt, generation_config=extraction_config,
            max_retries=6, base_sleep=1.0,
        )
        extraction_result = parse_gemini_json(extraction_text)
        if not isinstance(extraction_result, dict):
            raise ValueError("Extraction response is not a dict")

        _ensure_extraction_schema_fields(extraction_result)
        _normalize_multi_value_fields(extraction_result)

        result = {
            **extraction_result,
            'article_type': article_type,
            'relevance_score': relevance_score,
            'reason': reason,
            'manual_review': relevance_score < MANUAL_REVIEW_THRESHOLD,
        }

        logger.info(f"✓ {article_type} (score={relevance_score}): {reason[:80]}")

        # Attach metadata
        url = article.get('url', '')
        result['article_id'] = make_article_id(url)
        result['classified_at'] = classified_at
        result['raw_article'] = json.dumps(article)

        return result, _build_triage_audit_row(
            article,
            status='retained',
            article_type=article_type,
            relevance_score=relevance_score,
            manual_review=result['manual_review'],
            reason=reason,
            classified_at=classified_at,
            triage_flags=triage_flags,
        )

    except Exception as e:
        logger.error(f"Error classifying article: {e}")
        audit_row = _build_triage_audit_row(
            article,
            status='error',
            article_type=article_type,
            relevance_score=relevance_score,
            reason=reason,
            error_message=str(e),
            classified_at=classified_at,
            triage_flags=triage_flags,
        )
        raise ClassificationPipelineError(str(e), audit_row) from e


def classify_article(article: Dict) -> Optional[Dict[str, Any]]:
    """Backward-compatible wrapper that returns only retained classification rows."""
    row, _ = classify_article_with_audit(article)
    return row


def upload_to_bigquery(rows: List[Dict]) -> int:
    """Upload classified data to BigQuery in chunks to avoid 413 errors."""
    if not rows:
        logger.info("No rows to upload")
        return 0

    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"

    total_uploaded = 0
    chunk_size = 500
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        try:
            errors = client.insert_rows_json(table_id, chunk, row_ids=[r['article_id'] for r in chunk])
            if errors:
                logger.error(f"BigQuery insert errors (chunk {i//chunk_size + 1}): {errors}")
                raise Exception(f"BigQuery upload errors: {errors}")
            total_uploaded += len(chunk)
            logger.info(f"Uploaded chunk {i//chunk_size + 1}: {len(chunk)} rows (total: {total_uploaded})")
        except Exception as e:
            logger.error(f"Error uploading chunk: {e}")
            raise

    try:
        tracked_sources = sync_discovered_sources(rows)
        if tracked_sources:
            logger.info(f"Tracked {tracked_sources} newly validated discovered_sources")
    except Exception as e:
        logger.warning(f"Could not sync discovered_sources: {e}")

    return total_uploaded


def _normalize_title(title: str) -> str:
    """Lowercase, strip whitespace, collapse spaces — used for title-based dedup."""
    import re
    return re.sub(r'\s+', ' ', (title or '').lower().strip())


def _process_articles(articles: List[Dict], existing_ids: set,
                      existing_titles: set = None) -> tuple:
    """Classify a list of articles, skipping already-classified ones."""
    classified_rows = []
    audit_rows = []
    errors = []
    filtered_out = 0
    skipped = 0
    seen_titles = set(existing_titles) if existing_titles else set()

    for i, article in enumerate(articles, 1):
        url = article.get('url', '')
        article_id = make_article_id(url)

        if article_id in existing_ids:
            skipped += 1
            logger.info(f"[{i}/{len(articles)}] Skip (dup id): {article.get('title', '')[:50]}")
            continue

        norm_title = _normalize_title(article.get('title', ''))
        if norm_title and norm_title in seen_titles:
            skipped += 1
            logger.info(f"[{i}/{len(articles)}] Skip (dup title): {article.get('title', '')[:50]}")
            continue

        # Pre-filter: skip articles with no usable content (dead links, paywalls)
        title_text = (article.get('title') or '').strip()
        desc_text = (article.get('description') or '').strip()
        content_text = (article.get('content') or '').strip()
        combined_text = (title_text + desc_text + content_text).lower()
        if not title_text and not desc_text and not content_text:
            filtered_out += 1
            logger.info(f"[{i}/{len(articles)}] Skip (no content): empty article")
            audit_rows.append(_build_triage_audit_row(
                article,
                status='prefilter_skip',
                article_type='drop',
                reason='Skipped before triage: empty article',
            ))
            continue
        paywall_signals = ['subscribe to read', 'subscribe to continue', 'subscription required',
                           'subscribers only', 'sign in to read', 'create a free account to read',
                           '[+]', 'this content is for subscribers']
        if any(sig in combined_text for sig in paywall_signals) and len(combined_text) < 300:
            filtered_out += 1
            logger.info(f"[{i}/{len(articles)}] Skip (paywall): {title_text[:50]}")
            audit_rows.append(_build_triage_audit_row(
                article,
                status='prefilter_skip',
                article_type='drop',
                reason='Skipped before triage: likely paywalled article with too little visible text',
            ))
            continue

        try:
            logger.info(f"\n{'='*60}\nArticle {i}/{len(articles)}\n{'='*60}")

            row, audit_row = classify_article_with_audit(article)
            audit_rows.append(audit_row)
            if row:
                classified_rows.append(row)
                existing_ids.add(article_id)
                if norm_title:
                    seen_titles.add(norm_title)
            else:
                filtered_out += 1

            time.sleep(0.5)

        except ClassificationPipelineError as e:
            logger.error(f"Error processing article {i}: {e}")
            errors.append({"article_title": article.get('title', 'Unknown'), "error": str(e)})
            audit_rows.append(e.audit_row)
            continue
        except Exception as e:
            logger.error(f"Error processing article {i}: {e}")
            errors.append({"article_title": article.get('title', 'Unknown'), "error": str(e)})
            audit_rows.append(_build_triage_audit_row(
                article,
                status='error',
                article_type='',
                reason='',
                error_message=str(e),
            ))
            continue

    return classified_rows, audit_rows, errors, filtered_out, skipped



@app.route('/classify-raw', methods=['POST', 'GET'])
def classify_raw():
    """
    Classify articles already stored in raw_articles table.
    Does NOT call NewsAPI - reads directly from BigQuery.
    Skips articles already present in classified_articles.
    """
    try:
        data = request.get_json(silent=True) or {} if request.method == 'POST' else {}
        batch_size = int(data.get('batch', 100))
        offset = int(data.get('offset', 0))

        client = bigquery.Client(project=PROJECT_ID)
        raw_table = f"{PROJECT_ID}.{BIGQUERY_DATASET}.raw_articles"
        classified_table = f"{PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"

        # Load existing classified IDs and titles (for dedup)
        existing_ids = get_classified_article_ids()
        existing_titles_q = f"""
            SELECT DISTINCT LOWER(TRIM(
                COALESCE(JSON_EXTRACT_SCALAR(raw_article, '$.title'), '')
            )) AS norm_title
            FROM `{classified_table}`
            WHERE raw_article IS NOT NULL
              AND JSON_EXTRACT_SCALAR(raw_article, '$.title') IS NOT NULL
              AND JSON_EXTRACT_SCALAR(raw_article, '$.title') != ''
        """
        existing_titles = {
            r.norm_title for r in client.query(existing_titles_q).result()
            if r.norm_title
        }
        logger.info(f"Loaded {len(existing_titles)} existing classified titles for dedup")

        # Fetch unclassified raw articles — education sources first, then by recency
        # K-12 education sources are prioritised (score=0) over general news (score=1)
        # to maximise the pass-rate through Stage 1 and reduce wasted Gemini calls.
        EDUCATION_SOURCES = (
            'edsurge', 'edweek', 'chalkbeat', 'the74', '74million',
            'edutopia', 'eschoolnews', 'edtech', 'district administration',
            'k12dive', 'schoolleadersnow', 'weareteachers',
            'fairfax county', 'broward', 'houstonisd', 'dallasisd',
            'fcps', 'lausd', 'nyc schools', 'chicago public schools',
            'tnfirefly', 'gpb', 'mprnews', 'opb', 'kqed', 'wgbh',
        )
        education_filter = ' OR '.join(
            f"LOWER(r.source_name) LIKE '%{s}%'" for s in EDUCATION_SOURCES
        )
        content_table = f"{PROJECT_ID}.{BIGQUERY_DATASET}.raw_articles_content"
        url_hash_expr = _bq_url_hash_expr("r.url")
        query = f"""
            WITH classified_ids AS (
                SELECT DISTINCT article_id
                FROM `{classified_table}`
                WHERE article_id IS NOT NULL
            )
            SELECT
                r.article_id, r.url, r.title, r.description, r.source_name, r.publishedAt,
                r.collection_source,
                COALESCE(rc.content_full, r.content) AS content
            FROM `{raw_table}` r
            LEFT JOIN `{content_table}` rc
                ON r.article_id = rc.article_id AND rc.status = 'success'
            LEFT JOIN classified_ids c_id
                ON c_id.article_id = r.article_id
            LEFT JOIN classified_ids c_url
                ON r.url IS NOT NULL
               AND TRIM(r.url) != ''
               AND c_url.article_id = {url_hash_expr}
            WHERE c_id.article_id IS NULL
              AND c_url.article_id IS NULL
            ORDER BY
                (CASE WHEN rc.content_full IS NOT NULL THEN 0 ELSE 1 END),
                (CASE WHEN {education_filter} THEN 0 ELSE 1 END),
                (CASE WHEN r.description IS NOT NULL AND r.description != '' THEN 0 ELSE 1 END),
                r.publishedAt DESC
            LIMIT {batch_size}
            OFFSET {offset}
        """
        logger.info(f"Fetching up to {batch_size} unclassified raw articles (offset={offset})")
        rows = list(client.query(query).result())

        if not rows:
            return jsonify({"status": "success", "message": "No unclassified articles found", "processed": 0}), 200

        # Convert BQ rows to article dicts matching NewsAPI format
        articles = []
        for row in rows:
            article = {
                'url': row.get('url', ''),
                'title': row.get('title', ''),
                'description': row.get('description', ''),
                'content': row.get('content', ''),
                'publishedAt': str(row.get('publishedAt', '')),
                'source': {'name': row.get('source_name', '')},
                'collection_source': row.get('collection_source', ''),
            }
            articles.append(article)

        classified_rows, audit_rows, errors, filtered_out, skipped = _process_articles(
            articles, existing_ids, existing_titles)

        uploaded_count = 0
        if classified_rows:
            uploaded_count = upload_to_bigquery(classified_rows)
        audit_logged = write_local_triage_audit_rows(audit_rows)

        logger.info(f"classify-raw COMPLETE: input={len(articles)}, filtered={filtered_out}, "
                    f"classified={len(classified_rows)}, uploaded={uploaded_count}")

        return jsonify({
            "status": "success",
            "raw_articles_fetched": len(articles),
            "filtered_out": filtered_out,
            "skipped_dup": skipped,
            "classified": len(classified_rows),
            "uploaded": uploaded_count,
            "triage_audit_logged": audit_logged,
            "triage_audit_path": LOCAL_TRIAGE_AUDIT_PATH if ENABLE_LOCAL_TRIAGE_AUDIT else None,
            "errors": len(errors),
            "drop_debug_sample": _summarize_audit_rows(
                audit_rows, statuses=('drop', 'prefilter_skip', 'error'), limit=10),
            "next_offset": offset + batch_size,
        }), 200

    except Exception as e:
        logger.error(f"Fatal error in classify-raw: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/cleanup', methods=['POST', 'GET'])
def cleanup():
    """
    Deduplicate classified_articles and drop low-relevance-score rows.
    POST body (optional):
      { "min_score": 2, "dry_run": false }
    """
    try:
        data = request.get_json(silent=True) or {} if request.method == 'POST' else {}
        min_score = int(data.get('min_score', 2))
        dry_run = bool(data.get('dry_run', False))

        client = bigquery.Client(project=PROJECT_ID)
        table_id = f"{PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"

        total = list(client.query(f"SELECT COUNT(*) as cnt FROM `{table_id}`").result())[0].cnt
        unique = list(client.query(f"SELECT COUNT(DISTINCT article_id) as cnt FROM `{table_id}`").result())[0].cnt
        extra = total - unique

        score_dist = {
            str(r.relevance_score): r.cnt
            for r in client.query(f"""
                SELECT relevance_score, COUNT(*) as cnt
                FROM `{table_id}`
                GROUP BY relevance_score ORDER BY relevance_score
            """).result()
        }
        low_score_count = sum(
            cnt for score, cnt in score_dist.items()
            if score == 'None' or int(score) <= min_score
        )

        if dry_run:
            return jsonify({
                "status": "dry_run",
                "total_rows": total,
                "unique_article_ids": unique,
                "duplicate_rows": extra,
                "low_score_rows": low_score_count,
                "score_distribution": score_dist,
            }), 200

        # Step 1: Deduplicate — keep best row per article_id
        client.query(f"""
            CREATE OR REPLACE TABLE `{table_id}` AS
            SELECT * EXCEPT(row_num)
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY article_id
                    ORDER BY relevance_score DESC, classified_at DESC
                ) as row_num
                FROM `{table_id}`
            )
            WHERE row_num = 1
        """).result()

        after_dedup = list(client.query(f"SELECT COUNT(*) as cnt FROM `{table_id}`").result())[0].cnt

        # Step 2: Drop low-score rows
        client.query(f"""
            DELETE FROM `{table_id}`
            WHERE relevance_score <= {min_score} OR relevance_score IS NULL
        """).result()

        final = list(client.query(f"SELECT COUNT(*) as cnt FROM `{table_id}`").result())[0].cnt

        logger.info(f"Cleanup: {total} → {after_dedup} (dedup) → {final} (drop low score)")
        return jsonify({
            "status": "success",
            "before": total,
            "after_dedup": after_dedup,
            "after_drop_low": final,
            "removed_duplicates": total - after_dedup,
            "removed_low_score": after_dedup - final,
            "min_score_threshold": min_score,
        }), 200

    except Exception as e:
        logger.error(f"Fatal error in cleanup: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Return summary statistics for classified_articles and raw_articles."""
    try:
        client = bigquery.Client(project=PROJECT_ID)
        classified_table = f"{PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"
        raw_table = f"{PROJECT_ID}.{BIGQUERY_DATASET}.raw_articles"

        total_classified = list(client.query(
            f"SELECT COUNT(*) as cnt FROM `{classified_table}`").result())[0].cnt
        total_raw = list(client.query(
            f"SELECT COUNT(*) as cnt FROM `{raw_table}`").result())[0].cnt
        url_hash_expr = _bq_url_hash_expr("r.url")
        unclassified = list(client.query(f"""
            SELECT COUNT(*) as cnt FROM `{raw_table}` r
            WHERE NOT EXISTS (
                SELECT 1
                FROM `{classified_table}` c
                WHERE c.article_id = r.article_id
                   OR (
                        r.url IS NOT NULL
                    AND TRIM(r.url) != ''
                    AND c.article_id = {url_hash_expr}
                   )
            )
        """).result())[0].cnt

        avg_score = list(client.query(
            f"SELECT ROUND(AVG(relevance_score), 2) as v FROM `{classified_table}`").result())[0].v
        high_quality = list(client.query(
            f"SELECT COUNT(*) as cnt FROM `{classified_table}` WHERE relevance_score >= 6").result())[0].cnt

        by_type = {
            r.article_type or 'unknown': r.cnt
            for r in client.query(f"""
                SELECT article_type, COUNT(*) as cnt FROM `{classified_table}`
                GROUP BY article_type ORDER BY cnt DESC
            """).result()
        }

        top_states = {
            r.state: r.cnt
            for r in client.query(f"""
                SELECT state, COUNT(*) as cnt FROM `{classified_table}`
                WHERE state IS NOT NULL AND state NOT IN ('null', 'none', '')
                GROUP BY state ORDER BY cnt DESC LIMIT 10
            """).result()
        }

        score_dist = {
            str(r.relevance_score): r.cnt
            for r in client.query(f"""
                SELECT relevance_score, COUNT(*) as cnt FROM `{classified_table}`
                GROUP BY relevance_score ORDER BY relevance_score
            """).result()
        }

        return jsonify({
            "raw_articles": total_raw,
            "classified_articles": total_classified,
            "unclassified_raw": unclassified,
            "avg_relevance_score": avg_score,
            "high_quality_count": high_quality,
            "by_article_type": by_type,
            "top_states": top_states,
            "score_distribution": score_dist,
        }), 200

    except Exception as e:
        logger.error(f"Fatal error in stats: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/classify-sample', methods=['POST'])
def classify_sample():
    """
    Classify a specific list of article_ids from raw_articles.
    For manual assessment / professor review.
    POST body: { "article_ids": ["sha1...", ...] }
    """
    try:
        data = request.get_json(silent=True) or {}
        article_ids = data.get('article_ids', [])
        if not article_ids:
            return jsonify({"status": "error", "message": "article_ids required"}), 400

        client = bigquery.Client(project=PROJECT_ID)
        raw_table = f"{PROJECT_ID}.{BIGQUERY_DATASET}.raw_articles"

        content_table = f"{PROJECT_ID}.{BIGQUERY_DATASET}.raw_articles_content"
        ids_str = ", ".join(f'"{aid}"' for aid in article_ids)
        url_hash_expr = _bq_url_hash_expr("r.url")
        query = f"""
            SELECT
                r.article_id, r.url, r.title, r.description, r.source_name, r.publishedAt,
                r.collection_source,
                COALESCE(rc.content_full, r.content) AS content
            FROM `{raw_table}` r
            LEFT JOIN `{content_table}` rc
                ON r.article_id = rc.article_id AND rc.status = 'success'
            WHERE r.article_id IN ({ids_str})
               OR (
                    r.url IS NOT NULL
                AND TRIM(r.url) != ''
                AND {url_hash_expr} IN ({ids_str})
               )
        """
        rows = list(client.query(query).result())

        if not rows:
            return jsonify({"status": "error", "message": "No articles found"}), 404

        articles = []
        for row in rows:
            articles.append({
                'url': row.get('url', ''),
                'title': row.get('title', ''),
                'description': row.get('description', ''),
                'content': row.get('content', ''),
                'publishedAt': str(row.get('publishedAt', '')),
                'source': {'name': row.get('source_name', '')},
                'collection_source': row.get('collection_source', ''),
            })

        # Force re-classify even if already in classified_articles
        existing_ids = set()
        classified_rows, audit_rows, errors, filtered_out, _ = _process_articles(articles, existing_ids)

        if classified_rows:
            upload_to_bigquery(classified_rows)
        audit_logged = write_local_triage_audit_rows(audit_rows)
        audit_by_article_id = {row.get("article_id"): row for row in audit_rows}

        return jsonify({
            "status": "success",
            "total": len(articles),
            "filtered_out": filtered_out,
            "classified": len(classified_rows),
            "triage_audit_logged": audit_logged,
            "triage_audit_path": LOCAL_TRIAGE_AUDIT_PATH if ENABLE_LOCAL_TRIAGE_AUDIT else None,
            "errors": len(errors),
            "results": [
                {
                    "article_id": r["article_id"],
                    "title": json.loads(r["raw_article"]).get("title", ""),
                    "article_type": r.get("article_type"),
                    "relevance_score": r.get("relevance_score"),
                    "state": r.get("state"),
                    "AI_product": r.get("AI_product"),
                    "purpose_of_AI": r.get("purpose_of_AI"),
                    "impact": r.get("impact"),
                    "use_case_description": (r.get("use_case_description") or "")[:200],
                    "outcome": (r.get("outcome") or "")[:150],
                    "gate_flags": {
                        field: (audit_by_article_id.get(r["article_id"]) or {}).get(field)
                        for field in TRIAGE_FLAG_FIELDS
                    },
                }
                for r in classified_rows
            ],
            "drop_results": _summarize_audit_rows(
                audit_rows, statuses=('drop', 'prefilter_skip', 'error'), limit=100),
            "filtered_titles": [
                row["title"] for row in _summarize_audit_rows(
                    audit_rows, statuses=('drop', 'prefilter_skip', 'error'), limit=1000)
            ],
        }), 200

    except Exception as e:
        logger.error(f"Fatal error in classify-sample: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
