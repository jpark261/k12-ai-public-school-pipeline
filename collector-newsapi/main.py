"""
K-12 AI NewsAPI Collector
Fetches K-12 AI education articles from NewsAPI and stores them in raw_articles (BigQuery).
Does NOT classify — classification is handled by k12-classifier's /classify-raw endpoint.

Endpoints:
  POST/GET /          — collect articles (days_back, max_articles params)
  GET      /health    — health check

Cloud Scheduler: run daily, e.g. POST / {"days_back": 3}
"""

import os
import json
import logging
import time
import hashlib

from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import requests
from flask import Flask, request, jsonify
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

PROJECT_ID       = os.getenv('GCP_PROJECT', 'your-gcp-project-id')
BIGQUERY_DATASET = os.getenv('BIGQUERY_DATASET', 'k12_ai_dataset')
RAW_TABLE        = os.getenv('BIGQUERY_TABLE', 'raw_articles')
NEWS_API_KEY     = os.getenv('NEWS_API_KEY')

AI_CORE_TERMS = (
    '("artificial intelligence" OR "generative AI" OR ChatGPT OR Gemini OR '
    'Copilot OR Claude OR chatbot OR "AI assistant" OR "AI tool" OR '
    '"AI tutor" OR tutorbot OR Khanmigo OR MagicSchool OR SchoolAI OR '
    'Eduaide OR Diffit OR Brisk OR Curipod OR "ChatGPT Edu")'
)

EDUCATOR_ROLE_TERMS = (
    '("teacher" OR educator OR counselor OR librarian OR principal OR '
    'superintendent OR "instructional coach" OR psychologist OR '
    '"special education teacher")'
)

K12_CONTEXT_TERMS = (
    '(school OR classroom OR district OR "public schools" OR '
    '"school district" OR "K-12" OR "public education" OR '
    '"department of education" OR superintendent OR principal OR student)'
)

PUBLIC_SYSTEM_TERMS = (
    '("school district" OR "public schools" OR superintendent OR principal OR '
    '"board of education" OR "department of education" OR '
    '"state education department" OR "state board of education" OR '
    '"public instruction" OR "public charter school")'
)

NEWSAPI_QUERY_BUCKETS = [
    (
        "educator_workflow",
        f'{EDUCATOR_ROLE_TERMS} '
        f'AND {AI_CORE_TERMS} '
        'AND ("lesson plan" OR grading OR feedback OR rubric OR differentiate OR '
        'classroom OR "student support") '
        f'AND {K12_CONTEXT_TERMS}'
    ),
    (
        "educator_workflow_generic",
        f'{EDUCATOR_ROLE_TERMS} '
        'AND ("AI" OR "artificial intelligence" OR "generative AI" OR '
        '"AI assistant" OR chatbot) '
        'AND ("lesson plans" OR "lesson plan" OR grading OR feedback OR '
        'rubric OR writing OR tutoring OR translation OR "parent communication" OR '
        '"family communication" OR advising OR counseling OR instruction OR '
        'curriculum OR classroom) '
        f'AND {K12_CONTEXT_TERMS}'
    ),
    (
        "district_adoption",
        f'{PUBLIC_SYSTEM_TERMS} '
        f'AND {AI_CORE_TERMS} '
        'AND (adopted OR pilot OR piloted OR rollout OR "rolled out" OR deployed '
        'OR implemented OR partnership OR contract OR license OR licensed OR '
        'procured OR procurement OR purchase OR purchased)'
    ),
    (
        "state_agency_adoption",
        '("department of education" OR "state education department" OR '
        '"state board of education" OR "public instruction") '
        f'AND {AI_CORE_TERMS} '
        'AND (pilot OR piloted OR rollout OR "rolled out" OR partnership OR '
        'contract OR purchase OR license OR licensed OR deployed)'
    ),
    (
        "pilot_testing",
        f'{K12_CONTEXT_TERMS} '
        f'AND {AI_CORE_TERMS} '
        'AND (pilot OR piloted OR testing OR tested OR trial OR trialed OR '
        '"beta testing" OR rollout OR "rolled out" OR launched)'
    ),
    (
        "student_support",
        '(counselor OR counseling OR "student support" OR translation OR '
        '"family communication" OR multilingual OR attendance OR '
        '"mental health" OR wellbeing OR "special education" OR IEP) '
        f'AND {AI_CORE_TERMS} '
        f'AND {K12_CONTEXT_TERMS}'
    ),
    (
        "tutoring_intervention",
        '(tutoring OR tutor OR intervention OR "academic intervention" OR '
        'literacy OR writing OR math OR reading OR homework OR "personalized learning") '
        f'AND {AI_CORE_TERMS} '
        f'AND {K12_CONTEXT_TERMS}'
    ),
    (
        "operations_admin",
        f'{PUBLIC_SYSTEM_TERMS} '
        f'AND {AI_CORE_TERMS} '
        'AND (attendance OR transportation OR "bus routes" OR scheduling OR '
        'operations OR administrative OR paperwork OR enrollment OR '
        '"staff workflow" OR registration OR translation)'
    ),
    (
        "safety_security",
        f'{PUBLIC_SYSTEM_TERMS} '
        'AND ("artificial intelligence" OR "AI-powered" OR "AI-based" OR '
        'surveillance OR monitoring OR "threat detection" OR '
        '"weapons detection" OR "gun detection") '
        'AND (school OR district OR classroom OR campus)'
    ),
    (
        "product_specific",
        '(Khanmigo OR MagicSchool OR SchoolAI OR Eduaide OR Diffit OR Brisk OR '
        'Curipod OR "ChatGPT Edu" OR "Google Classroom" OR "Gemini for Education" OR '
        '"Microsoft Copilot" OR "Securly" OR "GoGuardian" OR "Lightspeed Alert") '
        f'AND {K12_CONTEXT_TERMS}'
    ),
]


def _make_id(url: str) -> str:
    return hashlib.sha1((url or '').strip().encode('utf-8')).hexdigest()


def _resolve_date_window(
    days_back: int,
    from_date: Optional[str],
    to_date: Optional[str],
) -> Tuple[str, str]:
    if from_date and to_date:
        return from_date, to_date
    if from_date and not to_date:
        return from_date, datetime.now().strftime('%Y-%m-%d')
    if to_date and not from_date:
        start = (datetime.strptime(to_date, '%Y-%m-%d') - timedelta(days=days_back)).strftime('%Y-%m-%d')
        return start, to_date

    resolved_from = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    resolved_to = datetime.now().strftime('%Y-%m-%d')
    return resolved_from, resolved_to


def _iter_date_windows(
    from_date: str,
    to_date: str,
    window_days: Optional[int],
) -> List[Tuple[str, str]]:
    if not window_days or window_days <= 0:
        return [(from_date, to_date)]

    start = datetime.strptime(from_date, '%Y-%m-%d')
    end = datetime.strptime(to_date, '%Y-%m-%d')
    windows = []
    cursor = start

    while cursor <= end:
        window_end = min(cursor + timedelta(days=window_days - 1), end)
        windows.append((
            cursor.strftime('%Y-%m-%d'),
            window_end.strftime('%Y-%m-%d'),
        ))
        cursor = window_end + timedelta(days=1)

    return windows


def fetch_newsapi(
    days_back: int = 3,
    max_total: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    sort_by: str = 'publishedAt',
    window_days: Optional[int] = None,
) -> List[dict]:
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY not set")

    from_date, to_date = _resolve_date_window(days_back, from_date, to_date)

    all_articles = []
    seen_urls = set()

    date_windows = _iter_date_windows(from_date, to_date, window_days)

    for bucket_name, query in NEWSAPI_QUERY_BUCKETS:
        bucket_added = 0

        for window_from, window_to in date_windows:
            page = 1

            while True:
                params = {
                    'q':        query,
                    'from':     window_from,
                    'to':       window_to,
                    'language': 'en',
                    'sortBy':   sort_by,
                    'searchIn': 'title,description,content',
                    'pageSize': 100,
                    'page':     page,
                    'apiKey':   NEWS_API_KEY,
                }
                try:
                    resp = requests.get('https://newsapi.org/v2/everything', params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    articles = data.get('articles', [])
                    if not articles:
                        break

                    new_articles = []
                    for article in articles:
                        url = (article.get('url') or '').strip()
                        if not url or url in seen_urls:
                            continue
                        seen_urls.add(url)
                        new_articles.append(article)

                    all_articles.extend(new_articles)
                    bucket_added += len(new_articles)
                    logger.info(
                        "Bucket %s window %s..%s page %s: %s new (%s raw, total %s/%s)",
                        bucket_name,
                        window_from,
                        window_to,
                        page,
                        len(new_articles),
                        len(articles),
                        len(all_articles),
                        data.get('totalResults', 0),
                    )

                    if len(articles) < 100:
                        break
                    if max_total and len(all_articles) >= max_total:
                        break

                    page += 1
                    time.sleep(0.5)
                except requests.RequestException as e:
                    logger.error(
                        "NewsAPI error bucket %s window %s..%s page %s: %s",
                        bucket_name,
                        window_from,
                        window_to,
                        page,
                        e,
                    )
                    break

            if max_total and len(all_articles) >= max_total:
                break

        logger.info("Bucket %s added %s unique articles", bucket_name, bucket_added)
        if max_total and len(all_articles) >= max_total:
            break

    if max_total:
        all_articles = all_articles[:max_total]
    logger.info(f"NewsAPI fetched: {len(all_articles)} articles")
    return all_articles


def save_to_bigquery(articles: List[dict]) -> int:
    if not articles:
        return 0

    client   = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{BIGQUERY_DATASET}.{RAW_TABLE}"
    now      = datetime.utcnow().isoformat()

    # Dedup against both article_id and URL so legacy rows with URL-based IDs
    # do not get re-inserted when collectors are re-run.
    candidate_ids = []
    candidate_urls = []
    for article in articles:
        url = (article.get('url') or '').strip()
        if not url:
            continue
        candidate_ids.append(_make_id(url))
        candidate_urls.append(url)

    existing_ids = set()
    existing_urls = set()
    if candidate_ids or candidate_urls:
        existing_q = f"""
            SELECT article_id, url
            FROM `{table_id}`
            WHERE article_id IN UNNEST(@article_ids)
               OR url IN UNNEST(@candidate_urls)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("article_ids", "STRING", candidate_ids),
                bigquery.ArrayQueryParameter("candidate_urls", "STRING", candidate_urls),
            ]
        )
        for row in client.query(existing_q, job_config=job_config).result():
            if row.article_id:
                existing_ids.add(row.article_id)
            if row.url:
                existing_urls.add(row.url)

    rows = []
    seen_urls = set()
    seen_article_ids = set()
    for a in articles:
        url = (a.get('url') or '').strip()
        if not url:
            continue
        article_id = _make_id(url)
        if url in seen_urls or article_id in seen_article_ids:
            continue
        if article_id in existing_ids or url in existing_urls:
            continue
        seen_urls.add(url)
        seen_article_ids.add(article_id)

        source = a.get('source', {})
        rows.append({
            'article_id':        article_id,
            'collected_at':      now,
            'source_id':         source.get('id') if isinstance(source, dict) else None,
            'source_name':       source.get('name', '') if isinstance(source, dict) else str(source),
            'author':            a.get('author'),
            'title':             a.get('title', ''),
            'description':       a.get('description', ''),
            'url':               url,
            'urlToImage':        a.get('urlToImage'),
            'publishedAt':       a.get('publishedAt', now),
            'content':           a.get('content', ''),
            'raw_json':          json.dumps(a),
            'collection_source': 'NewsAPI',
        })

    if not rows:
        logger.info("No new articles to insert (all duplicates)")
        return 0

    inserted = 0
    chunk_size = 500
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        errors = client.insert_rows_json(table_id, chunk,
                                         row_ids=[r['article_id'] for r in chunk])
        if errors:
            logger.error(f"BQ insert errors: {errors[:3]}")
        else:
            inserted += len(chunk)
    logger.info(f"Inserted {inserted} new articles into {RAW_TABLE}")
    return inserted


@app.route('/', methods=['POST', 'GET'])
def collect():
    data        = request.get_json(silent=True) or {} if request.method == 'POST' else {}
    days_back   = int(data.get('days_back', 3))
    max_articles = data.get('max_articles')
    from_date = data.get('from_date')
    to_date = data.get('to_date')
    sort_by = data.get('sort_by', 'publishedAt')
    window_days = data.get('window_days')
    window_days = int(window_days) if window_days is not None else None

    logger.info(
        "Collecting NewsAPI: days_back=%s from_date=%s to_date=%s sort_by=%s window_days=%s max=%s",
        days_back,
        from_date,
        to_date,
        sort_by,
        window_days,
        max_articles or 'unlimited',
    )

    articles = fetch_newsapi(
        days_back=days_back,
        max_total=max_articles,
        from_date=from_date,
        to_date=to_date,
        sort_by=sort_by,
        window_days=window_days,
    )
    inserted = save_to_bigquery(articles)

    return jsonify({
        'status':   'success',
        'fetched':  len(articles),
        'inserted': inserted,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
    }), 200


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
