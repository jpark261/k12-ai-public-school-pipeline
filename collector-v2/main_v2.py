import os
import json
import re
import hashlib
import time
import requests
from urllib.parse import urlparse, urljoin
from datetime import datetime
from email.utils import parsedate_to_datetime
from flask import Flask, request, jsonify
from google.cloud import bigquery
from bs4 import BeautifulSoup
import trafilatura

app = Flask(__name__)

GCP_PROJECT    = os.getenv("GCP_PROJECT", "your-gcp-project-id")
DATASET        = os.getenv("BIGQUERY_DATASET", "k12_ai_dataset")
RAW_TABLE      = os.getenv("BIGQUERY_TABLE", "raw_articles")
SERPAPI_KEY    = os.getenv("SERPAPI_KEY")
GNEWS_KEY      = os.getenv("GNEWS_KEY")

bq_client = bigquery.Client(project=GCP_PROJECT)

EDUCATOR_QUERIES = [
    # Teacher-specific AI adoption stories
    '"math teacher" OR "science teacher" OR "English teacher" "ChatGPT" OR "AI" "lesson plan" OR "curriculum" school -student -university',
    '"school principal" OR "superintendent" OR "curriculum director" "artificial intelligence" OR "ChatGPT" "implemented" OR "adopted" OR "introduced" district',
    'teacher "uses ChatGPT" OR "used ChatGPT" OR "using ChatGPT" "create" OR "design" OR "build" lesson classroom -university -college',
    'educator "AI tool" OR "generative AI" "save time" OR "grade" OR "feedback" OR "differentiate" K-12 school -market -invest',
    '"school district" "adopted" OR "piloted" OR "rolled out" "AI" OR "ChatGPT" teacher classroom -stock -CAGR',
    'teacher "MagicSchool" OR "Khanmigo" OR "Eduaide" OR "Diffit" OR "SchoolAI" "class" OR "students" OR "lesson" -market',
    '"instructional coach" OR "literacy coach" OR "department head" "AI" classroom school teacher',
    'teacher "writing feedback" OR "lesson planning" OR "quiz" OR "rubric" "ChatGPT" OR "AI" school -university',
    '"how I use AI" OR "how teachers use AI" classroom school lesson -university -college -student',
    'teacher "saved time" OR "more efficient" OR "personalized learning" "AI" OR "ChatGPT" school district -market',
]

def fetch_from_serpapi(days_back=30):
    if not SERPAPI_KEY:
        print("SERPAPI_KEY not set -- skipping SerpApi.")
        return []

    articles = []
    seen_urls = set()

    for query in EDUCATOR_QUERIES:
        for page in range(1, 4):
            params = {
                "engine":  "google_news",
                "q":       query,
                "gl":      "us",
                "hl":      "en",
                "start":   (page - 1) * 10,
                "api_key": SERPAPI_KEY,
            }
            try:
                resp = requests.get("https://serpapi.com/search", params=params, timeout=30)
                resp.raise_for_status()
                news_results = resp.json().get("news_results", [])
                if not news_results:
                    break
                for item in news_results:
                    url = item.get("link", "")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    now = datetime.utcnow().isoformat()
                    snippet = item.get("snippet", "")
                    full = _fetch_full_content(url)
                    articles.append({
                        "article_id":      _make_id(url),
                        "source_id":       None,
                        "source_name":     item.get("source", {}).get("name", "Google News"),
                        "author":          None,
                        "title":           item.get("title", ""),
                        "description":     snippet,
                        "url":             url,
                        "urlToImage":      item.get("thumbnail"),
                        "publishedAt":     now,
                        "content":         full or snippet,
                        "collected_at":    now,
                        "raw_json":        json.dumps(item),
                        "_discovery_tool": "SerpApi",
                        "_snippet":        snippet,
                    })
            except Exception as e:
                print(f"SerpApi error (query={query!r}, page={page}): {e}")
                break

    print(f"SerpApi collected: {len(articles)}")
    return articles



GNEWS_QUERIES = [
    'teacher "ChatGPT" "lesson plan" OR "curriculum" school -university',
    'educator "AI tool" classroom K-12 "adopted" OR "uses" OR "implemented"',
    'teacher "generative AI" "grade" OR "feedback" OR "differentiate" school district',
    'principal OR superintendent "artificial intelligence" school district pilot',
    'teacher MagicSchool OR Khanmigo OR Eduaide OR Diffit classroom lesson',
    '"how teachers use" OR "teachers using" AI school classroom -university -college',
    'teacher "save time" OR "more time" AI school lesson students',
]

def fetch_from_gnews(days_back=30):
    if not GNEWS_KEY:
        print("GNEWS_KEY not set -- skipping GNews.")
        return []

    articles = []
    seen_urls = set()

    for query in GNEWS_QUERIES:
        try:
            params = {
                "q":        query,
                "lang":     "en",
                "country":  "us",
                "max":      10,
                "token":    GNEWS_KEY,
            }
            resp = requests.get("https://gnews.io/api/v4/search", params=params, timeout=30)
            resp.raise_for_status()
            for item in resp.json().get("articles", []):
                url = item.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                now = datetime.utcnow().isoformat()
                description = item.get("description", "")
                full = _fetch_full_content(url)
                articles.append({
                    "article_id":      _make_id(url),
                    "source_id":       None,
                    "source_name":     item.get("source", {}).get("name", "GNews"),
                    "author":          None,
                    "title":           item.get("title", ""),
                    "description":     description,
                    "url":             url,
                    "urlToImage":      item.get("image"),
                    "publishedAt":     item.get("publishedAt", now),
                    "content":         full or item.get("content", "") or description,
                    "collected_at":    now,
                    "raw_json":        json.dumps(item),
                    "_discovery_tool": "GNews",
                    "_snippet":        description,
                })
        except Exception as e:
            print(f"GNews error (query={query!r}): {e}")

    print(f"GNews collected: {len(articles)}")
    return articles

def fetch_from_gdelt(days_back=30):
    print("Starting GDELT BigQuery collection...")
    interval = min(days_back, 90)
    bq_query = f"""
        SELECT
            DocumentIdentifier                               AS url,
            SourceCommonName                                 AS source_name,
            DATE(TIMESTAMP_TRUNC(
                PARSE_TIMESTAMP('%Y%m%d%H%M%S',
                    REGEXP_EXTRACT(GKGRECORDID, r'^(\\d{{15}})')),
                DAY))                                        AS published_date,
            V2Themes,
            V2Locations
        FROM `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE _PARTITIONTIME >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {interval} DAY)
          AND V2Themes LIKE '%EDUCATION%'
          AND (V2Themes LIKE '%ARTIFICIAL_INTELLIGENCE%' OR V2Themes LIKE '%CHATGPT%')
          AND (
              DocumentIdentifier LIKE '%.com/%'
           OR DocumentIdentifier LIKE '%.org/%'
           OR DocumentIdentifier LIKE '%.edu/%'
           OR DocumentIdentifier LIKE '%.gov/%'
          )

        LIMIT 500
    """
    articles = []
    try:
        for row in bq_client.query(bq_query).result():
            now = datetime.utcnow().isoformat()
            pub = str(row.published_date) + "T00:00:00Z" if row.published_date else now
            full = _fetch_full_content(row.url)
            articles.append({
                "article_id":      _make_id(row.url),
                "source_id":       None,
                "source_name":     row.source_name or "GDELT Source",
                "author":          None,
                "title":           "GDELT Article",
                "description":     "",
                "url":             row.url,
                "urlToImage":      None,
                "publishedAt":     pub,
                "content":         full,
                "collected_at":    now,
                "raw_json":        json.dumps({"themes": row.V2Themes, "locations": row.V2Locations}),
                "_discovery_tool": "GDELT",
                "_snippet":        "",
            })
    except Exception as e:
        print(f"GDELT Query Error: {e}")

    print(f"GDELT collected: {len(articles)}")
    return articles

def bq_insert_batch(table_ref, rows, batch_size=500):
    inserted = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i: i + batch_size]
        try:
            errs = bq_client.insert_rows_json(table_ref, chunk)
            if errs:
                print(f"Insert errors (batch {i}): {errs[0]}")
            else:
                inserted += len(chunk)
        except Exception as e:
            print(f"Batch upload error (batch {i}): {e}")
    return inserted


def process_and_save(articles):
    if not articles:
        return 0, 0

    # Strip internal fields before inserting into raw_articles, but save _discovery_tool as collection_source
    raw_rows = []
    for art in articles:
        row = {k: v for k, v in art.items() if not k.startswith("_")}
        row["collection_source"] = art.get("_discovery_tool", "Unknown")
        raw_rows.append(row)

    # Dedup against both article_id and URL so legacy rows with URL-based IDs do not re-enter
    if raw_rows:
        candidate_ids = [r["article_id"] for r in raw_rows if r.get("article_id")]
        candidate_urls = [r["url"] for r in raw_rows if r.get("url")]
        if candidate_ids or candidate_urls:
            try:
                existing_q = f"""
                    SELECT article_id, url
                    FROM `{GCP_PROJECT}.{DATASET}.{RAW_TABLE}`
                    WHERE article_id IN UNNEST(@article_ids)
                       OR url IN UNNEST(@candidate_urls)
                """
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ArrayQueryParameter("article_ids", "STRING", candidate_ids),
                        bigquery.ArrayQueryParameter("candidate_urls", "STRING", candidate_urls),
                    ]
                )
                existing_rows = list(bq_client.query(existing_q, job_config=job_config).result())
                existing_ids = {r.article_id for r in existing_rows if r.article_id}
                existing_urls = {r.url for r in existing_rows if r.url}
                raw_rows = [
                    r for r in raw_rows
                    if r.get("article_id") not in existing_ids and r.get("url") not in existing_urls
                ]
            except Exception as e:
                print(f"Dedup query error: {e}")

    inserted_raw = bq_insert_batch(f"{GCP_PROJECT}.{DATASET}.{RAW_TABLE}", raw_rows)

    # Source tracking is intentionally deferred until after classification so noisy
    # alternative-source candidates do not get promoted into discovered_sources.
    return inserted_raw, 0


# ─────────────────────────────────────────────────────────────
# Education Site Scraper
# ─────────────────────────────────────────────────────────────

SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# Education-focused RSS feeds not covered by NewsAPI (verified working)
EDUCATION_RSS_FEEDS = [
    # --- Original 10 ---
    ("Getting Smart",            "https://www.gettingsmart.com/feed/"),
    ("District Administration",  "https://districtadministration.com/feed/"),
    ("EdTech Magazine K12",      "https://edtechmagazine.com/k12/rss.xml"),
    ("eSchool News",             "https://www.eschoolnews.com/feed/"),
    ("K-12 Dive",                "https://www.k12dive.com/feeds/news/"),
    ("Cult of Pedagogy",         "https://www.cultofpedagogy.com/feed/"),
    ("We Are Teachers",          "https://www.weareteachers.com/feed/"),
    ("GovTech Education",        "https://www.govtech.com/rss/education.rss"),
    ("The Journal",              "https://thejournal.com/rss-feeds/all-articles.aspx"),
    ("Dallas ISD Hub",           "https://thehub.dallasisd.org/feed/"),
    # --- Added ~20 more credible K-12 / EdTech sources ---
    ("EdSurge",                  "https://www.edsurge.com/news.rss"),
    ("Education Week",           "https://www.edweek.org/feeds/rss/news.xml"),
    ("Chalkbeat",                "https://www.chalkbeat.org/arc/outboundfeeds/rss/"),
    ("The 74 Million",           "https://www.the74million.org/feed/"),
    ("Edutopia",                 "https://www.edutopia.org/rss.xml"),
    ("MindShift KQED",           "https://feeds.feedburner.com/MindShift-KQED"),
    ("NPR Education",            "https://feeds.npr.org/1013/rss.xml"),
    ("PBS NewsHour Education",   "https://www.pbs.org/newshour/feeds/rss/education"),
    ("ASCD SmartBrief",          "https://www.ascd.org/rss.xml"),
    ("ISTE",                     "https://www.iste.org/feed"),
    ("Common Sense Education",   "https://www.commonsense.org/education/articles/feed"),
    ("Learning Forward",         "https://learningforward.org/feed/"),
    ("EdWeek Market Brief",      "https://marketbrief.edweek.org/feed/"),
    ("Tech & Learning",          "https://www.techlearning.com/rss"),
    ("Campus Technology",        "https://campustechnology.com/rss-feeds/news.aspx"),
    ("Hechinger Report",         "https://hechingerreport.org/feed/"),
    ("Brookings Education",      "https://www.brookings.edu/topic/education/feed/"),
    ("Thomas B. Fordham Inst.",  "https://fordhaminstitute.org/national/feed"),
    ("Education Gadfly",         "https://fordhaminstitute.org/national/commentary/feed"),
    ("RAND Education",           "https://www.rand.org/topics/education.xml"),
    ("Education Next",           "https://www.educationnext.org/feed/"),
    ("EdSource",                 "https://edsource.org/feed"),
    ("EducationNC",              "https://www.ednc.org/feed/"),
    ("Digital Promise",          "https://digitalpromise.org/feed/"),
    ("SETDA",                    "https://www.setda.org/feed/"),
    ("CoSN",                     "https://www.cosn.org/feed/"),
    ("Project Tomorrow",         "https://tomorrow.org/feed/"),
    ("SchoolCEO",                "https://www.schoolceo.com/feed/"),
]

# District news pages (HTML scraping) — audited 2026-05-03 and kept only
# pages that returned 200, stayed on-domain, and exposed AI-related article links.
DISTRICT_NEWS_PAGES = [
    ("Fairfax County Public Schools",  "https://www.fcps.edu/news",                                    "https://www.fcps.edu"),
    ("Chicago Public Schools",         "https://www.cps.edu/news/",                                    "https://www.cps.edu"),
    ("NYC Department of Education",    "https://www.schools.nyc.gov/about-us/news/announcements",      "https://www.schools.nyc.gov"),
    ("Clark County School District",   "https://newsroom.ccsd.net",                                    "https://newsroom.ccsd.net"),
    ("Montgomery County Schools MD",   "https://www.montgomeryschoolsmd.org/news/",                    "https://www.montgomeryschoolsmd.org"),
    ("Broward County Schools",         "https://www.browardschools.com/news",                          "https://www.browardschools.com"),
    ("Wake County Schools",            "https://www.wcpss.net/news",                                   "https://www.wcpss.net"),
    ("Gwinnett County Schools",        "https://www.gcpsk12.org/get-connected/district-news-and-communication/all-news", "https://www.gcpsk12.org"),
    ("Miami-Dade County Schools",      "https://news.dadeschools.net",                                 "https://news.dadeschools.net"),
    ("Palm Beach County Schools",      "https://www.palmbeachschools.org/about-us/news-stories",       "https://www.palmbeachschools.org"),
    ("Dallas ISD",                     "https://www.dallasisd.org/news",                               "https://www.dallasisd.org"),
    ("Fort Worth ISD",                 "https://www.fwisd.org/news",                                   "https://www.fwisd.org"),
    ("Austin ISD",                     "https://www.austinisd.org/press-releases",                     "https://www.austinisd.org"),
    ("Aldine ISD",                     "https://www.aldineisd.org/news",                               "https://www.aldineisd.org"),
    ("Northside ISD San Antonio",      "https://www.nisd.net/news",                                    "https://www.nisd.net"),
    ("Jefferson County CO (JeffCo)",   "https://www.jeffcopublicschools.org/news",                     "https://www.jeffcopublicschools.org"),
    ("Aurora Public Schools CO",       "https://www.aurorak12.org/news",                               "https://www.aurorak12.org"),
    ("Detroit Public Schools",         "https://www.detroitk12.org/news",                              "https://www.detroitk12.org"),
    ("St. Paul Public Schools",        "https://www.spps.org/news",                                    "https://www.spps.org"),
    ("Guilford County Schools NC",     "https://www.gcsnc.com/news",                                   "https://www.gcsnc.com"),
    ("Greenville County Schools SC",   "https://www.greenville.k12.sc.us/news",                        "https://www.greenville.k12.sc.us"),
    ("Brevard County Schools FL",      "https://www.brevardschools.org/news",                          "https://www.brevardschools.org"),
    ("Pasco County Schools FL",        "https://www.pasco.k12.fl.us/news",                             "https://www.pasco.k12.fl.us"),
    ("Polk County Schools FL",         "https://www.polkschoolsfl.com/news",                           "https://www.polkschoolsfl.com"),
    ("Lee County Schools FL",          "https://www.leeschools.net/news",                              "https://www.leeschools.net"),
    ("Volusia County Schools FL",      "https://www.vcsedu.org/news",                                  "https://www.vcsedu.org"),
    ("Seminole County Schools FL",     "https://www.scps.k12.fl.us/news",                              "https://www.scps.k12.fl.us"),
    ("Prince William County VA",       "https://www.pwcs.edu/news",                                    "https://www.pwcs.edu"),
    ("Virginia Beach City Schools",    "https://www.vbschools.com/about/newsroom/pressreleases",       "https://www.vbschools.com"),
    ("Baltimore County Schools MD",    "https://www.bcps.org/news",                                    "https://www.bcps.org"),
    ("Houston ISD",                    "https://hisdnow.houstonisd.org/p/~board/district-news",        "https://hisdnow.houstonisd.org"),
    ("Orange County Public Schools",   "https://www.ocps.net/94816_3",                                 "https://www.ocps.net"),
    ("DeKalb County School District",  "https://www.dekalbschoolsga.org/",                             "https://www.dekalbschoolsga.org"),
    ("Charlotte-Mecklenburg Schools",  "https://www.cmsk12.org/",                                      "https://www.cmsk12.org"),
]

# State Department of Education news/press-release pages — audited 2026-05-03
# and kept only where the page returned 200, stayed on-domain, and exposed
# AI-related article links.
STATE_DOE_PAGES = [
    ("Alaska DOE",            "https://education.alaska.gov/news",                                          "https://education.alaska.gov"),
    ("California DOE",        "https://www.cde.ca.gov/nr/ne/yr24/",                                         "https://www.cde.ca.gov"),
    ("Connecticut DOE",       "https://portal.ct.gov/SDE/Press-Room/Press-Releases",                        "https://portal.ct.gov"),
    ("Florida DOE",           "https://www.fldoe.org/newsroom/latest-news/",                                "https://www.fldoe.org"),
    ("Hawaii DOE",            "https://www.hawaiipublicschools.org/ConnectWithUs/MediaRoom/Pages/Press-Releases.aspx", "https://www.hawaiipublicschools.org"),
    ("Indiana DOE",           "https://www.in.gov/doe/about/news/",                                         "https://www.in.gov"),
    ("Massachusetts DOE",     "https://www.doe.mass.edu/news/",                                             "https://www.doe.mass.edu"),
    ("Missouri DOE",          "https://dese.mo.gov/communications/news-releases",                           "https://dese.mo.gov"),
    ("Nevada DOE",            "https://doe.nv.gov/News__Media/Press_Releases/",                             "https://doe.nv.gov"),
    ("New Jersey DOE",        "https://www.nj.gov/education/news/",                                         "https://www.nj.gov/education"),
    ("New York DOE",          "https://www.nysed.gov/news",                                                 "https://www.nysed.gov"),
    ("North Carolina DOE",    "https://www.dpi.nc.gov/news",                                                "https://www.dpi.nc.gov"),
    ("Oklahoma DOE",          "https://oklahoma.gov/education/divisions/media/newsroom/2026/state-superintendent-releases-osde-2026-legislative-agenda-as-se.html", "https://oklahoma.gov"),
    ("South Carolina DOE",    "https://ed.sc.gov/newsroom/",                                                "https://ed.sc.gov"),
    ("Tennessee DOE",         "https://www.tn.gov/education/news/",                                         "https://www.tn.gov/education"),
    ("Texas DOE",             "https://tea.texas.gov/about-tea/news-and-multimedia/tea-news-releases",     "https://tea.texas.gov"),
    ("Vermont DOE",           "https://education.vermont.gov/news",                                         "https://education.vermont.gov"),
    ("Washington DOE",        "https://ospi.k12.wa.us/about-ospi/news-center/news-releases",               "https://ospi.k12.wa.us"),
    ("Wisconsin DOE",         "https://dpi.wi.gov/news",                                                    "https://dpi.wi.gov"),
    ("Wyoming DOE",           "https://edu.wyoming.gov/about/news/",                                        "https://edu.wyoming.gov"),
    ("DC OSSE",               "https://osse.dc.gov/newsroom",                                               "https://osse.dc.gov"),
]

AI_KEYWORDS = {
    "chatgpt", "artificial intelligence", " ai ", "generative ai",
    "magicschool", "khanmigo", "gemini", "copilot", "chatbot",
    "machine learning", "large language model", "ai tool", "ai tutor",
    "eduaide", "diffit", "schoolai", "brisk teaching", "curipod",
    "ai-powered", "ai-based", "ai assistant", "tutorbot", "chatgpt edu",
}

EDUCATOR_KEYWORDS = {
    "teacher", "principal", "superintendent", "librarian", "counselor",
    "classroom", "educator", "district", "school staff", "admin",
    "public schools", "school district", "board of education",
    "department of education", "instructional coach", "school psychologist",
    "special education", "curriculum director",
}


def _make_id(url: str) -> str:
    return hashlib.sha1((url or "").strip().encode("utf-8")).hexdigest()


def _is_ai_relevant(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in AI_KEYWORDS)


def _is_educator_relevant(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in EDUCATOR_KEYWORDS)


# Patterns that signal a specific educator ACTUALLY USING AI (not just opinion/policy)
_USE_CASE_PATTERNS = [
    re.compile(r'[A-Z][a-z]+\s+[A-Z][a-z]+[,\s]+(a |an |the )?(\w+ )?(teacher|principal|librarian|counselor|coach|educator|administrator)', re.I),
    re.compile(r'(teacher|educator|principal|librarian|counselor|coach)\s+\w+\s+(uses?|used|using)\s+(AI|ChatGPT|Gemini|Claude|MagicSchool|Khanmigo)', re.I),
    re.compile(r'\bI\s+(use|used|ask|asked|have been using)\s+(AI|ChatGPT|Gemini|Claude|MagicSchool|Copilot)', re.I),
    re.compile(r'(uses?|used|using)\s+(ChatGPT|MagicSchool|Khanmigo|Gemini|Claude|Copilot|Eduaide|Diffit|SchoolAI)\s+(to|for)\s+(create|generate|plan|grade|write|make|build|draft|differentiat)', re.I),
    re.compile(r'(lesson plan|grading|feedback|differentiat|quiz|assessment|rubric).{0,60}(ChatGPT|AI tool|MagicSchool|generative AI)', re.I),
]


def _is_potential_use_case(text: str) -> bool:
    """Heuristic: does this article likely contain a specific K-12 educator using AI?"""
    if not text or len(text) < 200:
        return False
    if not (_is_ai_relevant(text) and _is_educator_relevant(text)):
        return False
    return any(p.search(text) for p in _USE_CASE_PATTERNS)


def _fetch_full_content(url: str, max_chars: int = 4000) -> str:
    """Fetch and extract main article body text from URL."""
    try:
        resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        # Try common article body selectors
        body = (
            soup.find("article") or
            soup.find(class_=re.compile(
                r"article[-_]?(body|content|text)|post[-_]?content|entry[-_]?content|story[-_]?body",
                re.I
            )) or
            soup.find("main")
        )
        text = (body or soup).get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""


def _parse_date(raw: str) -> str:
    """Convert RFC 2822 or ISO 8601 date string to BQ-compatible ISO format."""
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        pass
    # Already ISO-ish (Atom format)
    try:
        raw = raw.rstrip("Z").split("+")[0]
        dt = datetime.strptime(raw[:19], "%Y-%m-%dT%H:%M:%S")
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def _parse_rss(xml_bytes: bytes, source_name: str) -> list:
    """Parse RSS 2.0 or Atom feed into article dicts using BeautifulSoup."""
    articles = []
    try:
        soup = BeautifulSoup(xml_bytes, "lxml-xml")
    except Exception:
        return articles

    # RSS 2.0 items
    for item in soup.find_all("item"):
        title = item.find("title")
        title = title.get_text("") if title else ""
        link  = item.find("link")
        link  = link.get_text("") if link else ""
        if not link:
            guid = item.find("guid")
            link = guid.get_text("") if guid else ""
        desc  = item.find("description")
        desc  = desc.get_text("") if desc else ""
        pub   = item.find("pubDate")
        pub   = pub.get_text("") if pub else ""

        if not link or not title:
            continue
        combined = title + " " + desc
        if not _is_ai_relevant(combined):
            continue
        articles.append({"title": title.strip(), "url": link.strip(),
                          "description": desc.strip(),
                          "publishedAt": _parse_date(pub),
                          "_source": source_name})

    # Atom entries (if no RSS items found)
    if not articles:
        for entry in soup.find_all("entry"):
            title = entry.find("title")
            title = title.get_text("") if title else ""
            link_el = entry.find("link")
            link  = link_el.get("href", "") if link_el else ""
            desc  = entry.find("summary") or entry.find("content")
            desc  = desc.get_text("") if desc else ""
            pub   = entry.find("published") or entry.find("updated")
            pub   = pub.get_text("") if pub else ""

            if not link or not title:
                continue
            combined = title + " " + desc
            if not _is_ai_relevant(combined):
                continue
            articles.append({"title": title.strip(), "url": link.strip(),
                              "description": desc.strip(),
                              "publishedAt": _parse_date(pub),
                              "_source": source_name})

    return articles


def fetch_from_education_rss() -> list:
    """Scrape education-focused RSS feeds for K-12 AI educator articles."""
    all_articles = []

    for source_name, feed_url in EDUCATION_RSS_FEEDS:
        try:
            resp = requests.get(feed_url, headers=SCRAPE_HEADERS, timeout=15)
            resp.raise_for_status()
            items = _parse_rss(resp.content, source_name)
            print(f"RSS {source_name}: {len(items)} relevant articles")
            all_articles.extend(items)
            time.sleep(0.5)
        except Exception as e:
            print(f"RSS error ({source_name}): {e}")

    # Fetch full content for matched articles
    now = datetime.utcnow().isoformat()
    results = []
    seen = set()
    for a in all_articles:
        url = a["url"].strip()
        if not url or url in seen:
            continue
        seen.add(url)
        full = _fetch_full_content(url)
        content = full or a["description"]
        combined = " ".join(filter(None, [a["title"], a["description"], content]))
        if not (_is_ai_relevant(combined) and _is_educator_relevant(combined)):
            continue
        use_case_tag = ":UseCase" if _is_potential_use_case(content) else ""
        results.append({
            "article_id":       _make_id(url),
            "source_id":        None,
            "source_name":      a["_source"],
            "author":           None,
            "title":            a["title"],
            "description":      a["description"],
            "url":              url,
            "urlToImage":       None,
            "publishedAt":      a.get("publishedAt") or now,
            "content":          content,
            "collected_at":     now,
            "raw_json":         json.dumps(a),
            "_discovery_tool":  "EducationRSS" + use_case_tag,
            "_snippet":         a["description"],
        })
        time.sleep(0.3)

    print(f"Education RSS total: {len(results)} articles")
    return results


def _scrape_site_list(site_list: list, tool_label: str) -> list:
    """Shared scraping logic for district and state DOE site lists."""
    all_articles = []
    now = datetime.utcnow().isoformat()

    for site_name, news_url, base_url in site_list:
        try:
            resp = requests.get(news_url, headers=SCRAPE_HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "lxml")

            seen = set()
            for a_tag in soup.find_all("a", href=True):
                href = a_tag.get("href", "").strip()
                text = a_tag.get_text(strip=True)

                # Resolve relative URLs
                full_url = urljoin(base_url, href)

                # Only follow same-domain links
                parsed = urlparse(full_url)
                base_parsed = urlparse(base_url)
                if parsed.netloc != base_parsed.netloc:
                    continue

                # Skip nav/footer boilerplate
                if len(text) < 20 or full_url in seen:
                    continue

                combined = text + " " + full_url
                if not _is_ai_relevant(combined):
                    continue

                seen.add(full_url)
                full_content = _fetch_full_content(full_url)
                if not full_content:
                    continue

                # Second pass: check full content is actually about educators
                if not (_is_ai_relevant(full_content) and _is_educator_relevant(full_content)):
                    continue

                use_case_tag = ":UseCase" if _is_potential_use_case(full_content) else ""
                all_articles.append({
                    "article_id":      _make_id(full_url),
                    "source_id":       None,
                    "source_name":     site_name,
                    "author":          None,
                    "title":           text[:200],
                    "description":     full_content[:500],
                    "url":             full_url,
                    "urlToImage":      None,
                    "publishedAt":     now,
                    "content":         full_content,
                    "collected_at":    now,
                    "raw_json":        json.dumps({"source": site_name, "url": full_url}),
                    "_discovery_tool": tool_label + use_case_tag,
                    "_snippet":        full_content[:300],
                })
                time.sleep(0.5)

            print(f"{tool_label} {site_name}: {len(seen)} AI links found")
        except Exception as e:
            print(f"{tool_label} scrape error ({site_name}): {e}")

    print(f"{tool_label} total: {len(all_articles)} articles")
    return all_articles


def fetch_from_district_sites() -> list:
    """Scrape major US school district news pages for AI-related articles."""
    return _scrape_site_list(DISTRICT_NEWS_PAGES, "DistrictScraper")


def fetch_from_doe_sites() -> list:
    """Scrape all 50 state Department of Education news pages for AI-related articles."""
    return _scrape_site_list(STATE_DOE_PAGES, "StateDOEScraper")


@app.route("/", methods=["POST", "GET"])
def run_collection():
    data = request.get_json(silent=True) or {}
    days_back = int(data.get("days_back", 30))

    serp_arts  = fetch_from_serpapi(days_back)
    gnews_arts = fetch_from_gnews(days_back)
    all_arts   = serp_arts + gnews_arts

    inserted_raw, inserted_sources = process_and_save(all_arts)

    return jsonify({
        "status":                 "success",
        "serpapi_count":          len(serp_arts),
        "gnews_count":            len(gnews_arts),
        "total_collected":        len(all_arts),
        "inserted_raw":           inserted_raw,
        "new_sources_discovered": inserted_sources,
        "timestamp":              datetime.utcnow().isoformat() + "Z",
    }), 200


@app.route("/collect-education", methods=["POST", "GET"])
def collect_education_sites():
    """Scrape education RSS feeds, district sites, and state DOE pages for K-12 AI articles."""
    data = request.get_json(silent=True) or {}
    sources = data.get("sources", "all")  # "rss", "district", "doe", or "all"

    rss_arts      = fetch_from_education_rss()      if sources in ("all", "rss")      else []
    district_arts = fetch_from_district_sites()     if sources in ("all", "district") else []
    doe_arts      = fetch_from_doe_sites()          if sources in ("all", "doe")      else []
    all_arts      = rss_arts + district_arts + doe_arts

    inserted_raw, inserted_sources = process_and_save(all_arts)

    return jsonify({
        "status":                 "success",
        "rss_count":              len(rss_arts),
        "district_count":         len(district_arts),
        "doe_count":              len(doe_arts),
        "total_collected":        len(all_arts),
        "inserted_raw":           inserted_raw,
        "new_sources_discovered": inserted_sources,
        "timestamp":              datetime.utcnow().isoformat() + "Z",
    }), 200


SOFT_PAYWALL_DOMAINS = [
    "nytimes.com", "washingtonpost.com", "wsj.com",
    "ft.com", "bloomberg.com", "theatlantic.com",
]


def _scrape_url_trafilatura(url: str) -> tuple:
    """
    Returns (status, content).
    status: 'success' | 'paywall' | 'failed' | 'timeout'
    Uses trafilatura for accurate main-body extraction.
    """
    try:
        resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=12, allow_redirects=True)

        if resp.status_code in (401, 403, 429):
            return "paywall", ""
        if resp.status_code == 404:
            return "failed", ""
        if resp.status_code != 200:
            return "failed", ""

        if any(d in resp.url for d in SOFT_PAYWALL_DOMAINS):
            return "paywall", ""

        text = trafilatura.extract(
            resp.text,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_recall=True,
        )
        if not text or len(text) < 100:
            return "failed", ""
        return "success", text

    except requests.Timeout:
        return "timeout", ""
    except Exception:
        return "failed", ""


@app.route("/fill-content", methods=["POST", "GET"])
def fill_content():
    """
    Scrape full article body for raw_articles that have no entry in raw_articles_content yet.
    Uses trafilatura for accurate main-body extraction.
    POST body (optional): { "batch": 500 }
    """
    data = request.get_json(silent=True) or {}
    batch_size = int(data.get("batch", 500))

    content_table = f"{GCP_PROJECT}.{DATASET}.raw_articles_content"

    rows = list(bq_client.query(f"""
        SELECT r.article_id, r.url
        FROM `{GCP_PROJECT}.{DATASET}.{RAW_TABLE}` r
        LEFT JOIN `{GCP_PROJECT}.{DATASET}.raw_articles_content` rc ON r.article_id = rc.article_id
        WHERE rc.article_id IS NULL
          AND r.url IS NOT NULL AND r.url != ''
        LIMIT {batch_size}
    """).result())

    print(f"fill-content: {len(rows)} articles to scrape")
    if not rows:
        return jsonify({"status": "success", "message": "Nothing to scrape", "scraped": 0}), 200

    results = []
    counts = {"success": 0, "paywall": 0, "failed": 0, "timeout": 0}
    now = datetime.utcnow().isoformat()

    for row in rows:
        status, text = _scrape_url_trafilatura(row.url)
        counts[status] = counts.get(status, 0) + 1
        results.append({
            "article_id":   row.article_id,
            "content_full": text if status == "success" else None,
            "scraped_at":   now,
            "status":       status,
            "char_count":   len(text) if text else 0,
        })
        time.sleep(0.3)

    inserted = bq_insert_batch(content_table, results)
    print(f"fill-content done: {counts}, inserted={inserted}")

    return jsonify({
        "status":   "success",
        "scraped":  len(rows),
        "inserted": inserted,
        **counts,
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
