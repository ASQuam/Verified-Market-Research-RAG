"""
Team 2 — Agentic Market Research (Final App, Week 8)

Full pipeline:
    Question -> Supervisor -> 3 Specialists (parallel) -> Shared Evidence Store
    -> Gap/Conflict Check -> (at most ONE extra research pass)
    -> Synthesis Agent -> Reviewer (at most ONE revision) -> Final Report

Run (from the folder that holds this file and your .env):
    python -m pip install -r requirements_team2.txt
    streamlit run team2_final_app.py

.env:
    TAVILY_API_KEY=your_key
    DEEPSEEK_API_KEY=your_key
"""

# ======================================================================
# IMPORTS
# ======================================================================
import json                     # Encode prompts as JSON and parse model replies
import os                       # Read API keys from environment variables
import re                       # Pull dates out of URLs (arXiv IDs, /YYYY/MM/ paths)
import sqlite3                  # Local database for evidence records and saved reports
import time                     # Time each pipeline step for the live progress display
from concurrent.futures import ThreadPoolExecutor, as_completed  # Run agents and extractions in parallel
from datetime import date, datetime, timezone  # Report timestamps and date validation
from email.utils import parsedate_to_datetime  # Parse Tavily dates like "Tue, 11 Mar 2025 17:00:00 GMT"
from typing import TypedDict    # Typed state schemas for the LangGraph graph
from urllib.parse import urlparse  # Read the domain of each source URL

import streamlit as st                          # Web UI framework
from dotenv import load_dotenv                  # Loads API keys from the .env file
from langgraph.graph import END, START, StateGraph  # Builds the multi-agent workflow graph
from openai import OpenAI                       # OpenAI-compatible client, pointed at DeepSeek
from tavily import TavilyClient                 # Web search API used by the specialists


# ======================================================================
# PAGE SETUP AND CONFIGURATION
# ======================================================================
# Must be the first Streamlit call on the page (sets the browser tab title and icon)
st.set_page_config(page_title="Team 2 — Agentic Research", page_icon="🧩")

load_dotenv()  # Pull TAVILY_API_KEY and DEEPSEEK_API_KEY from .env into the environment

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")      # Key for web search
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # Key for the language model

# Stop the app with a clear message if either key is missing
if not TAVILY_API_KEY or not DEEPSEEK_API_KEY:
    st.error(
        "Missing API key. Add TAVILY_API_KEY and DEEPSEEK_API_KEY "
        "to a .env file beside this script."
    )
    st.stop()

tavily = TavilyClient(api_key=TAVILY_API_KEY)  # Search client shared by all specialists
deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")  # Model client for every agent

EVIDENCE_DB = "team2_evidence.db"  # SQLite file holding the evidence and reports tables


# ======================================================================
# SHARED GRAPH STATE
# Each class extends the previous one, so the final graph carries every field.
# ======================================================================
class ResearchState(TypedDict):
    question: str                     # The user's research question
    assignments: list[dict]           # Supervisor's three specialist tasks
    specialist_results: list[dict]    # Each specialist's brief plus its raw Tavily sources


class Week4ResearchState(ResearchState):
    evidence_records: list[dict]      # Validated, structured evidence extracted from sources


class Week5ResearchState(Week4ResearchState):
    report: dict                      # The nine model-written report sections
    report_markdown: str              # Draft report rendered as markdown
    report_id: int                    # Row id of the saved draft in SQLite


class Week6ResearchState(Week5ResearchState):
    review: dict                      # Final reviewer verdict
    initial_review: dict              # First verdict, before any revision
    final_report_markdown: str        # Report rendered after review/revision
    revision_performed: bool          # Whether the one allowed revision ran


class Week7ResearchState(Week6ResearchState):
    gap_check: dict                       # Result of the gap/conflict check
    additional_research_performed: bool   # Whether the one extra research pass ran
    new_evidence_count: int               # Genuinely new records added by that pass


# ======================================================================
# MODEL HELPER
# ======================================================================
def call_model_json(system_prompt: str, user_text: str) -> dict:
    """Send one prompt to the model and return its reply parsed as JSON."""
    response = deepseek.chat.completions.create(
        model="deepseek-flash",                            # Model used by every agent
        messages=[
            {"role": "system", "content": system_prompt},  # The agent's role and rules
            {"role": "user", "content": user_text},        # The actual task input
        ],
        response_format={"type": "json_object"},           # Force the reply to be valid JSON
    )
    return json.loads(response.choices[0].message.content)  # Parse the JSON text into a dict


# ======================================================================
# STAGE 1 — SUPERVISOR
# Splits the question into three non-overlapping specialist assignments.
# ======================================================================
# Instructions for the supervisor: exactly three roles, strict JSON shape
TEAM_2_SYSTEM_PROMPT = """
You are the Supervisor for a multi-agent competitive-intelligence research system.

Divide the user's business research question across exactly three specialist roles:
Competitor, Market, and Tech/Regulatory.

Each specialist must receive a distinct, non-overlapping objective. Include a short
output_fields list describing what that specialist should return.

Each objective is used directly as a web search query against research papers and
formal reports only (peer-reviewed journals, conference papers, preprints, government
and regulatory documents, SEC filings, intergovernmental reports, research institutes,
and industry analyst reports). Write each objective as a concise, search-ready research
query under 200 characters, using the terms such documents would use. Do not phrase
objectives in ways that would match news, blogs, or social media.

Return only valid JSON in exactly this shape:
{
  "tasks": [
    {
      "agent": "Competitor",
      "objective": "...",
      "output_fields": ["...", "..."]
    },
    {
      "agent": "Market",
      "objective": "...",
      "output_fields": ["...", "..."]
    },
    {
      "agent": "Tech/Regulatory",
      "objective": "...",
      "output_fields": ["...", "..."]
    }
  ]
}

Use exactly these three agent names.
Do not include markdown or text outside the JSON object.
""".strip()


def supervisor_node(state: ResearchState) -> dict:
    """Ask the supervisor for three assignments and validate them."""
    output = call_model_json(TEAM_2_SYSTEM_PROMPT, state["question"])  # Ask the supervisor to plan
    tasks = output.get("tasks")  # The list of assignments it returned

    # There must be exactly three assignments
    if not isinstance(tasks, list) or len(tasks) != 3:
        raise ValueError("Supervisor must return exactly three assignments.")

    expected_agents = {"Competitor", "Market", "Tech/Regulatory"}  # The only allowed roles
    seen_agents = set()  # Roles already assigned, to catch duplicates
    cleaned = []         # Validated assignments go here

    for assignment in tasks:
        # Each assignment must be a JSON object
        if not isinstance(assignment, dict):
            raise ValueError("Each assignment must be a JSON object.")

        agent = str(assignment.get("agent", "")).strip()          # Specialist role name
        objective = str(assignment.get("objective", "")).strip()  # What that specialist researches
        output_fields = assignment.get("output_fields", [])      # What it should report back

        # Reject unknown roles, empty objectives, or a malformed field list
        if agent not in expected_agents:
            raise ValueError(f"Unexpected specialist role: {agent}")
        if not objective:
            raise ValueError(f"{agent} needs a non-empty objective.")
        if not isinstance(output_fields, list):
            raise ValueError(f"{agent} output_fields must be a list.")

        seen_agents.add(agent)  # Record that this role has been assigned
        cleaned.append(
            {
                "agent": agent,
                "objective": objective,
                "output_fields": [str(field) for field in output_fields],  # Force every field name to text
            }
        )

    # Every role must appear exactly once
    if seen_agents != expected_agents:
        raise ValueError(
            "Assignments must include Competitor, Market, and Tech/Regulatory exactly once."
        )

    return {"assignments": cleaned}  # Write the assignments into graph state


# ======================================================================
# SOURCE POLICY — research papers and formal reports only
# One allowlist is used twice: Tavily searches only these domains, and every
# result is re-checked in Python, so nothing else can reach the report.
# Edit this list to add or remove trusted publishers.
# ======================================================================
APPROVED_DOMAINS = [
    # Academic journals, publishers, conference proceedings, and paper repositories
    "arxiv.org", "nature.com", "science.org", "sciencedirect.com", "cell.com", "pnas.org",
    "springer.com", "wiley.com", "tandfonline.com", "sagepub.com", "academic.oup.com",
    "cambridge.org", "jstor.org", "ieee.org", "acm.org", "usenix.org", "openreview.net",
    "aclanthology.org", "proceedings.neurips.cc", "proceedings.mlr.press", "ssrn.com",
    "nber.org", "ncbi.nlm.nih.gov", "semanticscholar.org", "informs.org", "aeaweb.org",
    # Universities and university research centers
    "mit.edu", "stanford.edu", "harvard.edu", "berkeley.edu", "cmu.edu", "princeton.edu",
    "georgetown.edu", "ox.ac.uk", "cam.ac.uk", "psu.edu",
    # U.S. government and regulators (SEC filings live on sec.gov)
    "sec.gov", "ftc.gov", "justice.gov", "commerce.gov", "bis.gov", "bis.doc.gov", "nist.gov",
    "energy.gov", "eia.gov", "census.gov", "bls.gov", "bea.gov", "gao.gov", "cbo.gov",
    "congress.gov", "federalregister.gov", "whitehouse.gov", "usitc.gov", "ustr.gov",
    "uspto.gov", "fcc.gov", "federalreserve.gov", "treasury.gov", "nsf.gov",
    # International governments and intergovernmental organizations
    "europa.eu", "gov.uk", "oecd.org", "imf.org", "worldbank.org", "wto.org", "un.org",
    "itu.int", "iea.org", "bis.org", "wipo.int", "meti.go.jp",
    # Research institutes and industry associations that publish formal reports
    "brookings.edu", "rand.org", "csis.org", "cfr.org", "piie.com", "pewresearch.org",
    "itif.org", "carnegieendowment.org", "belfercenter.org", "epoch.ai", "epochai.org",
    "semiconductors.org", "semi.org",
    # Industry analyst and consulting research firms
    "mckinsey.com", "bcg.com", "bain.com", "deloitte.com", "pwc.com", "kpmg.com", "ey.com",
    "accenture.com", "gartner.com", "idc.com", "forrester.com", "trendforce.com",
    "counterpointresearch.com", "canalys.com", "omdia.tech.informa.com", "yolegroup.com",
    "techinsights.com", "spglobal.com", "moodys.com",
]

# Hosts on approved domains that publish news or marketing rather than research
BLOCKED_HOSTS = {"spectrum.ieee.org"}
BLOCKED_HOST_PREFIXES = ("news.", "blog.", "blogs.", "press.", "newsroom.", "careers.", "jobs.")

# The only source types allowed into the evidence store (the extractor must pick one)
ALLOWED_SOURCE_TYPES = {
    "peer_reviewed_paper",
    "conference_paper",
    "preprint",
    "working_paper",
    "government_or_regulatory",
    "regulatory_filing",
    "intergovernmental_report",
    "research_institute_report",
    "industry_analyst_report",
}


def is_approved_source(url: str) -> bool:
    """True only if the URL's host is on (or a subdomain of) an approved domain and isn't blocked."""
    host = (urlparse(url or "").hostname or "").lower()  # e.g. "www.nature.com"
    if host.startswith("www."):
        host = host[4:]                                   # Treat www.nature.com as nature.com
    if not host or host in BLOCKED_HOSTS or host.startswith(BLOCKED_HOST_PREFIXES):
        return False
    # Exact domain or any subdomain of it (link.springer.com matches springer.com)
    return any(host == domain or host.endswith("." + domain) for domain in APPROVED_DOMAINS)


def normalize_date(value) -> str | None:
    """Return a validated YYYY-MM-DD, YYYY-MM, or YYYY string, or None if not a real past date."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a", "unknown"}:
        return None

    today = date.today()

    def _check(year: int, month: int | None = None, day: int | None = None) -> str | None:
        # Reject impossible or implausible dates (before 1990 or in the future)
        try:
            d = date(year, month or 1, day or 1)
        except ValueError:
            return None
        if year < 1990 or d > today:
            return None
        if day:
            return d.strftime("%Y-%m-%d")
        if month:
            return d.strftime("%Y-%m")
        return str(year)

    # RFC 2822 style, which Tavily returns: "Tue, 11 Mar 2025 17:00:00 GMT"
    if "," in text and ":" in text:
        try:
            parsed = parsedate_to_datetime(text)
            return _check(parsed.year, parsed.month, parsed.day)
        except (TypeError, ValueError, IndexError, OverflowError):
            pass

    # ISO style: 2025-03-11, 2025-03-11T17:00:00Z, 2025-03, or 2025
    match = re.match(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?(?:$|[T\s])", text)
    if match:
        year, month, day = match.groups()
        return _check(int(year), int(month) if month else None, int(day) if day else None)

    # Written-out dates the model might return: "March 11, 2025", "11 March 2025", "March 2025"
    for fmt, has_day in (("%B %d, %Y", True), ("%d %B %Y", True), ("%b %d, %Y", True),
                         ("%d %b %Y", True), ("%B %Y", False), ("%b %Y", False)):
        try:
            parsed = datetime.strptime(text, fmt)
            return _check(parsed.year, parsed.month, parsed.day if has_day else None)
        except ValueError:
            continue
    return None


# arXiv IDs encode the submission year and month: arxiv.org/abs/2405.12345 = May 2024
_ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{2})(\d{2})\.\d{4,5}")
# Dated URL paths: /2024/05/12/ or /2024/05/
_URL_DATE = re.compile(r"/((?:19|20)\d{2})[/-](0[1-9]|1[0-2])(?:[/-](0[1-9]|[12]\d|3[01]))?(?=[/-]|$)")


def resolve_publication_date(source: dict, model_dates: list) -> str | None:
    """Pick the most reliable publication date for one source, in order of trust."""
    url = source.get("url") or ""

    # 1. arXiv ID: exact submission month, always correct for arXiv papers
    match = _ARXIV_ID.search(url)
    if match:
        found = normalize_date(f"20{match.group(1)}-{match.group(2)}")
        if found:
            return found

    # 2. A date written into the URL path, which publishers set at publication
    match = _URL_DATE.search(url)
    if match:
        found = normalize_date("-".join(part for part in match.groups() if part))
        if found:
            return found

    # 3. A date the extractor found stated explicitly in the document text
    for candidate in model_dates:
        found = normalize_date(candidate)
        if found:
            return found

    # 4. Tavily's estimate, which may be a last-updated date, so it's labeled as an estimate
    found = normalize_date(source.get("published_date"))
    return f"{found} (est.)" if found else None


def search_approved_sources(query: str) -> list[dict]:
    """Search ONLY approved research domains and drop anything that slips through."""
    params = {
        "query": query[:400],                  # Tavily rejects queries over 400 characters
        "search_depth": "advanced",            # Higher relevance, which matters with a narrow allowlist
        "max_results": 5,
        "include_domains": APPROVED_DOMAINS,   # Hard filter: only these domains are searched
        "include_raw_content": "text",         # Full page text, so the publication date can be read
    }
    try:
        # include_published_date asks Tavily for its own date estimate on every result
        response = tavily.search(**params, include_published_date=True)
    except TypeError:
        # Older tavily-python versions don't accept this option; search without it
        response = tavily.search(**params)
    results = response.get("results", []) or []
    # Second check in Python, so a filter change on Tavily's side can't let other sources in
    return [r for r in results if is_approved_source(r.get("url", ""))]


# Added to every specialist's instructions
SOURCE_POLICY_NOTE = """
The supplied sources are restricted to research papers and formal published reports.
Rely only on them. Say when a finding comes from a preprint (not yet peer reviewed)
or from an industry analyst estimate. If the sources don't cover something, say so.
""".strip()


# ======================================================================
# STAGE 2 — SPECIALISTS (run in parallel)
# Each specialist searches the web for its objective and writes a brief.
# ======================================================================
# Role-specific instructions so each specialist stays in its own lane
SPECIALIST_PROMPTS = {
    "Competitor": """
You are the Competitor specialist. Research competitors, products, positioning,
partnerships, and differentiators relevant to your assigned objective.
Stay within your assignment. Return a concise research brief grounded in the
provided Tavily results. Do not perform the Market or Tech/Regulatory agent's job.
""".strip(),
    "Market": """
You are the Market specialist. Research market direction, customers, adoption,
industry dynamics, and business context relevant to your assigned objective.
Stay within your assignment. Return a concise research brief grounded in the
provided Tavily results. Do not perform the Competitor or Tech/Regulatory agent's job.
""".strip(),
    "Tech/Regulatory": """
You are the Tech/Regulatory specialist. Research technology developments,
technical capabilities, emerging technology, regulation, and official policy
context relevant to your assigned objective.
Stay within your assignment. Return a concise research brief grounded in the
provided Tavily results. Do not perform the Competitor or Market agent's job.
""".strip(),
}


def research_assignment(assignment: dict, original_question: str) -> dict:
    """Execute ONE specialist's assignment: search, then summarize."""
    agent = assignment["agent"]                  # Which specialist this is
    objective = assignment["objective"]          # Its research objective
    output_fields = assignment["output_fields"]  # What it should report

    # Search approved research domains only, using the specialist's objective as the query
    tavily_results = search_approved_sources(objective)  # List of source dicts (title, url, content, ...)

    # No approved sources: report that honestly instead of letting the model write from nothing
    if not tavily_results:
        return {
            "agent": agent,
            "objective": objective,
            "output_fields": output_fields,
            "research": "No research papers or formal reports were found for this objective.",
            "sources": [],
        }

    # Turn the search results into readable numbered text for the model
    sources_text = ""
    for idx, src in enumerate(tavily_results, 1):
        title = src.get("title", "No Title")
        url = src.get("url", "No URL")
        content = src.get("content", "")
        sources_text += f"Source [{idx}]: {title}\nURL: {url}\nContent: {content}\n\n"

    # Prompt combining the question, the objective, and the search results
    prompt_text = f"""
    Original User Question: {original_question}
    Your Assigned Objective: {objective}
    Requested Output Fields: {', '.join(output_fields)} Tavily Research Results:
    {sources_text} Please provide a clear and concise research brief based on the sources provided above.
    """

    # This specialist's instructions plus the shared source policy
    system_prompt = SPECIALIST_PROMPTS.get(agent, "") + "\n\n" + SOURCE_POLICY_NOTE
    model_response = deepseek.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ],
    )
    research_text = model_response.choices[0].message.content  # The specialist's written brief

    # Return the brief plus the raw sources (the evidence stage reads the sources)
    return {
        "agent": agent,
        "objective": objective,
        "output_fields": output_fields,
        "research": research_text,
        "sources": tavily_results,
    }


def specialists_node(state: ResearchState) -> dict:
    """Run all three specialist assignments at the same time."""
    specialist_results = []  # Collected results from every specialist

    # One thread per specialist so all three search and write simultaneously
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(research_assignment, assignment, state["question"])
            for assignment in state["assignments"]
        ]

        # Collect each result as soon as it finishes
        for future in as_completed(futures):
            try:
                specialist_results.append(future.result())
            except Exception as e:
                # One failed specialist shouldn't kill the whole run
                st.error(f"Error during parallel specialist execution: {e}")

    return {"specialist_results": specialist_results}  # Write results into graph state


# ======================================================================
# STAGE 3 — SHARED EVIDENCE STORE
# Turns raw sources into structured, validated evidence and saves it to SQLite.
# ======================================================================
# Columns every evidence record has (same order as the database table)
EVIDENCE_FIELDS = [
    "claim",
    "source",
    "url",
    "publication_date",
    "source_type",
    "evidence",
    "confidence",
    "related_company",
    "report_section",
]


def init_evidence_db():
    """Create the evidence table if it doesn't exist yet."""
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim TEXT NOT NULL,
                source TEXT NOT NULL,
                url TEXT NOT NULL,
                publication_date TEXT,
                source_type TEXT NOT NULL,
                evidence TEXT NOT NULL,
                confidence TEXT,
                related_company TEXT,
                report_section TEXT,
                UNIQUE(claim, url)
            )
        """)  # UNIQUE(claim, url) blocks the same fact from the same page being stored twice
        conn.commit()


def save_evidence_records(records: list[dict]):
    """Save validated evidence to SQLite, skipping duplicates."""
    if not records:  # Nothing to save
        return

    columns = ", ".join(EVIDENCE_FIELDS)                             # "claim, source, url, ..."
    placeholders = ", ".join(f":{field}" for field in EVIDENCE_FIELDS)  # ":claim, :source, ..." named slots
    sql = f"INSERT OR IGNORE INTO evidence ({columns}) VALUES ({placeholders})"  # IGNORE = skip duplicates

    # Build one row per record with exactly the expected fields
    rows = [{field: record.get(field) for field in EVIDENCE_FIELDS} for record in records]

    conn = sqlite3.connect(EVIDENCE_DB)
    try:
        with conn:                        # Commits all inserts together
            conn.executemany(sql, rows)   # Insert every row in one batch
    finally:
        conn.close()                      # Always close the connection


init_evidence_db()  # Make sure the evidence table exists before anything is saved


# Instructions for turning one source into structured evidence records.
# The extractor also acts as a second gate: non-research pages yield no records.
EVIDENCE_SYSTEM_PROMPT = """
You are the evidence extraction step in a professional business research workflow.
Only research papers and formal published reports may be used as evidence.

Return ONLY valid JSON in exactly this shape:
{
  "records": [
    {
      "claim": "...",
      "source": "...",
      "url": "...",
      "publication_date": null,
      "source_type": "peer_reviewed_paper",
      "evidence": "...",
      "confidence": "high",
      "related_company": null,
      "report_section": null
    }
  ]
}

Step 1 — Decide whether the page is eligible.
Eligible: the page itself is a research paper (journal article, conference paper,
preprint, or working paper) or a formal published report (government or regulatory
document, regulatory filing such as a 10-K, intergovernmental report, research-institute
report, or industry analyst / consulting research report).
Not eligible: news articles, blog posts, press releases, marketing or product pages,
social media posts, forum threads, videos, event pages, job listings, author profiles,
or landing/index pages that only describe or link to a report.
If the page is not eligible, return {"records": []}.

Step 2 — For an eligible page, extract records using these rules:
- claim = a factual statement directly supported by the supplied text.
- evidence = the passage from the supplied text that supports the claim, quoted closely.
- source = "<publishing organization or journal>: <document title>".
- url = exactly the supplied url.
- publication_date = the document's own publication date, ONLY if it is explicitly
  stated in the supplied page_text or snippet (for example "Published 12 March 2024",
  a dated report cover, a journal issue date, or a filing date). Use YYYY-MM-DD, or
  YYYY-MM or YYYY if only that much is stated. Never use the date of an event the
  document describes, the date of a cited work, or a copyright footer year.
  If no publication date is explicitly stated, use null.
- source_type = exactly one of: peer_reviewed_paper, conference_paper, preprint,
  working_paper, government_or_regulatory, regulatory_filing, intergovernmental_report,
  research_institute_report, industry_analyst_report.
- confidence = high, medium, or low.
- Do not turn analysis, forecasts, or recommendations into facts; state forecasts
  as forecasts attributed to the source.
""".strip()


def extract_from_source(agent: str, objective: str, source: dict) -> list[dict]:
    """Extract evidence records from ONE approved source and attach a verified date."""
    # Everything the extractor needs about this one source
    source_payload = {
        "agent": agent,
        "objective": objective,
        "title": source.get("title"),
        "url": source.get("url"),
        "snippet": (source.get("content") or "")[:3000],       # Tavily's most relevant passages
        "page_text": (source.get("raw_content") or "")[:6000],  # Start of the full page, where dates usually appear
    }
    prompt = json.dumps(source_payload, ensure_ascii=False)  # Send it as JSON

    try:
        result = call_model_json(EVIDENCE_SYSTEM_PROMPT, prompt)
    except Exception:
        return []  # Skip a bad source instead of failing the whole run

    if not isinstance(result, dict):  # Unusable reply
        return []

    # Keep only well-formed record objects
    records = [r for r in result.get("records", []) if isinstance(r, dict)]

    # One date for the whole source, chosen in Python from the most reliable signal
    publication_date = resolve_publication_date(source, [r.get("publication_date") for r in records])

    for record in records:
        record["url"] = source.get("url")             # Always the real search URL, never a model-written one
        record["publication_date"] = publication_date  # Verified date, or None if none could be found

    return records


def extract_specialist_evidence(specialist_results: list[dict]) -> list[dict]:
    """Extract evidence from every source of every specialist, in parallel."""
    records = []  # All extracted records

    # One extraction call per source, up to 8 running at once
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                extract_from_source,
                specialist.get("agent"),
                specialist.get("objective"),
                source,
            )
            for specialist in specialist_results
            for source in specialist.get("sources", [])
        ]

        # Gather records as each extraction finishes
        for future in as_completed(futures):
            records.extend(future.result())

    return records


def validate_shared_evidence(records: list[dict]) -> list[dict]:
    """Drop incomplete records, non-approved sources, and exact duplicate claim/url pairs."""
    required = ["claim", "source", "url", "source_type", "evidence"]  # Fields that must be non-empty

    validated = []  # Records that pass every check
    seen = set()    # (claim, url) pairs already kept

    for record in records:
        if not isinstance(record, dict):  # Skip anything that isn't a record
            continue

        # Skip records missing any required field
        if not all(str(record.get(field) or "").strip() for field in required):
            continue

        # Source policy: approved domain AND an allowed research/report type
        if not is_approved_source(str(record.get("url") or "")):
            continue
        # Normalize spelling ("Peer-reviewed paper" -> "peer_reviewed_paper") before checking
        source_type = str(record.get("source_type") or "").strip().lower().replace("-", "_").replace(" ", "_")
        if source_type not in ALLOWED_SOURCE_TYPES:
            continue
        record["source_type"] = source_type

        key = (str(record["claim"]).lower(), str(record["url"]))  # Case-insensitive duplicate key
        if key in seen:  # Skip duplicates
            continue

        seen.add(key)
        validated.append(record)

    return validated


def evidence_store_node(state: Week4ResearchState) -> dict:
    """Graph node: extract, validate, and save the shared evidence."""
    records = extract_specialist_evidence(state["specialist_results"])  # Raw records from all sources
    validated = validate_shared_evidence(records)                       # Clean and de-duplicate
    save_evidence_records(validated)                                    # Persist to SQLite
    return {"evidence_records": validated}                              # Write into graph state


# ======================================================================
# STAGE 4 — GAP / CONFLICT CHECK (+ one bounded extra research pass)
# ======================================================================
# Instructions for deciding whether ONE targeted extra search is worth it
W7_T2_GAP_PROMPT = """
You are the existing supervisor's GAP AND CONFLICT CHECK, not a new agent.
Given the user's research question and evidence store, identify at most ONE
material research gap or meaningful contradiction that targeted research could fix.
Return JSON ONLY:
  {"needs_research": boolean, "reason": string,
   "targeted_question": string, "conflicts": [string], "gaps": [string]}
Missing evidence and genuinely conflicting facts matter; duplicate coverage
alone does not require searching. Do not invent missing facts or resolve a
conflict by guessing. Recommend NO further research when already adequate.
""".strip()


def week7_t2_find_gaps(state: dict) -> dict:
    """Graph node: check the evidence for material gaps or conflicts."""
    # Safe default: if anything goes wrong, recommend NO extra research
    no_research = {
        "needs_research": False,
        "reason": "",
        "targeted_question": "",
        "conflicts": [],
        "gaps": [],
    }

    # The question plus labeled evidence [E1], [E2]... for the check
    prompt = json.dumps(
        {
            "question": state["question"],
            "evidence": build_evidence_packet(state.get("evidence_records", [])),
        },
        ensure_ascii=False,
        indent=2,
    )

    # A failed model call means no research, not a crash
    try:
        result = call_model_json(W7_T2_GAP_PROMPT, prompt)
    except Exception as error:
        return {
            "gap_check": {**no_research, "reason": f"Gap check failed: {error}"},
            "additional_research_performed": False,
        }

    # A reply that isn't a dict is invalid, so it must not trigger research
    if not isinstance(result, dict):
        return {
            "gap_check": {**no_research, "reason": "Gap check returned invalid JSON."},
            "additional_research_performed": False,
        }

    targeted_question = str(result.get("targeted_question") or "").strip()  # The narrower search question
    conflicts = result.get("conflicts")  # Contradictions the check found
    gaps = result.get("gaps")            # Missing coverage the check found

    # Research only if the model said exactly true AND gave a question to search
    needs_research = result.get("needs_research") is True and bool(targeted_question)

    # Cleaned result with safe defaults for any malformed field
    gap_check = {
        "needs_research": needs_research,
        "reason": str(result.get("reason") or "").strip(),
        "targeted_question": targeted_question,
        "conflicts": [str(c) for c in conflicts] if isinstance(conflicts, list) else [],
        "gaps": [str(g) for g in gaps] if isinstance(gaps, list) else [],
    }

    # Research hasn't happened yet at this point
    return {"gap_check": gap_check, "additional_research_performed": False}


def week7_t2_route(state: dict) -> str:
    """Router: decide whether to run the extra research pass."""
    gap_check = state.get("gap_check") or {}
    # Only route to research if a real gap was found AND there's a question to search
    if gap_check.get("needs_research") is True and gap_check.get("targeted_question"):
        return "additional"
    return "synthesis"  # Otherwise go straight to writing the report


def week7_t2_additional_research(state: dict) -> dict:
    """Graph node: ONE extra research pass on the targeted question, merged with old evidence."""
    old_evidence = state.get("evidence_records", [])   # Evidence from the first pass
    old_results = state.get("specialist_results", [])  # Specialist results from the first pass
    targeted_question = (state.get("gap_check") or {}).get("targeted_question", "").strip()

    # Safe result if the extra pass can't run: keep everything exactly as it was
    unchanged = {
        "evidence_records": old_evidence,
        "specialist_results": old_results,
        "additional_research_performed": False,
        "new_evidence_count": 0,
    }

    if not targeted_question:  # Nothing to search
        return unchanged

    try:
        mini_state = {"question": targeted_question}           # Temporary state for the extra pass
        mini_state.update(supervisor_node(mini_state))         # Reuse the supervisor on the narrow question
        mini_state.update(specialists_node(mini_state))        # Reuse the specialists (parallel search)
        mini_state.update(evidence_store_node(mini_state))     # Reuse the evidence store (extract + save)
    except Exception:
        return unchanged  # If the extra pass fails, keep the evidence we already had

    # Old records first so existing [E#] labels keep their numbers; duplicates are dropped
    merged_evidence = validate_shared_evidence(old_evidence + mini_state["evidence_records"])

    return {
        "evidence_records": merged_evidence,
        "specialist_results": old_results + mini_state["specialist_results"],
        "additional_research_performed": True,
        "new_evidence_count": len(merged_evidence) - len(old_evidence),  # Only genuinely new records
    }


# ======================================================================
# STAGE 5 — SYNTHESIS
# Writes one unified report from the shared evidence and saves the draft.
# ======================================================================
def init_report_db():
    """Create the reports table if it doesn't exist yet."""
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                report_markdown TEXT NOT NULL,
                evidence_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def save_report(question: str, report_markdown: str, evidence_count: int) -> int:
    """Save a report to SQLite and return its new row id."""
    created_at = datetime.now(timezone.utc).isoformat()  # Current UTC time as text
    with sqlite3.connect(EVIDENCE_DB) as conn:
        # Parameterized insert: ? placeholders keep model text from breaking the SQL
        cursor = conn.execute(
            "INSERT INTO reports (question, report_markdown, evidence_count, created_at) "
            "VALUES (?, ?, ?, ?)",
            (question, report_markdown, evidence_count, created_at),
        )
    return cursor.lastrowid  # The id SQLite assigned to the new row


init_report_db()  # Make sure the reports table exists before anything is saved


# Shared report contract: (json_key, display title) for all ten sections.
# The model writes the first nine; Sources / References is built in Python.
REPORT_SECTION_KEYS = [
    ("executive_summary", "Executive Summary"),
    ("key_findings", "Key Findings"),
    ("competitor_comparison", "Competitor Comparison"),
    ("market_industry_trends", "Market / Industry Trends"),
    ("emerging_technologies", "Emerging Technologies"),
    ("regulatory_developments", "Regulatory Developments"),
    ("risks_opportunities", "Risks / Opportunities"),
    ("recommended_next_steps", "Recommended Next Steps"),
    ("limitations_information_gaps", "Limitations / Information Gaps"),
    ("sources_references", "Sources / References"),
]

# Just the nine keys the model is responsible for
MODEL_SECTION_KEYS = [key for key, _title in REPORT_SECTION_KEYS if key != "sources_references"]


# Instructions for the synthesis agent: evidence only, cite [E#], exact JSON keys
SYNTHESIS_SYSTEM_PROMPT = """
You are the SYNTHESIS AGENT in a multi-agent business research system.

Use ONLY the supplied Shared Evidence Store to create ONE coherent report.
Return ONLY valid JSON with exactly these keys:
{
  "executive_summary": "...",
  "key_findings": "...",
  "competitor_comparison": "...",
  "market_industry_trends": "...",
  "emerging_technologies": "...",
  "regulatory_developments": "...",
  "risks_opportunities": "...",
  "recommended_next_steps": "...",
  "limitations_information_gaps": "..."
}

Rules:
- Cite factual statements using evidence labels such as [E1].
- Reorganize findings by business topic, not agent name.
- The report should read as if one coordinated team produced it.
- Do not invent facts, dates, metrics, source names, URLs, or citations.
- Clearly distinguish facts, analysis, recommendations, and uncertainty.
- If evidence is insufficient, say so.
- Every evidence record comes from a research paper or formal report; its
  source_type says which kind. When a claim relies on a preprint or working paper,
  say it is not yet peer reviewed. Present figures from industry analyst or
  consulting reports as that firm's estimate, not as settled fact.
- For time-sensitive figures (market size, share, revenue, regulation status),
  include the source's publication date when the evidence record has one.
""".strip()


def build_evidence_packet(evidence_records: list[dict]) -> list[dict]:
    """Label each evidence record E1, E2, ... so the model can cite it."""
    packet = []
    for index, record in enumerate(evidence_records, start=1):
        packet.append({
            "evidence_id": f"E{index}",  # The citation label used in the report
            "claim": record.get("claim"),
            "source": record.get("source"),
            "url": record.get("url"),
            "publication_date": record.get("publication_date"),
            "source_type": record.get("source_type"),
            "evidence": record.get("evidence"),
            "confidence": record.get("confidence"),
            "related_company": record.get("related_company"),
            "report_section": record.get("report_section"),
        })
    return packet


def synthesize_report(question: str, evidence_records: list[dict]) -> dict:
    """Merge all specialist evidence into the common nine-section report."""
    evidence_packet = build_evidence_packet(evidence_records)  # Labeled evidence [E1], [E2], ...

    # The question plus the labeled evidence as clean JSON
    prompt = (
        f"Research question:\n{question}\n\n"
        f"Shared Evidence Store (cite using evidence_id labels such as [E1]):\n"
        f"{json.dumps(evidence_packet, ensure_ascii=False, indent=2)}"
    )

    report = call_model_json(SYNTHESIS_SYSTEM_PROMPT, prompt)  # Ask the synthesis agent to write

    if not isinstance(report, dict):  # Guard against an unusable reply
        report = {}

    # Guarantee all nine sections exist, in the shared order
    report = {
        key: report.get(key) or "Insufficient evidence found for this section."
        for key in MODEL_SECTION_KEYS
    }

    return report


def build_references(evidence_records: list[dict]) -> str:
    """Build the Sources / References list from real URLs (never model-invented)."""
    lines = []
    for index, record in enumerate(evidence_records, start=1):
        source = record.get("source") or "Unknown source"
        url = record.get("url") or "No URL"
        pub_date = record.get("publication_date") or "Date not stated"
        kind = str(record.get("source_type") or "").replace("_", " ")  # e.g. "peer reviewed paper"
        # Numbering matches the [E#] citations in the report
        lines.append(f"- [E{index}] {source} ({kind}, {pub_date}) — {url}")
    return "\n".join(lines) if lines else "No source records were available."


def render_report_markdown(report: dict, evidence_records: list[dict]) -> str:
    """Assemble the nine model sections plus the tenth Sources / References section."""
    blocks = []
    for key, title in REPORT_SECTION_KEYS:
        if key == "sources_references":  # Built below from real data, not by the model
            continue
        blocks.append(f"## {title}\n\n{report.get(key, '')}")  # One heading + body per section
    blocks.append("## Sources / References\n\n" + build_references(evidence_records))
    return "\n\n".join(blocks)


def synthesis_node(state):
    """Graph node: write the report only after all shared evidence exists."""
    evidence_records = state.get("evidence_records", [])  # Evidence from the store (and any extra pass)

    # Stop with a helpful error if there's nothing to write from
    if not evidence_records:
        raise ValueError(
            "synthesis_node got no evidence_records. Check that specialists_node "
            "returned results and evidence_store_node wrote them to state."
        )

    question = state["question"]
    report = synthesize_report(question, evidence_records)                    # Nine sections
    report_markdown = render_report_markdown(report, evidence_records)        # Ten-section markdown
    report_id = save_report(question, report_markdown, len(evidence_records)) # Save the draft

    return {
        "report": report,
        "report_markdown": report_markdown,
        "report_id": report_id,
    }


# ======================================================================
# STAGE 6 — REVIEWER (at most one revision)
# ======================================================================
# Instructions for the reviewer: six scores, PASS or NEEDS_REVISION
REVIEW_SYSTEM_PROMPT = """
You are the REVIEWER / CRITIC AGENT in a multi-agent business research system.
Evaluate the draft against the research question and Shared Evidence Store.
Do not conduct new research and do not silently rewrite the report.

Return ONLY valid JSON:
{
  "overall_status": "PASS" or "NEEDS_REVISION",
  "scores": {
    "completeness": 0,
    "citation_support": 0,
    "source_quality": 0,
    "consistency": 0,
    "business_usefulness": 0,
    "clarity": 0
  },
  "issues": [
    {
      "category": "completeness | citation_support | source_quality | consistency | business_usefulness | clarity",
      "section": "report section or overall",
      "explanation": "specific problem",
      "suggested_fix": "specific bounded fix"
    }
  ],
  "research_gaps": ["gap that may require more research later"]
}

Scoring source_quality: sources are restricted to research papers and formal reports.
Lower the score when key claims rest only on preprints or a single analyst estimate
without corroboration, or when time-sensitive figures are presented without dates.

PASS only when all six scores are at least 7 and there are no material citation-support issues.
""".strip()


# Instructions for the one controlled revision: fix wording/organization from existing evidence only
REVISION_SYSTEM_PROMPT = """
You are the SYNTHESIS AGENT performing ONE controlled revision after reviewer feedback.
Use ONLY the supplied shared evidence. Do not research new facts.
Keep the same nine-section JSON report contract used in Week 5.
If a true research gap cannot be fixed from existing evidence, acknowledge it in limitations.
""".strip()


def review_report(question: str, report: dict, evidence_records: list[dict]) -> dict:
    """Grade a report; any broken reviewer output counts as NEEDS_REVISION, never PASS."""
    # The six score categories from REVIEW_SYSTEM_PROMPT
    score_keys = ["completeness", "citation_support", "source_quality", "consistency", "business_usefulness", "clarity"]

    # Safe fallback used whenever the reviewer breaks
    failed = {
        "overall_status": "NEEDS_REVISION",
        "scores": {key: 0 for key in score_keys},
        "issues": [],
        "research_gaps": [],
    }

    # The question, the draft, and the labeled evidence for the reviewer
    prompt = json.dumps(
        {
            "question": question,
            "draft_report": report,
            "evidence": build_evidence_packet(evidence_records),
        },
        ensure_ascii=False,
        indent=2,
    )

    # An API or JSON error counts as a failed review
    try:
        result = call_model_json(REVIEW_SYSTEM_PROMPT, prompt)
    except Exception as error:
        return {**failed, "issues": [f"Reviewer call failed: {error}"]}

    # A reply without a scores dict is an invalid shape, so it fails
    if not isinstance(result, dict) or not isinstance(result.get("scores"), dict):
        return {**failed, "issues": ["Reviewer returned an invalid JSON shape."]}

    # Normalize each score to an integer from 0 to 10
    scores = {}
    for key in score_keys:
        try:
            value = int(round(float(result["scores"].get(key, 0))))  # Handles 7, "7", 7.6; missing = 0
        except (TypeError, ValueError):
            value = 0  # Unreadable values like "high" or None score 0
        scores[key] = max(0, min(10, value))  # Clamp into 0–10

    status = str(result.get("overall_status", "")).strip().upper()  # " pass " -> "PASS"
    issues = result.get("issues")         # The reviewer's specific problems
    gaps = result.get("research_gaps")    # Gaps that need more research later

    return {
        # Only PASS or NEEDS_REVISION are allowed; anything else is treated as not passed
        "overall_status": status if status in {"PASS", "NEEDS_REVISION"} else "NEEDS_REVISION",
        "scores": scores,
        "issues": issues if isinstance(issues, list) else [],
        "research_gaps": gaps if isinstance(gaps, list) else [],
    }


def revise_report_once(question: str, report: dict, evidence_records: list[dict], review: dict) -> dict:
    """ONE controlled revision using reviewer feedback and existing evidence only."""
    # Everything the reviser needs: question, current draft, feedback, labeled evidence, required keys
    prompt = json.dumps(
        {
            "question": question,
            "current_report": report,
            "reviewer_feedback": {
                "issues": review.get("issues", []),
                "research_gaps": review.get("research_gaps", []),
            },
            "evidence": build_evidence_packet(evidence_records),
            "required_keys": MODEL_SECTION_KEYS,  # The revision prompt doesn't list them, so name them here
        },
        ensure_ascii=False,
        indent=2,
    )

    # If the call breaks, keep the original report
    try:
        revised = call_model_json(REVISION_SYSTEM_PROMPT, prompt)
    except Exception:
        return report

    if not isinstance(revised, dict):  # Unusable reply: keep the original
        return report

    # All nine sections must exist; fall back to the original section if one is missing
    return {
        key: revised.get(key) or report.get(key) or "Insufficient evidence found for this section."
        for key in MODEL_SECTION_KEYS
    }


def reviewer_node(state):
    """Graph node: review the draft, allow at most ONE revision, render the final report."""
    question = state["question"]
    report = state["report"]                               # The synthesized draft
    evidence_records = state.get("evidence_records", [])

    initial_review = review_report(question, report, evidence_records)  # First verdict
    review = initial_review                                             # Final verdict (same unless revised)
    revision_performed = False

    # Any non-PASS verdict triggers exactly one revision and one re-review (no loop)
    if initial_review["overall_status"] != "PASS":
        report = revise_report_once(question, report, evidence_records, initial_review)
        revision_performed = True
        review = review_report(question, report, evidence_records)

    final_report_markdown = render_report_markdown(report, evidence_records)  # Ten-section final report

    return {
        "report": report,                          # Revised version if a revision happened
        "review": review,
        "initial_review": initial_review,
        "final_report_markdown": final_report_markdown,
        "revision_performed": revision_performed,
    }


# ======================================================================
# THE GRAPH
# START -> supervise -> specialists -> evidence_store -> gap_check
#   gap_check -> synthesis                     (no material gap)
#   gap_check -> additional -> synthesis       (one extra pass)
# synthesis -> reviewer -> END
# ======================================================================
def build_week7_graph():
    graph = StateGraph(Week7ResearchState)  # Use the full state so every field is tracked

    # Register every agent/step as a node
    graph.add_node("supervise", supervisor_node)
    graph.add_node("specialists", specialists_node)
    graph.add_node("evidence_store", evidence_store_node)
    graph.add_node("gap_check", week7_t2_find_gaps)
    graph.add_node("additional", week7_t2_additional_research)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("reviewer", reviewer_node)

    # Fixed path through research and evidence
    graph.add_edge(START, "supervise")
    graph.add_edge("supervise", "specialists")
    graph.add_edge("specialists", "evidence_store")
    graph.add_edge("evidence_store", "gap_check")

    # Branch: the router picks extra research or straight to synthesis
    graph.add_conditional_edges(
        "gap_check",
        week7_t2_route,
        {"additional": "additional", "synthesis": "synthesis"},
    )

    # The extra pass always goes forward to synthesis, so it can't loop
    graph.add_edge("additional", "synthesis")

    # Write, review, finish
    graph.add_edge("synthesis", "reviewer")
    graph.add_edge("reviewer", END)

    return graph.compile()  # Turn the definition into a runnable graph


week7_graph = build_week7_graph()  # Build once when the app loads


# ======================================================================
# WEEK 8 — APPROVAL GATE
# Fails closed: a report is approved only if EVERY check passes.
# ======================================================================
def _w8_t2_ready(result):
    """Conservative approval gate: every check must pass, otherwise the report is NOT approved."""
    # Result must be a dict
    if not isinstance(result, dict):
        return False

    # Reviewer must have said PASS
    review = result.get("review")
    if not isinstance(review, dict) or review.get("overall_status") != "PASS":
        return False

    # Final report must contain real text
    final_md = result.get("final_report_markdown")
    if not isinstance(final_md, str) or not final_md.strip():
        return False

    # At least one evidence record
    evidence = result.get("evidence_records")
    if not isinstance(evidence, list) or len(evidence) < 1:
        return False

    # At least three specialist outputs
    specialists = result.get("specialist_results")
    if not isinstance(specialists, list) or len(specialists) < 3:
        return False

    # Extra check: all three roles actually produced output (not just any 3 results)
    roles = {s.get("agent") for s in specialists if isinstance(s, dict)}
    if not {"Competitor", "Market", "Tech/Regulatory"} <= roles:
        return False

    # Extra check: all 10 report headings are present
    if not all(f"## {title}" in final_md for _key, title in REPORT_SECTION_KEYS):
        return False

    # Every check passed
    return True


# ======================================================================
# STREAMLIT UI — Final research workspace
# ======================================================================
ROLE_ORDER = ["Competitor", "Market", "Tech/Regulatory"]  # Display order for the specialists


def _step_message(node_name: str, node_output: dict) -> str:
    """Plain-language progress line for one finished pipeline step."""
    if node_name == "supervise":
        return "Supervisor split the question into three assignments"
    if node_name == "specialists":
        # Name each specialist that finished, so all three labels are visible
        roles = sorted(r.get("agent", "?") for r in node_output.get("specialist_results", []))
        return f"Specialists finished: {', '.join(roles) or 'none'}"
    if node_name == "evidence_store":
        return f"Evidence store built with {len(node_output.get('evidence_records', []))} records"
    if node_name == "gap_check":
        needs = (node_output.get("gap_check") or {}).get("needs_research")
        return "Gap check found a gap worth one extra search" if needs else "Gap check found the evidence sufficient"
    if node_name == "additional":
        return f"Extra research pass added {node_output.get('new_evidence_count', 0)} new records"
    if node_name == "synthesis":
        return "Synthesis agent drafted the report"
    if node_name == "reviewer":
        verdict = (node_output.get("review") or {}).get("overall_status", "unknown")
        revised = " after one revision" if node_output.get("revision_performed") else ""
        return f"Reviewer verdict: {verdict}{revised}"
    return node_name  # Fallback for any unexpected node


# ---------------- Header ----------------
st.title("Team 2 — Agentic Market Research")
st.caption(
    "A supervisor, three specialist agents, a shared evidence store, and a reviewer "
    "work together to answer one business research question."
)

# ---------------- Question form ----------------
# A form lets the user press Enter or click the button to submit
with st.form("research_form"):
    question = st.text_input(
        "Research question",
        value="How is NVIDIA positioning itself against AMD in the AI chip market?",
    )
    submitted = st.form_submit_button("Run research", type="primary")

# ---------------- Run the pipeline ----------------
if submitted:
    if not question.strip():
        st.warning("Enter a research question first.")
    else:
        # Starting state: every field the graph tracks, with empty defaults
        initial_state = {
            "question": question.strip(),
            "assignments": [],
            "specialist_results": [],
            "evidence_records": [],
            "report": {},
            "report_markdown": "",
            "report_id": 0,
            "review": {},
            "initial_review": {},
            "final_report_markdown": "",
            "revision_performed": False,
            "gap_check": {},
            "additional_research_performed": False,
            "new_evidence_count": 0,
        }

        result = dict(initial_state)  # Final state, built up as each step finishes
        start = time.perf_counter()   # Start time, for the duration metric

        # Live progress box: one line per finished step, with elapsed time
        with st.status("Researching...", expanded=True) as status:
            try:
                for update in week7_graph.stream(initial_state, stream_mode="updates"):
                    for node_name, node_output in update.items():
                        node_output = node_output or {}
                        result.update(node_output)  # Merge this step's output into the final state
                        elapsed = time.perf_counter() - start
                        status.write(f"{_step_message(node_name, node_output)} ({elapsed:.0f}s)")

                seconds = round(time.perf_counter() - start, 1)
                status.update(label=f"Finished in {seconds:.0f} seconds", state="complete", expanded=False)

                # Keep the result across reruns (e.g. clicking Download reruns the script)
                st.session_state["w8_t2_result"] = result
                st.session_state["w8_t2_seconds"] = seconds
            except Exception as exc:
                status.update(label="Research failed", state="error")
                st.session_state.pop("w8_t2_result", None)  # Never show a stale result after a failure
                st.error(f"Research failed and no report was produced: {exc}")

# ---------------- Results ----------------
if "w8_t2_result" in st.session_state:
    result = st.session_state["w8_t2_result"]
    approved = _w8_t2_ready(result)                         # The Week 8 quality gate
    review = result.get("review") or {}                     # Final reviewer verdict
    gap_check = result.get("gap_check") or {}               # Gap/conflict check result
    evidence = result.get("evidence_records") or []         # All evidence (both passes)
    specialists = result.get("specialist_results") or []    # All specialist outputs (both passes)
    report_md = result.get("final_report_markdown") or result.get("report_markdown") or ""

    # Verdict banner: the first thing the viewer sees
    if approved:
        st.success("Approved by the reviewer. This is prototype output and has not been independently fact-checked.")
    else:
        st.error("Not approved. The report did not pass the quality gate, so no download is offered.")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration", f"{st.session_state.get('w8_t2_seconds', 0):.0f}s")
    m2.metric("Evidence records", len(evidence))
    m3.metric("Specialist outputs", len(specialists))
    if result.get("additional_research_performed"):
        m4.metric("Extra research", f"+{result.get('new_evidence_count', 0)} records")
    else:
        m4.metric("Extra research", "Not needed")

    tab_report, tab_specialists, tab_gaps, tab_review, tab_evidence = st.tabs(
        ["Report", "Specialists", "Gap check", "Reviewer", "Evidence"]
    )

    # Report: shown openly with a download ONLY when approved
    with tab_report:
        display_md = report_md.replace("$", "\\$")  # Stop dollar amounts from rendering as math
        if approved:
            st.download_button(
                "Download report (.md)",
                report_md,                           # Raw markdown, without the display escaping
                file_name="elp_team2_report.md",
                mime="text/markdown",
                key="w8_t2_download",
            )
            st.markdown(display_md)
        else:
            with st.expander("Unapproved draft (internal debugging only)"):
                st.markdown(display_md)

    # Specialists: each role's objective and brief, first pass before the extra pass
    with tab_specialists:
        first_pass_objectives = {a.get("objective") for a in result.get("assignments", [])}

        def _sort_key(s):
            extra = s.get("objective") not in first_pass_objectives  # Extra-pass outputs go last
            role = ROLE_ORDER.index(s["agent"]) if s.get("agent") in ROLE_ORDER else len(ROLE_ORDER)
            return (extra, role)

        for s in sorted(specialists, key=_sort_key):
            label = s.get("agent", "Specialist")
            if s.get("objective") not in first_pass_objectives:
                label += " (extra research pass)"
            with st.expander(label):
                st.markdown(f"**Objective:** {s.get('objective', '')}")
                st.markdown((s.get("research") or "").replace("$", "\\$"))
                st.caption(f"{len(s.get('sources', []))} web sources")

    # Gap check: whether one extra research pass ran, and why
    with tab_gaps:
        if result.get("additional_research_performed"):
            st.info(
                f"One extra research pass ran on: {gap_check.get('targeted_question', '')}  \n"
                f"It added {result.get('new_evidence_count', 0)} new evidence records."
            )
        else:
            st.write("No extra research pass was needed.")

        st.markdown(f"**Reason:** {gap_check.get('reason') or 'No reason given.'}")

        st.markdown("**Gaps**")
        for g in gap_check.get("gaps") or ["None found."]:
            st.markdown(f"- {g}")

        st.markdown("**Conflicts**")
        for c in gap_check.get("conflicts") or ["None found."]:
            st.markdown(f"- {c}")

    # Reviewer: six scores against the pass bar of 7, then issues and gaps
    with tab_review:
        if result.get("revision_performed"):
            st.caption("The first draft did not pass, so one controlled revision was made and re-reviewed.")

        cols = st.columns(3)
        for i, (name, score) in enumerate((review.get("scores") or {}).items()):
            # Arrow shows distance from the pass bar (green above 7, red below)
            cols[i % 3].metric(name.replace("_", " ").title(), f"{score}/10", delta=score - 7)

        st.markdown("**Issues**")
        for issue in review.get("issues") or ["No issues reported."]:
            if isinstance(issue, dict):  # Normal reviewer issues are dicts
                st.markdown(
                    f"- **{issue.get('category', '')}** ({issue.get('section', 'overall')}): "
                    f"{issue.get('explanation', '')}  \n  Suggested fix: {issue.get('suggested_fix', '')}"
                )
            else:  # Failed reviews return plain-text issues
                st.markdown(f"- {issue}")

        st.markdown("**Research gaps**")
        for gap in review.get("research_gaps") or ["None reported."]:
            st.markdown(f"- {gap}")

        if result.get("revision_performed"):
            with st.expander("Initial review, before the revision"):
                st.json(result.get("initial_review") or {})

    # Evidence: one row per record, IDs match the [E#] citations in the report
    with tab_evidence:
        rows = [
            {
                "ID": f"E{i}",
                "Claim": rec.get("claim", ""),
                "Source": rec.get("source", ""),
                "Type": str(rec.get("source_type") or "").replace("_", " "),
                "Published": rec.get("publication_date") or "Not stated",
                "Confidence": rec.get("confidence", ""),
                "Link": rec.get("url", ""),
            }
            for i, rec in enumerate(evidence, start=1)
        ]
        st.dataframe(
            rows,
            hide_index=True,
            use_container_width=True,
            column_config={"Link": st.column_config.LinkColumn("Link")},  # Clickable source URLs
        )