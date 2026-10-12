"""ELP Week 3 — Team 2 Student Starter
Agentic Research Team: Supervisor + Specialist Agents + Parallelism

Week 2 already gave you:
    Question -> Supervisor -> first assignment -> Tavily -> Response

Week 3 goal:
    Question -> Supervisor -> Specialist Agents in Parallel -> Specialist Results

Your job is to complete the NEW Week 3 TODOs only.

Run:
    python -m pip install -r requirements_team2.txt
    streamlit run team2_week3_student.py

.env:
    TAVILY_API_KEY=your_key
    DEEPSEEK_API_KEY=your_key
"""

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

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from tavily import TavilyClient


# ---------------- Configuration ----------------
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not TAVILY_API_KEY or not DEEPSEEK_API_KEY:
    st.error(
        "Missing API key. Add TAVILY_API_KEY and DEEPSEEK_API_KEY "
        "to a .env file beside this script."
    )
    st.stop()

tavily = TavilyClient(api_key=TAVILY_API_KEY)
deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


# ---------------- Team 2 Supervisor ----------------
# This carries forward Week 2 homework: each assignment includes output_fields.
TEAM_2_SYSTEM_PROMPT = """
You are the Supervisor for a multi-agent competitive-intelligence research system.

Divide the user's business research question across exactly three specialist roles:
Competitor, Market, and Tech/Regulatory.

Each specialist must receive a distinct, non-overlapping objective. Include a short
output_fields list describing what that specialist should return.

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


class ResearchState(TypedDict):
    question: str
    assignments: list[dict]
    specialist_results: list[dict]

class Week4ResearchState(ResearchState):
    evidence_records: list[dict]


def call_model_json(system_prompt: str, user_text: str) -> dict:
    response = deepseek.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def supervisor_node(state: ResearchState) -> dict:
    output = call_model_json(TEAM_2_SYSTEM_PROMPT, state["question"])
    tasks = output.get("tasks")

    if not isinstance(tasks, list) or len(tasks) != 3:
        raise ValueError("Supervisor must return exactly three assignments.")

    expected_agents = {"Competitor", "Market", "Tech/Regulatory"}
    seen_agents = set()
    cleaned = []

    for assignment in tasks:
        if not isinstance(assignment, dict):
            raise ValueError("Each assignment must be a JSON object.")

        agent = str(assignment.get("agent", "")).strip()
        objective = str(assignment.get("objective", "")).strip()
        output_fields = assignment.get("output_fields", [])

        if agent not in expected_agents:
            raise ValueError(f"Unexpected specialist role: {agent}")
        if not objective:
            raise ValueError(f"{agent} needs a non-empty objective.")
        if not isinstance(output_fields, list):
            raise ValueError(f"{agent} output_fields must be a list.")

        seen_agents.add(agent)
        cleaned.append(
            {
                "agent": agent,
                "objective": objective,
                "output_fields": [str(field) for field in output_fields],
            }
        )

    if seen_agents != expected_agents:
        raise ValueError(
            "Assignments must include Competitor, Market, and Tech/Regulatory exactly once."
        )

    return {"assignments": cleaned}


# ---------------- Week 3 specialist execution ----------------
def research_assignment(assignment: dict, original_question: str) -> dict:
    """Execute ONE specialist's assignment."""
    # 1. Read assignment metadata
    agent = assignment["agent"]
    objective = assignment["objective"]
    output_fields = assignment["output_fields"]

    # 2. Search Tavily using the specialist's objective
    tavily_response = tavily.search(query=objective, max_results=5)
    tavily_results = tavily_response.get("results", [])

    # 3. Format Tavily results into readable source text
    sources_text = ""
    for idx, src in enumerate(tavily_results, 1):
        title = src.get("title", "No Title")
        url = src.get("url", "No URL")
        content = src.get("content", "")
        sources_text += f"Source [{idx}]: {title}\nURL: {url}\nContent: {content}\n\n"

    # 4. Call DeepSeek with specialist prompt & context
    prompt_text = f"""
    Original User Question: {original_question}
    Your Assigned Objective: {objective}
    Requested Output Fields: {', '.join(output_fields)} Tavily Research Results:
    {sources_text} Please provide a clear and concise research brief based on the sources provided above.
    """

    system_prompt = SPECIALIST_PROMPTS.get(agent, "")
    model_response = deepseek.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ],
    )
    research_text = model_response.choices[0].message.content

    # 5. Return formatted result
    return {
        "agent": agent,
        "objective": objective,
        "output_fields": output_fields,
        "research": research_text,
        "sources": tavily_results,
    }


def specialists_node(state: ResearchState) -> dict:
    """Run all three independent specialist assignments in parallel."""
    specialist_results = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(research_assignment, assignment, state["question"])
            for assignment in state["assignments"]
        ]

        for future in as_completed(futures):
            try:
                result = future.result()
                specialist_results.append(result)
            except Exception as e:
                st.error(f"Error during parallel specialist execution: {e}")

    return {"specialist_results": specialist_results}


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("supervise", supervisor_node)
    graph.add_node("specialists", specialists_node)

    graph.add_edge(START, "supervise")
    graph.add_edge("supervise", "specialists")
    graph.add_edge("specialists", END)

    return graph.compile()


research_graph = build_graph()


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="ELP Team 2 — Week 3", page_icon="🧩")
st.title("Team 2 — Agentic Research Team")
st.caption("Supervisor → Specialist Agents → Parallel Research")

question = st.text_input(
    "Enter a business research question",
    value="How is NVIDIA positioning itself against AMD in the AI chip market?",
)

if st.button("Research", type="primary") and question.strip():
    initial_state: ResearchState = {
        "question": question.strip(),
        "assignments": [],
        "specialist_results": [],
    }

    try:
        with st.spinner("Supervising and running specialist research..."):
            result = research_graph.invoke(initial_state)
    except Exception as error:
        st.error(f"The workflow could not finish: {error}")
    else:
        st.subheader("Supervisor assignments")
        st.json(result["assignments"])

        st.subheader("Independent specialist results")
        for item in sorted(
            result["specialist_results"],
            key=lambda x: x.get("agent", ""),
        ):
            with st.expander(item.get("agent", "Specialist"), expanded=True):
                st.write(f"**Objective:** {item.get('objective', '')}")
                st.markdown(item.get("research", "").replace("$", "\\$"))
                st.caption(f"{len(item.get('sources', []))} Tavily sources returned")

"""
ELP Week 4 — Team 2 Student Extension
Evidence & Traceability + SQLite
"""

# ---------------- Week 4: Shared Evidence Store ----------------

import sqlite3

EVIDENCE_DB = "team2_evidence.db"



def init_evidence_db():
    """Create the Week 4 shared evidence table if it does not exist."""
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
        """)
        conn.commit()


def save_evidence_records(records: list[dict]):
    """Save validated specialist evidence to SQLite."""
    if not records:
        return

    columns = ", ".join(EVIDENCE_FIELDS)
    placeholders = ", ".join(f":{field}" for field in EVIDENCE_FIELDS)
    sql = f"INSERT OR IGNORE INTO evidence ({columns}) VALUES ({placeholders})"

    rows = [{field: record.get(field) for field in EVIDENCE_FIELDS} for record in records]

    conn = sqlite3.connect(EVIDENCE_DB)
    try:
        with conn:
            conn.executemany(sql, rows)
    finally:
        conn.close()


def load_evidence_records() -> list[dict]:
    """Read the shared evidence store back from SQLite."""
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM evidence ORDER BY id").fetchall()
        return [dict(row) for row in rows]


init_evidence_db()


EVIDENCE_SYSTEM_PROMPT = """
You are the evidence extraction step in a business research workflow.

Convert useful specialist research sources into structured Evidence Records.

Return ONLY valid JSON in exactly this shape:
{
  "records": [
    {
      "claim": "...",
      "source": "...",
      "url": "...",
      "publication_date": null,
      "source_type": "web",
      "evidence": "...",
      "confidence": "high",
      "related_company": null,
      "report_section": null
    }
  ]
}

Rules:
- claim = a factual statement directly supported by the source.
- evidence = the passage that supports the claim.
- source and url must come from the supplied Tavily result.
- Do not guess a publication date.
- confidence must be high, medium, or low.
- Do not turn analysis or recommendations into facts.
""".strip()


def extract_from_source(agent: str, objective: str, source: dict) -> list[dict]:
    """Extract Evidence Records from ONE Tavily source."""
    source_payload = {
        "agent": agent,
        "objective": objective,
        "title": source.get("title"),
        "url": source.get("url"),
        "content": (source.get("content") or "")[:6000],
        "published_date": source.get("published_date"),
    }
    prompt = json.dumps(source_payload, ensure_ascii=False)

    try:
        result = call_model_json(EVIDENCE_SYSTEM_PROMPT, prompt)
    except Exception:
        return []  # skip a bad source instead of failing the whole run

    return [r for r in result.get("records", []) if isinstance(r, dict)]


def extract_specialist_evidence(specialist_results: list[dict]) -> list[dict]:
    """Convert specialist source results into the shared Evidence Record format."""
    records = []

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

        for future in as_completed(futures):
            records.extend(future.result())

    return records


def validate_shared_evidence(records: list[dict]) -> list[dict]:
    """Validate records and remove exact duplicate claim/source pairs."""
    required = ["claim", "source", "url", "source_type", "evidence"]

    validated = []
    seen = set()

    for record in records:
        if not isinstance(record, dict):
            continue

        if not all(str(record.get(field) or "").strip() for field in required):
            continue

        key = (record["claim"].lower(), record["url"])
        if key in seen:
            continue

        seen.add(key)
        validated.append(record)

    return validated

def evidence_store_node(state: Week4ResearchState) -> dict:
    """Week 4: build and persist the shared evidence store."""
    records = extract_specialist_evidence(state["specialist_results"])
    validated = validate_shared_evidence(records)
    save_evidence_records(validated)
    return {"evidence_records": validated}


def build_week4_graph():
    """
    Extend the existing Week 3 Team 2 flow:

    Question -> Supervisor -> Specialist Agents -> Evidence Records -> SQLite Store
    """
    graph = StateGraph(Week4ResearchState)

    graph.add_node("supervise", supervisor_node)
    graph.add_node("specialists", specialists_node)
    graph.add_node("evidence_store", evidence_store_node)

    graph.add_edge(START, "supervise")
    graph.add_edge("supervise", "specialists")
    graph.add_edge("specialists", "evidence_store")
    graph.add_edge("evidence_store", END)

    return graph.compile()


week4_graph = build_week4_graph()


# ---------------- Week 4 UI ----------------

st.divider()
st.subheader("Week 4 — Shared Evidence Store")
st.caption("Specialist research → Evidence Records → SQLite Shared Store")

if st.button("Build Shared Evidence Store", type="primary") and question.strip():
    week4_state = {
        "question": question.strip(),
        "assignments": [],
        "specialist_results": [],
        "evidence_records": [],
    }

    try:
        with st.spinner("Running specialists and building the shared evidence store..."):
            evidence_result = week4_graph.invoke(week4_state)
    except Exception as error:
        st.error(f"The Week 4 evidence workflow could not finish: {error}")
    else:
        st.subheader("New Evidence Records")
        st.json(evidence_result["evidence_records"])

        stored_records = load_evidence_records()
        st.subheader("Shared Evidence Store (SQLite)")
        st.dataframe(stored_records, use_container_width=True)
        st.caption(f"{len(stored_records)} evidence records stored in {EVIDENCE_DB}")


# UTC timestamp is used when each report is stored in SQLite.
from datetime import datetime, timezone

# ---------------- Week 5: Save reports in SQLite ----------------

# WHAT THIS DOES — DATABASE SETUP: Creates reports table if missing; preserves all Week 4 evidence rows.
def init_report_db():
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


# WHAT THIS DOES — DATABASE WRITE: Saves question, generated markdown, evidence count, timestamp; returns report ID.
def save_report(question: str, report_markdown: str, evidence_count: int) -> int:
    """Save the synthesized report and return its database id."""
    # STUDENT: Replace the `pass` in this function with real code.
    # TODO 0 — REPORT PERSISTENCE
    # Insert question, report_markdown, evidence_count, and a current UTC timestamp
    # into reports using parameterized SQL. Return cursor.lastrowid.
    created_at = datetime.now(timezone.utc).isoformat() 
    with sqlite3.connect(EVIDENCE_DB) as conn:
        cursor = conn.execute(
            "INSERT INTO reports(question, report_markdown, evidence_count, created_at ) "
            "VALUES (?, ?, ?, ?)",
            (question, report_markdown, evidence_count, created_at))
    return cursor.lastrowid


# WHAT THIS DOES — DATABASE READ: Loads previously generated reports, most recent first, for the UI.
def load_reports() -> list[dict]:
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM reports ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]


init_report_db()  # Ensure reports table exists before any report is saved.


# SHARED OUTPUT CONTRACT: Both teams use the SAME nine model-written sections;
# Sources / References is appended in Python as section number ten.
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


# MODEL INSTRUCTIONS: The model must work from evidence only and return structured JSON.
# Week 6, not this file, handles independent citation validation and report review.
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
- Do NOT add the Reviewer/Critic today. That is Week 6.
""".strip()


# WHAT THIS DOES — EVIDENCE PREP: Assigns [E1], [E2] labels to this run’s evidence, keeps sources and excerpts.
def build_evidence_packet(evidence_records: list[dict]) -> list[dict]:
    packet = []
    for index, record in enumerate(evidence_records, start=1):
        packet.append({
            "evidence_id": f"E{index}",
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


# WHAT THIS DOES — TEAM 2 SYNTHESIS AGENT: Combines all specialist evidence into ONE unified report.
def synthesize_report(question: str, evidence_records: list[dict]) -> dict:
    """Merge all specialist evidence into the common report structure."""

    # Build the numbered evidence packet.
    evidence_packet = build_evidence_packet(evidence_records)
    # Put the research question + shared evidence into the prompt.
    prompt = (
        f"Research question:\n{question}\n\n"
        f"Evidence (cite using the [E#] labels):\n{evidence_packet}\n\n"
        f"Return a JSON object with these keys: {', '.join(REPORT_SECTION_KEYS)}"
    )
    # Call call_model_json(SYNTHESIS_SYSTEM_PROMPT, prompt)
    report = call_model_json(SYNTHESIS_SYSTEM_PROMPT, prompt)

    # Make sure all REPORT_SECTION_KEYS exist
    if not isinstance(report, dict):
        report = {}

    report = {
        key: report.get(key) or "Insufficient evidence found for this section."
        for key in REPORT_SECTION_KEYS
    }

    # return the report dict
    return report


# WHAT THIS DOES — REFERENCES: Creates citation legend from original URLs, not invented model references.
def build_references(evidence_records: list[dict]) -> str:
    lines = []
    for index, record in enumerate(evidence_records, start=1):
        source = record.get("source") or "Unknown source"
        url = record.get("url") or "No URL"
        date = record.get("publication_date") or "Date unavailable"
        lines.append(f"- [E{index}] {source} — {date} — {url}")
    return "\n".join(lines) if lines else "No source records were available."


# WHAT THIS DOES — REPORT FORMAT: Assembles nine model sections and the tenth Sources / References section.
def render_report_markdown(report: dict, evidence_records: list[dict]) -> str:
    blocks = []
    for key, title in REPORT_SECTION_KEYS:
        blocks.append(f"## {title}\n\n{report.get(key, '')}")
    blocks.append("## Sources / References\n\n" + build_references(evidence_records))
    return "\n\n".join(blocks)


class Week5ResearchState(Week4ResearchState):
    report: dict
    report_markdown: str
    report_id: int


# WHAT THIS DOES — TEAM 2 GRAPH NODE: Runs synthesis, formats it, saves to SQLite.
def synthesis_node(state):
    """Synthesize only after all shared evidence has been produced."""
    # TODO 3 — CONNECT SHARED EVIDENCE TO SYNTHESIS
    # 1. Read state["evidence_records"].
    # 2. Stop with a helpful error if empty.
    # 3. Call synthesize_report(...).
    # 4. Render markdown.
    # 5. Save the report to SQLite.
    # 6. Return report, report_markdown, and report_id.
    


# WORKFLOW WIRING: Reuses Week 3/4 nodes and adds Week 5 report/synthesis at the end.
def build_week5_graph():
    """Question -> Supervisor -> Specialists -> Shared Evidence -> Synthesis -> END"""
    graph = StateGraph(Week5ResearchState)

    graph.add_node("supervise", supervisor_node)
    graph.add_node("specialists", specialists_node)
    graph.add_node("evidence_store", evidence_store_node)
    graph.add_node("synthesis", synthesis_node)

    graph.add_edge(START, "supervise")
    graph.add_edge("supervise", "specialists")
    graph.add_edge("specialists", "evidence_store")
    graph.add_edge("evidence_store", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()



week5_graph = build_week5_graph()


# ---------------- Week 5 UI ----------------
# UI: Adds a separate Week 5 button and shows the report + evidence + saved history.
# A full run invokes Week 3 planning/research, Week 4 evidence, THEN Week 5 writing.

st.divider()
st.subheader("Week 5 — Multi-Agent Synthesis Pipeline")
st.caption("Question → Supervisor → Specialists → Shared Evidence → Synthesis Agent → Report")

if st.button("Run Full Synthesis Pipeline", type="primary") and question.strip():
    # TODO 5 — RUN THE COMPLETE PIPELINE
    # Create the initial state, invoke week5_graph, and display:
    #   - the synthesized report with st.markdown(...)
    #   - the shared evidence used
    #   - saved reports from SQLite
    pass

