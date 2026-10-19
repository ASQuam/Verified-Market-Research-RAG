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

MODEL_SECTION_KEYS = [key for key, _title in REPORT_SECTION_KEYS if key != "sources_references"]


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


def synthesize_report(question: str, evidence_records: list[dict]) -> dict:
    """Merge all specialist evidence into the common report structure."""
    # 1. Build the numbered evidence packet
    evidence_packet = build_evidence_packet(evidence_records)

    # 2. Question + shared evidence into the prompt
    prompt = (
        f"Research question:\n{question}\n\n"
        f"Shared Evidence Store (cite using evidence_id labels such as [E1]):\n"
        f"{json.dumps(evidence_packet, ensure_ascii=False, indent=2)}"
    )

    # 3. Call the model
    report = call_model_json(SYNTHESIS_SYSTEM_PROMPT, prompt)

    # 4. Make sure every model-written section exists
    if not isinstance(report, dict):
        report = {}
    report = {
        key: report.get(key) or "Insufficient evidence found for this section."
        for key in MODEL_SECTION_KEYS
    }

    # 5. Return the report dict
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


def render_report_markdown(report: dict, evidence_records: list[dict]) -> str:
    blocks = []
    for key, title in REPORT_SECTION_KEYS:
        if key == "sources_references":
            continue
        blocks.append(f"## {title}\n\n{report.get(key, '')}")
    blocks.append("## Sources / References\n\n" + build_references(evidence_records))
    return "\n\n".join(blocks)


class Week5ResearchState(Week4ResearchState):
    report: dict
    report_markdown: str
    report_id: int


def synthesis_node(state):
    """Synthesize only after all shared evidence has been produced."""
    # 1. Read evidence from state
    evidence_records = state.get("evidence_records", [])

    # 2. Stop with a helpful error if empty
    if not evidence_records:
        raise ValueError(
            "synthesis_node got no evidence_records. Check that specialists_node "
            "returned results and evidence_store_node wrote them to state."
        )

    # 3. Synthesize
    question = state["question"]
    report = synthesize_report(question, evidence_records)

    # 4. Render markdown
    report_markdown = render_report_markdown(report, evidence_records)

    # 5. Save to SQLite
    report_id = save_report(question, report_markdown, len(evidence_records))

    # 6. Return the new state keys
    return {
        "report": report,
        "report_markdown": report_markdown,
        "report_id": report_id,
    }
    


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


if st.button("Run Full Synthesis Pipeline", type="primary") and question.strip():
    initial_state = {"question": question.strip()}

    try:
        with st.spinner("Running supervisor, specialists, and synthesis..."):
            result = week5_graph.invoke(initial_state)
    except ValueError as e:
        st.error(f"Pipeline stopped: {e}")
        st.stop()

    # Synthesized report
    st.success(f"Report saved (id {result['report_id']})")
    st.markdown(result["report_markdown"])

    # Shared evidence used
    evidence = result.get("evidence_records", [])
    with st.expander(f"Shared evidence used ({len(evidence)} records)"):
        for i, rec in enumerate(evidence, start=1):
            st.markdown(f"**[E{i}] {rec.get('title', 'Untitled')}**  \n{rec.get('url', 'no url')}")
            st.caption((rec.get("content") or "")[:300])

    # Saved reports from SQLite
    with sqlite3.connect(EVIDENCE_DB) as conn:
        rows = conn.execute(
            "SELECT id, question, evidence_count, created_at "
            "FROM reports ORDER BY id DESC LIMIT 10"
        ).fetchall()
    st.markdown("#### Saved reports")
    st.dataframe(
        [{"id": r[0], "question": r[1], "evidence": r[2], "created": r[3]} for r in rows],
        use_container_width=True,
    )


    """
ELP WEEK 6 — TEAM 2 — STUDENT EXERCISE
Multi-agent reviewer / quality control

==================== EXACT SETUP INSTRUCTIONS ====================
1. OPEN the existing, combined TEAM 2 Streamlit Python app that contains your
   completed Week 3, Week 4, and Week 5 work IN THAT ORDER.
2. CONFIRM the Week 5 button runs and produces a 10-section draft report.
   Complete Week 5 TODOs before moving on; Week 6 cannot fix missing Week 5 logic.
3. SCROLL TO THE VERY END of that SAME .py app file, AFTER the last line of Week 5
   (including the Week 5 Streamlit UI). Leave two blank lines.
4. COPY THIS ENTIRE Week 6 extension (including imports, functions, graph, UI)
   and PASTE IT BELOW Week 5. Do not paste into a new standalone .py script.
5. DO NOT DELETE or replace Week 3/4/5. DO NOT mix Team 1 and Team 2 code.
6. If you previously pasted a Week 6 extension, REPLACE that Week 6 block;
   do NOT append duplicate Week 6 copies.
7. SAVE your combined app, then run from a terminal in its directory:
       streamlit run YOUR_EXISTING_APP_FILE.py
8. Enter your research question in the EXISTING question input. Click:
       Run Reviewed Synthesis Pipeline
9. Inspect the quality result. PASS permits final display; failures must be
   clearly flagged, with the draft accessible only for inspection.

IMPORTANT:
- Week 6 is NOT standalone; it calls existing Week 5 functions, graph nodes,
  shared state, report/evidence helpers, model client, SQLite, and Streamlit.
- You need the SAME working API keys and Python dependencies as Week 5.
- Week 5 may still save a DRAFT to SQLite before Week 6 checks it;
  that saved draft is NOT automatically an approved final report.
- Use the new Week 6 button, not the older Week 5 button, for the full flow.
- This is a live end-to-end prototype; it is not guaranteed to identify every
  unsupported statement or broken URL. Inspect the verdict yourself.
- NO follow-up Q&A or automatic extra research until Week 7.
- Team 2: reviewer is an agent after synthesis; at most ONE revision.

STUDENT: Complete all numbered TODOs before the NEXT meeting, at your own pace.
Existing `pass` placeholders are exercises and the extension WILL NOT work until finished.
END-OF-WEEK GOAL: The report is checked before being presented as trusted.
===================================================================
"""


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

PASS only when all six scores are at least 7 and there are no material citation-support issues.
""".strip()


REVISION_SYSTEM_PROMPT = """
You are the SYNTHESIS AGENT performing ONE controlled revision after reviewer feedback.
Use ONLY the supplied shared evidence. Do not research new facts.
Keep the same nine-section JSON report contract used in Week 5.
If a true research gap cannot be fixed from existing evidence, acknowledge it in limitations.
""".strip()


def review_report(question: str, report: dict, evidence_records: list[dict]) -> dict:
    # The six score categories the reviewer grades (copy names from REVIEW_SYSTEM_PROMPT)
    score_keys = ["completeness", "citation_support", "source_quality", "consistency", "business_usefulness", "clarity"]

    # Safe fallback result: used whenever the reviewer breaks, so a failure is never a PASS
    failed = {
        "overall_status": "NEEDS_REVISION",
        "scores": {key: 0 for key in score_keys},
        "issues": [],
        "research_gaps": [],
    }

    # Package the question, draft report, and labeled evidence [E1], [E2]... as JSON for the reviewer
    prompt = json.dumps(
        {
            "question": question,
            "draft_report": report,
            "evidence": build_evidence_packet(evidence_records),
        },
        ensure_ascii=False,
        indent=2,
    )

    # Ask the reviewer model to grade the report; an API or JSON parse error counts as a failed review
    try:
        result = call_model_json(REVIEW_SYSTEM_PROMPT, prompt)
    except Exception as error:
        return {**failed, "issues": [f"Reviewer call failed: {error}"]}

    # Reject responses that aren't a dict or lack a scores dict (invalid shape = failed review)
    if not isinstance(result, dict) or not isinstance(result.get("scores"), dict):
        return {**failed, "issues": ["Reviewer returned an invalid JSON shape."]}

    # Normalize each score to an integer from 0 to 10
    scores = {}
    for key in score_keys:
        try:
            # Handles 7, "7", and 7.6; a missing score defaults to 0
            value = int(round(float(result["scores"].get(key, 0))))
        except (TypeError, ValueError):
            # Unreadable values like "high" or None score 0
            value = 0
        # Clamp out-of-range values (e.g. 12 becomes 10, -3 becomes 0)
        scores[key] = max(0, min(10, value))

    # Clean up the status text so " pass " and "Pass" both become "PASS"
    status = str(result.get("overall_status", "")).strip().upper()

    # Pull out the reviewer's lists of problems and missing research
    issues = result.get("issues")
    gaps = result.get("research_gaps")

    # Return the four required fields, with safe defaults for anything malformed
    return {
        "overall_status": status if status in {"PASS", "NEEDS_REVISION"} else "NEEDS_REVISION",  # unknown status = FAIL
        "scores": scores,
        "issues": issues if isinstance(issues, list) else [],
        "research_gaps": gaps if isinstance(gaps, list) else [],
    }


def revise_report_once(question: str, report: dict, evidence_records: list[dict], review: dict) -> dict:
    # Package everything the reviser needs: question, current draft, reviewer feedback, labeled evidence
    prompt = json.dumps(
        {
            "question": question,
            "current_report": report,
            "reviewer_feedback": {
                "issues": review.get("issues", []),
                "research_gaps": review.get("research_gaps", []),
            },
            "evidence": build_evidence_packet(evidence_records),
            # The revision prompt doesn't list the nine keys, so tell the model exactly which to return
            "required_keys": MODEL_SECTION_KEYS,
        },
        ensure_ascii=False,
        indent=2,
    )

    # Ask the synthesis model for ONE revision; if the call breaks, keep the original report
    try:
        revised = call_model_json(REVISION_SYSTEM_PROMPT, prompt)
    except Exception:
        return report

    # A response that isn't a dict is unusable, so keep the original report
    if not isinstance(revised, dict):
        return report

    # Make sure all nine model-written sections exist; fall back to the original section if one is missing
    return {
        key: revised.get(key) or report.get(key) or "Insufficient evidence found for this section."
        for key in MODEL_SECTION_KEYS
    }



class Week6ResearchState(Week5ResearchState):
    review: dict
    initial_review: dict
    final_report_markdown: str
    revision_performed: bool


def reviewer_node(state):
    """Review the Week 5 draft, allow at most ONE revision, then render the final report."""
    question = state["question"]
    report = state["report"]
    evidence_records = state.get("evidence_records", [])

    # 1. Review the Week 5 synthesized report
    initial_review = review_report(question, report, evidence_records)
    review = initial_review
    revision_performed = False

    # 2. If PASS, keep the report as-is (nothing to do)
    # 3. If NEEDS_REVISION, revise ONCE using only the existing evidence
    if initial_review["overall_status"] != "PASS":
        report = revise_report_once(question, report, evidence_records, initial_review)
        revision_performed = True

        # 4. Review the revised report exactly once more (no second revision, whatever the result)
        review = review_report(question, report, evidence_records)

    # 5. Render the final markdown, with Sources / References built from the real URLs
    final_report_markdown = render_report_markdown(report, evidence_records)

    # Return the new Week 6 state keys; "report" now holds the revised version if one was made
    return {
        "report": report,
        "review": review,
        "initial_review": initial_review,
        "final_report_markdown": final_report_markdown,
        "revision_performed": revision_performed,
    }



def build_week6_graph():
    graph = StateGraph(Week6ResearchState)

    graph.add_node("supervise", supervisor_node)
    graph.add_node("specialists", specialists_node)
    graph.add_node("evidence_store", evidence_store_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_edge(START, "supervise")
    graph.add_edge("supervise", "specialists")
    graph.add_edge("specialists", "evidence_store")
    graph.add_edge("evidence_store", "synthesis")
    graph.add_edge("synthesis", "reviewer")
    graph.add_edge("reviewer", END)

    return graph.compile()


week6_graph = build_week6_graph()


st.divider()
st.subheader("Week 6 — Reviewed Synthesis Pipeline")
st.caption("Question → Supervisor → Specialists → Shared Evidence → Synthesis → Reviewer → Final Report")

if st.button("Run Reviewed Synthesis Pipeline", type="primary") and question.strip():
    # Starting state, same pattern as the Week 4 button
    week6_state = {
        "question": question.strip(),
        "assignments": [],
        "specialist_results": [],
        "evidence_records": [],
    }

    # Run the whole pipeline; any failure shows a clean error instead of a traceback
    try:
        with st.spinner("Running research, synthesis, and review..."):
            result = week6_graph.invoke(week6_state)
    except Exception as error:
        st.error(f"The Week 6 reviewed workflow could not finish: {error}")
    else:
        review = result["review"]
        # Escape $ so dollar amounts don't render as LaTeX math
        report_md = result["final_report_markdown"].replace("$", "\\$")

        # Final report: shown openly ONLY on PASS; otherwise labeled unapproved and tucked in an expander
        if review["overall_status"] == "PASS":
            st.success("Report approved by the reviewer.")
            st.markdown(report_md)
        else:
            st.error("UNAPPROVED: this report did not pass review and should not be treated as trusted.")
            with st.expander("Unapproved draft (for inspection only)"):
                st.markdown(report_md)

        if result.get("revision_performed"):
            st.info("One controlled revision was performed after the initial review.")

        # Reviewer scorecard: six scores in a 3-column grid
        st.subheader("Reviewer scorecard")
        cols = st.columns(3)
        for i, (name, score) in enumerate(review["scores"].items()):
            cols[i % 3].metric(name.replace("_", " ").title(), f"{score}/10")

        # Issues: the prompt returns dicts, but a failed review returns plain strings, so handle both
        st.subheader("Issues")
        if review["issues"]:
            for issue in review["issues"]:
                if isinstance(issue, dict):
                    st.markdown(
                        f"- **{issue.get('category', '')}** ({issue.get('section', 'overall')}): "
                        f"{issue.get('explanation', '')}  \n  *Suggested fix:* {issue.get('suggested_fix', '')}"
                    )
                else:
                    st.markdown(f"- {issue}")
        else:
            st.caption("No issues reported.")

        # Research gaps the reviewer flagged for later weeks
        st.subheader("Research gaps")
        if review["research_gaps"]:
            for gap in review["research_gaps"]:
                st.markdown(f"- {gap}")
        else:
            st.caption("No research gaps reported.")

        # Initial review, shown only if a revision happened, so you can compare before and after
        if result.get("revision_performed"):
            with st.expander("Initial review (before revision)"):
                st.json(result["initial_review"])

        # Shared evidence, numbered to match the [E#] citations in the report
        evidence = result.get("evidence_records", [])
        with st.expander(f"Shared evidence used ({len(evidence)} records)"):
            for i, rec in enumerate(evidence, start=1):
                st.markdown(
                    f"**[E{i}]** {rec.get('claim', '')}  \n"
                    f"*{rec.get('source', 'Unknown source')}* — {rec.get('url', 'no url')}"
                )


