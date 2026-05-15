from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """You are a data analysis agent. You must respond with exactly one ```json fenced code block containing a JSON object with keys "thought", "action", and "action_input". No other text outside the JSON block.

Format:
```json
{"thought": "reasoning", "action": "tool_name", "action_input": {"param": "value"}}
```

Strategy:
1. Context is pre-loaded in the task prompt (files, CSV columns, DB schema, knowledge.md). Skip list_context unless you need more detail.
2. Read knowledge.md FIRST if it exists - it contains critical metric definitions, SQL hints, and field semantics.
3. Use execute_python as your PRIMARY tool for data analysis.
4. For databases (.db files): use sqlite3 in execute_python.
5. For CSV files: use pandas in execute_python.
6. For large files (>5MB): use chunked reading or SQL, never load entirely.

CRITICAL RULES FOR CORRECT ANSWERS:

COLUMN RULES (most important for scoring - extra columns REDUCE your score):
- Return ONLY the exact columns the question asks for. NEVER add extra context columns.
  * "What is the average weight?" → 1 column: the average value. NOT (name, weight, average)
  * "Which event has the lowest cost?" → 1 column: event name(s). NOT (event_name, cost)
  * "List the names of Y" → 1 column: names only. NOT (id, name, description)
  * "Tally the element of each molecule" → 1 column: element only. NOT (molecule_id, element)
  * "What is the name and age?" → exactly 2 columns: name and age
  * "How many X?" → 1 column with 1 number
- NEVER add ID, rank, molecule_id, or descriptive columns unless the question explicitly asks for them.
- If the question mentions something (like "lowest cost") as a FILTER condition, do NOT include it as a column.
- "Tally/list/identify X of Y" → return ONLY X, not Y.

VALUE RULES:
- Return ALL rows that match the criteria. If multiple items tie (e.g., same lowest cost), return ALL of them.
- NEVER round or truncate numeric values. Return the EXACT computation result with full precision.
  * WRONG: 182.28 (rounded)  CORRECT: 182.2832618025751 (full precision)
- NEVER add % signs, unit suffixes, or currency symbols to numbers.
- Keep first_name and last_name as SEPARATE columns (do NOT concatenate into full_name).
- For percentage calculations: return the raw decimal unless the question explicitly says "in percentage".

QUERY RULES:
- Read knowledge.md carefully - it often contains exact SQL patterns, metric definitions, and field semantics.
- Pay attention to ALL conditions/filters in the question (year, category, threshold, etc.).
- YEAR FILTER: When filtering by year (e.g., "in 2013"), check the date format. If dates are 'YYYYMM', filter with LIKE '2013%' or SUBSTR(date,1,4)='2013'.
- MONTHLY vs YEARLY: When the question says "monthly" but data is yearly, divide by 12. When "annual" but data is monthly, multiply by 12.
- Use DISTINCT when counting unique items (e.g., COUNT(DISTINCT id)).
- Double-check JOIN conditions and WHERE clauses against the schema.
- When question asks for "type" of something, look for a 'type' column in the relevant table.
- Always print() your final result and verify it makes sense before submitting.
- If stuck after 2 attempts, try a completely different approach.
- TIED ROWS: When filtering for min/max, use WHERE value = (SELECT MIN(value) ...) not LIMIT 1.

GROUPING RULES:
- "Identify the type of expenses" → GROUP BY the 'type' column, NOT by expense_description.
- "List countries of X" → return DISTINCT countries, not all rows.
- When aggregating (SUM, AVG, COUNT), make sure you're grouping by the right column.

DOC PARSING RULES (for .md/.txt files with narrative data):
- Use regex or string parsing in execute_python to extract structured data from narrative documents.
- Look for patterns like "identifier X", "registered as Y", "value is Z" to extract fields.
- When data is in narrative form, write Python code to parse ALL relevant entries systematically.
- Example: re.findall(r'Race ID:\\s*(\\d+)', text) to extract numeric IDs.

Answer submission:
- action="answer", action_input={"columns": [...], "rows": [[...], ...]}
- ALL values in rows must be strings.
- Only include columns the question explicitly asks for. When in doubt, use FEWER columns.""".strip()


RESPONSE_EXAMPLE = """
Example - query database with full precision:
```json
{"thought": "Query the database and print full result.", "action": "execute_python", "action_input": {"code": "import sqlite3\\nimport pandas as pd\\npd.set_option('display.float_format', lambda x: f'{x}')\\nconn = sqlite3.connect('db/data.db')\\nresult = pd.read_sql('SELECT AVG(weight_kg) as avg_weight FROM superhero', conn)\\nprint(result)\\nconn.close()"}}
```

Example - answer with ONLY the asked column (question: "What is the average weight?"):
WRONG: {"columns": ["hero_name", "AVG(weight_kg)"], "rows": [["Superman", "60.779"]]}  ← extra column + rounded
CORRECT:
```json
{"thought": "Return only the single value asked for, with full precision.", "action": "answer", "action_input": {"columns": ["AVG(weight_kg)"], "rows": [["60.77956989247312"]]}}
```

Example - answer for "Which events have the lowest cost?" (return ALL tied rows, NO cost column):
WRONG: {"columns": ["event_name", "cost"], "rows": [["Event A", "6.0"]]}  ← extra column + only 1 row
CORRECT:
```json
{"thought": "3 events tied for lowest cost. Return all of them, only the event name column.", "action": "answer", "action_input": {"columns": ["event_name"], "rows": [["Event A"], ["Event B"], ["Event C"]]}}
```

Example - answer for "Tally the element of each carcinogenic molecule" (NO molecule_id column):
WRONG: {"columns": ["molecule_id", "element"], "rows": [["TR001", "c"], ["TR002", "br"]]}  ← extra molecule_id column
CORRECT:
```json
{"thought": "Return only the element column, not molecule_id.", "action": "answer", "action_input": {"columns": ["element"], "rows": [["c"], ["br"], ["cl"]]}}
```

Example - answer for "List full name and total cost" (keep first_name and last_name SEPARATE):
WRONG: {"columns": ["full_name", "total_cost"], "rows": [["John Smith", "100.0"]]}  ← concatenated name
CORRECT:
```json
{"thought": "Keep first_name and last_name as separate columns.", "action": "answer", "action_input": {"columns": ["first_name", "last_name", "SUM(cost)"], "rows": [["John", "Smith", "100.0"]]}}
```

Example - answer for "What is the average monthly consumption?" (divide yearly total by 12):
WRONG: {"columns": ["avg_monthly"], "rows": [["82027220.3"]]}  ← forgot to divide by 12
CORRECT: First check knowledge.md for the formula, then: AVG(annual_consumption) / 12

Example - answer for "How many patients have abnormal levels?" (single count, full precision):
```json
{"thought": "The count is 4.", "action": "answer", "action_input": {"columns": ["COUNT(DISTINCT ID)"], "rows": [["4"]]}}
```""".strip()

FORMAT_REMINDER = (
    "Respond with ONLY a ```json fenced block. "
    "Keys: thought (string), action (string), action_input (object)."
)


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        f"Available tools:\n{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLE}\n\n"
        f"{FORMAT_REMINDER}"
    )


def build_task_prompt(task: PublicTask, context_info: str | None = None) -> str:
    parts = [f"Question: {task.question}"]
    if context_info:
        parts.append(f"\nContext files:\n{context_info}")
    parts.append(
        "\nAll file paths are relative to the task context directory. "
        "When you have the final answer table, call the `answer` tool. "
        "REMEMBER: (1) Only include columns the question asks for. "
        "(2) Return ALL matching rows (check for ties). "
        "(3) Keep full numeric precision. "
        "(4) If knowledge.md exists, read it first."
    )
    return "\n".join(parts)


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    if len(rendered) > 8000:
        rendered = rendered[:7500] + "\n... [truncated, use more specific queries]"
    return f"Observation:\n{rendered}"


def build_format_correction_prompt() -> str:
    return (
        "Your previous response was not valid JSON. "
        "You MUST respond with exactly one ```json fenced code block containing: "
        '{"thought": "...", "action": "tool_name", "action_input": {...}}'
    )


def build_reflection_prompt(reason: str) -> str:
    return f"Reflection: {reason} Try a different approach."


def build_force_answer_prompt() -> str:
    return (
        "CRITICAL: You MUST submit your answer NOW using action=\"answer\". You are running out of steps. "
        "RULES FOR YOUR FINAL ANSWER: "
        "1) Include ONLY the columns the question asks for (fewer columns = better score). "
        "2) Include ALL rows that match the criteria (check for ties). "
        "3) Keep FULL numeric precision. Do NOT round. Do NOT add % signs. "
        "4) All values must be strings in the rows array. "
        'Format: {"thought": "...", "action": "answer", "action_input": {"columns": ["col1"], "rows": [["val1"]]}}. '
        "Even a partial answer scores better than no answer. Submit NOW."
    )


def prepare_task_context(task: PublicTask) -> str:
    """Pre-analyze task context to inject into the user prompt.
    Reads file list, sizes, CSV headers, DB schema, JSON structure, and knowledge.md.
    """
    context_dir = task.assets.context_dir
    if not context_dir.exists():
        return ""

    parts = []

    # List all files with sizes
    files_info = []
    csv_files = []
    db_files = []
    doc_files = []
    json_files = []
    knowledge_path = None

    for f in sorted(context_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(context_dir)
        size = f.stat().st_size
        if size < 1024:
            size_str = f"{size}B"
        elif size < 1024 * 1024:
            size_str = f"{size//1024}KB"
        else:
            size_str = f"{size//(1024*1024)}MB"
        files_info.append(f"  {rel} ({size_str})")

        suffix = f.suffix.lower()
        name = f.name.lower()
        if name == "knowledge.md":
            knowledge_path = f
        elif suffix == ".csv":
            csv_files.append(f)
        elif suffix in (".db", ".sqlite", ".sqlite3"):
            db_files.append(f)
        elif suffix in (".md", ".txt") and "doc" in str(rel):
            doc_files.append(f)
        elif suffix == ".json":
            json_files.append(f)

    if files_info:
        parts.append("Files:\n" + "\n".join(files_info))

    # CSV headers
    csv_headers = []
    for csv_path in csv_files:
        try:
            if csv_path.stat().st_size > 10 * 1024 * 1024:
                continue
            with open(csv_path, encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                headers = next(reader, [])
            rel = csv_path.relative_to(context_dir)
            csv_headers.append(f"  {rel}: [{', '.join(headers)}]")
        except Exception:
            pass
    if csv_headers:
        parts.append("CSV columns:\n" + "\n".join(csv_headers))

    # DB schema
    db_schemas = []
    for db_path in db_files:
        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [r[0] for r in cur.fetchall()]
            rel = db_path.relative_to(context_dir)
            schema_lines = [f"  {rel}:"]
            for table in tables:
                cur.execute(f"PRAGMA table_info({table})")
                cols = [f"{r[1]}({r[2]})" for r in cur.fetchall()]
                schema_lines.append(f"    {table}: {', '.join(cols)}")
            conn.close()
            db_schemas.append("\n".join(schema_lines))
        except Exception:
            pass
    if db_schemas:
        parts.append("DB schema:\n" + "\n".join(db_schemas))

    # JSON structure
    json_hints = []
    for json_path in json_files:
        try:
            if json_path.stat().st_size > 5 * 1024 * 1024:
                continue
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            rel = json_path.relative_to(context_dir)
            if isinstance(raw, dict) and "records" in raw:
                records = raw["records"]
                table_name = raw.get("table", json_path.stem)
                if records:
                    cols = list(records[0].keys())
                    json_hints.append(f"  {rel}: table='{table_name}', {len(records)} records, columns: {cols}")
            elif isinstance(raw, list) and raw:
                cols = list(raw[0].keys()) if isinstance(raw[0], dict) else []
                json_hints.append(f"  {rel}: {len(raw)} records, columns: {cols}")
        except Exception:
            pass
    if json_hints:
        parts.append("JSON structure:\n" + "\n".join(json_hints))

    # Doc file previews
    doc_previews = []
    for doc_path in doc_files:
        try:
            rel = doc_path.relative_to(context_dir)
            text = doc_path.read_text(encoding="utf-8", errors="replace")
            preview = text[:400].replace("\n", " ").strip()
            doc_previews.append(f"  {rel}: {preview}...")
        except Exception:
            pass
    if doc_previews:
        parts.append("Doc previews:\n" + "\n".join(doc_previews))

    # Knowledge.md (highest priority, up to 8000 chars)
    if knowledge_path:
        try:
            content = knowledge_path.read_text(encoding="utf-8", errors="replace")
            if len(content) > 8000:
                content = content[:8000] + "\n... [truncated]"
            parts.append(f"knowledge.md:\n{content}")
        except Exception:
            pass

    return "\n\n".join(parts)
