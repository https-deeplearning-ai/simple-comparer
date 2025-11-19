# app/tool_router_logic.py
import json
import asyncio
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from fastapi import HTTPException

from duel_core import call_openai  # reutilizamos el caller


# Catalog of tools the LLM can choose from
TOOL_ROUTER_CATALOG = [
    {
        "name": "web_search",
        "description": "General web or news search for up-to-date information.",
        "default_auto_trigger": True,
    },
    {
        "name": "arxiv_search",
        "description": "Search scientific, academic, or research papers on arXiv.",
        "default_auto_trigger": True,
    },
    {
        "name": "wikipedia_summary",
        "description": "Get background or encyclopedic information from Wikipedia.",
        "default_auto_trigger": True,
    },
    {
        "name": "weather_lookup",
        "description": "Get current or forecast weather for a specific location.",
        "default_auto_trigger": True,
    },
    {
        "name": "connector_lookup",
        "description": "Query internal documents, tables, or files through a connector.",
        "default_auto_trigger": False,  # manual toggle in UI
    },
    {
        "name": "python_eval",
        "description": "Run simple math or small code-like calculations.",
        "default_auto_trigger": True,
    },
]

TOOL_ROUTER_BY_NAME = {t["name"]: t for t in TOOL_ROUTER_CATALOG}


class ToolRouterAnalyzeRequest(BaseModel):
    query: str = Field(description="User query text to analyze for tool activation.")
    model: Optional[str] = Field(
        default="gpt-4.1-mini",
        description="Routing model used to decide tools (e.g., gpt-4.1-mini, gpt-4o-mini)."
    )


class ToolRouterToolSelection(BaseModel):
    name: str
    reason: str
    auto_trigger: bool
    description: str
    # Respuesta corta de la herramienta
    result_preview: Optional[str] = None


class ToolRouterSpan(BaseModel):
    start: int
    end: int
    tool: str


class ToolRouterAnalyzeResponse(BaseModel):
    query: str
    tools: List[ToolRouterToolSelection]
    spans: List[ToolRouterSpan]


TOOL_ROUTER_SYSTEM_PROMPT = """You are a query-to-tool router.
Your job is to:
1) Read the user query.
2) Decide which tools from the catalog are relevant.
3) Mark which text spans in the query activate which tools.
4) Explain briefly why each tool is selected.

Tools you can use:
- web_search: For general web/news/price/information queries that need up-to-date or broad web data.
- arxiv_search: For scientific, academic, or research paper questions.
- wikipedia_summary: For definitions, background info, or encyclopedic knowledge.
- weather_lookup: For anything about weather, temperature, or forecast in a location.
- connector_lookup: For anything like "my files", "my database", "my reports", internal tables, or connectors.
- python_eval: For math, numeric reasoning, or code-like calculations.

Output STRICTLY in JSON with this exact structure:

{
  "tools": [
    {
      "name": "web_search",
      "reason": "Short explanation.",
      "auto_trigger": true
    }
  ],
  "spans": [
    {
      "start": 10,
      "end": 35,
      "tool": "web_search"
    }
  ]
}

Rules:
- "start" and "end" are zero-based character indices into the original query string.
- Spans must not overlap.
- If no tool is needed, return "tools": [] and "spans": [].
- Only use tool names from the catalog. No invented tool names.
- Be conservative: prefer fewer tools over many.
"""


async def run_tool_router_llm(query: str, model_name: str = "gpt-4.1-mini") -> Dict[str, Any]:
    """
    Use an LLM (router model configurable) to decide which tools to use
    and which spans trigger them.
    """
    router_prompt = f"""{TOOL_ROUTER_SYSTEM_PROMPT}

User query:
\"\"\"{query}\"\"\"

Return only valid JSON as specified."""
    text = await call_openai(model_name, router_prompt, temperature=0.1, max_tokens=400)

    try:
        block = text.strip()
        i, j = block.find("{"), block.rfind("}")
        if i != -1 and j != -1:
            block = block[i:j+1]
        data = json.loads(block)
        return data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse tool-router JSON: {e} | raw={text}",
        )


# --------- Tool "backend" execution (llamando al modelo) ---------

TOOL_BACKEND_PROMPT = """You are the backend for a tool named: {tool_name}.

Your job is to respond to the original user query in the style and capabilities of this tool:

- web_search:
  - Provide a concise answer with 3–5 bullet points.
  - Mention plausible sources or site names (e.g., 'Wikipedia', 'official docs') but keep URLs generic.
- arxiv_search:
  - Suggest 3–5 relevant paper titles with year and 1–2 sentence summaries.
- wikipedia_summary:
  - Give a short definition and 2–4 key facts, written like a compact encyclopedic entry.
- weather_lookup:
  - Provide a short, friendly forecast-style answer for the location and time mentioned, making it clear it's an approximate description rather than real-time data.
- connector_lookup:
  - Explain briefly what kind of internal tables/files/reports you would query and summarize the type of insights you would return.
- python_eval:
  - Interpret any expression(s) in the query.
  - Show the main steps of the calculation and the final numeric result.

Keep the response direct and helpful.

Original user query:
\"\"\"{query}\"\"\"

Tool-style response:
"""


async def run_single_tool_backend(tool_name: str, query: str, model_name: str) -> str:
    """
    Ejecuta una respuesta estilo 'tool' usando el modelo dado.
    """
    prompt = TOOL_BACKEND_PROMPT.format(tool_name=tool_name, query=query)
    text = await call_openai(model_name, prompt, temperature=0.2, max_tokens=400)
    return (text or "").strip()


# --------- High-level: analyze + run tools ---------

async def analyze_tools(req: ToolRouterAnalyzeRequest) -> ToolRouterAnalyzeResponse:
    """
    High-level helper:
      1) Llama al router LLM para decidir tools + spans.
      2) Ejecuta cada tool con un backend LLM y rellena result_preview.
    """
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    model_name = req.model or "gpt-4.1-mini"

    # 1) Router decide qué tools y spans
    llm_data = await run_tool_router_llm(query, model_name)

    tools_raw = llm_data.get("tools", []) or []
    spans_raw = llm_data.get("spans", []) or []

    tools: List[ToolRouterToolSelection] = []
    for t in tools_raw:
        name = t.get("name")
        if name not in TOOL_ROUTER_BY_NAME:
            continue

        meta = TOOL_ROUTER_BY_NAME[name]
        reason = (t.get("reason") or "").strip() or f"The router decided {name} is relevant."
        auto_trigger = bool(t.get("auto_trigger")) if "auto_trigger" in t else meta["default_auto_trigger"]

        tools.append(
            ToolRouterToolSelection(
                name=name,
                reason=reason,
                auto_trigger=auto_trigger,
                description=meta["description"],
                result_preview=None,
            )
        )

    # 2) Ejecutar las herramientas seleccionadas (para este demo, todas las tools detectadas)
    async def run_for_selection(sel: ToolRouterToolSelection):
        try:
            preview = await run_single_tool_backend(sel.name, query, model_name)
            sel.result_preview = preview
        except Exception as e:
            sel.result_preview = f"[Error running tool '{sel.name}': {e}]"

    if tools:
        await asyncio.gather(*(run_for_selection(t) for t in tools))

    # 3) Construir spans
    spans: List[ToolRouterSpan] = []
    for s in spans_raw:
        tool_name = s.get("tool")
        if tool_name not in TOOL_ROUTER_BY_NAME:
            continue
        try:
            start = int(s.get("start", 0))
            end = int(s.get("end", 0))
        except Exception:
            continue

        if 0 <= start < end <= len(query):
            spans.append(
                ToolRouterSpan(
                    start=start,
                    end=end,
                    tool=tool_name,
                )
            )

    return ToolRouterAnalyzeResponse(
        query=query,
        tools=tools,
        spans=spans,
    )
