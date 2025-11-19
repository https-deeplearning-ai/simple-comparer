# app/main.py
import os
from collections import deque
from typing import Deque, Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from duel_core import (
    APP_NAME,
    HISTORY_MAX_LEN,
    MODEL_POOL,
    CompareRequest,
    JudgeRequest,
    run_compare,
    run_judge,
    export_history_csv,
)
from tool_router_core import (
    ToolRouterAnalyzeRequest,
    analyze_tools,
)
from prompt_eval_core import (
    PromptEvalRequest,
    PromptEvalResponse,
    evaluate_prompt,
)
from context_viz_core import (
    ContextVizRequest,
    ContextVizResponse,
    analyze_context,
)

# ------- App & filesystem setup -------

app = FastAPI(title=APP_NAME)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")
os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# In-memory history for duel
history: Deque[Dict[str, Any]] = deque(maxlen=HISTORY_MAX_LEN)


# ------- Pages -------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Index = pestaña Duel por defecto
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_name": APP_NAME,
            "active_tab": "duel",
        },
    )


@app.get("/duel", response_class=HTMLResponse)
async def duel_page(request: Request):
    return templates.TemplateResponse(
        "duel.html",
        {
            "request": request,
            "app_name": APP_NAME,
            "active_tab": "duel",
        },
    )


@app.get("/tool-router", response_class=HTMLResponse)
async def tool_router_page(request: Request):
    return templates.TemplateResponse(
        "router.html",
        {
            "request": request,
            "app_name": APP_NAME,
            "active_tab": "router",
        },
    )


@app.get("/prompt-eval", response_class=HTMLResponse)
async def prompt_eval_page(request: Request):
    return templates.TemplateResponse(
        "prompt.html",
        {
            "request": request,
            "app_name": APP_NAME,
            "active_tab": "prompt",
        },
    )


@app.get("/context-viz", response_class=HTMLResponse)
async def context_viz_page(request: Request):
    return templates.TemplateResponse(
        "context.html",
        {
            "request": request,
            "app_name": APP_NAME,
            "active_tab": "context",
        },
    )


# ------- Duel: models, compare, judge, history, export -------

@app.get("/models")
async def list_models():
    return {"models": MODEL_POOL}


@app.post("/compare")
async def compare(req: CompareRequest):
    result = await run_compare(req)
    history.append(result.model_dump())
    return result


@app.post("/judge")
async def judge(req: JudgeRequest):
    data = await run_judge(req, history)
    return data


@app.get("/history")
async def get_history():
    return list(history)


@app.get("/export.csv")
async def export_csv():
    return export_history_csv(history)


# ------- Tool Router endpoints -------

@app.post("/tool-router/analyze")
async def tool_router_analyze(req: ToolRouterAnalyzeRequest):
    resp = await analyze_tools(req)
    return resp


# ------- Prompt eval -------

@app.post("/prompt-eval/analyze", response_model=PromptEvalResponse)
async def prompt_eval_analyze(req: PromptEvalRequest):
    return await evaluate_prompt(req)


# ------- Context viz -------

@app.post("/context-viz/analyze", response_model=ContextVizResponse)
async def context_viz_analyze(req: ContextVizRequest):
    return await analyze_context(req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)