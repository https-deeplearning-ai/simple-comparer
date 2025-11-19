# app/duel_logic.py
import os
import time
import json
import math
import asyncio
from typing import Literal, Optional, Dict, Any, Deque, List
from collections import deque

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

load_dotenv()

APP_NAME = "LLM Duel – OpenAI (topics + compact charts)"
HISTORY_MAX_LEN = 500

# ---- Model pool (id, label, pricing USD per 1K tokens)
MODEL_POOL = [
    {"id": "gpt-4o",        "label": "GPT-4o (general)",          "pricing": {"in": 5.00,  "out": 15.00}},
    {"id": "gpt-4o-mini",   "label": "GPT-4o mini (fast/cheap)",  "pricing": {"in": 0.15,  "out": 0.60}},
    {"id": "gpt-4.1",       "label": "GPT-4.1 (reasoning)",       "pricing": {"in": 5.00,  "out": 15.00}},
    {"id": "gpt-4.1-mini",  "label": "GPT-4.1 mini (balanced)",   "pricing": {"in": 0.30,  "out": 1.25}},
    {"id": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo (legacy)",    "pricing": {"in": 0.50,  "out": 1.50}},
]
COSTS = {m["id"]: m["pricing"] for m in MODEL_POOL}

# ---------- OpenAI / AISuite clients ----------

USE_AISUITE = True
_ais_client = None
try:
    from aisuite import Client as AISClient  # pip install aisuite
    _ais_client = AISClient()
except Exception:
    USE_AISUITE = False

from openai import OpenAI  # OpenAI official SDK (v1+)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Please export your OpenAI API key in the environment."
    )

_openai_client = OpenAI(api_key=OPENAI_API_KEY)

TaskType = Literal["factual", "creative", "reasoning", "code"]


class ModelCfg(BaseModel):
    model: str = Field(description="OpenAI model id (e.g., gpt-4o-mini)")
    temperature: float = 0.7
    max_tokens: int = 512


class CompareRequest(BaseModel):
    prompt: str
    task_type: TaskType = "factual"
    a: ModelCfg
    b: ModelCfg


class Metrics(BaseModel):
    latency_ms: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    char_len: int
    paragraphs: int
    url_count: int
    instruction_overlap: float


class ModelRun(BaseModel):
    name: str
    temperature: float
    max_tokens: int
    answer: str
    metrics: Metrics


class CompareResult(BaseModel):
    id: str
    timestamp: float
    task_type: str
    prompt: str
    model_a: ModelRun
    model_b: ModelRun
    auto_judge: Optional[Dict[str, Any]] = None


class JudgeRequest(BaseModel):
    prompt: str
    task_type: TaskType
    answer_a: str
    answer_b: str
    judge_model: Optional[ModelCfg] = None  # if not provided, default


# -------- utils

def approx_token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))  # rough chars→tokens


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p = COSTS.get(model, {"in": 0.0, "out": 0.0})
    return round((tokens_in / 1000.0) * p["in"] + (tokens_out / 1000.0) * p["out"], 6)


def compute_metrics(prompt: str, answer: str, model_name: str, t0: float, t1: float) -> Metrics:
    tokens_in = approx_token_count(prompt)
    tokens_out = approx_token_count(answer)
    cost = estimate_cost(model_name, tokens_in, tokens_out)
    char_len = len(answer or "")
    paragraphs = sum(1 for ln in (answer or "").splitlines() if ln.strip())
    url_count = (answer or "").count("http://") + (answer or "").count("https://")
    p_words = {w.strip(".,:;!?'\"()[]").lower() for w in prompt.split() if len(w) > 3}
    a_words = {w.strip(".,:;!?'\"()[]").lower() for w in (answer or "").split() if len(w) > 3}
    overlap = round(len(p_words & a_words) / len(p_words), 3) if p_words else 0.0
    return Metrics(
        latency_ms=int((t1 - t0) * 1000),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        char_len=char_len,
        paragraphs=paragraphs,
        url_count=url_count,
        instruction_overlap=overlap
    )


# -------- OpenAI callers --------

async def call_openai_via_aisuite(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    """
    Uses aisuite if available.
    """
    resp = await _ais_client.achat(
        model=f"openai/{model}",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if isinstance(resp, dict):
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            return str(resp)
    return getattr(resp, "content", str(resp))


async def call_openai_sdk(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    """
    Call OpenAI SDK in a thread to avoid blocking.
    """
    if _openai_client is None:
        raise RuntimeError("OpenAI client not initialized")

    def _do_call():
        r = _openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return r.choices[0].message.content or ""

    return await asyncio.to_thread(_do_call)


async def call_openai(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    """
    Router: try aisuite first, fallback to SDK.
    """
    if USE_AISUITE and _ais_client is not None:
        try:
            return await call_openai_via_aisuite(model, prompt, temperature, max_tokens)
        except Exception:
            return await call_openai_sdk(model, prompt, max_tokens=max_tokens, temperature=temperature)
    return await call_openai_sdk(model, prompt, temperature, max_tokens)


# -------- Judge logic --------

JUDGE_PROMPT = """You are a strict evaluator. Task type: {task_type}
User prompt:
---
{prompt}
---

Option 1:
---
{a}
---

Option 2:
---
{b}
---

First, score each option from 1-5 on:
1) Clarity
2) Task Fit
3) Structure
4) Safety
{extra}

Second, extract topics (3–6 concise phrases per option) that summarize WHAT each option talks about.

Third, compute topic coverage vs the user prompt (0.0–1.0) for each option. Also list up to 3 missing topics per option.

Finally, decide: WINNER = "Option 1", "Option 2", or "Tie".

Return STRICT JSON with keys:
{
  "o1": {"clarity":_, "fit":_, "structure":_, "safety":_, "correctness":_?},
  "o2": {"clarity":_, "fit":_, "structure":_, "safety":_, "correctness":_?},
  "topics_A": ["...", "..."],
  "topics_B": ["...", "..."],
  "shared_topics": ["...", "..."],
  "coverage_A": 0.0,
  "coverage_B": 0.0,
  "missing_A": ["..."],
  "missing_B": ["..."],
  "winner": "Option 1|Option 2|Tie",
  "rationale_short": "one-paragraph explanation"
}
"""


def heuristic_judge(task_type: str, prompt: str, a: str, b: str) -> Dict[str, Any]:
    def score(ans: str) -> Dict[str, float]:
        if not ans:
            return {"clarity": 0, "fit": 0, "structure": 0, "safety": 5.0, "correctness": 0}
        clarity = min(5, 2 + ans.count("\n\n") + len(ans) / 400)
        overlap_ratio = len(set(prompt.lower().split()) &
                            set((ans or "").lower().split())) / max(1, len(prompt.split()))
        fit = 1 + 4 * overlap_ratio
        structure = min(5, 1 + ans.count("- ") + ans.count("* ") + ans.count("1. "))
        safety = 5.0
        s = {"clarity": float(clarity), "fit": float(fit), "structure": float(structure), "safety": safety}
        if task_type == "factual":
            s["correctness"] = 3.0
        return s

    def topics(ans: str):
        import re
        words = [w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", ans or "")]
        stop = set("about into there which their would could should other these those being while where under over after before because within without between across using among since often".split())
        freq = {}
        for w in words:
            if w in stop:
                continue
            freq[w] = freq.get(w, 0) + 1
        ranked = sorted(freq, key=freq.get, reverse=True)
        return [w for w in ranked[:5]]

    o1, o2 = score(a), score(b)
    t1, t2 = sum(o1.values()), sum(o2.values())
    winner = "Tie" if abs(t1 - t2) < 0.75 else ("Option 1" if t1 > t2 else "Option 2")

    topics_A = topics(a)
    topics_B = topics(b)
    shared = [t for t in topics_A if t in topics_B]
    coverage_A = min(1.0, len(shared) / max(1, len(set(topics_A + topics_B))))
    coverage_B = coverage_A

    return {
        "o1": o1,
        "o2": o2,
        "winner": winner,
        "topics_A": topics_A,
        "topics_B": topics_B,
        "shared_topics": shared,
        "coverage_A": round(coverage_A, 3),
        "coverage_B": round(coverage_B, 3),
        "missing_A": [],
        "missing_B": [],
        "rationale_short": "Heuristic verdict based on structure and lexical overlap."
    }


def _normalize_scores(raw: Dict[str, Any], fallback: Dict[str, float]) -> Dict[str, float]:
    raw = raw or {}
    out = {}

    def pick(key: str, *aliases: str) -> float:
        for k in (key, *aliases):
            if k in raw and raw[k] is not None:
                try:
                    val = float(raw[k])
                    if val == 0.0 and key in fallback and fallback[key] > 0:
                        return float(fallback[key])
                    return val
                except Exception:
                    continue
        return float(fallback.get(key, 0.0))

    out["clarity"] = pick("clarity")
    out["fit"] = pick("fit", "task_fit", "taskFit", "task_fit_score")
    out["structure"] = pick("structure", "organization")
    out["safety"] = pick("safety")
    out["correctness"] = pick("correctness", "accuracy")
    return out


async def model_judge(judge_model: str, task_type: str, prompt: str, a: str, b: str) -> Dict[str, Any]:
    extra = "5) Apparent Correctness (if factual)."
    jp = JUDGE_PROMPT.format(
        task_type=task_type,
        prompt=prompt,
        a=a,
        b=b,
        extra=extra if task_type == "factual" else ""
    )

    text = await call_openai(judge_model, jp, temperature=0.0, max_tokens=700)
    heur = heuristic_judge(task_type, prompt, a, b)

    try:
        block = text.strip()
        i, j = block.find("{"), block.rfind("}")
        if i != -1 and j != -1:
            block = block[i:j+1]
        data = json.loads(block)

        raw_o1 = data.get("o1", {}) or {}
        raw_o2 = data.get("o2", {}) or {}

        o1 = _normalize_scores(raw_o1, heur.get("o1", {}))
        o2 = _normalize_scores(raw_o2, heur.get("o2", {}))

        def _safe_list(x):
            if isinstance(x, list):
                return [str(i) for i in x][:8]
            return []

        out = {
            "o1": o1,
            "o2": o2,
            "topics_A": _safe_list(data.get("topics_A", [])),
            "topics_B": _safe_list(data.get("topics_B", [])),
            "shared_topics": _safe_list(data.get("shared_topics", [])),
            "coverage_A": float(data.get("coverage_A", heur.get("coverage_A", 0.0))),
            "coverage_B": float(data.get("coverage_B", heur.get("coverage_B", 0.0))),
            "missing_A": _safe_list(data.get("missing_A", heur.get("missing_A", []))),
            "missing_B": _safe_list(data.get("missing_B", heur.get("missing_B", []))),
            "winner": data.get("winner", heur.get("winner", "Tie")),
            "rationale_short": str(data.get("rationale_short", heur.get("rationale_short", "")))[:800],
        }
        return out
    except Exception:
        return heur


# -------- High-level helpers used by FastAPI --------

async def run_compare(req: CompareRequest) -> CompareResult:
    t0 = time.time()

    async def run_one(cfg: ModelCfg) -> ModelRun:
        start = time.time()
        try:
            answer = await call_openai(cfg.model, req.prompt, cfg.temperature, cfg.max_tokens)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling {cfg.model}: {e}")
        end = time.time()
        metrics = compute_metrics(req.prompt, answer, cfg.model, start, end)
        return ModelRun(
            name=f"{cfg.model}",
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            answer=answer,
            metrics=metrics
        )

    a_run, b_run = await asyncio.gather(run_one(req.a), run_one(req.b))
    return CompareResult(
        id=f"cmp_{int(t0 * 1000)}",
        timestamp=t0,
        task_type=req.task_type,
        prompt=req.prompt,
        model_a=a_run,
        model_b=b_run
    )


async def run_judge(req: JudgeRequest, history: Deque[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        judge_model_name = (req.judge_model.model if req.judge_model else "gpt-4o-mini")
        data = await model_judge(judge_model_name, req.task_type, req.prompt, req.answer_a, req.answer_b)
    except Exception:
        data = heuristic_judge(req.task_type, req.prompt, req.answer_a, req.answer_b)

    for it in reversed(history):
        if (
            it["prompt"] == req.prompt
            and it["model_a"]["answer"] == req.answer_a
            and it["model_b"]["answer"] == req.answer_b
        ):
            it["auto_judge"] = data
            break
    return data


def export_history_csv(history: Deque[Dict[str, Any]]) -> StreamingResponse:
    headers = [
        "id","timestamp","task_type","prompt",
        "a_name","a_latency_ms","a_tokens_in","a_tokens_out","a_cost_usd","a_char_len","a_paragraphs","a_url_count","a_overlap",
        "b_name","b_latency_ms","b_tokens_in","b_tokens_out","b_cost_usd","b_char_len","b_paragraphs","b_url_count","b_overlap",
        "auto_winner"
    ]

    def iter_rows():
        yield ",".join(headers) + "\n"
        for it in history:
            a, b = it["model_a"], it["model_b"]
            row = [
                it["id"], str(it["timestamp"]), it["task_type"], json.dumps(it["prompt"]).replace(",", " "),
                a["name"], str(a["metrics"]["latency_ms"]), str(a["metrics"]["tokens_in"]), str(a["metrics"]["tokens_out"]), str(a["metrics"]["cost_usd"]), str(a["metrics"]["char_len"]), str(a["metrics"]["paragraphs"]), str(a["metrics"]["url_count"]), str(a["metrics"]["instruction_overlap"]),
                b["name"], str(b["metrics"]["latency_ms"]), str(b["metrics"]["tokens_in"]), str(b["metrics"]["tokens_out"]), str(b["metrics"]["cost_usd"]), str(b["metrics"]["char_len"]), str(b["metrics"]["paragraphs"]), str(b["metrics"]["url_count"]), str(b["metrics"]["instruction_overlap"]),
                it.get("auto_judge", {}).get("winner", "")
            ]
            yield ",".join(row) + "\n"

    return StreamingResponse(
        iter_rows(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=duel_export.csv"}
    )
