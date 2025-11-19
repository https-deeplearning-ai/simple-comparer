# app/context_viz_logic.py

import math
from typing import List, Optional, Dict, Any, Literal

import json
import re

from fastapi import HTTPException
from pydantic import BaseModel, Field

from duel_core import call_openai  # mismo caller que ya usas en el proyecto

# Intentamos usar tiktoken para tokenizar de verdad
try:
    import tiktoken

    def get_tokenizer_for_model(model_name: str):
        # Ajusta mapping según tus modelos
        if "gpt-4o" in model_name or "gpt-4.1" in model_name:
            return tiktoken.get_encoding("o200k_base")
        # fallback razonable para otros
        return tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str, model_name: str) -> int:
        enc = get_tokenizer_for_model(model_name)
        return len(enc.encode(text or ""))
except Exception:
    # Fallback super simple si no está tiktoken (no recomendado en producción)
    def count_tokens(text: str, model_name: str) -> int:
        # muy rough: ~4 chars por token
        return max(1, math.ceil(len(text or "") / 4))


# ---------- Models ----------

SegmentType = Literal["system", "user", "assistant", "web", "retrieved", "file", "other"]


class ContextSegment(BaseModel):
    id: str
    role: str = Field(..., description="Role in the chat sense (system/user/assistant/tool).")
    type: SegmentType = Field(..., description="High-level segment type (system/user/assistant/web/retrieved/file/other).")
    source: str = Field(..., description="Label or source of the segment, e.g., 'sys-1', 'user-1', 'web-1'.")
    text: str

    token_count: int
    start_token: int
    end_token: int

    decay_weight: float
    preview: Optional[str] = None  # Short LLM-generated summary (optional)


class TokenBreakdown(BaseModel):
    system: int = 0
    user: int = 0
    assistant: int = 0
    web: int = 0
    retrieved: int = 0
    file: int = 0
    other: int = 0


class ContextVizRequest(BaseModel):
    model: str = Field(
        default="gpt-4o-mini",
        description="Model / tokenizer name used for counting tokens."
    )
    max_context_tokens: int = Field(
        default=8000,
        gt=0,
        le=128000,
        description="Max context window size for visualization."
    )
    scenario_id: Optional[str] = Field(
        default=None,
        description="Optional predefined scenario id (e.g., 'rag_web_retrieved')."
    )
    manual_text: Optional[str] = Field(
        default=None,
        description="If provided & non-empty, this overrides the example and is used as a single user segment."
    )
    use_llm_previews: bool = Field(
        default=False,
        description="If true, calls the LLM to generate short previews per segment."
    )
    use_llm_rot_explanation: bool = Field(
        default=False,
        description="If true, calls the LLM to generate a global explanation of context rot for this scenario."
    )
    simulate_decay: bool = Field(
        default=True,
        description="If true, compute a decay_weight based on recency."
    )


class ContextVizResponse(BaseModel):
    model: str
    max_context_tokens: int
    total_tokens: int
    usage_pct: float
    token_breakdown: TokenBreakdown
    segments: List[ContextSegment]
    rot_explanation: Optional[str] = None


# ---------- Example scenarios ----------

def _scenario_rag_web_retrieved() -> List[Dict[str, Any]]:
    """
    RAG scenario similar to the one que estabas describiendo:
    - system
    - user
    - web search snippet
    - retrieved doc
    - assistant answer
    """
    return [
        {
            "id": "sys-1",
            "role": "system",
            "type": "system",
            "source": "system",
            "text": (
                "You are a research assistant. When answering, rely mainly on the retrieved "
                "documents and cite them inline as [doc1], [doc2], etc. If something is not "
                "supported by the documents, say that explicitly."
            ),
        },
        {
            "id": "user-1",
            "role": "user",
            "type": "user",
            "source": "user",
            "text": (
                "What are the main limitations of diffusion models for image generation in terms "
                "of compute and data requirements?"
            ),
        },
        {
            "id": "web-1",
            "role": "tool",
            "type": "web",
            "source": "web",
            "text": (
                "Web search result: Blog article summarizing diffusion models, noting that they "
                "require large-scale training data and expensive GPU compute, but can be fine-tuned "
                "for specific tasks. Mentions that sampling can be slow at inference time."
            ),
        },
        {
            "id": "retrieved-1",
            "role": "tool",
            "type": "retrieved",
            "source": "retrieved",
            "text": (
                "Retrieved paper [doc1]: Highlights that diffusion models need many training steps, "
                "high VRAM usage, and heavy sampling costs at inference time. Discusses challenges "
                "with scaling to higher resolutions."
            ),
        },
        {
            "id": "assistant-1",
            "role": "assistant",
            "type": "assistant",
            "source": "assistant",
            "text": (
                "Diffusion models have several limitations around compute and data. First, training "
                "from scratch typically requires large datasets and long training runs [doc1]. "
                "They also consume significant GPU memory and can be slow at inference time due to "
                "the many sampling steps.\n\nIn practice, teams often rely on pre-trained models "
                "and fine-tuning to reduce compute requirements, but high-resolution generation "
                "can still be expensive and time-consuming."
            ),
        },
    ]


def _scenario_chatbot_customer_support() -> List[Dict[str, Any]]:
    return [
        {
            "id": "sys-1",
            "role": "system",
            "type": "system",
            "source": "system",
            "text": (
                "You are a helpful customer support agent for an e-commerce store. "
                "Always answer in a friendly and concise way. Ask for clarification if "
                "the user question is ambiguous."
            ),
        },
        {
            "id": "user-1",
            "role": "user",
            "type": "user",
            "source": "user",
            "text": "Hi, my order hasn't arrived yet. It was supposed to come last Friday.",
        },
        {
            "id": "assistant-1",
            "role": "assistant",
            "type": "assistant",
            "source": "assistant",
            "text": (
                "I'm sorry to hear your order is delayed. Could you share your order ID "
                "so I can check the status for you?"
            ),
        },
        {
            "id": "user-2",
            "role": "user",
            "type": "user",
            "source": "user",
            "text": "Sure, it's #12345. It was a pair of running shoes.",
        },
        {
            "id": "assistant-2",
            "role": "assistant",
            "type": "assistant",
            "source": "assistant",
            "text": (
                "Thanks! I'm checking order #12345. It looks like the package is in transit "
                "and should arrive in the next 1–2 business days. I'll also send you a "
                "tracking link."
            ),
        },
    ]


def _scenario_long_system_prompt() -> List[Dict[str, Any]]:
    return [
        {
            "id": "sys-1",
            "role": "system",
            "type": "system",
            "source": "system",
            "text": (
                "You are an AI coding assistant that strictly follows style guidelines:\n"
                "- Use Python 3.11 features when appropriate.\n"
                "- Add type hints to all functions and methods.\n"
                "- Prefer list comprehensions over manual loops when it improves readability.\n"
                "- Always include a short docstring for public functions.\n"
                "- Avoid global state and prefer dependency injection.\n"
                "- Never write code that accesses the network or the local filesystem.\n"
                "- When unsure, ask clarifying questions before answering.\n"
            ),
        },
        {
            "id": "user-1",
            "role": "user",
            "type": "user",
            "source": "user",
            "text": "Write a function to compute the moving average over a list of floats.",
        },
    ]


SCENARIOS: Dict[str, Any] = {
    "rag_web_retrieved": _scenario_rag_web_retrieved,
    "customer_support": _scenario_chatbot_customer_support,
    "long_system_prompt": _scenario_long_system_prompt,
}


# ---------- Decay / context rot ----------

def compute_decay_weights(segments_tokens: List[int]) -> List[float]:
    """
    Simple recency-based decay:
    - Segments más cercanos al final tienen peso ~1.0
    - Segments antiguos se van acercando a un mínimo, p.ej. 0.15
    """
    n = len(segments_tokens)
    if n == 0:
        return []

    # Indices 0..n-1, asumimos 0 = más antiguo, n-1 = más reciente
    # Normalizamos posición 0..1
    weights: List[float] = []
    min_w = 0.15
    max_w = 1.0
    lam = 2.0  # controla qué tan rápido cae

    for i in range(n):
        # distance_from_end: 0 (último) → 1 (primero)
        distance_from_end = (n - 1 - i) / max(1, n - 1)
        # Decay exponencial inversa
        w = max_w * math.exp(-lam * distance_from_end)
        # clamp para que no se vaya muy abajo
        w_clamped = max(min_w, min(max_w, w))
        weights.append(round(w_clamped, 3))
    return weights


# ---------- LLM helpers ----------

async def _generate_segment_previews(
    segments: List[Dict[str, Any]],
    model_name: str,
    max_chars: int = 500,
    max_segments: int = 8,
) -> List[Optional[str]]:
    """
    For each segment, optionally ask the LLM for a very short preview (1–2 sentences).
    We limit number of segments and truncate text for efficiency.
    """
    previews: List[Optional[str]] = [None] * len(segments)
    if not segments:
        return previews

    # Preparamos prompts por segmento (hasta max_segments)
    tasks = []
    idxs = []
    for idx, seg in enumerate(segments):
        if len(idxs) >= max_segments:
            break
        text = seg.get("text", "") or ""
        short = text[:max_chars]

        prompt = (
            "Summarize the following context segment in 1–2 sentences. "
            "Focus only on WHAT the segment is about, not on giving instructions to the user.\n\n"
            f"Segment text:\n\"\"\"{short}\"\"\"\n\n"
            "Return only the summary, no preamble."
        )

        tasks.append(
            call_openai(
                model=model_name,
                prompt=prompt,
                temperature=0.0,
                max_tokens=80,
            )
        )
        idxs.append(idx)

    if not tasks:
        return previews

    # Ejecutamos en paralelo
    from asyncio import gather
    results = await gather(*tasks, return_exceptions=True)

    for idx_local, res in enumerate(results):
        seg_idx = idxs[idx_local]
        if isinstance(res, Exception):
            previews[seg_idx] = None
        else:
            previews[seg_idx] = (res or "").strip()

    return previews


async def _generate_rot_explanation(
    segments: List[Dict[str, Any]],
    decay_weights: List[float],
    model_name: str,
) -> Optional[str]:
    """
    Optional: ask the LLM to explain 'context rot' for this specific scenario.
    """
    try:
        summary_lines = []
        for seg, w in zip(segments, decay_weights):
            short_text = (seg.get("text") or "")[:120].replace("\n", " ")
            summary_lines.append(
                f"- id={seg['id']}, type={seg['type']}, role={seg['role']}, "
                f"tokens={seg.get('token_count', '?')}, decay_weight={w}: {short_text}..."
            )

        joined = "\n".join(summary_lines)
        prompt = (
            "You are explaining how context windows work in a large language model to a technical audience.\n"
            "Below is a list of context segments currently in memory, each with a decay_weight between 0.15 and 1.0.\n"
            "Explain in 1–2 short paragraphs:\n"
            "- What 'context rot' means in this scenario.\n"
            "- Which segments will influence the next answer the most and why.\n"
            "- Why older segments might be effectively ignored.\n\n"
            f"Segments summary:\n{joined}\n\n"
            "Keep the explanation concise and clear."
        )

        text = await call_openai(
            model=model_name,
            prompt=prompt,
            temperature=0.2,
            max_tokens=240,
        )
        return (text or "").strip()
    except Exception:
        return None


# ---------- Core logic ----------

def _build_base_segments(req: ContextVizRequest) -> List[Dict[str, Any]]:
    """
    Decide qué segmentos usar:
    - Si hay manual_text -> un solo segmento user manual.
    - Si no, scenario_id -> usa uno de los escenarios.
    - Si no hay nada -> escenario por defecto RAG.
    """
    manual = (req.manual_text or "").strip()
    if manual:
        return [
            {
                "id": "manual-1",
                "role": "user",
                "type": "user",
                "source": "manual",
                "text": manual,
            }
        ]

    scenario_id = req.scenario_id or "rag_web_retrieved"
    builder = SCENARIOS.get(scenario_id, _scenario_rag_web_retrieved)
    return builder()


def _attach_token_positions(
    raw_segments: List[Dict[str, Any]],
    model_name: str,
    max_context_tokens: int,
    simulate_decay: bool,
) -> (List[Dict[str, Any]], TokenBreakdown, int, List[float]):
    """
    Calcula token_count, start_token, end_token por segmento,
    y breakdown por tipo + decay.
    """
    segments_out: List[Dict[str, Any]] = []
    total_tokens = 0

    breakdown = TokenBreakdown()

    # 1) contar tokens y posiciones
    for raw in raw_segments:
        txt = raw.get("text") or ""
        tk = count_tokens(txt, model_name=model_name)

        start = total_tokens
        end = start + tk
        total_tokens = end

        seg_type = raw.get("type", "other")
        if seg_type not in breakdown.__fields__:
            seg_type = "other"

        # sumamos al breakdown
        current_val = getattr(breakdown, seg_type)
        setattr(breakdown, seg_type, current_val + tk)

        seg_with_pos = {
            **raw,
            "token_count": tk,
            "start_token": start,
            "end_token": end,
        }
        segments_out.append(seg_with_pos)

    # 2) decay weights
    if simulate_decay:
        weights = compute_decay_weights([s["token_count"] for s in segments_out])
    else:
        weights = [1.0] * len(segments_out)

    # attach weights
    for seg, w in zip(segments_out, weights):
        seg["decay_weight"] = w

    # clip total_tokens by max_context_tokens only for visualization (no discard here)
    # Solo informativo; el frontend ya verá % de uso.
    return segments_out, breakdown, total_tokens, weights


async def analyze_context(req: ContextVizRequest) -> ContextVizResponse:
    # 1) construir segmentos base
    base_segments = _build_base_segments(req)
    if not base_segments:
        raise HTTPException(status_code=400, detail="No segments available for this scenario/request.")

    # 2) token counts + positions + decay
    segments_with_pos, breakdown, total_tokens, decay_weights = _attach_token_positions(
        base_segments,
        model_name=req.model,
        max_context_tokens=req.max_context_tokens,
        simulate_decay=req.simulate_decay,
    )

    # 3) optional LLM previews
    previews: List[Optional[str]] = []
    if req.use_llm_previews:
        previews = await _generate_segment_previews(segments_with_pos, model_name=req.model)
    else:
        previews = [None] * len(segments_with_pos)

    # 4) optional LLM explanation of context rot
    rot_explanation: Optional[str] = None
    if req.use_llm_rot_explanation:
        rot_explanation = await _generate_rot_explanation(segments_with_pos, decay_weights, model_name=req.model)

    # 5) build ContextSegment list
    ctx_segments: List[ContextSegment] = []
    for seg, prev in zip(segments_with_pos, previews):
        ctx_segments.append(
            ContextSegment(
                id=seg["id"],
                role=seg["role"],
                type=seg["type"],
                source=seg["source"],
                text=seg["text"],
                token_count=seg["token_count"],
                start_token=seg["start_token"],
                end_token=seg["end_token"],
                decay_weight=seg["decay_weight"],
                preview=prev,
            )
        )

    usage_pct = round(100.0 * total_tokens / max(1, req.max_context_tokens), 2)

    return ContextVizResponse(
        model=req.model,
        max_context_tokens=req.max_context_tokens,
        total_tokens=total_tokens,
        usage_pct=usage_pct,
        token_breakdown=breakdown,
        segments=ctx_segments,
        rot_explanation=rot_explanation,
    )
