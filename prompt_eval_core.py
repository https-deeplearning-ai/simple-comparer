# app/prompt_eval_logic.py

import json
import re
from typing import List, Optional, Dict, Any, Tuple

from fastapi import HTTPException
from pydantic import BaseModel, Field

from duel_core import call_openai  # reutilizamos el mismo caller asíncrono


# --------- Pydantic models ---------


class PromptEvalDimensionScores(BaseModel):
    clarity: int       # 1–5
    specificity: int   # 1–5
    structure: int     # 1–5
    tone: int          # 1–5
    safety: int        # 1–5


class SinglePromptEval(BaseModel):
    prompt: str
    score: int  # 0–100
    dimensions: PromptEvalDimensionScores
    suggestions: List[str]
    explanation: str


class PromptEvalRequest(BaseModel):
    prompt: str = Field(..., description="Original prompt to evaluate.")
    revised_prompt: Optional[str] = Field(
        default=None,
        description="Optional revised version of the prompt for comparison."
    )
    model: Optional[str] = Field(
        default="gpt-4o-mini",
        description="Model used for evaluation (e.g., gpt-4.1-mini, gpt-4o-mini, gpt-4.1, gpt-4o)."
    )


class PromptEvalResponse(BaseModel):
    original: SinglePromptEval
    revised: Optional[SinglePromptEval] = None
    delta_score: Optional[int] = None


# --------- Heuristic layer ---------


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def _has_role_instruction(text: str) -> bool:
    text_low = (text or "").lower()
    return any(
        token in text_low
        for token in ["you are", "act as", "role:", "system:", "assistant:"]
    )


def _has_constraints(text: str) -> bool:
    text_low = (text or "").lower()
    patterns = [
        "no more than",
        "at most",
        "exactly",
        "in bullet points",
        "in bullets",
        "use json",
        "return json",
        "step by step",
        "limit to",
        "between",
        "do not include",
        "avoid",
    ]
    return any(p in text_low for p in patterns)


def _structure_score(text: str) -> int:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    bullet_like = sum(1 for ln in lines if ln.startswith(("-", "*", "1.", "2.", "3.")))
    if bullet_like >= 4:
        return 5
    if bullet_like >= 2:
        return 4
    if "\n\n" in (text or ""):
        return 3
    return 2 if len(lines) > 1 else 1


def _tone_and_safety(text: str) -> (int, int):
    text_low = (text or "").lower()
    toxic_words = [
        "idiot", "stupid", "dumb", "kill", "hate",
        "hijo de puta", "pendejo", "vete a la mierda",
    ]
    has_toxic = any(bad in text_low for bad in toxic_words)

    tone = 4
    safety = 5
    if has_toxic:
        tone = 2
        safety = 2
    return tone, safety


def heuristic_evaluate(prompt: str) -> Dict[str, Any]:
    """
    Heurística simple para tener un score de respaldo y mezclar con el LLM.
    Devuelve dims 1–5 y un score 0–100.
    """
    wc = _word_count(prompt)
    role_instr = _has_role_instruction(prompt)
    has_constr = _has_constraints(prompt)
    structure = _structure_score(prompt)
    tone, safety = _tone_and_safety(prompt)

    # Clarity: longitud razonable + rol
    if wc < 5:
        clarity = 1
    elif wc < 15:
        clarity = 2
    elif wc > 250:
        clarity = 3
    else:
        clarity = 4 if role_instr else 3

    # Specificity: constraints + longitud
    if has_constr and 20 <= wc <= 250:
        specificity = 5
    elif has_constr:
        specificity = 4
    elif wc > 200:
        specificity = 3
    else:
        specificity = 2

    dims = {
        "clarity": int(clarity),
        "specificity": int(specificity),
        "structure": int(structure),
        "tone": int(tone),
        "safety": int(safety),
    }

    avg = sum(dims.values()) / 5.0  # 1–5
    score = int(round(avg * 20))    # 0–100

    suggestions: List[str] = []
    if wc < 15:
        suggestions.append("Add more context so the model understands the task.")
    if not role_instr:
        suggestions.append("Define the assistant role or perspective (e.g., 'You are a data engineer...').")
    if not has_constr:
        suggestions.append("Add explicit constraints (format, length, style) to reduce ambiguity.")
    if structure < 3:
        suggestions.append("Use bullet points or numbered steps to structure the instructions.")
    if safety < 4:
        suggestions.append("Avoid offensive or overly aggressive language to keep the tone safe.")

    explanation = (
        "Heuristic evaluation based on length, presence of role instructions, "
        "constraints, basic structure, and potential tone/safety issues."
    )

    return {
        "dims": dims,
        "score": score,
        "suggestions": suggestions,
        "explanation": explanation,
    }


# --------- LLM-based evaluation ---------


PROMPT_EVAL_SYSTEM = """You are a prompt evaluation assistant.
Your job is to evaluate how good a prompt is for use with a large language model.

You MUST:
- Analyze clarity, specificity, structure, tone, and safety.
- Give each dimension a score from 1 to 5 (5 = excellent).
- Provide an overall score from 0 to 100.
- Suggest concrete improvements.
- Explain briefly WHY the prompt scores that way.

Output STRICT JSON with this schema:

{
  "clarity": 1-5,
  "specificity": 1-5,
  "structure": 1-5,
  "tone": 1-5,
  "safety": 1-5,
  "score_overall": 0-100,
  "suggestions": ["short suggestion 1", "short suggestion 2", ...],
  "explanation": "1–3 sentences explaining the main strengths and weaknesses."
}
"""


async def llm_evaluate_prompt(prompt: str, model_name: str) -> Optional[Dict[str, Any]]:
    """
    Llama al modelo para evaluar el prompt. Si algo falla, devuelve None.
    """
    user_msg = (
        f"{PROMPT_EVAL_SYSTEM}\n\n"
        f"Prompt to evaluate:\n\"\"\"{prompt}\"\"\"\n\n"
        f"Return ONLY the JSON, no extra text."
    )
    try:
        text = await call_openai(
            model=model_name,
            prompt=user_msg,
            temperature=0.0,
            max_tokens=400,
        )
    except Exception:
        return None

    if not text:
        return None

    try:
        block = text.strip()
        i, j = block.find("{"), block.rfind("}")
        if i != -1 and j != -1:
            block = block[i:j+1]
        data = json.loads(block)
        return data
    except Exception:
        return None


def _merge_scores(
    heur: Dict[str, Any],
    llm: Optional[Dict[str, Any]],
    max_suggestions: int = 6,
) -> Tuple[PromptEvalDimensionScores, int, List[str], str]:
    """
    Combina heurística y LLM en un solo conjunto de resultados.
    Retorna:
      - PromptEvalDimensionScores
      - score_final (0–100)
      - suggestions (list[str])
      - explanation (str)
    """
    dims_h = heur["dims"]
    score_h = heur["score"]

    if llm is None:
        dims_final = dims_h
        score_final = score_h
        suggestions_llm: List[str] = []
        explanation_llm = ""
    else:
        def _get_dim(name: str) -> int:
            try:
                val = int(llm.get(name, 0) or 0)
            except Exception:
                val = 0
            if not (1 <= val <= 5):
                # fallback a heurístico si está raro
                return dims_h[name]
            return val

        dims_final = {
            "clarity": _get_dim("clarity"),
            "specificity": _get_dim("specificity"),
            "structure": _get_dim("structure"),
            "tone": _get_dim("tone"),
            "safety": _get_dim("safety"),
        }

        try:
            score_llm = int(llm.get("score_overall", 0) or 0)
        except Exception:
            score_llm = score_h

        # mezcla: LLM pesa más (70%) que la heurística (30%)
        score_final = int(round(0.3 * score_h + 0.7 * score_llm))

        suggestions_llm = []
        if isinstance(llm.get("suggestions"), list):
            suggestions_llm = [str(s).strip() for s in llm["suggestions"] if str(s).strip()]

        explanation_llm = str(llm.get("explanation", "")).strip()

    # mezclar sugerencias (heur + llm) sin duplicar demasiado
    suggestions_all: List[str] = []
    for s in heur["suggestions"]:
        if s not in suggestions_all:
            suggestions_all.append(s)
    for s in suggestions_llm:
        if s not in suggestions_all:
            suggestions_all.append(s)
    suggestions_all = suggestions_all[:max_suggestions]

    explanation_parts = [heur["explanation"]]
    if explanation_llm:
        explanation_parts.append(explanation_llm)
    explanation_final = " ".join(explanation_parts)

    dim_model = PromptEvalDimensionScores(**dims_final)

    return dim_model, score_final, suggestions_all, explanation_final


async def _evaluate_single_prompt(prompt: str, model_name: str) -> SinglePromptEval:
    heur = heuristic_evaluate(prompt)
    llm_raw = await llm_evaluate_prompt(prompt, model_name)

    dims_model, score_final, suggestions, explanation = _merge_scores(heur, llm_raw)

    return SinglePromptEval(
        prompt=prompt,
        score=score_final,
        dimensions=dims_model,
        suggestions=suggestions,
        explanation=explanation,
    )


# --------- Public API (llamado desde FastAPI) ---------


async def evaluate_prompt(req: PromptEvalRequest) -> PromptEvalResponse:
    text = (req.prompt or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    model_name = req.model or "gpt-4o-mini"

    original_eval = await _evaluate_single_prompt(text, model_name)

    revised_eval: Optional[SinglePromptEval] = None
    delta_score: Optional[int] = None

    if req.revised_prompt:
        rev_text = req.revised_prompt.strip()
        if rev_text:
            revised_eval = await _evaluate_single_prompt(rev_text, model_name)
            delta_score = revised_eval.score - original_eval.score

    return PromptEvalResponse(
        original=original_eval,
        revised=revised_eval,
        delta_score=delta_score,
    )
