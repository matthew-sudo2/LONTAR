from __future__ import annotations

import json
import os
from typing import Iterable, Optional

from pydantic import BaseModel, Field, ValidationError

from src.config import DEFAULT_GROQ_MODEL
from src.relevance import expand_ingredient_terms


class BenefitSummary(BaseModel):
    """Structured, health-benefit-first summary of a single source record."""

    ingredient: str
    benefit_summary: str = Field(
        description="2-3 sentences describing what this study found about the "
        "ingredient's effect on human health. Written for a health-claim audience, "
        "not a general abstract paraphrase."
    )
    direction: str = Field(
        description="One of: benefit, risk, neutral, unrelated"
    )
    mechanism: Optional[str] = Field(
        default=None, description="Biological mechanism/pathway, if the abstract states one."
    )
    population: Optional[str] = Field(
        default=None, description="Study population, e.g. 'human, n=42', 'in-vitro', 'rat model'."
    )
    confidence: str = Field(
        default="low", description="One of: low, medium, high"
    )


VALID_DIRECTIONS = {"benefit", "risk", "neutral", "unrelated"}
VALID_CONFIDENCE = {"low", "medium", "high"}


def build_summary_prompt(record: dict, ingredient: str) -> str:
    title = record.get("title") or ""
    abstract = record.get("abstract") or ""
    schema = BenefitSummary.model_json_schema()
    return (
        "You are a nutritional-science research analyst. Read the abstract below "
        f"and summarize it with a specific focus on health benefits of '{ingredient}'. "
        "Do not summarize the paper generally - extract only what is relevant to "
        "whether and how this ingredient affects human health.\n\n"
        "Rules:\n"
        "- If the abstract reports a positive health effect (e.g. lowers cholesterol, "
        "reduces inflammation, improves metabolic markers), set direction='benefit'.\n"
        "- If it reports an adverse or harmful effect (toxicity, allergic reaction, "
        "negative interaction), set direction='risk'.\n"
        "- If it studies the ingredient but finds no significant health effect either way, "
        "set direction='neutral'.\n"
        "- If the abstract is not actually about a health effect of this ingredient "
        "(e.g. materials science, agriculture-only, unrelated compound), set "
        "direction='unrelated' and keep benefit_summary short.\n"
        "- Only state what the abstract actually supports. Do not invent dosages, "
        "mechanisms, or populations that are not mentioned.\n"
        "- confidence should reflect study strength as described (e.g. large human RCT "
        "= high, small in-vitro/animal-only or unclear methodology = low).\n\n"
        "Return JSON only, matching this schema exactly:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"Title: {title}\n"
        f"Abstract:\n{abstract}\n"
    )


def _coerce_summary(payload: dict, ingredient: str) -> BenefitSummary:
    payload = dict(payload)
    payload.setdefault("ingredient", ingredient)
    if payload.get("direction") not in VALID_DIRECTIONS:
        payload["direction"] = "unrelated"
    if payload.get("confidence") not in VALID_CONFIDENCE:
        payload["confidence"] = "low"
    return BenefitSummary.model_validate(payload)


def summarize_record_groq(
    record: dict,
    ingredient: str,
    *,
    model_name: str = DEFAULT_GROQ_MODEL,
) -> BenefitSummary:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for summarization.")

    from groq import Groq

    client = Groq(api_key=api_key)
    prompt = build_summary_prompt(record, ingredient)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "Return JSON only. Do not include markdown or explanations.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw_text = response.choices[0].message.content or ""
    payload = json.loads(raw_text)
    return _coerce_summary(payload, ingredient)


def primary_ingredient_for_record(record: dict, ingredients: Iterable[str]) -> str:
    """Pick the ingredient (from the requested list) that best matches this record."""
    ingredients = list(ingredients)
    if len(ingredients) == 1:
        return ingredients[0]

    title_abstract = f"{record.get('title', '')} {record.get('abstract', '')}".lower()
    for ingredient in ingredients:
        for alias in expand_ingredient_terms([ingredient]):
            if alias.lower() in title_abstract:
                return ingredient
    return ingredients[0]


def summarize_records(
    records: list[dict],
    *,
    ingredients: Iterable[str],
    model_name: str = DEFAULT_GROQ_MODEL,
    drop_directions: Optional[set[str]] = None,
    on_error: str = "keep",
) -> tuple[list[dict], list[dict]]:
    """Summarize each record with a health-benefit focus.

    Returns (summarized_records, skipped_records). Each summarized record gets
    benefit_summary/direction/mechanism/population/confidence fields merged in.
    """
    drop_directions = drop_directions or set()
    ingredients = list(ingredients)

    kept: list[dict] = []
    skipped: list[dict] = []

    for record in records:
        ingredient = primary_ingredient_for_record(record, ingredients)
        try:
            summary = summarize_record_groq(record, ingredient, model_name=model_name)
        except (RuntimeError, json.JSONDecodeError, ValidationError, Exception) as exc:
            annotated = dict(record)
            annotated["summarization_error"] = str(exc)
            if on_error == "drop":
                skipped.append(annotated)
            else:
                kept.append(annotated)
            continue

        annotated = dict(record)
        annotated.update(
            {
                "benefit_summary": summary.benefit_summary,
                "direction": summary.direction,
                "mechanism": summary.mechanism,
                "population": summary.population,
                "confidence": summary.confidence,
                "summarized_ingredient": summary.ingredient,
            }
        )

        if summary.direction in drop_directions:
            skipped.append(annotated)
        else:
            kept.append(annotated)

    return kept, skipped
