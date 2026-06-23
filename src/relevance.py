from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional


DEFAULT_FOCUS = "Broad/Generic (No Filter)"

INGREDIENT_ALIASES: dict[str, list[str]] = {
    "macadamia": [
        "macadamia",
        "macadamia nut",
        "macadamia nuts",
        "macadamia oil",
        "macadamia integrifolia",
        "macadamia tetraphylla",
    ],
    "curcuma longa": ["curcuma longa", "turmeric", "curcumin"],
    "turmeric": ["turmeric", "curcuma longa", "curcumin"],
    "curcumin": ["curcumin", "turmeric", "curcuma longa"],
    "piper nigrum": ["piper nigrum", "black pepper", "piperine"],
    "black pepper": ["black pepper", "piper nigrum", "piperine"],
    "pepperberry": ["pepperberry", "tasmannia lanceolata", "mountain pepper"],
}


@dataclass(frozen=True)
class FocusProfile:
    positive_terms: tuple[str, ...] = ()
    negative_terms: tuple[str, ...] = ()
    min_score: int = 20
    require_positive_context: bool = False


FOCUS_PROFILES: dict[str, FocusProfile] = {
    "Health & Clinical Therapy": FocusProfile(
        positive_terms=(
            "health",
            "clinical",
            "human",
            "humans",
            "patient",
            "patients",
            "therapy",
            "therapeutic",
            "medicinal",
            "pharmacology",
            "pharmacological",
            "disease",
            "anti-inflammatory",
            "antioxidant",
            "cardiovascular",
            "cholesterol",
            "lipid",
            "lipids",
            "serum",
            "metabolic",
            "diabetes",
            "cancer",
            "neuroprotective",
            "bioactive",
            "nutrition",
            "nutritional",
            "diet",
            "dietary",
            "supplement",
            "toxicity",
            "safety",
            "bioavailability",
        ),
        negative_terms=(
            "wastewater",
            "waste water",
            "activated carbon",
            "adsorption",
            "adsorbent",
            "adsorbents",
            "nanomaterial",
            "nanomaterials",
            "carbon dot",
            "carbon dots",
            "photocatalytic",
            "photocatalysis",
            "heavy metal",
            "heavy metals",
            "dye removal",
            "pollutant",
            "pollutants",
            "effluent",
            "aqueous solution",
            "kinetic and equilibrium",
            "nut shell",
            "nut shells",
            "nutshell",
            "nutshells",
        ),
        min_score=45,
        require_positive_context=True,
    ),
    "Agriculture & Botany": FocusProfile(
        positive_terms=(
            "agriculture",
            "botany",
            "cultivation",
            "farming",
            "genetics",
            "crop",
            "crops",
            "plant",
            "plants",
            "seed",
            "seeds",
            "soil",
            "harvest",
            "yield",
            "horticulture",
            "germination",
        ),
        negative_terms=(
            "clinical trial",
            "patient",
            "patients",
            "chemotherapy",
            "drug delivery",
        ),
        min_score=35,
        require_positive_context=False,
    ),
}


def clean_ingredient_terms(ingredients: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    for item in ingredients:
        term = " ".join(str(item).strip().split())
        if term:
            cleaned.append(term)
    return cleaned


def expand_ingredient_terms(ingredients: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for term in clean_ingredient_terms(ingredients):
        variants = INGREDIENT_ALIASES.get(term.lower(), [term])
        for variant in variants:
            normalized = " ".join(variant.strip().split())
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                expanded.append(normalized)
    return expanded


def format_query_term(term: str) -> str:
    if " " in term and not (term.startswith('"') and term.endswith('"')):
        return f'"{term}"'
    return term


def group_query_terms(ingredients: Iterable[str]) -> str:
    groups: list[str] = []
    for ingredient in clean_ingredient_terms(ingredients):
        variants = INGREDIENT_ALIASES.get(ingredient.lower(), [ingredient])
        terms = " OR ".join(format_query_term(term) for term in variants)
        if terms:
            groups.append(f"({terms})" if len(variants) > 1 else terms)
    return " OR ".join(groups)


def focus_query_terms(focus: Optional[str]) -> str:
    if focus == "Health & Clinical Therapy":
        terms = (
            "health",
            "clinical",
            "medicinal",
            "therapeutic",
            "nutrition",
            "dietary",
            "cholesterol",
            "lipid",
            "cardiovascular",
            "metabolic",
            "antioxidant",
            "anti-inflammatory",
            "bioactive",
            "safety",
            "bioavailability",
        )
        return " OR ".join(terms)
    if focus == "Agriculture & Botany":
        terms = (
            "agriculture",
            "botany",
            "cultivation",
            "farming",
            "genetics",
            "crop",
            "horticulture",
            "seed",
            "harvest",
        )
        return " OR ".join(terms)
    return ""


def _contains_term(text: str, term: str) -> bool:
    normalized = term.lower().strip()
    if not normalized:
        return False
    if re.fullmatch(r"[a-z0-9-]+", normalized):
        return re.search(rf"(?<![a-z0-9-]){re.escape(normalized)}(?![a-z0-9-])", text) is not None
    return normalized in text


def _matching_terms(text: str, terms: Iterable[str]) -> list[str]:
    return [term for term in terms if _contains_term(text, term)]


def relevance_profile(focus: Optional[str]) -> FocusProfile:
    return FOCUS_PROFILES.get(focus or DEFAULT_FOCUS, FocusProfile())


def evaluate_record_relevance(
    record: dict,
    *,
    ingredients: Optional[Iterable[str]] = None,
    focus: Optional[str] = None,
) -> dict:
    title = str(record.get("title") or "")
    abstract = str(record.get("abstract") or "")
    title_text = title.lower()
    abstract_text = abstract.lower()
    full_text = f"{title_text} {abstract_text}"

    score = 0
    reasons: list[str] = []
    reject_reasons: list[str] = []

    ingredient_terms = expand_ingredient_terms(ingredients or [])
    profile = relevance_profile(focus)
    if not ingredient_terms and not profile.positive_terms and not profile.negative_terms:
        return {
            "relevance_score": 0,
            "relevance_status": "accepted",
            "relevance_reasons": ["relevance guardrails not configured"],
            "relevance_reject_reasons": [],
        }
    if ingredient_terms:
        title_hits = _matching_terms(title_text, ingredient_terms)
        abstract_hits = _matching_terms(abstract_text, ingredient_terms)
        if title_hits:
            score += 55
            reasons.append(f"ingredient in title: {', '.join(title_hits[:3])}")
        if abstract_hits:
            score += 25
            reasons.append(f"ingredient in abstract: {', '.join(abstract_hits[:3])}")
        if not title_hits and not abstract_hits:
            reject_reasons.append("no requested ingredient term found in title or abstract")
            score -= 80

    profile = relevance_profile(focus)
    positive_title_hits = _matching_terms(title_text, profile.positive_terms)
    positive_abstract_hits = _matching_terms(abstract_text, profile.positive_terms)
    positive_hits = sorted(set(positive_title_hits + positive_abstract_hits))
    negative_title_hits = _matching_terms(title_text, profile.negative_terms)
    negative_abstract_hits = _matching_terms(abstract_text, profile.negative_terms)
    negative_hits = sorted(set(negative_title_hits + negative_abstract_hits))

    if positive_title_hits:
        score += min(36, len(set(positive_title_hits)) * 12)
    if positive_abstract_hits:
        score += min(30, len(set(positive_abstract_hits)) * 6)
    if positive_hits:
        reasons.append(f"focus context: {', '.join(positive_hits[:5])}")
    elif profile.require_positive_context and ingredient_terms:
        reject_reasons.append("no health or clinical context found")
        score -= 30

    if negative_title_hits:
        score -= min(80, len(set(negative_title_hits)) * 28)
    if negative_abstract_hits:
        score -= min(60, len(set(negative_abstract_hits)) * 15)
    if negative_hits:
        reasons.append(f"off-focus context: {', '.join(negative_hits[:5])}")
    if focus == "Health & Clinical Therapy" and negative_hits:
        if len(negative_hits) >= 2 or not positive_hits:
            reject_reasons.append("environmental/materials context conflicts with health focus")

    accepted = not reject_reasons and score >= profile.min_score
    if not accepted and not reject_reasons:
        reject_reasons.append(f"relevance score below threshold ({score} < {profile.min_score})")

    return {
        "relevance_score": score,
        "relevance_status": "accepted" if accepted else "rejected",
        "relevance_reasons": reasons,
        "relevance_reject_reasons": reject_reasons,
    }


def annotate_record_relevance(
    record: dict,
    *,
    ingredients: Optional[Iterable[str]] = None,
    focus: Optional[str] = None,
) -> dict:
    annotated = dict(record)
    annotated.update(
        evaluate_record_relevance(record, ingredients=ingredients, focus=focus)
    )
    return annotated


def triage_records(
    records: Iterable[dict],
    *,
    ingredients: Optional[Iterable[str]] = None,
    focus: Optional[str] = None,
) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    for record in records:
        annotated = annotate_record_relevance(record, ingredients=ingredients, focus=focus)
        if annotated["relevance_status"] == "accepted":
            accepted.append(annotated)
        else:
            rejected.append(annotated)

    accepted.sort(key=lambda item: item.get("relevance_score", 0), reverse=True)
    rejected.sort(key=lambda item: item.get("relevance_score", 0), reverse=True)
    return accepted, rejected
