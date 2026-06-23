from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from src.relevance import evaluate_record_relevance, group_query_terms


class HealthClaim(BaseModel):
    ingredient: str
    claim: str
    evidence_dois: list[str] = Field(default_factory=list)
    evidence_titles: list[str] = Field(default_factory=list)


class SynergyCoefficient(BaseModel):
    ingredients: list[str]
    mechanism: str
    evidence_dois: list[str] = Field(default_factory=list)


class SafetyDosage(BaseModel):
    ingredient: str
    guidance: str
    evidence_dois: list[str] = Field(default_factory=list)


class Report(BaseModel):
    health_claim_matrix: list[HealthClaim]
    synergy_coefficients: list[SynergyCoefficient]
    safety_dosage_dossier: list[SafetyDosage]
    limitations: list[str] = Field(default_factory=list)


def build_query(ingredients: Iterable[str]) -> str:
    return group_query_terms(ingredients)


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    doi = doi.strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return doi or None


def retrieve_sources(
    ingredients: Iterable[str],
    *,
    chroma_path: Path,
    collection_name: str,
    cohere_model: str,
    top_k: int,
    focus: Optional[str] = "Health & Clinical Therapy",
) -> list[dict]:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY is required for retrieval embeddings.")

    query_text = build_query(ingredients)
    if not query_text:
        raise RuntimeError("No valid ingredient terms provided.")

    import cohere
    import chromadb

    client = cohere.Client(api_key)
    embed_response = client.embed(
        texts=[query_text],
        model=cohere_model,
        input_type="search_query",
    )
    embedding = embed_response.embeddings[0]

    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    collection = chroma_client.get_or_create_collection(name=collection_name)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=max(top_k * 3, top_k),
        include=["documents", "metadatas", "distances"],
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    sources: list[dict] = []
    for doc, metadata, distance in zip(documents, metadatas, distances, strict=False):
        metadata = metadata or {}
        doi = normalize_doi(metadata.get("doi"))
        source = {
            "title": metadata.get("title"),
            "doi": doi,
            "year": metadata.get("year"),
            "citation_count": metadata.get("citation_count"),
            "abstract": doc,
            "url": metadata.get("url"),
            "distance": distance,
        }
        relevance = evaluate_record_relevance(
            source,
            ingredients=ingredients,
            focus=focus,
        )
        if relevance["relevance_status"] != "accepted":
            continue
        source.update(relevance)
        sources.append(source)
        if len(sources) >= top_k:
            break
    return sources

def build_prompt(ingredients: Iterable[str], sources: list[dict]) -> str:
    ingredient_text = ", ".join(ingredients)
    source_blocks: list[str] = []
    for idx, source in enumerate(sources, start=1):
        block = [
            f"[SOURCE DOCUMENT {idx}]",
            f"Title: {source.get('title') or ''}",
            f"DOI: {source.get('doi') or ''}",
            f"Year: {source.get('year') or ''}",
            f"Citation Count: {source.get('citation_count') or ''}",
            f"URL: {source.get('url') or ''}",
            "Abstract:",
            source.get("abstract") or "",
        ]
        source_blocks.append("\n".join(block))

    schema = Report.model_json_schema()
    joined_sources_text = "\n\n".join(source_blocks)
    return (
        "You are a scientific report generator. "
        "Return JSON only. Do not include markdown or explanations.\n\n"
        f"Ingredients: {ingredient_text}\n\n"
        "Use ONLY the source documents provided below. "
        "If a claim is not supported, omit it. "
        "DOI values must be plain DOI strings without URLs.\n\n"
        "JSON schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Source documents:\n"
        f"{joined_sources_text}"
    )


def generate_report(
    prompt: str,
    *,
    model_name: str,
    max_retries: int,
    backoff_seconds: float,
) -> Report:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for report generation.")

    from google import genai

    client = genai.Client(api_key=api_key)
    def parse_retry_delay(message: str) -> Optional[float]:
        marker = "retrydelay"
        lowered = message.lower()
        if marker not in lowered:
            return None
        start = lowered.find(marker)
        snippet = lowered[start : start + 40]
        digits = "".join(ch for ch in snippet if ch.isdigit())
        if digits:
            return float(digits)
        return None

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            break
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if attempt >= max_retries:
                raise
            if "429" not in message and "quota" not in message and "rate" not in message:
                raise
            retry_delay = parse_retry_delay(message)
            delay = retry_delay if retry_delay is not None else backoff_seconds * (2**attempt)
            print(f"Gemini rate limit hit; retrying in {delay:.1f}s...")
            import time

            time.sleep(delay)
    else:
        if last_error:
            raise last_error
        raise RuntimeError("Gemini generation failed unexpectedly.")

    raw_text = response.text or ""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini output was not valid JSON.") from exc

    try:
        return Report.model_validate(payload)
    except ValidationError as exc:
        raise RuntimeError("Gemini output did not match schema.") from exc


def generate_report_groq(
    prompt: str,
    *,
    model_name: str,
) -> Report:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for Groq report generation.")

    from groq import Groq

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "Return JSON only. Do not include markdown or explanations.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content or ""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Groq output was not valid JSON.") from exc

    try:
        return Report.model_validate(payload)
    except ValidationError as exc:
        raise RuntimeError("Groq output did not match schema.") from exc


def render_markdown(report: Report) -> str:
    lines: list[str] = ["# LONTAR Evidence Report", ""]

    lines.append("## Commercial Health Claim Matrix")
    lines.append("| Ingredient | Claim | Evidence DOIs | Evidence Titles |")
    lines.append("| --- | --- | --- | --- |")
    for item in report.health_claim_matrix:
        doi_links = []
        for doi in item.evidence_dois:
            doi_value = normalize_doi(doi) or doi
            doi_links.append(f"https://doi.org/{doi_value}")
        lines.append(
            "| "
            + " | ".join(
                [
                    item.ingredient,
                    item.claim,
                    ", ".join(doi_links),
                    "; ".join(item.evidence_titles),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Synergy Coefficients")
    for synergy in report.synergy_coefficients:
        lines.append(f"- Ingredients: {', '.join(synergy.ingredients)}")
        lines.append(f"  - Mechanism: {synergy.mechanism}")
        lines.append(f"  - Evidence DOIs: {', '.join(synergy.evidence_dois)}")

    lines.append("")
    lines.append("## Safety & Dosage Dossier")
    for safety in report.safety_dosage_dossier:
        lines.append(f"- Ingredient: {safety.ingredient}")
        lines.append(f"  - Guidance: {safety.guidance}")
        lines.append(f"  - Evidence DOIs: {', '.join(safety.evidence_dois)}")

    if report.limitations:
        lines.append("")
        lines.append("## Limitations")
        for item in report.limitations:
            lines.append(f"- {item}")

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3: generate the final report.")
    parser.add_argument("ingredients", nargs="*", help="Ingredient terms to search for.")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--focus",
        default="Health & Clinical Therapy",
        choices=[
            "Health & Clinical Therapy",
            "Agriculture & Botany",
            "Broad/Generic (No Filter)",
        ],
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path("data") / "chroma",
        help="Local Chroma persistence path.",
    )
    parser.add_argument(
        "--collection",
        default="lontar_ingredient_research",
    )
    parser.add_argument(
        "--cohere-model",
        default="embed-multilingual-v3.0",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.0-flash",
    )
    parser.add_argument("--gemini-retries", type=int, default=2)
    parser.add_argument("--gemini-backoff", type=float, default=2.5)
    parser.add_argument(
        "--llm-provider",
        choices=["gemini", "groq"],
        default="gemini",
    )
    parser.add_argument(
        "--groq-model",
        default="llama-3.3-70b-versatile",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("outputs") / "report.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("outputs") / "report.md",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    ingredients = args.ingredients or ["Macadamia", "Turmeric", "Pepperberry"]

    sources = retrieve_sources(
        ingredients,
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        cohere_model=args.cohere_model,
        top_k=args.top_k,
        focus=args.focus,
    )
    if not sources:
        raise RuntimeError("No sources retrieved from Chroma.")

    prompt = build_prompt(ingredients, sources)
    if args.llm_provider == "groq":
        report = generate_report_groq(
            prompt,
            model_name=args.groq_model,
        )
    else:
        report = generate_report(
            prompt,
            model_name=args.gemini_model,
            max_retries=args.gemini_retries,
            backoff_seconds=args.gemini_backoff,
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")

    print(f"Generated report to {args.out_md} and {args.out_json}.")


if __name__ == "__main__":
    main()
