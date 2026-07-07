from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .phase2 import filter_records, build_chunks, embed_and_upsert
from .ingestion import ingest_ingredients, save_records
from .phase3 import build_prompt, generate_report_groq, render_markdown, retrieve_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1-3 pipeline.")
    parser.add_argument("ingredients", nargs="*", help="Ingredient terms to search for.")
    parser.add_argument("--openalex-pages", type=int, default=2)
    parser.add_argument("--semantic-pages", type=int, default=2)
    parser.add_argument("--pubmed-pages", type=int, default=2)
    parser.add_argument("--core-pages", type=int, default=2)
    parser.add_argument("--per-page", type=int, default=25)
    parser.add_argument("--min-citations", type=int, default=5)
    parser.add_argument("--recent-year", type=int, default=2024)
    parser.add_argument("--collection", default="lontar_ingredient_research")
    parser.add_argument("--chroma-path", type=Path, default=Path("data") / "chroma")
    parser.add_argument("--cohere-model", default="embed-multilingual-v3.0")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    parser.add_argument("--out-json", type=Path, default=Path("outputs") / "report.json")
    parser.add_argument("--out-md", type=Path, default=Path("outputs") / "report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingredients = args.ingredients or [
        "macadamia nuts",
        "hemp seeds",
        "nutritional yeast",
        "yellow mustard powder",
        "Himalayan salt",
        "turmeric",
        "ginger",
        "cinnamon",
        "pepperberry",
    ]

    records = asyncio.run(
        ingest_ingredients(
            ingredients,
            openalex_pages=args.openalex_pages,
            semantic_scholar_pages=args.semantic_pages,
            pubmed_pages=args.pubmed_pages,
            core_pages=args.core_pages,
            per_page=args.per_page,
        )
    )
    save_records(records, Path("data") / "ingestion.json")

    record_dicts = [record.model_dump() for record in records]
    filtered = filter_records(
        record_dicts,
        min_citations=args.min_citations,
        recent_year=args.recent_year,
    )
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("data") .joinpath("filtered.json").write_text(
        __import__("json").dumps(filtered, indent=2),
        encoding="utf-8",
    )

    chunks = build_chunks(filtered)
    embed_and_upsert(
        chunks,
        collection_name=args.collection,
        chroma_path=args.chroma_path,
        cohere_model=args.cohere_model,
        batch_size=48,
    )

    retrieval = retrieve_sources(
        ingredients,
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        cohere_model=args.cohere_model,
        top_k=args.top_k,
    )
    if not retrieval.sources:
        raise RuntimeError("No sources retrieved from Chroma.")

    prompt = build_prompt(ingredients, retrieval.sources)
    report = generate_report_groq(prompt, model_name=args.groq_model)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")

    print(f"Pipeline complete. Report written to {args.out_md} and {args.out_json}.")


if __name__ == "__main__":
    main()
