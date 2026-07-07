from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv

from src.config import (
    CHROMA_PATH,
    COHERE_EMBED_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBED_FIELD,
    DEFAULT_SUMMARY_MODEL,
    FOCUS_OPTIONS,
    LONTAR_COLLECTION,
    MIN_CITATIONS,
    RECENT_YEAR,
)
from src.relevance import evaluate_record_relevance
from src.summarize import summarize_records


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    doi = doi.strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return doi or None


def record_key(record: dict) -> Optional[str]:
    doi = normalize_doi(record.get("doi"))
    if doi:
        return f"doi:{doi.lower()}"

    external_ids = record.get("external_ids") or {}
    for key in ("openalex", "pmid", "coreId"):
        value = external_ids.get(key)
        if value:
            return f"{key}:{str(value).lower()}"

    title = (record.get("title") or "").strip().lower()
    year = record.get("year")
    if title:
        return f"title:{title}|{year or ''}"
    return None


def filter_records(
    records: Iterable[dict],
    *,
    min_citations: int,
    recent_year: int,
    ingredients: Optional[Iterable[str]] = None,
    focus: Optional[str] = None,
) -> list[dict]:
    seen: set[str] = set()
    filtered: list[dict] = []

    for record in records:
        abstract = (record.get("abstract") or "").strip()
        if not abstract:
            continue

        relevance = evaluate_record_relevance(
            record,
            ingredients=ingredients,
            focus=focus,
        )
        if relevance["relevance_status"] != "accepted":
            continue

        citation_count = record.get("citation_count")
        year = record.get("year")
        is_recent = isinstance(year, int) and year >= recent_year
        meets_citations = isinstance(citation_count, int) and citation_count >= min_citations

        if not (meets_citations or is_recent):
            continue

        key = record_key(record)
        if key and key in seen:
            continue
        if key:
            seen.add(key)

        accepted_record = dict(record)
        accepted_record["abstract"] = abstract
        accepted_record.update(relevance)
        filtered.append(accepted_record)

    filtered.sort(key=lambda item: item.get("relevance_score", 0), reverse=True)
    return filtered


def build_chunks(records: Iterable[dict], *, embed_field: str = DEFAULT_EMBED_FIELD) -> list[dict]:
    """Build embeddable chunks.

    embed_field controls what text gets embedded:
    - "auto" (default): use benefit_summary if present, else raw abstract
    - "abstract": always embed the raw abstract
    - "benefit_summary": always embed benefit_summary (skips records without one)
    """
    chunks: list[dict] = []
    for record in records:
        abstract = record.get("abstract") or ""
        benefit_summary = record.get("benefit_summary") or ""

        if embed_field == "abstract":
            text = abstract
        elif embed_field == "benefit_summary":
            text = benefit_summary
        else:
            text = benefit_summary or abstract

        if not text:
            continue

        doi = normalize_doi(record.get("doi"))
        source_id = record_key(record) or ""
        raw_id = f"{record.get('source','')}|{doi or source_id}|{record.get('title','')}"
        chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()

        metadata = {}
        raw_metadata_fields = [
            ("source", record.get("source")),
            ("title", record.get("title")),
            ("doi", doi),
            ("year", record.get("year")),
            ("citation_count", record.get("citation_count")),
            ("url", record.get("url")),
            ("relevance_score", record.get("relevance_score")),
            ("abstract", abstract or None),
            ("benefit_summary", benefit_summary or None),
            ("direction", record.get("direction")),
            ("mechanism", record.get("mechanism")),
            ("population", record.get("population")),
            ("confidence", record.get("confidence")),
        ]
        for key, value in raw_metadata_fields:
            if value is not None:
                metadata[key] = value
        chunks.append({"id": chunk_id, "text": text, "metadata": metadata})
    return chunks


def summarize_for_health_benefits(
    records: list[dict],
    *,
    ingredients: Optional[Iterable[str]],
    summary_model: str,
    drop_risk_and_unrelated: bool,
) -> tuple[list[dict], list[dict]]:
    """Run AI benefit summarization over guardrail-accepted records."""
    if not ingredients:
        return records, []

    drop_directions = {"risk", "unrelated"} if drop_risk_and_unrelated else set()
    return summarize_records(
        records,
        ingredients=ingredients,
        model_name=summary_model,
        drop_directions=drop_directions,
    )


SUMMARY_FIELDS = (
    "benefit_summary",
    "direction",
    "mechanism",
    "population",
    "confidence",
    "summarized_ingredient",
)


def apply_summaries_to_records(
    ingested: list[dict],
    summarized: list[dict],
) -> list[dict]:
    """Merge benefit-summary fields from Tab 2 back into the ingested record list."""
    summary_by_key: dict[str, dict] = {}
    for record in summarized:
        key = record_key(record)
        if key:
            summary_by_key[key] = record

    updated: list[dict] = []
    for record in ingested:
        merged = dict(record)
        key = record_key(record)
        if key and key in summary_by_key:
            source = summary_by_key[key]
            for field in SUMMARY_FIELDS:
                if source.get(field) is not None:
                    merged[field] = source[field]
        updated.append(merged)
    return updated

def batch_iter(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def embed_and_upsert(
    chunks: list[dict],
    *,
    collection_name: str,
    chroma_path: Path,
    cohere_model: str,
    batch_size: int,
) -> None:
    load_dotenv()
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY is required for embeddings.")

    import chromadb
    import cohere

    client = cohere.Client(api_key)
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    collection = chroma_client.get_or_create_collection(name=collection_name)

    for batch in batch_iter(chunks, batch_size):
        texts = [item["text"] for item in batch]
        ids = [item["id"] for item in batch]
        metadatas = [item["metadata"] for item in batch]

        response = client.embed(
            texts=texts,
            model=cohere_model,
            input_type="search_document",
        )
        embeddings = response.embeddings
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: filter and embed ingestion data.")
    parser.add_argument("--input", type=Path, default=Path("data") / "ingestion.json")
    parser.add_argument("--filtered-out", type=Path, default=Path("data") / "filtered.json")
    parser.add_argument("--min-citations", type=int, default=MIN_CITATIONS)
    parser.add_argument("--recent-year", type=int, default=RECENT_YEAR)
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=CHROMA_PATH,
        help="Local Chroma persistence path.",
    )
    parser.add_argument(
        "--collection",
        default=LONTAR_COLLECTION,
    )
    parser.add_argument(
        "--cohere-model",
        default=COHERE_EMBED_MODEL,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--ingredients", nargs="*", default=None)
    parser.add_argument(
        "--focus",
        default=None,
        choices=list(FOCUS_OPTIONS),
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Run AI health-benefit summarization before embedding.",
    )
    parser.add_argument(
        "--summary-model",
        default=DEFAULT_SUMMARY_MODEL,
        help="Groq model used for benefit summarization.",
    )
    parser.add_argument(
        "--drop-risk-and-unrelated",
        action="store_true",
        help="Discard records classified as risk/unrelated by the summarizer.",
    )
    parser.add_argument(
        "--embed-field",
        default=DEFAULT_EMBED_FIELD,
        choices=["auto", "abstract", "benefit_summary"],
        help="Which text to embed (default: benefit_summary if available, else abstract).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    records = load_records(args.input)
    filtered = filter_records(
        records,
        min_citations=args.min_citations,
        recent_year=args.recent_year,
        ingredients=args.ingredients,
        focus=args.focus,
    )
    if args.summarize:
        filtered, skipped = summarize_for_health_benefits(
            filtered,
            ingredients=args.ingredients,
            summary_model=args.summary_model,
            drop_risk_and_unrelated=args.drop_risk_and_unrelated,
        )
        if skipped:
            print(f"Summarizer set aside {len(skipped)} risk/unrelated records.")

    args.filtered_out.parent.mkdir(parents=True, exist_ok=True)
    args.filtered_out.write_text(json.dumps(filtered, indent=2), encoding="utf-8")

    embed_field = args.embed_field if args.summarize else "abstract"
    chunks = build_chunks(filtered, embed_field=embed_field)
    embed_and_upsert(
        chunks,
        collection_name=args.collection,
        chroma_path=args.chroma_path,
        cohere_model=args.cohere_model,
        batch_size=args.batch_size,
    )

    print(
        f"Filtered {len(filtered)} records, embedded {len(chunks)} chunks into "
        f"{args.collection} at {args.chroma_path}."
    )


if __name__ == "__main__":
    main()
