from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Optional

import chromadb
import cohere
from dotenv import load_dotenv


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
) -> list[dict]:
    seen: set[str] = set()
    filtered: list[dict] = []

    for record in records:
        abstract = (record.get("abstract") or "").strip()
        if not abstract:
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

        record["abstract"] = abstract
        filtered.append(record)

    return filtered


def build_chunks(records: Iterable[dict]) -> list[dict]:
    chunks: list[dict] = []
    for record in records:
        text = record.get("abstract") or ""
        if not text:
            continue
        doi = normalize_doi(record.get("doi"))
        source_id = record_key(record) or ""
        raw_id = f"{record.get('source','')}|{doi or source_id}|{record.get('title','')}"
        chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()

        metadata = {
            "source": record.get("source"),
            "title": record.get("title"),
            "doi": doi,
            "year": record.get("year"),
            "citation_count": record.get("citation_count"),
            "url": record.get("url"),
        }
        chunks.append({"id": chunk_id, "text": text, "metadata": metadata})
    return chunks


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
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY is required for embeddings.")

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
    parser.add_argument("--min-citations", type=int, default=5)
    parser.add_argument("--recent-year", type=int, default=2024)
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
    parser.add_argument("--batch-size", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    records = load_records(args.input)
    filtered = filter_records(
        records,
        min_citations=args.min_citations,
        recent_year=args.recent_year,
    )
    args.filtered_out.parent.mkdir(parents=True, exist_ok=True)
    args.filtered_out.write_text(json.dumps(filtered, indent=2), encoding="utf-8")

    chunks = build_chunks(filtered)
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
