from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Iterable, Optional
from xml.etree import ElementTree

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.relevance import focus_query_terms, group_query_terms

OPENALEX_BASE_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CORE_BASE_URL = "https://api.core.ac.uk/v3/search/works"


class StudyRecord(BaseModel):
    source: str
    title: str
    doi: Optional[str] = None
    year: Optional[int] = None
    citation_count: Optional[int] = None
    abstract: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    url: Optional[str] = None
    external_ids: dict[str, str | int] = Field(default_factory=dict)


def build_query(ingredients: Iterable[str], focus: Optional[str] = None) -> str:
    base_query = group_query_terms(ingredients)
    if not base_query:
        return ""

    focus_terms = focus_query_terms(focus)
    if focus_terms:
        return f"({base_query}) AND ({focus_terms})"

    return base_query


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    doi = doi.strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return doi or None


def inflate_abstract(inverted_index: Optional[dict[str, list[int]]]) -> Optional[str]:
    if not inverted_index:
        return None
    max_index = max(pos for positions in inverted_index.values() for pos in positions)
    words = [""] * (max_index + 1)
    # Rebuild the abstract from OpenAlex's inverted index positions.
    for word, positions in inverted_index.items():
        for pos in positions:
            if 0 <= pos < len(words):
                words[pos] = word
    text = " ".join(word for word in words if word)
    return text or None


async def request_with_retries(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: Optional[dict[str, str]] = None,
    params: Optional[dict[str, str | int]] = None,
    retries: int = 3,
    backoff: float = 1.0,
) -> httpx.Response:
    for attempt in range(retries + 1):
        try:
            response = await client.request(method, url, headers=headers, params=params)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    await asyncio.sleep(int(retry_after))
                else:
                    await asyncio.sleep(backoff * (2**attempt))
                continue
            response.raise_for_status()
            return response
        except httpx.RequestError:
            if attempt >= retries:
                raise
            await asyncio.sleep(backoff * (2**attempt))
    raise RuntimeError("Unreachable retry exhaustion.")


def map_openalex_record(record: dict) -> StudyRecord:
    authorships = record.get("authorships", [])
    authors = [
        item.get("author", {}).get("display_name")
        for item in authorships
        if item.get("author", {}).get("display_name")
    ]
    raw_ids = record.get("ids") or {}
    external_ids = {k: str(v) for k, v in raw_ids.items() if v is not None}
    return StudyRecord(
        source="openalex",
        title=record.get("title") or "",
        doi=normalize_doi(record.get("doi")),
        year=record.get("publication_year"),
        citation_count=record.get("cited_by_count"),
        abstract=inflate_abstract(record.get("abstract_inverted_index")),
        authors=authors,
        url=(record.get("primary_location", {}) or {}).get("landing_page_url")
        or record.get("id"),
        external_ids=external_ids,
    )


def map_semantic_scholar_record(record: dict) -> StudyRecord:
    raw_ids = record.get("externalIds") or {}
    external_ids = {k: str(v) for k, v in raw_ids.items() if v is not None}
    authors = [item.get("name") for item in record.get("authors", []) if item.get("name")]
    return StudyRecord(
        source="semantic_scholar",
        title=record.get("title") or "",
        doi=normalize_doi(external_ids.get("DOI")),
        year=record.get("year"),
        citation_count=record.get("citationCount"),
        abstract=record.get("abstract"),
        authors=authors,
        url=record.get("url"),
        external_ids=external_ids,
    )


def map_pubmed_record(record: dict) -> StudyRecord:
    return StudyRecord(
        source="pubmed",
        title=record.get("title") or "",
        doi=normalize_doi(record.get("doi")),
        year=record.get("year"),
        citation_count=None,
        abstract=record.get("abstract"),
        authors=record.get("authors", []),
        url=record.get("url"),
        external_ids=record.get("external_ids", {}),
    )


def map_core_record(record: dict) -> StudyRecord:
    authors = record.get("authors") or []
    doi = record.get("doi")
    if not doi:
        identifiers = record.get("identifiers") or []
        for item in identifiers:
            if isinstance(item, str) and item.lower().startswith("10."):
                doi = item
                break
    return StudyRecord(
        source="core",
        title=record.get("title") or "",
        doi=normalize_doi(doi),
        year=record.get("yearPublished"),
        citation_count=record.get("citationCount"),
        abstract=record.get("abstract"),
        authors=[item for item in authors if isinstance(item, str)],
        url=record.get("downloadUrl") or record.get("url"),
        external_ids={"coreId": str(record.get("id"))} if record.get("id") else {},
    )


async def fetch_openalex(
    query: str,
    *,
    per_page: int = 25,
    max_pages: int = 2,
    mailto: Optional[str] = None,
    timeout: float = 30.0,
) -> list[dict]:
    params: dict[str, str | int] = {
        "search": query,
        "per_page": per_page,
        "cursor": "*",
    }
    if mailto:
        params["mailto"] = mailto

    results: list[dict] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for _ in range(max_pages):
            response = await request_with_retries(client, "GET", OPENALEX_BASE_URL, params=params)
            payload = response.json()
            page_results = payload.get("results", [])
            results.extend(page_results)

            next_cursor = (payload.get("meta") or {}).get("next_cursor")
            if not next_cursor:
                break
            params["cursor"] = next_cursor

    return results


async def fetch_semantic_scholar(
    query: str,
    *,
    per_page: int = 25,
    max_pages: int = 2,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> list[dict]:
    if not api_key:
        return []

    fields = "title,year,abstract,externalIds,citationCount,url,authors"
    headers = {"User-Agent": "LONTAR/1.0"}
    headers["x-api-key"] = api_key

    results: list[dict] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for page in range(max_pages):
            params = {
                "query": query,
                "limit": per_page,
                "offset": page * per_page,
                "fields": fields,
            }
            try:
                response = await request_with_retries(
                    client,
                    "GET",
                    SEMANTIC_SCHOLAR_BASE_URL,
                    params=params,
                    headers=headers,
                )
            except httpx.HTTPStatusError as exc:
                if exc.response is not None:
                    if exc.response.status_code == 429:
                        print("Semantic Scholar rate limit hit; skipping remaining pages.")
                        break
                    if exc.response.status_code == 400:
                        # Offset out of bounds (no more results)
                        break
                raise

            payload = response.json()
            page_results = payload.get("data", [])
            if not page_results:
                break
            results.extend(page_results)
            await asyncio.sleep(0.5)

    return results


async def enrich_missing_dois(
    records: list[StudyRecord],
    *,
    api_key: str,
    max_records: int = 50,
    timeout: float = 30.0,
) -> None:
    fields = "title,externalIds,paperId"
    headers = {"User-Agent": "LONTAR/1.0", "x-api-key": api_key}
    pending = [record for record in records if not record.doi and record.title]
    if not pending:
        return

    async with httpx.AsyncClient(timeout=timeout) as client:
        for record in pending[:max_records]:
            params = {
                "query": record.title,
                "limit": 1,
                "fields": fields,
            }
            try:
                response = await request_with_retries(
                    client,
                    "GET",
                    SEMANTIC_SCHOLAR_BASE_URL,
                    params=params,
                    headers=headers,
                )
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code == 429:
                    print("Semantic Scholar rate limit hit; stopping DOI enrichment.")
                    break
                raise

            payload = response.json()
            hits = payload.get("data") or []
            if hits:
                hit = hits[0]
                external_ids = hit.get("externalIds") or {}
                doi = normalize_doi(external_ids.get("DOI"))
                if doi:
                    record.doi = doi
                    record.external_ids.update(external_ids)
                    paper_id = hit.get("paperId")
                    if paper_id:
                        record.external_ids["semantic_scholar"] = paper_id
            await asyncio.sleep(0.2)


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    root = ElementTree.fromstring(xml_text)
    records: list[dict] = []
    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle") or ""
        abstract_parts = [
            elem.text.strip()
            for elem in article.findall(".//Abstract/AbstractText")
            if elem.text
        ]
        abstract = " ".join(abstract_parts) if abstract_parts else None
        authors = [
            " ".join(filter(None, [
                author.findtext("ForeName"),
                author.findtext("LastName"),
            ]))
            for author in article.findall(".//Author")
        ]
        authors = [item for item in authors if item]
        year_text = article.findtext(".//PubDate/Year")
        doi = None
        for id_node in article.findall(".//ArticleId"):
            if id_node.get("IdType") == "doi" and id_node.text:
                doi = id_node.text
                break
        url = None
        pmid = article.findtext(".//PMID")
        if pmid:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        external_ids = {"PMID": pmid} if pmid else {}

        records.append(
            {
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": int(year_text) if year_text and year_text.isdigit() else None,
                "doi": doi,
                "url": url,
                "external_ids": external_ids,
            }
        )
    return records


async def fetch_pubmed(
    query: str,
    *,
    per_page: int = 25,
    max_pages: int = 2,
    timeout: float = 30.0,
) -> list[dict]:
    results: list[dict] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for page in range(max_pages):
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": per_page,
                "retstart": page * per_page,
                "retmode": "json",
            }
            search_response = await request_with_retries(
                client,
                "GET",
                PUBMED_ESEARCH_URL,
                params=params,
            )
            search_payload = search_response.json()
            id_list = (search_payload.get("esearchresult") or {}).get("idlist", [])
            if not id_list:
                break

            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
            }
            fetch_response = await request_with_retries(
                client,
                "GET",
                PUBMED_EFETCH_URL,
                params=fetch_params,
            )
            results.extend(parse_pubmed_xml(fetch_response.text))
    return results


async def fetch_core(
    query: str,
    *,
    per_page: int = 25,
    max_pages: int = 2,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> list[dict]:
    if not api_key:
        return []

    headers = {"Authorization": f"Bearer {api_key}"}
    results: list[dict] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for page in range(1, max_pages + 1):
            params = {
                "q": query,
                "page": page,
                "pageSize": per_page,
            }
            response = await request_with_retries(
                client,
                "GET",
                CORE_BASE_URL,
                params=params,
                headers=headers,
            )
            payload = response.json()
            page_results = payload.get("results") or []
            if not page_results:
                break
            results.extend(page_results)
    return results


async def ingest_ingredients(
    ingredients: Iterable[str],
    *,
    focus: Optional[str] = None,
    openalex_pages: int = 2,
    semantic_scholar_pages: int = 2,
    pubmed_pages: int = 2,
    core_pages: int = 2,
    per_page: int = 25,
    enrich_semantic_doi: bool = True,
) -> list[StudyRecord]:
    load_dotenv()

    openalex_mailto = os.getenv("OPENALEX_EMAIL")
    semantic_scholar_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    core_key = os.getenv("CORE_API_KEY")

    ingredients_list = [ing.strip() for ing in ingredients if ing.strip()]
    if not ingredients_list:
        return []

    mapped: list[StudyRecord] = []
    for ingredient in ingredients_list:
        query = build_query([ingredient], focus=focus)
        if not query:
            continue
        print(f"Fetching literature for: {ingredient}...")
        openalex_task = fetch_openalex(
            query,
            per_page=per_page,
            max_pages=openalex_pages,
            mailto=openalex_mailto,
        )
        sem_task = fetch_semantic_scholar(
            query,
            per_page=per_page,
            max_pages=semantic_scholar_pages,
            api_key=semantic_scholar_key,
        )
        pubmed_task = fetch_pubmed(
            query,
            per_page=per_page,
            max_pages=pubmed_pages,
        )
        core_task = fetch_core(
            query,
            per_page=per_page,
            max_pages=core_pages,
            api_key=core_key,
        )

        openalex_results, semantic_results, pubmed_results, core_results = await asyncio.gather(
            openalex_task,
            sem_task,
            pubmed_task,
            core_task,
        )

        mapped.extend(map(map_openalex_record, openalex_results))
        mapped.extend(map(map_semantic_scholar_record, semantic_results))
        mapped.extend(map(map_pubmed_record, pubmed_results))
        mapped.extend(map(map_core_record, core_results))
        
        await asyncio.sleep(1.0)  # Respect rate limits between ingredients

    if semantic_scholar_key and enrich_semantic_doi:
        await enrich_missing_dois(mapped, api_key=semantic_scholar_key)
    return dedupe_records(mapped)


def dedupe_records(records: Iterable[StudyRecord]) -> list[StudyRecord]:
    seen: set[tuple[str, str]] = set()
    unique: list[StudyRecord] = []
    for record in records:
        doi = normalize_doi(record.doi)
        if doi:
            key = ("doi", doi.lower())
        else:
            title_key = record.title.strip().lower() if record.title else ""
            year_key = str(record.year) if record.year is not None else ""
            key = ("title", f"{title_key}|{year_key}")
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def records_to_dataframe(records: Iterable[StudyRecord]):
    import pandas as pd

    return pd.DataFrame([record.model_dump() for record in records])


def records_to_dicts(records: Iterable[StudyRecord]) -> list[dict]:
    return [record.model_dump() for record in records]


def save_records(records: Iterable[StudyRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        df = records_to_dataframe(records)
        df.to_csv(output_path, index=False)
        return

    payload = records_to_dicts(records)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch literature metadata from OpenAlex and Semantic Scholar."
    )
    parser.add_argument(
        "ingredients",
        nargs="*",
        help="Ingredient terms to search for (e.g., Macadamia Turmeric Pepperberry).",
    )
    parser.add_argument("--openalex-pages", type=int, default=2)
    parser.add_argument("--semantic-pages", type=int, default=2)
    parser.add_argument("--pubmed-pages", type=int, default=2)
    parser.add_argument("--core-pages", type=int, default=2)
    parser.add_argument("--per-page", type=int, default=25)
    parser.add_argument(
        "--focus",
        default=None,
        choices=[
            "Health & Clinical Therapy",
            "Agriculture & Botany",
            "Broad/Generic (No Filter)",
        ],
        help="Optional research focus used to shape repository queries.",
    )
    parser.add_argument(
        "--semantic-doi-enrich",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Semantic Scholar to fill missing DOIs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data") / "ingestion.json",
        help="Output file path (.json or .csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingredients = args.ingredients or ["Macadamia", "Turmeric", "Pepperberry"]
    fetched = asyncio.run(
        ingest_ingredients(
            ingredients,
            openalex_pages=args.openalex_pages,
            semantic_scholar_pages=args.semantic_pages,
            pubmed_pages=args.pubmed_pages,
            core_pages=args.core_pages,
            per_page=args.per_page,
            focus=args.focus,
            enrich_semantic_doi=args.semantic_doi_enrich,
        )
    )
    save_records(fetched, args.out)
    print(f"Fetched {len(fetched)} records and wrote {args.out}.")


if __name__ == "__main__":
    main()
