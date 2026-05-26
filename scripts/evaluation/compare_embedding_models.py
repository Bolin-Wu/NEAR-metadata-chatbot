"""Compare semantic retrieval behavior across two embedding models.

This script is designed for the NEAR aging metadata project and compares:
- sentence-transformers/all-MiniLM-L6-v2 on ./chroma_azure_db
- text-embedding-3-small on ./chroma_azure_db

It runs one or more queries and prints side-by-side top results (up to 3)
from each model to support quick manual comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL_NAME = "text-embedding-3-small"
HF_DB_DEFAULT = "./chroma_azure_db"
OPENAI_DB_DEFAULT = "./chroma_azure_db"
DEFAULT_QUERIES_FILE = "scripts/evaluation/aging_semantic_queries.json"
DEFAULT_K = 8
DEFAULT_PREVIEW_CHARS = 220


@dataclass
class EvalQuery:
    query: str
    database: str


@dataclass
class Hit:
    rank: int
    score: float
    variable_name: str
    source: str
    content: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare semantic search quality between MiniLM and text-embedding-3-small."
    )
    parser.add_argument(
        "--queries-file",
        default=DEFAULT_QUERIES_FILE,
        help=f"JSON file containing aging/cohort queries (default: {DEFAULT_QUERIES_FILE}).",
    )
    parser.add_argument(
        "--query",
        action="append",
        help=(
            "Optional ad-hoc query to evaluate. Can be repeated. "
            "If provided, these queries are used instead of --queries-file."
        ),
    )
    parser.add_argument(
        "--database",
        default="SNAC-K",
        help="Database used with --query mode (default: SNAC-K).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Top-k retrieved docs per query (default: {DEFAULT_K}).",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=DEFAULT_PREVIEW_CHARS,
        help=f"Content preview size per hit (default: {DEFAULT_PREVIEW_CHARS}).",
    )
    parser.add_argument("--hf-db", default=HF_DB_DEFAULT)
    parser.add_argument("--hf-model", default=HF_MODEL_NAME)
    parser.add_argument("--openai-db", default=OPENAI_DB_DEFAULT)
    parser.add_argument("--openai-model", default=OPENAI_MODEL_NAME)
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--openai-base-url", default="")

    return parser.parse_args()


def load_streamlit_secrets(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    secrets: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            secrets[key] = value
    return secrets


def normalize_base_url(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if value.endswith("/openai/v1") or value.endswith("/openai/v1/"):
        return value.rstrip("/") + "/"
    return value.rstrip("/") + "/openai/v1/"


def load_queries(queries_file: Path) -> list[EvalQuery]:
    if not queries_file.exists():
        raise FileNotFoundError(f"Query file not found: {queries_file}")

    payload = json.loads(queries_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Query JSON must be a list.")

    queries: list[EvalQuery] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Entry #{idx} must be an object.")

        query = str(item.get("query", "")).strip()
        database = str(item.get("database", "")).strip()
        if not query or not database:
            raise ValueError(f"Entry #{idx} must include non-empty query and database.")
        queries.append(
            EvalQuery(
                query=query,
                database=database,
            )
        )

    return queries


def build_openai_embeddings(args: argparse.Namespace, secrets: dict[str, str]) -> OpenAIEmbeddings:
    api_key = (
        args.openai_api_key
        or os.getenv("AZURE_api_key")
        or os.getenv("AZURE_API_KEY")
        or os.getenv("AZURE_OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or secrets.get("AZURE_api_key")
        or secrets.get("AZURE_API_KEY")
        or secrets.get("AZURE_OPENAI_API_KEY")
        or secrets.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "Missing API key for text-embedding-3-small. Provide --openai-api-key or set env/secrets."
        )

    base_url = (
        args.openai_base_url
        or os.getenv("AZURE_FOUNDRY_BASE_URL")
        or os.getenv("AZURE_openai_endpoint")
        or os.getenv("AZURE_OPENAI_ENDPOINT")
        or secrets.get("AZURE_FOUNDRY_BASE_URL")
        or secrets.get("AZURE_openai_endpoint")
        or secrets.get("AZURE_OPENAI_ENDPOINT")
    )
    base_url = normalize_base_url(base_url)

    kwargs: dict[str, Any] = {
        "model": args.openai_model,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAIEmbeddings(**kwargs)


def get_store(
    persist_dir: Path,
    collection_name: str,
    embedding_fn: Any,
) -> Chroma:
    if not persist_dir.exists():
        raise FileNotFoundError(f"Persist directory not found: {persist_dir}")

    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding_fn,
        collection_name=collection_name,
    )


def retrieve_hits(store: Chroma, query: str, k: int) -> list[Hit]:
    scored = store.similarity_search_with_score(
        query,
        k=k,
        filter={"type": "variable_definitions"},
    )

    hits: list[Hit] = []
    for idx, (doc, score) in enumerate(scored, start=1):
        metadata = doc.metadata or {}
        hits.append(
            Hit(
                rank=idx,
                score=float(score),
                variable_name=str(metadata.get("variable_name", "N/A")),
                source=str(metadata.get("source", "Unknown")),
                content=doc.page_content or "",
            )
        )
    return hits


def short_preview(text: str, limit: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def print_hits(label: str, hits: list[Hit], preview_chars: int) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    for hit in hits:
        print(
            f"#{hit.rank:02d} | score={hit.score:.5f} | var={hit.variable_name} | source={hit.source}"
        )
        print(f"     {short_preview(hit.content, preview_chars)}")


def run() -> None:
    load_dotenv()
    args = parse_args()
    secrets = load_streamlit_secrets(Path(".streamlit/secrets.toml"))

    if args.query:
        queries = [
            EvalQuery(query=item, database=args.database)
            for item in args.query
        ]
    else:
        queries = load_queries(Path(args.queries_file))

    hf_embeddings = HuggingFaceEmbeddings(model_name=args.hf_model)
    openai_embeddings = build_openai_embeddings(args, secrets)

    hf_db = Path(args.hf_db)
    openai_db = Path(args.openai_db)

    print("=" * 88)
    print("NEAR Semantic Search Embedding Comparison")
    print("=" * 88)
    print(f"Queries: {len(queries)}")
    print(f"Top-k: {args.k}")
    print(f"Model A: {args.hf_model} | DB: {hf_db}")
    print(f"Model B: {args.openai_model} | DB: {openai_db}")

    for idx, item in enumerate(queries, start=1):
        collection_name = f"{item.database.lower()}_metadata"

        store_a = get_store(hf_db, collection_name, hf_embeddings)
        store_b = get_store(openai_db, collection_name, openai_embeddings)

        hits_a = retrieve_hits(store_a, item.query, args.k)
        hits_b = retrieve_hits(store_b, item.query, args.k)

        print("\n" + "=" * 88)
        print(f"[{idx}/{len(queries)}] Database: {item.database}")
        print(f"Query: {item.query}")
        print_hits(f"Model A: {args.hf_model}", hits_a, args.preview_chars)
        print_hits(f"Model B: {args.openai_model}", hits_b, args.preview_chars)


if __name__ == "__main__":
    run()
