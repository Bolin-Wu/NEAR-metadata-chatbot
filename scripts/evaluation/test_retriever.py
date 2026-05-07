"""Interactive retriever test script for NEAR Chroma collections.

Use this to inspect semantic search results after migrating XML ingestion to
one-variable-per-document.

Examples:
    python scripts/evaluation/test_retriever.py --database Betula --query "blood pressure"
    python scripts/evaluation/test_retriever.py --database SNAC-K --query "frailty" --k 20
    python scripts/evaluation/test_retriever.py --database H70 --query "cohort background" --type database_description
"""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHROMA_DB = "./chroma_prod_db"
DEFAULT_RESULT_K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test semantic retrieval results from NEAR Chroma collections."
    )
    parser.add_argument(
        "--database",
        required=True,
        help="Database name, e.g. Betula, SNAC-K, GAS_SNAC_S.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Search query for semantic retrieval.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_RESULT_K,
        help=f"Top-k results to retrieve (default: {DEFAULT_RESULT_K}).",
    )
    parser.add_argument(
        "--type",
        dest="doc_type",
        default="variable_definitions",
        choices=["variable_definitions", "database_description", "all"],
        help=(
            "Metadata type filter. Use 'variable_definitions' for one-variable-per-document "
            "testing (default), 'database_description' for cohort background docs, or 'all'."
        ),
    )
    parser.add_argument(
        "--persist-dir",
        default=DEFAULT_CHROMA_DB,
        help=f"Path to Chroma persist directory (default: {DEFAULT_CHROMA_DB}).",
    )
    parser.add_argument(
        "--show-content-chars",
        type=int,
        default=500,
        help="Max chars to show per result content preview (default: 500).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    persist_dir = Path(args.persist_dir)
    if not persist_dir.exists():
        raise FileNotFoundError(f"Chroma directory not found: {persist_dir}")

    collection_name = f"{args.database.lower()}_metadata"

    print("=" * 80)
    print("NEAR RETRIEVER TEST")
    print("=" * 80)
    print(f"Database:      {args.database}")
    print(f"Collection:    {collection_name}")
    print(f"Persist dir:   {persist_dir}")
    print(f"Query:         {args.query}")
    print(f"k:             {args.k}")
    print(f"Type filter:   {args.doc_type}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    total_docs = vectorstore._collection.count()
    print(f"Collection size: {total_docs} docs")

    search_kwargs = {"k": args.k}
    if args.doc_type != "all":
        search_kwargs["filter"] = {"type": args.doc_type}

    print("\nRunning similarity_search_with_score...\n")
    scored_docs = vectorstore.similarity_search_with_score(args.query, **search_kwargs)

    if not scored_docs:
        print("No documents retrieved.")
        return

    print(f"Retrieved {len(scored_docs)} documents")
    print("Lower score means closer semantic match for Chroma distance metrics.")

    for idx, (doc, score) in enumerate(scored_docs, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source", "Unknown")
        var_name = metadata.get("variable_name", "N/A")
        db_name = metadata.get("database", "N/A")
        doc_type = metadata.get("type", "N/A")

        content_preview = doc.page_content[: args.show_content_chars].strip()
        if len(doc.page_content) > args.show_content_chars:
            content_preview += "..."

        print("-" * 80)
        print(f"Result #{idx}")
        print(f"Score:         {score:.6f}")
        print(f"Type:          {doc_type}")
        print(f"Database:      {db_name}")
        print(f"Source:        {source}")
        print(f"Variable:      {var_name}")
        print(f"Content chars: {len(doc.page_content)}")
        print("Content preview:")
        print(content_preview)

    # Quick signal for one-variable-per-document behavior.
    if args.doc_type == "variable_definitions":
        with_variable_name = sum(
            1 for doc, _ in scored_docs if (doc.metadata or {}).get("variable_name")
        )
        print("\n" + "=" * 80)
        print(
            "Variable metadata coverage: "
            f"{with_variable_name}/{len(scored_docs)} have 'variable_name' metadata"
        )
        if with_variable_name < len(scored_docs):
            print(
                "Note: Some retrieved docs do not contain variable_name metadata. "
                "This may indicate mixed legacy entries in the collection."
            )


if __name__ == "__main__":
    main()
