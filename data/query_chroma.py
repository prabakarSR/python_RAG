#!/usr/bin/env python3
"""Simple example: query the Chroma collection and retrieve relevant documents using Ollama embeddings.

Usage:
  - Install Ollama: https://ollama.ai
  - Pull embedding model: `ollama pull embeddinggemma:300m`
  - Run:
      python data/query_chroma.py --query "your search query" --persist-dir "./chroma_db"
"""

import argparse
from pathlib import Path
import chromadb
import ollama  # <--- Updated


def embed_query(query: str, model: str = "embeddinggemma:300m"):
    resp = ollama.embeddings(model=model, prompt=query)
    return resp["embedding"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--persist-dir", default="./chroma_db")
    parser.add_argument("--collection", default="documents")
    parser.add_argument("--model", default="embeddinggemma:300m")
    parser.add_argument("--n-results", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading Chroma collection from {args.persist_dir}...")
    client = chromadb.PersistentClient(path=args.persist_dir)
    collection = client.get_or_create_collection(name=args.collection)

    print(f"Embedding query: '{args.query}'")
    query_embedding = embed_query(args.query, model=args.model)

    print(f"\nSearching for {args.n_results} most relevant documents...\n")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=args.n_results
    )

    if not results["ids"] or not results["ids"][0]:
        print("No results found.")
        return

    for i, (doc_id, doc_text, distance) in enumerate(
        zip(results["ids"][0], results["documents"][0], results["distances"][0])
    ):
        similarity = 1 - distance
        print(f"--- Result {i + 1} (similarity: {similarity:.4f}) ---")
        print(f"ID: {doc_id}")
        print(f"Text: {doc_text[:500]}...")
        print()


if __name__ == "__main__":
    main()
