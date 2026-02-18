#!/usr/bin/env python3
"""Embed markdown files in `data/raw` using Ollama embeddings and store them in Chroma DB.

Usage:
  - Install Ollama: https://ollama.ai
  - Pull embedding model: `ollama pull embeddinggemma:300m`
  - Install dependencies: `pip install -r requirements.txt`
  - Run: `python data/embed_to_chroma.py --raw-dir data/raw --persist-dir ./chroma_db`
"""

import argparse
import os
from pathlib import Path
from typing import List

from tqdm import tqdm
import chromadb
import ollama  # <--- New

def list_text_files(raw_dir: Path) -> List[Path]:
    exts = {".md", ".markdown", ".txt"}
    return [p for p in raw_dir.iterdir() if p.suffix.lower() in exts and p.is_file()]

def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="latin-1")

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= L:
            break
        start = end - overlap
    return chunks

def batch(iterable, n=100):
    l = len(iterable)
    for i in range(0, l, n):
        yield iterable[i : i + n]

def embed_texts(texts: List[str], model: str = "embeddinggemma:300m"):
    embeddings = []
    for txt in texts:
        resp = ollama.embeddings(model=model, prompt=txt)
        embeddings.append(resp["embedding"])
    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--persist-dir", default="./chroma_db")
    parser.add_argument("--collection", default="documents")
    parser.add_argument("--model", default="embeddinggemma:300m")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise SystemExit(f"Raw directory not found: {raw_dir}")

    print("Initializing Chroma client (persistent)...")
    persist_dir = str(Path(args.persist_dir).resolve())
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=args.collection)

    files = list_text_files(raw_dir)
    if not files:
        print("No markdown/text files found in", raw_dir)
        return

    all_ids = []
    all_docs = []
    all_metas = []

    for f in files:
        text = read_file(f)
        chunks = chunk_text(text, max_chars=args.chunk_size, overlap=args.chunk_overlap)
        for i, chunk in enumerate(chunks):
            doc_id = f"{f.name}__{i}"
            meta = {"source": str(f), "chunk_index": i}
            all_ids.append(doc_id)
            all_docs.append(chunk)
            all_metas.append(meta)

    print(f"Prepared {len(all_docs)} chunks from {len(files)} files.")

    # Compute embeddings in batches and add to Chroma
    for ids_batch in batch(all_ids, args.batch_size):
        start = all_ids.index(ids_batch[0])
        end = start + len(ids_batch)

        docs_batch = all_docs[start:end]
        metas_batch = all_metas[start:end]

        print(f"Embedding {len(docs_batch)} documents...")
        embeddings = embed_texts(docs_batch, model=args.model)

        collection.add(
            ids=ids_batch,
            documents=docs_batch,
            metadatas=metas_batch,
            embeddings=embeddings,
        )

    try:
        client.persist()
    except Exception:
        pass

    print(f"Stored {len(all_docs)} vectors in Chroma collection '{args.collection}' at {persist_dir}")

if __name__ == "__main__":
    main()
