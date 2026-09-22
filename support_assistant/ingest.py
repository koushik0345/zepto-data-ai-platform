from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "all-MiniLM-L6-v2"


def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    text = text.strip()

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        start = end

    return [chunk for chunk in chunks if chunk]


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Opening ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    documents = []
    ids = []
    metadatas = []

    for doc_path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = doc_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks, start=1):
            chunk_id = f"{doc_path.stem}_chunk_{index:02d}"

            documents.append(chunk)
            ids.append(chunk_id)
            metadatas.append(
                {
                    "document_id": doc_path.stem,
                    "chunk_id": chunk_id,
                    "source": doc_path.name,
                }
            )

    if not documents:
        raise RuntimeError("No policy documents found in support_assistant/docs")

    print(f"Found {len(documents)} chunks.")

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    # Rebuild the collection so ingestion is deterministic.
    existing_ids = collection.get()["ids"]

    if existing_ids:
        collection.delete(ids=existing_ids)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Stored {collection.count()} chunks in ChromaDB.")
    print(f"ChromaDB path: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
