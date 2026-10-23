"""RAG service: chunk text and generate answers from retrieved context."""

import hashlib
from typing import Protocol

from app.core.errors import ExternalServiceError


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


def compute_doc_id(raw: bytes) -> str:
    """Compute content-derived document ID (SHA-256 truncated)."""
    return hashlib.sha256(raw).hexdigest()[:12]


def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into word-boundary chunks. Never cuts words in half."""
    words = text.split(" ")
    chunks = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > chunk_size and current:
            chunks.append(current)
            current = word
        else:
            current = current + " " + word if current else word
    if current:
        chunks.append(current)
    return chunks


def ingest_document(raw: bytes, ocr, settings, embedder, store) -> int:
    """OCR all pages, chunk, embed, and store into the vector store.
    Returns the number of chunks stored."""
    doc_id = compute_doc_id(raw)
    npages = ocr.page_count(raw)
    all_chunks = []
    all_metadatas = []

    for page_index in range(npages):
        text, _ = ocr.ocr_page(raw, page_index, settings.ocr_dpi)
        page_chunks = chunk_text(text, settings.chunk_size)
        for idx, chunk in enumerate(page_chunks):
            if chunk.strip():
                all_chunks.append(chunk)
                all_metadatas.append({"doc_id": doc_id, "page": page_index, "chunk_index": idx})

    # Embed and store
    embeddings = embedder.embed_documents(all_chunks)
    ids = []
    for i, meta in enumerate(all_metadatas):
        ids.append(f"{meta['doc_id']}:{meta['page']}:{meta['chunk_index']}")

    store.add_chunks(ids, embeddings, all_chunks, all_metadatas)
    return len(all_chunks)


def generate_answer(question: str, embedder, store, llm, top_k) -> dict:
    """Retrieve relevant chunks and generate a grounded answer."""
    query_embedding = embedder.embed_query(question)
    results = store.query(query_embedding, top_k)

    # Build context from retrieved chunks
    context_parts = []
    sources = []
    for r in results:
        text = r.get("text", "")
        sources.append(text)
        context_parts.append(f"[{r.get('id', '')}] {text}")

    if not context_parts:
        # No relevant content found — abstain
        return {"answer": "No relevant content was found in the ingested documents to answer this question.",
                "sources": []}

    context = "\n".join(context_parts)
    prompt = (f"You are a document Q&A assistant. Answer the question using only the "
             f"retrieved context below. If the context does not contain the answer, say so. "
             f"List the sources you used.\n\n"
             f"Context:\n{context}\n\n"
             f"Question: {question}\n\n"
             f"Answer:")

    try:
        answer = llm.complete(prompt)
    except Exception as e:
        raise ExternalServiceError(f"LLM call failed: {e}") from e

    return {"answer": answer, "sources": sources}
