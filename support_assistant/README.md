# Module 3 — Zepto Support Assistant

## Overview

This module implements a RAG-based Zepto policy support assistant using:

- Sentence Transformers (`all-MiniLM-L6-v2`) for embeddings
- ChromaDB for vector storage and retrieval
- LangGraph for the assistant workflow
- FastAPI for the REST API
- Pydantic for structured responses
- Ollama as an optional local LLM
- A deterministic mock mode for reproducible evaluation

The assistant uses the eight supplied Zepto policy documents in `docs/`.

---

## Architecture

```text
Policy Documents
      |
      v
Chunking
      |
      v
all-MiniLM-L6-v2
      |
      v
ChromaDB
      |
      v
User Query
      |
      v
classify_intent
      |
      +----------------------+
      |                      |
      v                      v
policy_question        general_question
      |                      |
      v                      v
retrieve_and_answer    direct_answer
      |
      v
Top-3 retrieved chunks
      |
      v
Structured response

---

## Project Files

```text
support_assistant/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── ingest.py
├── rag.py
├── main.py
├── requirements.txt
├── Dockerfile
└── chroma_db/
---

## Installation

From the project root:

    source .venv/bin/activate
    pip install -r support_assistant/requirements.txt

## Ingest the Policy Documents

Run:

    python support_assistant/ingest.py

The ingestion process loads all eight policy documents, chunks them, generates embeddings using all-MiniLM-L6-v2, and stores the embeddings and source metadata in ChromaDB.

## Running the API

From the support_assistant directory:

    uvicorn main:app --host 0.0.0.0 --port 7860

The API runs at:

    http://127.0.0.1:7860

## API Endpoint

### POST /ask

Retrieval example:

    curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d '{"query":"What is the delivery policy?"}'

Example response:

    {
      "answer": "Based on the retrieved context: ...",
      "sources": ["doc_01_chunk_01", "doc_02_chunk_01", "doc_05_chunk_01"],
      "confidence": 1.0
    }

General-question example:

    curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d '{"query":"What is the capital of India?"}'

Example response:

    {
      "answer": "I can only answer questions about Zepto policies right now.",
      "sources": [],
      "confidence": 1.0
    }

## LangGraph Workflow

The graph contains three nodes:

1. classify_intent
2. retrieve_and_answer
3. direct_answer

In default mock mode, these keywords are classified as policy questions:

    delivery
    return
    refund
    membership
    tracking
    cancel
    gift card
    support hours

Policy questions are routed to retrieve_and_answer. Other questions are routed to direct_answer.

For policy questions, the query is embedded using all-MiniLM-L6-v2 and ChromaDB retrieves the top three chunks. Their chunk IDs are returned as sources.

For general questions, the mock response is:

    I can only answer questions about Zepto policies right now.

## Mock and Real LLM Modes

The default mode is deterministic mock mode:

    MOCK_LLM=1

An optional real LLM mode is available:

    MOCK_LLM=0

When real mode is enabled, the application uses Ollama.

Configure the Ollama model with:

    export OLLAMA_MODEL=llama3.2

The default Ollama endpoint is:

    http://localhost:11434/api/generate

The real LLM response is validated using Pydantic. Validation is attempted up to three times in total.

## Response Schema

Every response contains:

    {
      "answer": "string",
      "sources": ["chunk_id"],
      "confidence": 0.0
    }

The confidence value is constrained between 0 and 1.

## Docker

Build the image from the support_assistant directory:

    docker build -t zepto-support-assistant .

Run it with:

    docker run --rm -p 7860:7860 zepto-support-assistant

The API will be available at:

    http://127.0.0.1:7860

## Design Decisions

- ChromaDB uses cosine similarity.
- The same Sentence Transformer model is used for document and query embeddings.
- Top-3 chunks are retrieved for policy questions.
- Source IDs are preserved through ChromaDB metadata.
- Mock mode provides deterministic evaluation without requiring an external LLM.
- Ollama is optional and runs locally when MOCK_LLM=0.
- The prompt contains an explicit negative constraint against unsupported information.
