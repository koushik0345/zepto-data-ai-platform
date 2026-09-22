# Module 3 — Zepto Support Assistant

A retrieval-augmented (RAG) support assistant that answers questions about Zepto's delivery, returns, membership and support policies. The answers are grounded in the eight supplied policy documents.

Stack: `sentence-transformers` (`all-MiniLM-L6-v2`) for embeddings, ChromaDB for vector storage, a LangGraph `StateGraph` for routing, Pydantic for the response schema, and FastAPI + uvicorn for the API.

**Graded mode is fully offline.** With `MOCK_LLM` unset (or `MOCK_LLM=1`), no LLM is called and no API key or network access to an LLM provider is needed. Only the embedding model is downloaded once from Hugging Face on first run.

## Install and run

From the repository root, with the virtual environment active:

```bash
pip install -r support_assistant/requirements.txt

cd support_assistant
python ingest.py                       # 1. embed the 8 docs into ChromaDB (run once)
uvicorn main:app --port 7860           # 2. start the API (MOCK_LLM left unset)
```

`ingest.py` must run before the API is started for the first time: it creates the local `chroma_db/` folder, which is not committed.

## Example calls (recorded with `MOCK_LLM` left at its default)

**1. Policy question → retrieval.** Contains the keyword "refund", so it is routed to `retrieve_and_answer`:

```bash
curl -X POST http://127.0.0.1:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the refund policy for damaged groceries?"}'
```

```json
{"answer":"Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unop","sources":["doc_02_chunk_01","doc_06_chunk_01","doc_05_chunk_01"],"confidence":1.0}
```

The top chunk is `doc_02` (Returns & Refunds), the correct source document, and the second is `doc_06` (Damaged or Missing Items).

**2. General question → no retrieval.** Contains no policy keyword, so it is routed to `direct_answer`:

```bash
curl -X POST http://127.0.0.1:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the capital of India?"}'
```

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

**3. Another policy question** ("membership" and "cancel"):

```bash
curl -X POST http://127.0.0.1:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"How do I cancel my Zepto Pass membership?"}'
```

```json
{"answer":"Based on the retrieved context: Zepto offers three account tiers: Basic (free, default tier, standard delivery fees apply), Zepto Pass (INR 49 per month, free standard delivery on all orders and 5% off select categories), and Zepto ","sources":["doc_03_chunk_01","doc_05_chunk_01","doc_08_chunk_01"],"confidence":1.0}
```

Top chunk: `doc_03` (Membership Tiers), whose last sentence covers cancellation.

## Architecture

```text
 INGESTION            EMBEDDING                 STORAGE
 docs/doc_01..08.txt  all-MiniLM-L6-v2          ChromaDB collection "zepto_policies"
 ingest.py            ingest.py                 chroma_db/  (cosine distance)
 chunk_text() ──────▶ model.encode(...) ──────▶ ids + vectors + text + metadata
                                                          │
 ─────────────────────────── query time ──────────────────┼────────────────────
                                                          │
 POST /ask {"query"} ─▶ LangGraph StateGraph (rag.py)     │
 (main.py)                │                               │
                          ▼                               │
                   classify_intent                        │
                    │          │                          │
       policy_question        general_question            │
                    ▼          ▼                          │
      retrieve_and_answer    direct_answer                │
      RETRIEVAL: embed query, top-3 ◀─────────────────────┘
      GENERATION: answer from top chunk      GENERATION: fixed canned string
                    │          │
                    ▼          ▼
         AssistantResponse {answer, sources, confidence}  (Pydantic)
                          │
                          ▼
                 JSON response from /ask
```

The four stages, in order:

1. **Ingestion — `ingest.py`.** Reads the eight files in `docs/`, one document per file. `chunk_text()` splits text into chunks of at most 1,000 characters. Every document is shorter than that, so each becomes a single chunk with a stable id such as `doc_02_chunk_01`. The source file name and chunk id are stored as metadata.
2. **Embedding — `ingest.py`.** `SentenceTransformer("all-MiniLM-L6-v2").encode(..., normalize_embeddings=True)` turns each chunk into a 384-dimensional vector, run locally on CPU. The vectors, chunk text and metadata are written to the persistent ChromaDB collection **`zepto_policies`** in `chroma_db/`, which is configured for **cosine** distance (`hnsw:space = cosine`). Re-running ingestion clears the collection first, so it is deterministic.
3. **Retrieval — the `retrieve_and_answer` LangGraph node** (`retrieve()` in `rag.py`). The query is embedded with the same model and ChromaDB returns the **top 3** most similar chunks by cosine similarity, with their ids. Retrieval always runs for real in both modes, since it needs no API key.
4. **Generation.** In the `retrieve_and_answer` node for policy questions, and in the `direct_answer` node for general questions. The output is validated into the Pydantic `AssistantResponse` model and returned by FastAPI.

Data flows through the graph as one `AssistantState` `TypedDict` (`query → intent → retrieved_chunks → answer, sources, confidence`). Each node reads the state and returns an updated copy.

### What `MOCK_LLM` changes

Only the LLM steps branch on `MOCK_LLM`. Ingestion, embedding, retrieval and the routing edge behave the same in both modes.

| Step | `MOCK_LLM` unset / `1` (default, graded) | `MOCK_LLM=0` (optional) |
|---|---|---|
| `classify_intent` | Keyword heuristic: `policy_question` if the lowercased query contains `delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card` or `support hours`; otherwise `general_question`. No LLM call. | The LLM classifies the query into one of the two intents |
| `retrieve_and_answer` (generation) | `"Based on the retrieved context: " + first 200 characters of the top chunk`; `sources` = the 3 retrieved chunk ids; `confidence = 1.0`. No LLM call. | The LLM answers from the 3 retrieved chunks using the structured prompt |
| `direct_answer` | Fixed string `"I can only answer questions about Zepto policies right now."`; `sources = []`; `confidence = 1.0`. No LLM call. | The LLM answers directly, with no retrieval |

## LangGraph graph

Built in `create_response()` in `rag.py`:

- **State:** `AssistantState(TypedDict)` with `query`, `intent`, `retrieved_chunks`, `answer`, `sources`, `confidence`.
- **Nodes:** `classify_intent`, `retrieve_and_answer`, `direct_answer`.
- **Edges:** `START → classify_intent`. Then a **conditional edge**: `route_intent()` returns `"retrieve_and_answer"` for `policy_question` and `"direct_answer"` otherwise. Both branches go to `END`. The routing reads only the `intent` field, so it does not depend on `MOCK_LLM`.

## Structured prompt template

`build_policy_prompt()` in `rag.py` contains the full template as text. The optional `MOCK_LLM=0` path uses it. Its sections:

- **ROLE:** "You are a Zepto Policy Support Assistant."
- **CONTEXT:** the three retrieved chunks, each prefixed with its id.
- **TASK:** answer using only the supplied policy context.
- **FORMAT:** JSON with `answer`, `sources` and `confidence`.
- **LENGTH:** concise and directly relevant.
- **NEGATIVE CONSTRAINT:** "Do not invent, assume, or use information that is not supported by the retrieved policy context."
- **FEW-SHOT EXAMPLES:** one grounded answer (the delivery fee on a INR 120 order, citing `doc_01_chunk_01`) and one refusal where the context does not cover the question.

## Response schema and retries

```python
class AssistantResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
```

- **Mock mode:** the fields are filled deterministically by code: `sources` holds the retrieved chunk ids (empty for general questions) and `confidence = 1.0`. There is no LLM output that could fail validation.
- **`MOCK_LLM=0`:** `validated_llm_response()` parses the LLM's JSON and validates it with `AssistantResponse.model_validate`. On failure it **retries up to 2 more times** (3 attempts in total), each time adding a corrective instruction that restates the exact schema. If all attempts fail, it returns a clearly marked error response (`"LLM response validation failed: ..."`, `confidence = 0.0`).

`main.py` exposes `POST /ask` with a Pydantic request model `AskRequest {query: str}` and `response_model=AskResponse`. FastAPI therefore validates both the request and the response.

## Docker

The `Dockerfile` installs CPU-only PyTorch and the requirements, copies the app, and **runs `ingest.py` at build time**, so the ChromaDB index and the embedding model are baked into the image. It serves the API with `uvicorn main:app --host 0.0.0.0 --port 7860`. `MOCK_LLM=1` is set as the image default.

```bash
cd support_assistant
docker build -t zepto-support-assistant .
docker run --rm -p 7860:7860 zepto-support-assistant

# in another terminal
curl -X POST http://127.0.0.1:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the refund policy for damaged groceries?"}'
```

The build needs internet access once, to download Python packages and the embedding model. The running container then answers in mock mode with no network access.

## Optional extension: real LLM with Ollama (`MOCK_LLM=0`, ungraded)

The optional real-LLM path uses [Ollama](https://ollama.com), which runs an open model locally for free, with no account, API key or payment:

```bash
ollama pull llama3.2
MOCK_LLM=0 OLLAMA_MODEL=llama3.2 uvicorn main:app --port 7860
```

`OLLAMA_URL` defaults to `http://localhost:11434/api/generate`. This path is not part of the graded baseline, and every example above was recorded with `MOCK_LLM` unset.

## Design decisions

- **One chunk per document.** Each policy is a single short paragraph (under 1,000 characters), so splitting it further would separate related rules, such as a return window and its exceptions. Retrieval results map directly to source documents.
- **Normalized embeddings + cosine distance.** Cosine similarity is the standard metric for `all-MiniLM-L6-v2`, and the same model embeds both documents and queries.
- **Top-3 retrieval.** The answer uses the single best chunk, but returning the ids of the top 3 as `sources` shows which neighbouring policies were considered (e.g. Damaged or Missing Items alongside Returns & Refunds).
- **Mock mode by default.** Keeps the graded behaviour deterministic, free and offline. The real-LLM path is opt-in only.
- **Index built at Docker build time.** A fresh container never starts with an empty collection.
