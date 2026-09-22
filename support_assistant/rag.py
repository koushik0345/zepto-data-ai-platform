import json
import os
import re
from pathlib import Path
from typing import TypedDict

import chromadb
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "all-MiniLM-L6-v2"

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)


# ---------------------------------------------------------
# Pydantic response schema
# ---------------------------------------------------------

class AssistantResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------

class AssistantState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_chunks: list[dict]
    answer: str
    sources: list[str]
    confidence: float


# ---------------------------------------------------------
# Embedding model + ChromaDB
# ---------------------------------------------------------

_model = None
_collection = None


def get_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def get_collection():
    global _collection

    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    return _collection


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve(query: str, top_k: int = 3) -> list[dict]:
    model = get_model()
    collection = get_collection()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        retrieved.append(
            {
                "id": metadata["chunk_id"],
                "document": document,
                "source": metadata["source"],
                "distance": float(distance),
            }
        )

    return retrieved


# ---------------------------------------------------------
# Intent classification
# ---------------------------------------------------------

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


def classify_intent(query: str) -> str:
    """
    Default MOCK_LLM behavior required by the assignment.

    Policy keywords -> policy_question
    Everything else -> general_question
    """

    if MOCK_LLM:
        query_lower = query.lower()

        if any(keyword in query_lower for keyword in POLICY_KEYWORDS):
            return "policy_question"

        return "general_question"

    # Optional real Ollama classification.
    prompt = f"""
Classify the user's question into exactly one category:

policy_question
general_question

Return JSON only:
{{"intent": "policy_question"}}

User question:
{query}
"""

    result = call_ollama(prompt)

    try:
        data = json.loads(result)
        intent = data.get("intent")

        if intent in {"policy_question", "general_question"}:
            return intent

    except (json.JSONDecodeError, TypeError):
        pass

    return "general_question"


# ---------------------------------------------------------
# Prompt skeleton
# ---------------------------------------------------------

def build_policy_prompt(query: str, context: str) -> str:
    return f"""
ROLE:
You are a Zepto Policy Support Assistant.

CONTEXT:
{context}

TASK:
Answer the user's question using only the supplied policy context.

FORMAT:
Return JSON with:
- answer: string
- sources: list of chunk IDs
- confidence: number from 0 to 1

LENGTH:
Keep the answer concise and directly relevant.

NEGATIVE CONSTRAINT:
Do not invent, assume, or use information that is not supported by the retrieved policy context.

FEW-SHOT EXAMPLES:
Example 1 (answer is in the context)
Context:
[doc_01_chunk_01] ... Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. ...
User:
What is the delivery fee on a INR 120 order?
Assistant:
{{"answer":"Orders below INR 149 incur a flat INR 25 delivery fee, so a INR 120 order is charged INR 25.","sources":["doc_01_chunk_01"],"confidence":0.95}}

Example 2 (answer is NOT in the context)
Context:
[doc_08_chunk_01] Zepto customer support is available via in-app chat 24 hours a day, 7 days a week ...
User:
Can I pay with cryptocurrency?
Assistant:
{{"answer":"The provided Zepto policies do not cover this, so I cannot answer it.","sources":[],"confidence":0.2}}

USER QUESTION:
{query}
"""


def build_direct_prompt(query: str) -> str:
    return f"""
ROLE:
You are a Zepto Policy Support Assistant.

TASK:
Answer the user's question directly.

FORMAT:
Return JSON with:
- answer: string
- sources: list
- confidence: number from 0 to 1

LENGTH:
Keep the answer concise.

NEGATIVE CONSTRAINT:
Do not pretend to know Zepto policy information that is not available.

USER QUESTION:
{query}
"""


# ---------------------------------------------------------
# Ollama
# ---------------------------------------------------------

def call_ollama(prompt: str) -> str:
    import requests

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data.get("response", "")


# ---------------------------------------------------------
# LLM response validation
# ---------------------------------------------------------

def extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("No JSON object found in LLM response.")

    return json.loads(match.group(0))


def validated_llm_response(
    prompt: str,
    fallback_sources: list[str],
) -> AssistantResponse:

    last_error = None

    # Initial attempt + 2 retries = maximum 3 attempts.
    for attempt in range(3):
        try:
            raw = call_ollama(prompt)
            data = extract_json(raw)

            if "sources" not in data:
                data["sources"] = fallback_sources

            if "confidence" not in data:
                data["confidence"] = 0.8

            result = AssistantResponse.model_validate(data)

            return result

        except Exception as exc:
            last_error = exc

            prompt = f"""
The previous response failed validation.

Return ONLY valid JSON matching this exact schema:
{{
  "answer": "string",
  "sources": ["string"],
  "confidence": 0.0
}}

The confidence value must be between 0 and 1.

Original task:
{prompt}
"""

    return AssistantResponse(
        answer=f"LLM response validation failed: {last_error}",
        sources=fallback_sources,
        confidence=0.0,
    )


# ---------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------

def classify_intent_node(state: AssistantState) -> AssistantState:
    intent = classify_intent(state["query"])

    return {
        **state,
        "intent": intent,
    }


def retrieve_and_answer_node(state: AssistantState) -> AssistantState:
    query = state["query"]

    retrieved = retrieve(query, top_k=3)

    if not retrieved:
        return {
            **state,
            "retrieved_chunks": [],
            "answer": "No relevant policy information was found.",
            "sources": [],
            "confidence": 0.0,
        }

    sources = [item["id"] for item in retrieved]

    if MOCK_LLM:
        top_snippet = retrieved[0]["document"][:200]

        answer = (
            "Based on the retrieved context: "
            + top_snippet
        )

        return {
            **state,
            "retrieved_chunks": retrieved,
            "answer": answer,
            "sources": sources,
            "confidence": 1.0,
        }

    context = "\n\n".join(
        f"[{item['id']}] {item['document']}"
        for item in retrieved
    )

    prompt = build_policy_prompt(query, context)

    result = validated_llm_response(
        prompt,
        sources,
    )

    return {
        **state,
        "retrieved_chunks": retrieved,
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
    }


def direct_answer_node(state: AssistantState) -> AssistantState:
    query = state["query"]

    if MOCK_LLM:
        return {
            **state,
            "answer": (
                "I can only answer questions about "
                "Zepto policies right now."
            ),
            "sources": [],
            "confidence": 1.0,
        }

    prompt = build_direct_prompt(query)

    result = validated_llm_response(
        prompt,
        [],
    )

    return {
        **state,
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
    }


# ---------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------

def route_intent(state: AssistantState) -> str:
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ---------------------------------------------------------
# Public assistant function
# ---------------------------------------------------------

def create_response(query: str) -> AssistantResponse:
    from langgraph.graph import StateGraph, START, END

    graph = StateGraph(AssistantState)

    graph.add_node(
        "classify_intent",
        classify_intent_node,
    )

    graph.add_node(
        "retrieve_and_answer",
        retrieve_and_answer_node,
    )

    graph.add_node(
        "direct_answer",
        direct_answer_node,
    )

    graph.add_edge(
        START,
        "classify_intent",
    )

    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    graph.add_edge(
        "retrieve_and_answer",
        END,
    )

    graph.add_edge(
        "direct_answer",
        END,
    )

    app = graph.compile()

    result = app.invoke(
        {
            "query": query,
        }
    )

    return AssistantResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
    )
