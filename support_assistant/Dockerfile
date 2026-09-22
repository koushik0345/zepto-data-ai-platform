FROM python:3.11-slim

WORKDIR /app

# MOCK_LLM=1 is the graded, fully offline default.
ENV PYTHONUNBUFFERED=1 \
    MOCK_LLM=1

COPY requirements.txt .

# CPU-only PyTorch (needed by sentence-transformers) keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the ChromaDB index and cache the embedding model inside the image,
# so the container can answer policy questions as soon as it starts.
RUN python ingest.py

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
