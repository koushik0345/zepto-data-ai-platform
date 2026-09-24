# Zepto Data & AI Platform

Repository: <https://github.com/koushik0345/zepto-data-ai-platform>

An end-to-end AI/ML capstone in one repository, with three connected modules:

1. **[`data_pipeline/`](data_pipeline/)**: scrapes catalogue data, cleans it, converts prices to INR and loads it into a normalized SQLite database for SQL and pandas analysis.
2. **[`analytics/`](analytics/)**: profiles and cleans a passenger dataset (Titanic), tells a visual story about who survived, then builds, tunes and evaluates predictive models on the same data.
3. **[`support_assistant/`](support_assistant/)**: a grounded GenAI support service. It retrieves from Zepto's own policy documents through a LangGraph flow and serves structured answers from a FastAPI endpoint.

Each module has its own README with the full write-up: every required interpretation, table and recorded output. This root README covers setup, how to run everything, and a summary of the design decisions.

## Repository structure

```text
zepto-data-ai-platform/
├── README.md                    <- this file
├── requirements.txt             <- installs all three modules
├── data_pipeline/
│   ├── README.md                <- Module 1 write-up
│   ├── requirements.txt
│   ├── scrape_books.py          <- scrape  -> books_raw.csv
│   ├── data_cleaning.py         <- clean   -> books_clean.csv
│   ├── create_database.py       <- load    -> books.db (regenerated, not committed)
│   ├── sql_analysis.py          <- query   -> sql_outputs/
│   └── sql_outputs/             <- query strings, outputs, query_results.md
├── analytics/
│   ├── README.md                <- Module 2 write-up (all interpretations + model table)
│   ├── requirements.txt
│   ├── titanic_analysis.py      <- Part A: the single dataset load, profiling, cleaning, EDA
│   ├── modeling.py              <- Part B: classifiers, tuning, imbalance, regression
│   ├── titanic.csv              <- committed offline copy of the loaded dataset
│   ├── models/                  <- complete fitted pipelines (.joblib)
│   └── figures/                 <- generated charts
└── support_assistant/
    ├── README.md                <- Module 3 write-up (example calls + architecture)
    ├── requirements.txt
    ├── docs/doc_01..08.txt      <- the eight Zepto policy documents
    ├── ingest.py                <- chunk + embed + store in ChromaDB
    ├── rag.py                   <- LangGraph graph, retrieval, prompt, schema
    ├── main.py                  <- FastAPI app (POST /ask)
    └── Dockerfile
```

## Setup

Requires Python 3.11 or newer (developed on Python 3.13 on macOS).

**Requirements layout:** each module has **its own `requirements.txt`**, and the root `requirements.txt` installs all three through `-r` includes. Install everything at once:

```bash
git clone https://github.com/koushik0345/zepto-data-ai-platform.git
cd zepto-data-ai-platform

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

No paid services, API keys or accounts are needed. Internet access is needed on the first run only, to scrape books.toscrape.com, to download the Titanic dataset through Seaborn, and to download the `all-MiniLM-L6-v2` embedding model.

## Running each module end to end

All commands start from the repository root with the virtual environment active.

### Module 1 — Data pipeline

```bash
cd data_pipeline
python scrape_books.py       # 60 books across 3 categories -> books_raw.csv
python data_cleaning.py      # typed columns + price_inr     -> books_clean.csv
python create_database.py    # two-table SQLite schema       -> books.db
python sql_analysis.py       # 6 SQL queries + pd.merge check -> sql_outputs/
cd ..
```

The currency conversion uses the fixed project baseline rate **1 GBP = 105.50 INR**.

### Module 2 — Analytics

```bash
python analytics/titanic_analysis.py   # loads the dataset once, saves titanic.csv, runs the EDA
python analytics/modeling.py           # reads titanic.csv; models, tuning, regression, saved pipeline
```

Run `titanic_analysis.py` first. It makes the only `sns.load_dataset("titanic")` call, and `modeling.py` continues from the `titanic.csv` it saves. `titanic.csv` is committed, so `modeling.py` also works offline.

### Module 3 — Support assistant

```bash
cd support_assistant
python ingest.py                       # build the ChromaDB index (once)
uvicorn main:app --port 7860           # serve POST /ask; leave MOCK_LLM unset

# in a second terminal
curl -X POST http://127.0.0.1:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the refund policy for damaged groceries?"}'
```

Or with Docker:

```bash
cd support_assistant
docker build -t zepto-support-assistant .
docker run --rm -p 7860:7860 zepto-support-assistant
```

## Design decisions

### Module 1 — Data pipeline ([full write-up](data_pipeline/README.md))

- **Scope:** three categories (Mystery, Historical Fiction, Sequential Art) × 20 books = **60 books**.
- **Raw first, parse later.** The scraper stores fields exactly as listed on the site (`£47.82`, `Four`, `In stock`). All type conversion happens in one place, `data_cleaning.py`, which produces `price_gbp` (float), `rating` (int 1–5), `in_stock` (bool) and `price_inr` (float).
- **Messy rows are dropped, not imputed.** Imputing a median price or rating would invent data for a real product and distort price benchmarks, and availability text cannot be median-imputed. Unparseable rows are dropped and printed. On the current site none fail (60 → 60).
- **Normalized schema:** `categories(category_id PK, category_name UNIQUE)` and `products(product_id PK, …, category_id FK → categories)`. Foreign keys are enforced with `PRAGMA foreign_keys = ON`, and `rating`/`in_stock` are protected by `CHECK` constraints.
- **Verified SQL ↔ pandas equivalence.** The JOIN query is reproduced with `pd.merge` on in-memory DataFrames and checked with `pd.testing.assert_frame_equal`: all 10 rows × 7 columns are identical.

### Module 2 — Analytics ([full write-up](analytics/README.md))

- **One load.** `sns.load_dataset("titanic")` runs once and is saved immediately to `titanic.csv`; all modeling continues from that file.
- **Threshold-based cleaning:** `embarked` (0.22% missing) → drop rows; `age` (19.87%) → median imputation; `deck` (77.22%) → drop the column, because its missingness mostly just means "not first class", which `pclass` already records.
- **Leakage-safe modeling.** A stratified split (61.6% / 38.4% class balance) comes before any preprocessing. Imputers, the one-hot encoder and the scaler live in a `ColumnTransformer` inside each `Pipeline`, so they are fit on the training split only. SMOTE is applied to the training fold only.
- **Results:** Logistic Regression has the best ROC-AUC (0.850) and recall (0.768), and is the recommended and saved model. The GridSearch-tuned Random Forest (`max_depth=5`, `max_features="sqrt"`, `n_estimators=200`) reaches an OOB score of 0.824. SMOTE gave the best F1 in the imbalance comparison (0.761 vs 0.729 baseline). The fare regression has R² 0.396 and clear heteroscedasticity.
- **Deployable artifact:** `models/best_pipeline.joblib` is the complete fitted pipeline. After reloading, it predicts on raw rows, including one with a missing value.

### Module 3 — Support assistant ([full write-up](support_assistant/README.md))

- **RAG pipeline:** ingestion and chunking (one chunk per document, since each policy is a single short paragraph) → `all-MiniLM-L6-v2` embeddings → ChromaDB collection `zepto_policies` (cosine) → top-3 retrieval → generation.
- **LangGraph routing:** `classify_intent` → conditional edge → `retrieve_and_answer` or `direct_answer`, over a `TypedDict` state.
- **Offline, deterministic by default.** With `MOCK_LLM` unset, intent comes from the keyword heuristic, policy answers use the `"Based on the retrieved context: …"` template, and general questions get a fixed string. No LLM is called. Retrieval runs for real in both modes.
- **Optional real LLM:** with `MOCK_LLM=0`, the structured role–context–task–format–length prompt (with a negative constraint and few-shot examples) is sent to a local **Ollama** model, which is free and needs no API key. Invalid JSON is retried up to 2 more times before a marked error response is returned.
- **Pydantic everywhere:** the request model `{query}` and the response model `{answer, sources, confidence ∈ [0, 1]}` are both validated by FastAPI.
- **Container-ready:** the Dockerfile builds the ChromaDB index at image build time, so `docker run` serves working answers immediately.

## Git workflow

Work was done on feature branches and merged back into `main` with merge commits. `feature/data-pipeline` and `feature/submission-polish` each have several commits before their merge. `git log --graph --all --oneline` shows the branches and merges.
