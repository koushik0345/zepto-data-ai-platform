# Module 1 — Data Pipeline

Scrapes book listings from [books.toscrape.com](https://books.toscrape.com/), cleans them into typed columns, converts prices to INR at a fixed rate, loads them into a normalized SQLite database, and queries it with both SQL and pandas.

## Install and run

From the repository root, with the virtual environment active:

```bash
pip install -r data_pipeline/requirements.txt

cd data_pipeline
python scrape_books.py      # -> books_raw.csv
python data_cleaning.py     # -> books_clean.csv
python create_database.py   # -> books.db
python sql_analysis.py      # -> sql_outputs/
```

Each script reads the previous script's output, so run them in this order. `books.db` is not committed: `create_database.py` recreates it from scratch (it drops and recreates both tables every time it runs).

| File | Stage | Output |
|---|---|---|
| `scrape_books.py` | Scrape raw fields with `requests` + `BeautifulSoup` | `books_raw.csv` |
| `data_cleaning.py` | Parse types, convert GBP → INR, drop unparseable rows | `books_clean.csv` |
| `create_database.py` | Create the two-table schema and insert rows | `books.db` |
| `sql_analysis.py` | Run 6 SQL queries, reproduce the JOIN with `pd.merge` | `sql_outputs/` |

## Scraping

Scope: all books on the first listing pages of three categories: **Mystery, Historical Fiction and Sequential Art**, 20 books each, **60 books in total**. The scraper follows the "next" pagination link if a category page has fewer than 20 books.

For each book the scraper stores the raw text exactly as listed, with no conversion:

| Raw column | Example |
|---|---|
| `title` | `Sharp Objects` |
| `price` | `£47.82` |
| `star_rating` | `Four` |
| `availability` | `In stock` |
| `category` | `Mystery` |

It also stores `product_id`, the book's URL slug (e.g. `sharp-objects_997`), which is unique per book and is used as the primary key.

## Cleaning decisions

| Clean column | Type | Rule |
|---|---|---|
| `price_gbp` | `float` | Strip the `£` symbol and parse the number |
| `rating` | `int` (1–5) | Map `One`…`Five` to 1…5 |
| `in_stock` | `bool` | `True` if the text starts with "In stock", `False` if it starts with "Out of stock" |
| `price_inr` | `float` | `price_gbp × 105.50`, rounded to 2 decimals |

**Currency conversion:** the fixed, project-defined baseline rate **1 GBP = 105.50 INR** is used. It is a constant in `data_cleaning.py` (`GBP_TO_INR`), not a live or historical rate, so no API call or network access is needed for the conversion.

**Messy rows are dropped, not imputed.** If `price`, `star_rating` or `availability` fails to parse (for example, an unexpected rating word or a price with no number), the row is dropped and printed so the drop is visible. Reasons:

- Median-imputing a price or rating would invent a value for a real product and skew price benchmarks, which is the whole purpose of this dataset.
- `availability` is text, not numeric, so median imputation does not apply to it.
- Dropping a rare bad row costs little at this sample size.

On the current site no rows fail to parse (60 raw → 60 clean). All 60 scraped books are listed as "In stock", so `in_stock` is `True` for every row. That is what the site shows, not a parsing default: unrecognised availability text would become a dropped row, not `True`.

## Database schema

Two normalized tables linked by a primary/foreign key. `PRAGMA foreign_keys = ON` is set before any insert so SQLite enforces the relationship.

```sql
CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
);

CREATE TABLE products (
    product_id   TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    category_id  INTEGER NOT NULL,
    price_gbp    REAL NOT NULL,
    price_inr    REAL NOT NULL,
    rating       INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock     INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    availability TEXT NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
```

Category names are stored once in `categories`, and each product refers to its category by id. `in_stock` is stored as `0`/`1` because SQLite has no boolean type.

## SQL queries

Six queries run against `books.db`. Together they cover every required clause:

| Query | Clauses demonstrated |
|---|---|
| `query1` | `JOIN`, `GROUP BY`, `ORDER BY`, `LIMIT` |
| `query2` | `DISTINCT`, `WHERE`, `IN` |
| `query3` | `WHERE`, `BETWEEN`, `ORDER BY`, `LIMIT` |
| `query4` | `WHERE`, `IN`, `ORDER BY`, `LIMIT` |
| `query5` | `JOIN`, `GROUP BY`, `AVG`, `ORDER BY` |
| `query6` | `INNER JOIN`, `ORDER BY`, `LIMIT` |

**Every query string and its full printed output are in [`sql_outputs/query_results.md`](sql_outputs/query_results.md).** Each query is also saved as `sql_outputs/queryN.sql`, with its result as `sql_outputs/queryN_output.csv`.

All six results are read back into pandas DataFrames with `pd.read_sql(query, connection)`.

Selected results:

- **query5** (average price per category): Historical Fiction INR 3,732.49, Sequential Art INR 3,473.01, Mystery INR 3,459.77. Average ratings are almost identical (2.90–2.95), so category explains little of the price or rating difference in this sample.
- **query6** (the JOIN): the most expensive book is *Boar Island (Anna Pigeon #19)* (Mystery) at £59.48 = INR 6,275.14.

## SQL JOIN vs `pd.merge`

`sql_analysis.py` reproduces `query6` without SQL:

1. Build an in-memory `categories` DataFrame from the unique category names in `books_clean.csv`. They are sorted the same way `create_database.py` inserts them, so the ids match.
2. Build an in-memory `products` DataFrame that holds `category_id`.
3. `pd.merge(products, categories, on="category_id", how="inner")`, sort by `price_inr` descending, and take the top 10.

The script prints both results side by side and checks them with `pd.testing.assert_frame_equal`. **All 10 rows × 7 columns are identical.** The side-by-side output is recorded at the end of [`sql_outputs/query_results.md`](sql_outputs/query_results.md).
