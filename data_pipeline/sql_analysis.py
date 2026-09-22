from pathlib import Path
import sqlite3

import pandas as pd


BASE_DIR = Path(__file__).parent
DB_FILE = BASE_DIR / "books.db"
CLEAN_CSV = BASE_DIR / "books_clean.csv"
OUTPUT_DIR = BASE_DIR / "sql_outputs"
REPORT_FILE = OUTPUT_DIR / "query_results.md"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 45)


# Each query lists the SQL clauses it demonstrates.
QUERIES = {
    "query1": (
        "JOIN + GROUP BY + ORDER BY + LIMIT — books per category",
        """
        SELECT
            c.category_name,
            COUNT(p.product_id) AS product_count
        FROM categories c
        JOIN products p
            ON c.category_id = p.category_id
        GROUP BY c.category_name
        ORDER BY product_count DESC, c.category_name
        LIMIT 10;
        """,
    ),
    "query2": (
        "DISTINCT + WHERE + IN + ORDER BY — distinct ratings of in-stock books in two categories",
        """
        SELECT DISTINCT rating
        FROM products
        WHERE category_id IN (1, 3)
          AND in_stock = 1
        ORDER BY rating;
        """,
    ),
    "query3": (
        "WHERE + BETWEEN + ORDER BY + LIMIT — books priced INR 3000-5000",
        """
        SELECT
            title,
            price_gbp,
            price_inr
        FROM products
        WHERE price_inr BETWEEN 3000 AND 5000
        ORDER BY price_inr DESC
        LIMIT 10;
        """,
    ),
    "query4": (
        "WHERE + IN + ORDER BY + LIMIT — highest-rated books that are in stock",
        """
        SELECT
            title,
            rating,
            price_inr,
            in_stock
        FROM products
        WHERE rating IN (4, 5)
          AND in_stock = 1
        ORDER BY rating DESC, price_inr DESC
        LIMIT 10;
        """,
    ),
    "query5": (
        "JOIN + GROUP BY + aggregate + ORDER BY — average price and rating per category",
        """
        SELECT
            c.category_name,
            ROUND(AVG(p.price_inr), 2) AS average_price_inr,
            ROUND(AVG(p.rating), 2) AS average_rating
        FROM categories c
        JOIN products p
            ON c.category_id = p.category_id
        GROUP BY c.category_name
        ORDER BY average_price_inr DESC;
        """,
    ),
    "query6": (
        "INNER JOIN + ORDER BY + LIMIT — 10 most expensive books with category names "
        "(reproduced below with pd.merge)",
        """
        SELECT
            p.product_id,
            p.title,
            c.category_name,
            p.price_gbp,
            p.price_inr,
            p.rating,
            p.in_stock
        FROM products p
        INNER JOIN categories c
            ON p.category_id = c.category_id
        ORDER BY p.price_inr DESC, p.product_id
        LIMIT 10;
        """,
    ),
}

JOIN_QUERY = "query6"


def run_sql_queries(connection, report):
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("SQL ANALYSIS")
    print("=" * 70)

    report.append("## SQL queries and their output\n")

    results = {}

    for name, (description, query) in QUERIES.items():
        query = "\n".join(line[8:] if line.startswith(" " * 8) else line
                          for line in query.strip("\n").splitlines())

        # Every result is read back into a DataFrame with pd.read_sql.
        result = pd.read_sql(query, connection)
        results[name] = result

        print(f"\n{name}: {description}")
        print("-" * 70)
        print(query)
        print(result.to_string(index=False))

        (OUTPUT_DIR / f"{name}.sql").write_text(query + "\n")
        result.to_csv(OUTPUT_DIR / f"{name}_output.csv", index=False)

        report.append(f"### {name} — {description}\n")
        report.append(f"```sql\n{query}\n```\n")
        report.append(f"```text\n{result.to_string(index=False)}\n```\n")

    return results


def reproduce_join_with_pandas(sql_result, report):
    """Rebuild the two tables in memory from the cleaned CSV and
    reproduce the JOIN query with pd.merge — no SQL involved."""
    print("\n" + "=" * 70)
    print("PANDAS MERGE REPRODUCTION OF THE JOIN QUERY")
    print("=" * 70)

    clean = pd.read_csv(CLEAN_CSV)

    # In-memory 'categories' table: same sorted order as create_database.py,
    # so the generated category_id values match the SQLite ids.
    categories = pd.DataFrame(
        {"category_name": sorted(clean["category"].unique())}
    )
    categories["category_id"] = range(1, len(categories) + 1)

    # In-memory 'products' table, holding only the foreign key.
    products = clean.merge(
        categories,
        left_on="category",
        right_on="category_name",
    )[[
        "product_id",
        "title",
        "category_id",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
    ]]
    products["in_stock"] = products["in_stock"].astype(int)

    # INNER JOIN ... ORDER BY price_inr DESC, product_id LIMIT 10
    merged = (
        products.merge(categories, on="category_id", how="inner")
        .sort_values(["price_inr", "product_id"], ascending=[False, True])
        .head(10)
        .reset_index(drop=True)
    )[list(sql_result.columns)]

    merged.to_csv(OUTPUT_DIR / "pandas_merge_output.csv", index=False)

    try:
        pd.testing.assert_frame_equal(
            sql_result.reset_index(drop=True),
            merged,
            check_dtype=False,
        )
        match = True
    except AssertionError as error:
        print(error)
        match = False

    side_by_side = pd.concat(
        {
            "pd.read_sql (SQL JOIN)": sql_result[["title", "category_name", "price_inr"]],
            "pd.merge (in-memory)": merged[["title", "category_name", "price_inr"]],
        },
        axis=1,
    )

    print(side_by_side.to_string(index=False))
    print(f"\nAll {len(merged)} rows x {len(merged.columns)} columns identical: {match}")

    report.append("## pd.read_sql vs pd.merge for the JOIN query\n")
    report.append(
        f"`{JOIN_QUERY}` was read with `pd.read_sql`, then reproduced with "
        "`pd.merge` on in-memory `categories` and `products` DataFrames built "
        "from `books_clean.csv` (no SQL). Key columns side by side:\n"
    )
    report.append(f"```text\n{side_by_side.to_string(index=False)}\n```\n")
    report.append(
        f"`pd.testing.assert_frame_equal` over all {len(merged)} rows and "
        f"{len(merged.columns)} columns: **{'identical' if match else 'MISMATCH'}**.\n"
    )

    return match


def main():
    report = [
        "# Module 1 — SQL query log\n",
        "Generated by `python sql_analysis.py` against `books.db`.\n",
    ]

    connection = sqlite3.connect(DB_FILE)

    try:
        results = run_sql_queries(connection, report)
    finally:
        connection.close()

    match = reproduce_join_with_pandas(results[JOIN_QUERY], report)

    REPORT_FILE.write_text("\n".join(report))
    print(f"\nQuery log written to: {REPORT_FILE}")

    if not match:
        raise SystemExit("pd.merge output does not match the SQL JOIN output.")


if __name__ == "__main__":
    main()
