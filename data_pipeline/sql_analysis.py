from pathlib import Path
import sqlite3

import pandas as pd


BASE_DIR = Path(__file__).parent
DB_FILE = BASE_DIR / "books.db"
OUTPUT_DIR = BASE_DIR / "sql_outputs"


def run_sql_queries(connection):
    OUTPUT_DIR.mkdir(exist_ok=True)

    queries = {
        "query1": """
            SELECT
                c.category_name,
                COUNT(p.product_id) AS product_count
            FROM categories c
            JOIN products p
                ON c.category_id = p.category_id
            GROUP BY c.category_name
            ORDER BY product_count DESC
            LIMIT 10;
        """,
        "query2": """
            SELECT DISTINCT category_id
            FROM products
            WHERE category_id IN (1, 2, 3)
            ORDER BY category_id;
        """,
        "query3": """
            SELECT
                title,
                price_gbp,
                price_inr
            FROM products
            WHERE price_inr BETWEEN 3000 AND 5000
            ORDER BY price_inr DESC
            LIMIT 10;
        """,
        "query4": """
            SELECT
                title,
                rating,
                price_inr
            FROM products
            WHERE rating IN (4, 5)
            ORDER BY rating DESC, price_inr DESC
            LIMIT 10;
        """,
        "query5": """
            SELECT
                c.category_name,
                ROUND(AVG(p.price_inr), 2) AS average_price_inr
            FROM categories c
            JOIN products p
                ON c.category_id = p.category_id
            GROUP BY c.category_name
            ORDER BY average_price_inr DESC;
        """,
        "query6": """
            SELECT
                p.product_id,
                p.title,
                c.category_name,
                p.price_gbp,
                p.price_inr,
                p.availability,
                p.rating
            FROM products p
            INNER JOIN categories c
                ON p.category_id = c.category_id
            ORDER BY p.price_inr DESC
            LIMIT 10;
        """,
    }

    print("=" * 70)
    print("SQL ANALYSIS")
    print("=" * 70)

    all_results = {}

    for name, query in queries.items():
        print(f"\n{name}")
        print("-" * 70)
        print(query.strip())

        result = pd.read_sql_query(query, connection)
        all_results[name] = result

        print(result)

        # Save both query text and result.
        (OUTPUT_DIR / f"{name}.sql").write_text(query.strip() + "\n")
        result.to_csv(OUTPUT_DIR / f"{name}_output.csv", index=False)

    return all_results["query6"]


def run_pandas_join():
    print("\n" + "=" * 70)
    print("PANDAS JOIN / MERGE")
    print("=" * 70)

    df = pd.read_csv(BASE_DIR / "books_clean.csv")

    categories = (
        df[["category"]]
        .drop_duplicates()
        .sort_values("category")
        .reset_index(drop=True)
    )

    categories["category_id"] = range(1, len(categories) + 1)

    products = df.merge(
        categories,
        on="category",
        how="inner",
    )

    products = products.rename(
        columns={"category": "category_name"}
    )

    print(products.head(10))
    print(f"\nPandas merged rows: {len(products)}")

    products.to_csv(
        OUTPUT_DIR / "pandas_merge_output.csv",
        index=False,
    )


def main():
    connection = sqlite3.connect(DB_FILE)

    try:
        run_sql_queries(connection)
    finally:
        connection.close()

    run_pandas_join()


if __name__ == "__main__":
    main()
