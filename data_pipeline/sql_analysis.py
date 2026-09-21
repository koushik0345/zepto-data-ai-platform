from pathlib import Path
import sqlite3

import pandas as pd


DB_FILE = Path(__file__).parent / "books.db"


def run_sql_queries(connection):
    print("=" * 70)
    print("SQL ANALYSIS")
    print("=" * 70)

    # Query 1: Count products by category.
    query1 = """
        SELECT
            c.category_name,
            COUNT(p.product_id) AS product_count
        FROM categories c
        JOIN products p
            ON c.category_id = p.category_id
        GROUP BY c.category_name
        ORDER BY product_count DESC;
    """

    print("\n1. Product count by category")
    print(pd.read_sql_query(query1, connection))

    # Query 2: Books above a price threshold.
    query2 = """
        SELECT
            title,
            category_id,
            price_gbp,
            price_inr
        FROM products
        WHERE price_inr > 4000
        ORDER BY price_inr DESC;
    """

    print("\n2. Products above ₹4,000")
    print(pd.read_sql_query(query2, connection).head(10))

    # Query 3: Average price by category.
    query3 = """
        SELECT
            c.category_name,
            ROUND(AVG(p.price_inr), 2) AS average_price_inr
        FROM categories c
        JOIN products p
            ON c.category_id = p.category_id
        GROUP BY c.category_name
        ORDER BY average_price_inr DESC;
    """

    print("\n3. Average price by category")
    print(pd.read_sql_query(query3, connection))

    # Query 4: Highest-rated books.
    query4 = """
        SELECT
            title,
            rating,
            price_inr
        FROM products
        WHERE rating = 5
        ORDER BY price_inr DESC;
    """

    print("\n4. Five-star books")
    print(pd.read_sql_query(query4, connection).head(10))

    # Query 5: Required JOIN.
    query5 = """
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
        ORDER BY p.price_inr DESC;
    """

    print("\n5. Products joined with categories")
    join_result = pd.read_sql_query(query5, connection)
    print(join_result.head(10))

    return join_result


def run_pandas_join():
    print("\n" + "=" * 70)
    print("PANDAS JOIN / MERGE")
    print("=" * 70)

    df = pd.read_csv(Path(__file__).parent / "books_clean.csv")

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


def main():
    connection = sqlite3.connect(DB_FILE)

    try:
        run_sql_queries(connection)
    finally:
        connection.close()

    run_pandas_join()


if __name__ == "__main__":
    main()
