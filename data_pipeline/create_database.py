from pathlib import Path
import sqlite3

import pandas as pd


BASE_DIR = Path(__file__).parent
CSV_FILE = BASE_DIR / "books_clean.csv"
DB_FILE = BASE_DIR / "books.db"


def create_database():
    df = pd.read_csv(CSV_FILE)

    connection = sqlite3.connect(DB_FILE)

    cursor = connection.cursor()

    # Start clean if the script is run again.
    cursor.execute("DROP TABLE IF EXISTS products")
    cursor.execute("DROP TABLE IF EXISTS categories")

    # Category table.
    cursor.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
        """
    )

    # Product table.
    cursor.execute(
        """
        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            availability TEXT NOT NULL,
            rating INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
        """
    )

    # Insert categories.
    categories = sorted(df["category"].unique())

    cursor.executemany(
        "INSERT INTO categories (category_name) VALUES (?)",
        [(category,) for category in categories],
    )

    # Create category lookup.
    category_lookup = {
        row[1]: row[0]
        for row in cursor.execute(
            "SELECT category_id, category_name FROM categories"
        )
    }

    # Insert products.
    products = []

    for _, row in df.iterrows():
        products.append(
            (
                row["product_id"],
                row["title"],
                category_lookup[row["category"]],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                row["availability"],
                int(row["rating"]),
            )
        )

    cursor.executemany(
        """
        INSERT INTO products (
            product_id,
            title,
            category_id,
            price_gbp,
            price_inr,
            availability,
            rating
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        products,
    )

    connection.commit()

    # Enable foreign-key enforcement.
    cursor.execute("PRAGMA foreign_keys = ON")

    print("=" * 60)
    print("DATABASE CREATED")
    print("=" * 60)
    print(f"Database: {DB_FILE}")

    category_count = cursor.execute(
        "SELECT COUNT(*) FROM categories"
    ).fetchone()[0]

    product_count = cursor.execute(
        "SELECT COUNT(*) FROM products"
    ).fetchone()[0]

    print(f"Categories: {category_count}")
    print(f"Products: {product_count}")

    print()
    print("Tables:")

    tables = cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
        """
    ).fetchall()

    for table in tables:
        print(f"- {table[0]}")

    connection.close()


if __name__ == "__main__":
    create_database()
