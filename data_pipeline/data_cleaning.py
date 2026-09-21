from pathlib import Path

import pandas as pd


INPUT_FILE = Path(__file__).parent / "books_raw.csv"
OUTPUT_FILE = Path(__file__).parent / "books_clean.csv"

GBP_TO_INR = 105.50


def clean_data(df):
    df = df.copy()

    # Clean text fields.
    text_columns = ["title", "category", "availability"]

    for column in text_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    # Clean price.
    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce",
    )

    # Fixed assignment conversion rate.
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    # Clean rating.
    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce",
    )

    # Remove rows missing essential fields.
    df = df.dropna(
        subset=[
            "product_id",
            "title",
            "category",
            "price_gbp",
            "price_inr",
            "availability",
            "rating",
        ]
    )

    # Remove duplicate products.
    df = df.drop_duplicates(subset=["product_id"])

    return df


def main():
    print(f"Reading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Raw rows: {len(df)}")

    clean_df = clean_data(df)

    clean_df.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"Rows after cleaning: {len(clean_df)}")
    print(f"Columns: {len(clean_df.columns)}")
    print(f"GBP to INR rate: {GBP_TO_INR}")
    print(f"Saved to: {OUTPUT_FILE}")

    print()
    print("Rows by category:")
    print(clean_df.groupby("category").size())

    print()
    print("Price validation:")
    print(
        clean_df[
            ["title", "price_gbp", "price_inr"]
        ].head()
    )


if __name__ == "__main__":
    main()
