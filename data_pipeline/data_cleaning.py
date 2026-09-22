from pathlib import Path

import pandas as pd


INPUT_FILE = Path(__file__).parent / "books_raw.csv"
OUTPUT_FILE = Path(__file__).parent / "books_clean.csv"

# Fixed, project-defined baseline rate (not a live market rate).
GBP_TO_INR = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def parse_price(series):
    # "£51.77" -> 51.77. Anything that does not contain a number becomes NaN.
    return pd.to_numeric(
        series.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0],
        errors="coerce",
    )


def parse_rating(series):
    # "Three" -> 3. Unknown words become NaN.
    return series.astype(str).str.strip().map(RATING_MAP)


def parse_in_stock(series):
    # "In stock (19 available)" -> True, "Out of stock" -> False.
    # Text matching neither pattern becomes NaN.
    text = series.astype(str).str.strip().str.lower()

    in_stock = pd.Series(pd.NA, index=series.index, dtype="boolean")
    in_stock[text.str.startswith("in stock")] = True
    in_stock[text.str.startswith("out of stock")] = False

    return in_stock


def clean_data(df):
    df = df.copy()

    # Normalise whitespace in text fields.
    for column in ["product_id", "title", "category", "availability"]:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    df["price_gbp"] = parse_price(df["price"])
    df["rating"] = parse_rating(df["star_rating"])
    df["in_stock"] = parse_in_stock(df["availability"])

    # Messy-row policy: DROP rows where any required field fails to parse.
    # Median imputation would invent a price or rating for a real product,
    # which would silently distort price benchmarks; availability is not
    # numeric, so it cannot be median-imputed at all.
    required = ["price_gbp", "rating", "in_stock"]
    failed = df[required].isna().any(axis=1)

    if failed.any():
        print(f"Dropping {failed.sum()} row(s) that failed to parse:")
        print(df.loc[failed, ["title", "price", "star_rating", "availability"]])

    df = df.loc[~failed].copy()

    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    df = df.drop_duplicates(subset=["product_id"])

    return df[
        [
            "product_id",
            "title",
            "category",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "availability",
        ]
    ]


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
    print(f"Rows dropped: {len(df) - len(clean_df)}")
    print(f"GBP to INR rate: 1 GBP = {GBP_TO_INR:.2f} INR")
    print(f"Saved to: {OUTPUT_FILE}")

    print()
    print("Column types:")
    print(clean_df.dtypes)

    print()
    print("Rows by category:")
    print(clean_df.groupby("category").size())

    print()
    print("In stock counts:")
    print(clean_df["in_stock"].value_counts())

    print()
    print("Price validation:")
    print(clean_df[["title", "price_gbp", "price_inr", "rating", "in_stock"]].head())


if __name__ == "__main__":
    main()
