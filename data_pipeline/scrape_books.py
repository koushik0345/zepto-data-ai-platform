import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://books.toscrape.com/"
OUTPUT_FILE = Path(__file__).parent / "books_raw.csv"

CATEGORIES = {
    "Mystery": "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
    "Historical Fiction": "https://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html",
    "Sequential Art": "https://books.toscrape.com/catalogue/category/books/sequential-art_5/index.html",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36"
}


def rating_to_number(rating_text):
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5,
    }
    return rating_map.get(rating_text, None)


def scrape_category(category_name, category_url, target_books=20):
    books = []
    next_url = category_url

    while next_url and len(books) < target_books:
        print(f"Scraping {category_name}: {next_url}")

        response = requests.get(
            next_url,
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for article in soup.select("article.product_pod"):
            if len(books) >= target_books:
                break

            title_tag = article.select_one("h3 a")
            price_tag = article.select_one(".price_color")
            availability_tag = article.select_one(".availability")
            rating_tag = article.select_one("p.star-rating")

            title = title_tag.get("title", "").strip()
            price_text = price_tag.get_text(strip=True)
            availability_text = availability_tag.get_text(" ", strip=True)

            price_match = re.search(r"([\d.]+)", price_text)
            price_gbp = float(price_match.group(1)) if price_match else None

            rating_class = rating_tag.get("class", [])
            rating_word = next(
                (item for item in rating_class if item != "star-rating"),
                None,
            )

            product_url = title_tag.get("href", "").strip()

            books.append(
                {
                    "product_id": product_url,
                    "title": title,
                    "category": category_name,
                    "price_gbp": price_gbp,
                    "availability": availability_text,
                    "rating": rating_to_number(rating_word),
                }
            )

        next_link = soup.select_one("li.next a")

        if next_link and len(books) < target_books:
            next_href = next_link.get("href")
            current_page = next_url.rsplit("/", 1)[0]
            next_url = f"{current_page}/{next_href}"
        else:
            next_url = None

        time.sleep(0.5)

    return books


def main():
    all_books = []

    for category_name, category_url in CATEGORIES.items():
        category_books = scrape_category(category_name, category_url, target_books=20)
        all_books.extend(category_books)
        print(f"{category_name}: {len(category_books)} books collected")

    df = pd.DataFrame(all_books)

    # Remove accidental duplicates.
    df = df.drop_duplicates(subset=["product_id"])

    # Keep the required columns in a clear order.
    df = df[
        [
            "product_id",
            "title",
            "category",
            "price_gbp",
            "availability",
            "rating",
        ]
    ]

    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 60)
    print(f"Total books collected: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 60)

    print()
    print(df.groupby("category").size())
    print()
    print(df.head())


if __name__ == "__main__":
    main()
