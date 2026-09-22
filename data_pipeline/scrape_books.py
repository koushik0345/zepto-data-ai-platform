import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


OUTPUT_FILE = Path(__file__).parent / "books_raw.csv"

CATEGORIES = {
    "Mystery": "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
    "Historical Fiction": "https://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html",
    "Sequential Art": "https://books.toscrape.com/catalogue/category/books/sequential-art_5/index.html",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36"
}


def scrape_category(category_name, category_url, target_books=20):
    """Scrape raw, unparsed book fields from one category.

    Every field is stored exactly as listed on the site (price with its
    currency symbol, star rating as a word, availability as text).
    All type conversion happens later in data_cleaning.py.
    """
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

        # The site serves UTF-8 without declaring it, so set it explicitly
        # to keep the "£" symbol intact.
        response.encoding = "utf-8"

        soup = BeautifulSoup(response.text, "html.parser")

        for article in soup.select("article.product_pod"):
            if len(books) >= target_books:
                break

            title_tag = article.select_one("h3 a")
            price_tag = article.select_one(".price_color")
            availability_tag = article.select_one(".availability")
            rating_tag = article.select_one("p.star-rating")

            # The star rating is stored as a CSS class, e.g.
            # <p class="star-rating Three">.
            rating_classes = rating_tag.get("class", []) if rating_tag else []
            star_rating = next(
                (item for item in rating_classes if item != "star-rating"),
                None,
            )

            # Relative links look like ../../../sharp-objects_997/index.html;
            # the folder name is a stable unique product id.
            product_url = urljoin(next_url, title_tag.get("href", ""))
            product_id = product_url.rstrip("/").split("/")[-2]

            books.append(
                {
                    "product_id": product_id,
                    "title": title_tag.get("title", "").strip(),
                    "price": price_tag.get_text(strip=True) if price_tag else None,
                    "star_rating": star_rating,
                    "availability": (
                        availability_tag.get_text(" ", strip=True)
                        if availability_tag
                        else None
                    ),
                    "category": category_name,
                }
            )

        next_link = soup.select_one("li.next a")

        if next_link and len(books) < target_books:
            next_url = urljoin(next_url, next_link.get("href"))
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
