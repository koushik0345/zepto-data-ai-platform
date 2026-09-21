# Module 1 — Data Pipeline

## Objective

This module demonstrates a complete data pipeline using Books to Scrape.

The pipeline performs:

1. Web scraping
2. Data cleaning
3. GBP to INR currency conversion
4. SQLite database creation
5. SQL analysis
6. Equivalent pandas analysis

## Source

Books to Scrape:

https://books.toscrape.com/

## Dataset

The scraper collects 60 books from three categories:

- Mystery — 20 books
- Historical Fiction — 20 books
- Sequential Art — 20 books

## Fields

The scraped dataset contains:

- `product_id`
- `title`
- `category`
- `price_gbp`
- `availability`
- `rating`

The cleaned dataset additionally contains:

- `price_inr`

## Currency Conversion

The assignment specifies:

```text
1 GBP = 105.50 INR
