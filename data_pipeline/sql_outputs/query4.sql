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

