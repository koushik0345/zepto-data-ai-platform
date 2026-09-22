SELECT DISTINCT rating
FROM products
WHERE category_id IN (1, 3)
  AND in_stock = 1
ORDER BY rating;

