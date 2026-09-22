SELECT
    c.category_name,
    COUNT(p.product_id) AS product_count
FROM categories c
JOIN products p
    ON c.category_id = p.category_id
GROUP BY c.category_name
ORDER BY product_count DESC, c.category_name
LIMIT 10;

