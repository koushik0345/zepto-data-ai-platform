SELECT DISTINCT category_id
            FROM products
            WHERE category_id IN (1, 2, 3)
            ORDER BY category_id;
