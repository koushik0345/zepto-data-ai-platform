SELECT
                c.category_name,
                ROUND(AVG(p.price_inr), 2) AS average_price_inr
            FROM categories c
            JOIN products p
                ON c.category_id = p.category_id
            GROUP BY c.category_name
            ORDER BY average_price_inr DESC;
