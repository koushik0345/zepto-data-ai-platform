SELECT
                p.product_id,
                p.title,
                c.category_name,
                p.price_gbp,
                p.price_inr,
                p.availability,
                p.rating
            FROM products p
            INNER JOIN categories c
                ON p.category_id = c.category_id
            ORDER BY p.price_inr DESC
            LIMIT 10;
