SELECT
                title,
                rating,
                price_inr
            FROM products
            WHERE rating IN (4, 5)
            ORDER BY rating DESC, price_inr DESC
            LIMIT 10;
