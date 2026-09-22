SELECT
                title,
                price_gbp,
                price_inr
            FROM products
            WHERE price_inr BETWEEN 3000 AND 5000
            ORDER BY price_inr DESC
            LIMIT 10;
