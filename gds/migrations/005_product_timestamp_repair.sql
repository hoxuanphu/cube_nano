-- Repair rows written by the pre-remediation product lifecycle.
-- The operation is forward-only and idempotent: published products keep their
-- artifact identity while timestamps are made transition-consistent.
UPDATE products
SET
    verified_at_us = COALESCE(verified_at_us, published_at_us, created_at_us),
    published_at_us = COALESCE(published_at_us, verified_at_us, created_at_us),
    retention_until_us = CASE
        WHEN retention_until_us IS NULL
            OR retention_until_us <= COALESCE(published_at_us, verified_at_us, created_at_us)
        THEN COALESCE(published_at_us, verified_at_us, created_at_us) + 2592000000000
        ELSE retention_until_us
    END
WHERE state IN ('VERIFIED', 'PUBLISHED');
