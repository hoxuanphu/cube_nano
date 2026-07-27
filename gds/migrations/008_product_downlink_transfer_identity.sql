-- APID 3 must never claim a retry merely because its transfer_id column has
-- not been populated yet.  A flight transfer identity belongs to one durable
-- downlink admission for a spacecraft instance.
CREATE UNIQUE INDEX ux_product_downlink_transfer_identity
    ON product_downlink_ledger(product_spacecraft_instance_id, transfer_id)
    WHERE transfer_id IS NOT NULL;

-- File data can arrive before the APID 2 receipt that names its downlink
-- RequestKey.  Retain only an exact ProductRef + transfer identity until the
-- receipt assigns that transfer; never infer ownership from a NULL retry row.
CREATE TABLE product_downlink_pending_files (
    product_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(product_spacecraft_instance_id) = 'blob' AND length(product_spacecraft_instance_id) = 8),
    origin_boot_id INTEGER NOT NULL CHECK (origin_boot_id BETWEEN 0 AND 4294967295),
    product_id INTEGER NOT NULL CHECK (product_id BETWEEN 0 AND 4294967295),
    transfer_id INTEGER NOT NULL CHECK (transfer_id BETWEEN 0 AND 4294967295),
    transfer_state TEXT NOT NULL CHECK (
        transfer_state IN ('RECEIVING', 'VERIFIED', 'FAILED', 'CANCELED')
    ),
    first_observed_at_us INTEGER NOT NULL CHECK (first_observed_at_us >= 0),
    updated_at_us INTEGER NOT NULL CHECK (updated_at_us >= first_observed_at_us),
    expires_at_us INTEGER NOT NULL CHECK (expires_at_us >= updated_at_us),
    PRIMARY KEY (
        product_spacecraft_instance_id, origin_boot_id, product_id, transfer_id
    )
);

CREATE INDEX ix_product_downlink_pending_expiry
    ON product_downlink_pending_files(expires_at_us);
