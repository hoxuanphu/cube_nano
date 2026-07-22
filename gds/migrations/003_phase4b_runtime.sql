ALTER TABLE catalog_snapshots ADD COLUMN source_boot_id INTEGER;
ALTER TABLE catalog_snapshots ADD COLUMN source_link_session_id BLOB;
ALTER TABLE catalog_snapshots ADD COLUMN is_active INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1));
ALTER TABLE catalog_snapshots ADD COLUMN verified_at_us INTEGER;
ALTER TABLE catalog_snapshots ADD COLUMN retired_at_us INTEGER;

ALTER TABLE scenes ADD COLUMN source_stat_json TEXT;
ALTER TABLE scenes ADD COLUMN sidecar_stat_json TEXT;
ALTER TABLE scenes ADD COLUMN invalid_reason TEXT;
ALTER TABLE scenes ADD COLUMN ingested_at_us INTEGER;
ALTER TABLE scenes ADD COLUMN active_preview_generation INTEGER NOT NULL DEFAULT 0;

ALTER TABLE products ADD COLUMN manifest_sha256 BLOB;
ALTER TABLE products ADD COLUMN verified_at_us INTEGER;
ALTER TABLE products ADD COLUMN published_at_us INTEGER;
ALTER TABLE products ADD COLUMN evicted_at_us INTEGER;
ALTER TABLE products ADD COLUMN retention_until_us INTEGER;
ALTER TABLE products ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0, 1));
ALTER TABLE products ADD COLUMN eviction_reason TEXT;
ALTER TABLE products ADD COLUMN file_checksum INTEGER;
ALTER TABLE products ADD COLUMN origin_request_key_json TEXT;

ALTER TABLE product_transfers ADD COLUMN link_session_id BLOB;
ALTER TABLE product_transfers ADD COLUMN file_epoch_id BLOB;
ALTER TABLE product_transfers ADD COLUMN expected_bundle_sha256 BLOB;
ALTER TABLE product_transfers ADD COLUMN expected_file_checksum INTEGER;
ALTER TABLE product_transfers ADD COLUMN part_path TEXT;
ALTER TABLE product_transfers ADD COLUMN staging_path TEXT;
ALTER TABLE product_transfers ADD COLUMN last_activity_us INTEGER;
ALTER TABLE product_transfers ADD COLUMN terminal_reason TEXT;
ALTER TABLE product_transfers ADD COLUMN verified_at_us INTEGER;

CREATE TABLE scene_packages (
    source_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(source_spacecraft_instance_id) = 'blob' AND length(source_spacecraft_instance_id) = 8),
    catalog_epoch INTEGER NOT NULL CHECK (catalog_epoch BETWEEN 0 AND 4294967295),
    scene_id INTEGER NOT NULL CHECK (scene_id BETWEEN 0 AND 4294967295),
    scene_revision INTEGER NOT NULL CHECK (scene_revision BETWEEN 0 AND 4294967295),
    package_sha256 BLOB NOT NULL
        CHECK (typeof(package_sha256) = 'blob' AND length(package_sha256) = 32),
    root_path TEXT NOT NULL,
    source_path TEXT NOT NULL,
    sidecar_path TEXT NOT NULL,
    source_stat_json TEXT NOT NULL,
    sidecar_stat_json TEXT NOT NULL,
    read_only INTEGER NOT NULL CHECK (read_only IN (0, 1)),
    state TEXT NOT NULL CHECK (state IN ('STAGED', 'PUBLISHED', 'INVALID', 'RETIRED')),
    created_at_us INTEGER NOT NULL CHECK (created_at_us >= 0),
    PRIMARY KEY (source_spacecraft_instance_id, catalog_epoch, scene_id, scene_revision),
    FOREIGN KEY (source_spacecraft_instance_id, catalog_epoch, scene_id, scene_revision)
        REFERENCES scenes(source_spacecraft_instance_id, catalog_epoch, scene_id, scene_revision)
);

CREATE TABLE file_reassemblies (
    source_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(source_spacecraft_instance_id) = 'blob' AND length(source_spacecraft_instance_id) = 8),
    link_session_id BLOB NOT NULL
        CHECK (typeof(link_session_id) = 'blob' AND length(link_session_id) = 8),
    file_epoch_id BLOB NOT NULL
        CHECK (typeof(file_epoch_id) = 'blob' AND length(file_epoch_id) = 8),
    transfer_id INTEGER NOT NULL CHECK (transfer_id BETWEEN 0 AND 4294967295),
    product_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(product_spacecraft_instance_id) = 'blob' AND length(product_spacecraft_instance_id) = 8),
    origin_boot_id INTEGER NOT NULL CHECK (origin_boot_id BETWEEN 0 AND 4294967295),
    product_id INTEGER NOT NULL CHECK (product_id BETWEEN 0 AND 4294967295),
    expected_size INTEGER NOT NULL CHECK (expected_size >= 0),
    expected_file_checksum INTEGER NOT NULL CHECK (expected_file_checksum BETWEEN 0 AND 4294967295),
    expected_bundle_sha256 BLOB NOT NULL
        CHECK (typeof(expected_bundle_sha256) = 'blob' AND length(expected_bundle_sha256) = 32),
    source_name TEXT NOT NULL,
    destination_name TEXT NOT NULL,
    part_path TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('RECEIVING', 'VERIFIED', 'INCOMPLETE', 'CHECKSUM_FAILED', 'CANCELED', 'CONFLICT')),
    ranges_json TEXT NOT NULL,
    sequence_map_json TEXT NOT NULL,
    start_payload BLOB NOT NULL,
    terminal_reason TEXT,
    created_at_us INTEGER NOT NULL CHECK (created_at_us >= 0),
    updated_at_us INTEGER NOT NULL CHECK (updated_at_us >= created_at_us),
    verified_at_us INTEGER,
    PRIMARY KEY (source_spacecraft_instance_id, link_session_id, file_epoch_id),
    UNIQUE (source_spacecraft_instance_id, transfer_id),
    FOREIGN KEY (product_spacecraft_instance_id, origin_boot_id, product_id)
        REFERENCES products(spacecraft_instance_id, origin_boot_id, product_id)
);

CREATE TABLE product_artifacts (
    spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(spacecraft_instance_id) = 'blob' AND length(spacecraft_instance_id) = 8),
    origin_boot_id INTEGER NOT NULL CHECK (origin_boot_id BETWEEN 0 AND 4294967295),
    product_id INTEGER NOT NULL CHECK (product_id BETWEEN 0 AND 4294967295),
    artifact_path TEXT NOT NULL,
    artifact_size INTEGER NOT NULL CHECK (artifact_size >= 0),
    artifact_sha256 BLOB NOT NULL
        CHECK (typeof(artifact_sha256) = 'blob' AND length(artifact_sha256) = 32),
    state TEXT NOT NULL CHECK (state IN ('VERIFIED', 'EVICTED')),
    PRIMARY KEY (spacecraft_instance_id, origin_boot_id, product_id, artifact_path),
    FOREIGN KEY (spacecraft_instance_id, origin_boot_id, product_id)
        REFERENCES products(spacecraft_instance_id, origin_boot_id, product_id)
        ON DELETE CASCADE
);

CREATE INDEX ix_catalog_active
    ON catalog_snapshots(source_spacecraft_instance_id, is_active, catalog_epoch, catalog_revision);
CREATE UNIQUE INDEX uq_catalog_one_active
    ON catalog_snapshots(source_spacecraft_instance_id)
    WHERE is_active = 1;
CREATE INDEX ix_scenes_state
    ON scenes(source_spacecraft_instance_id, state, catalog_epoch, scene_id, scene_revision);
CREATE INDEX ix_products_retention
    ON products(spacecraft_instance_id, state, pinned, retention_until_us);
CREATE INDEX ix_file_reassemblies_state
    ON file_reassemblies(source_spacecraft_instance_id, state, updated_at_us);
