CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY CHECK (version > 0),
    name TEXT NOT NULL UNIQUE,
    checksum_sha256 TEXT NOT NULL CHECK (length(checksum_sha256) = 64),
    applied_at_us INTEGER NOT NULL CHECK (applied_at_us >= 0)
);

CREATE TABLE ground_namespaces (
    ground_instance_id BLOB PRIMARY KEY
        CHECK (typeof(ground_instance_id) = 'blob' AND length(ground_instance_id) = 8),
    gds_installation_epoch BLOB NOT NULL
        CHECK (typeof(gds_installation_epoch) = 'blob' AND length(gds_installation_epoch) = 8),
    next_request_id INTEGER NOT NULL
        CHECK (next_request_id BETWEEN 1 AND 4294967296),
    state TEXT NOT NULL CHECK (state IN ('ACTIVE', 'DRAINING', 'RETIRED')),
    created_at_us INTEGER NOT NULL CHECK (created_at_us >= 0),
    retired_at_us INTEGER CHECK (retired_at_us IS NULL OR retired_at_us >= created_at_us)
);

CREATE UNIQUE INDEX uq_ground_namespaces_active
    ON ground_namespaces(state) WHERE state = 'ACTIVE';

CREATE TABLE gds_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    gds_installation_epoch BLOB NOT NULL UNIQUE
        CHECK (typeof(gds_installation_epoch) = 'blob' AND length(gds_installation_epoch) = 8),
    active_ground_instance_id BLOB NOT NULL
        CHECK (typeof(active_ground_instance_id) = 'blob' AND length(active_ground_instance_id) = 8),
    created_at_us INTEGER NOT NULL CHECK (created_at_us >= 0),
    FOREIGN KEY (active_ground_instance_id)
        REFERENCES ground_namespaces(ground_instance_id)
);

CREATE TABLE spacecraft_instances (
    spacecraft_instance_id BLOB PRIMARY KEY
        CHECK (typeof(spacecraft_instance_id) = 'blob' AND length(spacecraft_instance_id) = 8),
    link_generation BLOB NOT NULL
        CHECK (typeof(link_generation) = 'blob' AND length(link_generation) = 8),
    link_session_id BLOB
        CHECK (link_session_id IS NULL OR
               (typeof(link_session_id) = 'blob' AND length(link_session_id) = 8)),
    state TEXT NOT NULL CHECK (state IN ('ACTIVE', 'RETIRED')),
    first_seen_at_us INTEGER NOT NULL CHECK (first_seen_at_us >= 0),
    last_seen_at_us INTEGER NOT NULL CHECK (last_seen_at_us >= first_seen_at_us),
    rebaseline_reason TEXT
);

CREATE TABLE commands (
    ground_instance_id BLOB NOT NULL
        CHECK (typeof(ground_instance_id) = 'blob' AND length(ground_instance_id) = 8),
    request_id INTEGER NOT NULL CHECK (request_id BETWEEN 0 AND 4294967295),
    target_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(target_spacecraft_instance_id) = 'blob' AND
               length(target_spacecraft_instance_id) = 8),
    gds_installation_epoch BLOB NOT NULL
        CHECK (typeof(gds_installation_epoch) = 'blob' AND length(gds_installation_epoch) = 8),
    principal TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    http_digest BLOB NOT NULL
        CHECK (typeof(http_digest) = 'blob' AND length(http_digest) = 32),
    semantic_body_jcs BLOB NOT NULL CHECK (typeof(semantic_body_jcs) = 'blob'),
    opcode INTEGER NOT NULL CHECK (opcode BETWEEN 0 AND 4294967295),
    mission_arguments BLOB NOT NULL CHECK (typeof(mission_arguments) = 'blob'),
    mission_digest BLOB NOT NULL
        CHECK (typeof(mission_digest) = 'blob' AND length(mission_digest) = 32),
    delivery_mode TEXT NOT NULL CHECK (delivery_mode IN ('immediate', 'next_contact')),
    effective_expires_at_us INTEGER NOT NULL CHECK (effective_expires_at_us >= 0),
    command_state TEXT NOT NULL CHECK (
        command_state IN ('ADMITTED', 'ACKED', 'REJECTED', 'EXECUTED', 'FAILED', 'CANCELED')
    ),
    created_at_us INTEGER NOT NULL CHECK (created_at_us >= 0),
    updated_at_us INTEGER NOT NULL CHECK (updated_at_us >= created_at_us),
    terminal_at_us INTEGER CHECK (terminal_at_us IS NULL OR terminal_at_us >= created_at_us),
    PRIMARY KEY (ground_instance_id, request_id),
    UNIQUE (target_spacecraft_instance_id, ground_instance_id, request_id),
    UNIQUE (gds_installation_epoch, principal, idempotency_key),
    FOREIGN KEY (ground_instance_id) REFERENCES ground_namespaces(ground_instance_id)
);

CREATE INDEX ix_commands_target_created
    ON commands(target_spacecraft_instance_id, created_at_us, ground_instance_id, request_id);
CREATE INDEX ix_commands_state_expiry
    ON commands(command_state, effective_expires_at_us);

CREATE TABLE command_outbox (
    ground_instance_id BLOB NOT NULL
        CHECK (typeof(ground_instance_id) = 'blob' AND length(ground_instance_id) = 8),
    request_id INTEGER NOT NULL CHECK (request_id BETWEEN 0 AND 4294967295),
    target_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(target_spacecraft_instance_id) = 'blob' AND
               length(target_spacecraft_instance_id) = 8),
    state TEXT NOT NULL CHECK (
        state IN ('HELD_NO_CONTACT', 'OUTBOX_PENDING', 'DISPATCHING', 'SENT',
                  'ACKED', 'EXPIRED', 'DELIVERY_FAILED', 'CANCELED')
    ),
    available_at_us INTEGER NOT NULL CHECK (available_at_us >= 0),
    expires_at_us INTEGER NOT NULL CHECK (expires_at_us >= 0),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count BETWEEN 0 AND 4294967295),
    lease_owner TEXT,
    lease_expires_at_us INTEGER CHECK (lease_expires_at_us IS NULL OR lease_expires_at_us >= 0),
    last_error_code TEXT,
    created_at_us INTEGER NOT NULL CHECK (created_at_us >= 0),
    updated_at_us INTEGER NOT NULL CHECK (updated_at_us >= created_at_us),
    PRIMARY KEY (ground_instance_id, request_id),
    FOREIGN KEY (target_spacecraft_instance_id, ground_instance_id, request_id)
        REFERENCES commands(target_spacecraft_instance_id, ground_instance_id, request_id)
        ON DELETE CASCADE
);

CREATE INDEX ix_command_outbox_dispatch
    ON command_outbox(state, available_at_us, expires_at_us);
CREATE INDEX ix_command_outbox_target_state
    ON command_outbox(target_spacecraft_instance_id, state);

CREATE TABLE command_attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ground_instance_id BLOB NOT NULL
        CHECK (typeof(ground_instance_id) = 'blob' AND length(ground_instance_id) = 8),
    request_id INTEGER NOT NULL CHECK (request_id BETWEEN 0 AND 4294967295),
    target_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(target_spacecraft_instance_id) = 'blob' AND
               length(target_spacecraft_instance_id) = 8),
    attempt_number INTEGER NOT NULL CHECK (attempt_number BETWEEN 1 AND 4294967295),
    link_generation BLOB NOT NULL
        CHECK (typeof(link_generation) = 'blob' AND length(link_generation) = 8),
    link_session_id BLOB NOT NULL
        CHECK (typeof(link_session_id) = 'blob' AND length(link_session_id) = 8),
    apid INTEGER NOT NULL CHECK (apid BETWEEN 0 AND 2047),
    packet_sequence INTEGER NOT NULL CHECK (packet_sequence BETWEEN 0 AND 16383),
    frame_sequence INTEGER NOT NULL CHECK (frame_sequence BETWEEN 0 AND 255),
    encoded_tc BLOB NOT NULL CHECK (typeof(encoded_tc) = 'blob'),
    encoded_tc_sha256 BLOB NOT NULL
        CHECK (typeof(encoded_tc_sha256) = 'blob' AND length(encoded_tc_sha256) = 32),
    send_result TEXT NOT NULL,
    sent_at_us INTEGER,
    created_at_us INTEGER NOT NULL CHECK (created_at_us >= 0),
    UNIQUE (ground_instance_id, request_id, attempt_number),
    FOREIGN KEY (target_spacecraft_instance_id, ground_instance_id, request_id)
        REFERENCES commands(target_spacecraft_instance_id, ground_instance_id, request_id)
        ON DELETE CASCADE
);

CREATE TABLE system_state (
    state_key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at_us INTEGER NOT NULL CHECK (updated_at_us >= 0)
);

CREATE TABLE catalog_snapshots (
    source_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(source_spacecraft_instance_id) = 'blob' AND
               length(source_spacecraft_instance_id) = 8),
    catalog_epoch INTEGER NOT NULL CHECK (catalog_epoch BETWEEN 0 AND 4294967295),
    catalog_revision INTEGER NOT NULL CHECK (catalog_revision BETWEEN 0 AND 4294967295),
    snapshot_sha256 BLOB NOT NULL
        CHECK (typeof(snapshot_sha256) = 'blob' AND length(snapshot_sha256) = 32),
    state TEXT NOT NULL CHECK (state IN ('STAGING', 'VERIFIED', 'RETIRED', 'INVALID')),
    manifest_json TEXT NOT NULL,
    synced_at_us INTEGER NOT NULL CHECK (synced_at_us >= 0),
    PRIMARY KEY (source_spacecraft_instance_id, catalog_epoch, catalog_revision)
);

CREATE TABLE scenes (
    source_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(source_spacecraft_instance_id) = 'blob' AND
               length(source_spacecraft_instance_id) = 8),
    catalog_epoch INTEGER NOT NULL CHECK (catalog_epoch BETWEEN 0 AND 4294967295),
    scene_id INTEGER NOT NULL CHECK (scene_id BETWEEN 0 AND 4294967295),
    scene_revision INTEGER NOT NULL CHECK (scene_revision BETWEEN 0 AND 4294967295),
    catalog_revision INTEGER NOT NULL CHECK (catalog_revision BETWEEN 0 AND 4294967295),
    source_sha256 BLOB NOT NULL
        CHECK (typeof(source_sha256) = 'blob' AND length(source_sha256) = 32),
    sidecar_sha256 BLOB NOT NULL
        CHECK (typeof(sidecar_sha256) = 'blob' AND length(sidecar_sha256) = 32),
    metadata_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('ACTIVE', 'STALE', 'INVALID', 'UNSUPPORTED')),
    active_preview_spacecraft_instance_id BLOB
        CHECK (active_preview_spacecraft_instance_id IS NULL OR
               (typeof(active_preview_spacecraft_instance_id) = 'blob' AND
                length(active_preview_spacecraft_instance_id) = 8)),
    active_preview_origin_boot_id INTEGER
        CHECK (active_preview_origin_boot_id IS NULL OR
               active_preview_origin_boot_id BETWEEN 0 AND 4294967295),
    active_preview_product_id INTEGER
        CHECK (active_preview_product_id IS NULL OR
               active_preview_product_id BETWEEN 0 AND 4294967295),
    PRIMARY KEY (source_spacecraft_instance_id, catalog_epoch, scene_id, scene_revision),
    FOREIGN KEY (source_spacecraft_instance_id, catalog_epoch, catalog_revision)
        REFERENCES catalog_snapshots(source_spacecraft_instance_id, catalog_epoch, catalog_revision)
);

CREATE TABLE telemetry_samples (
    source_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(source_spacecraft_instance_id) = 'blob' AND
               length(source_spacecraft_instance_id) = 8),
    source_boot_id INTEGER NOT NULL CHECK (source_boot_id BETWEEN 0 AND 4294967295),
    simulation_run_id BLOB NOT NULL
        CHECK (typeof(simulation_run_id) = 'blob' AND length(simulation_run_id) = 8),
    direction TEXT NOT NULL CHECK (direction IN ('UPLINK', 'DOWNLINK')),
    link_session_id BLOB NOT NULL
        CHECK (typeof(link_session_id) = 'blob' AND length(link_session_id) = 8),
    link_frame_id BLOB NOT NULL
        CHECK (typeof(link_frame_id) = 'blob' AND length(link_frame_id) = 8),
    copy_index INTEGER NOT NULL CHECK (copy_index BETWEEN 0 AND 4294967295),
    sample_ordinal INTEGER NOT NULL CHECK (sample_ordinal BETWEEN 0 AND 4294967295),
    apid INTEGER NOT NULL CHECK (apid BETWEEN 0 AND 2047),
    channel_id INTEGER NOT NULL CHECK (channel_id BETWEEN 0 AND 4294967295),
    satellite_time_us INTEGER,
    received_at_us INTEGER NOT NULL CHECK (received_at_us >= 0),
    raw_value BLOB NOT NULL CHECK (typeof(raw_value) = 'blob'),
    decoded_value_json TEXT NOT NULL,
    payload_sha256 BLOB NOT NULL
        CHECK (typeof(payload_sha256) = 'blob' AND length(payload_sha256) = 32),
    PRIMARY KEY (source_spacecraft_instance_id, simulation_run_id, direction,
                 link_frame_id, copy_index, sample_ordinal)
);

CREATE INDEX ix_telemetry_channel_time
    ON telemetry_samples(source_spacecraft_instance_id, channel_id, received_at_us);
CREATE INDEX ix_telemetry_packet_time
    ON telemetry_samples(source_spacecraft_instance_id, direction, received_at_us, apid);

CREATE TABLE telemetry_rollups (
    source_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(source_spacecraft_instance_id) = 'blob' AND
               length(source_spacecraft_instance_id) = 8),
    channel_id INTEGER NOT NULL CHECK (channel_id BETWEEN 0 AND 4294967295),
    bucket_start_us INTEGER NOT NULL CHECK (bucket_start_us >= 0),
    sample_count INTEGER NOT NULL CHECK (sample_count > 0),
    min_value REAL,
    max_value REAL,
    mean_value REAL,
    last_value_json TEXT NOT NULL,
    source_retention_revision INTEGER NOT NULL CHECK (source_retention_revision >= 0),
    PRIMARY KEY (source_spacecraft_instance_id, channel_id, bucket_start_us)
);

CREATE TABLE events (
    event_id BLOB PRIMARY KEY
        CHECK (typeof(event_id) = 'blob' AND length(event_id) = 8),
    source_spacecraft_instance_id BLOB
        CHECK (source_spacecraft_instance_id IS NULL OR
               (typeof(source_spacecraft_instance_id) = 'blob' AND
                length(source_spacecraft_instance_id) = 8)),
    target_spacecraft_instance_id BLOB
        CHECK (target_spacecraft_instance_id IS NULL OR
               (typeof(target_spacecraft_instance_id) = 'blob' AND
                length(target_spacecraft_instance_id) = 8)),
    source_boot_id INTEGER CHECK (source_boot_id IS NULL OR
                                  source_boot_id BETWEEN 0 AND 4294967295),
    ground_instance_id BLOB
        CHECK (ground_instance_id IS NULL OR
               (typeof(ground_instance_id) = 'blob' AND length(ground_instance_id) = 8)),
    request_id INTEGER CHECK (request_id IS NULL OR request_id BETWEEN 0 AND 4294967295),
    severity TEXT NOT NULL,
    event_name TEXT NOT NULL,
    dictionary_version TEXT,
    message_json TEXT NOT NULL,
    server_time_us INTEGER NOT NULL CHECK (server_time_us >= 0)
);

CREATE INDEX ix_events_server_time ON events(server_time_us, event_id);
CREATE INDEX ix_events_request ON events(ground_instance_id, request_id, event_id);

CREATE TABLE link_frames (
    simulation_run_id BLOB NOT NULL
        CHECK (typeof(simulation_run_id) = 'blob' AND length(simulation_run_id) = 8),
    direction TEXT NOT NULL CHECK (direction IN ('UPLINK', 'DOWNLINK')),
    link_frame_id BLOB NOT NULL
        CHECK (typeof(link_frame_id) = 'blob' AND length(link_frame_id) = 8),
    copy_index INTEGER NOT NULL CHECK (copy_index BETWEEN 0 AND 4294967295),
    source_spacecraft_instance_id BLOB
        CHECK (source_spacecraft_instance_id IS NULL OR
               (typeof(source_spacecraft_instance_id) = 'blob' AND
                length(source_spacecraft_instance_id) = 8)),
    target_spacecraft_instance_id BLOB
        CHECK (target_spacecraft_instance_id IS NULL OR
               (typeof(target_spacecraft_instance_id) = 'blob' AND
                length(target_spacecraft_instance_id) = 8)),
    link_session_id BLOB NOT NULL
        CHECK (typeof(link_session_id) = 'blob' AND length(link_session_id) = 8),
    apid INTEGER CHECK (apid IS NULL OR apid BETWEEN 0 AND 2047),
    vcid INTEGER CHECK (vcid IS NULL OR vcid BETWEEN 0 AND 63),
    frame_sequence INTEGER,
    crc_valid INTEGER NOT NULL CHECK (crc_valid IN (0, 1)),
    fault_json TEXT NOT NULL,
    segment_path TEXT NOT NULL,
    segment_offset INTEGER NOT NULL CHECK (segment_offset >= 0),
    segment_length INTEGER NOT NULL CHECK (segment_length > 0),
    received_at_us INTEGER NOT NULL CHECK (received_at_us >= 0),
    PRIMARY KEY (simulation_run_id, direction, link_frame_id, copy_index)
);

CREATE TABLE jobs (
    source_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(source_spacecraft_instance_id) = 'blob' AND
               length(source_spacecraft_instance_id) = 8),
    ground_instance_id BLOB NOT NULL
        CHECK (typeof(ground_instance_id) = 'blob' AND length(ground_instance_id) = 8),
    request_id INTEGER NOT NULL CHECK (request_id BETWEEN 0 AND 4294967295),
    state TEXT NOT NULL,
    roi_json TEXT,
    config_snapshot_json TEXT NOT NULL,
    model_identity_json TEXT NOT NULL,
    progress_bp INTEGER NOT NULL DEFAULT 0 CHECK (progress_bp BETWEEN 0 AND 10000),
    result_json TEXT,
    error_code TEXT,
    updated_at_us INTEGER NOT NULL CHECK (updated_at_us >= 0),
    PRIMARY KEY (source_spacecraft_instance_id, ground_instance_id, request_id)
);

CREATE TABLE products (
    spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(spacecraft_instance_id) = 'blob' AND length(spacecraft_instance_id) = 8),
    origin_boot_id INTEGER NOT NULL CHECK (origin_boot_id BETWEEN 0 AND 4294967295),
    product_id INTEGER NOT NULL CHECK (product_id BETWEEN 0 AND 4294967295),
    origin_ground_instance_id BLOB NOT NULL
        CHECK (typeof(origin_ground_instance_id) = 'blob' AND
               length(origin_ground_instance_id) = 8),
    origin_request_id INTEGER NOT NULL CHECK (origin_request_id BETWEEN 0 AND 4294967295),
    product_type TEXT NOT NULL,
    state TEXT NOT NULL,
    bundle_size INTEGER CHECK (bundle_size IS NULL OR bundle_size >= 0),
    bundle_sha256 BLOB CHECK (bundle_sha256 IS NULL OR
                              (typeof(bundle_sha256) = 'blob' AND length(bundle_sha256) = 32)),
    manifest_json TEXT,
    local_path TEXT,
    created_at_us INTEGER NOT NULL CHECK (created_at_us >= 0),
    PRIMARY KEY (spacecraft_instance_id, origin_boot_id, product_id)
);

CREATE TABLE product_transfers (
    spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(spacecraft_instance_id) = 'blob' AND length(spacecraft_instance_id) = 8),
    transfer_id INTEGER NOT NULL CHECK (transfer_id BETWEEN 0 AND 4294967295),
    origin_boot_id INTEGER NOT NULL CHECK (origin_boot_id BETWEEN 0 AND 4294967295),
    product_id INTEGER NOT NULL CHECK (product_id BETWEEN 0 AND 4294967295),
    state TEXT NOT NULL,
    attempt INTEGER NOT NULL CHECK (attempt > 0),
    bytes_received INTEGER NOT NULL DEFAULT 0 CHECK (bytes_received >= 0),
    expected_size INTEGER CHECK (expected_size IS NULL OR expected_size >= 0),
    checksum_sha256 BLOB CHECK (checksum_sha256 IS NULL OR
                               (typeof(checksum_sha256) = 'blob' AND length(checksum_sha256) = 32)),
    updated_at_us INTEGER NOT NULL CHECK (updated_at_us >= 0),
    PRIMARY KEY (spacecraft_instance_id, transfer_id),
    FOREIGN KEY (spacecraft_instance_id, origin_boot_id, product_id)
        REFERENCES products(spacecraft_instance_id, origin_boot_id, product_id)
);

CREATE TABLE simulation_runs (
    simulation_run_id BLOB PRIMARY KEY
        CHECK (typeof(simulation_run_id) = 'blob' AND length(simulation_run_id) = 8),
    release_identity TEXT NOT NULL,
    profile_sha256 BLOB NOT NULL
        CHECK (typeof(profile_sha256) = 'blob' AND length(profile_sha256) = 32),
    state TEXT NOT NULL CHECK (
        state IN ('OPEN', 'FINAL', 'INCOMPLETE_CRASH', 'INCOMPLETE_STORAGE')
    ),
    replay_state TEXT NOT NULL CHECK (replay_state IN ('PRESENT', 'PINNED', 'EVICTED')),
    artifact_size INTEGER CHECK (artifact_size IS NULL OR artifact_size >= 0),
    artifact_sha256 BLOB CHECK (artifact_sha256 IS NULL OR
                               (typeof(artifact_sha256) = 'blob' AND length(artifact_sha256) = 32)),
    created_at_us INTEGER NOT NULL CHECK (created_at_us >= 0),
    finalized_at_us INTEGER
);

CREATE TABLE replay_segments (
    simulation_run_id BLOB NOT NULL
        CHECK (typeof(simulation_run_id) = 'blob' AND length(simulation_run_id) = 8),
    segment_ordinal INTEGER NOT NULL CHECK (segment_ordinal >= 0),
    segment_path TEXT NOT NULL,
    segment_size INTEGER NOT NULL CHECK (segment_size >= 0),
    segment_sha256 BLOB NOT NULL
        CHECK (typeof(segment_sha256) = 'blob' AND length(segment_sha256) = 32),
    PRIMARY KEY (simulation_run_id, segment_ordinal),
    FOREIGN KEY (simulation_run_id) REFERENCES simulation_runs(simulation_run_id)
        ON DELETE CASCADE
);

CREATE TABLE http_idempotency_retired (
    gds_installation_epoch BLOB NOT NULL
        CHECK (typeof(gds_installation_epoch) = 'blob' AND length(gds_installation_epoch) = 8),
    principal TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    http_digest BLOB NOT NULL
        CHECK (typeof(http_digest) = 'blob' AND length(http_digest) = 32),
    original_ground_instance_id BLOB NOT NULL
        CHECK (typeof(original_ground_instance_id) = 'blob' AND
               length(original_ground_instance_id) = 8),
    original_request_id INTEGER NOT NULL
        CHECK (original_request_id BETWEEN 0 AND 4294967295),
    retained_until_us INTEGER NOT NULL CHECK (retained_until_us >= 0),
    PRIMARY KEY (gds_installation_epoch, principal, idempotency_key)
);

CREATE TABLE product_tombstones (
    spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(spacecraft_instance_id) = 'blob' AND length(spacecraft_instance_id) = 8),
    origin_boot_id INTEGER NOT NULL CHECK (origin_boot_id BETWEEN 0 AND 4294967295),
    product_id INTEGER NOT NULL CHECK (product_id BETWEEN 0 AND 4294967295),
    eviction_reason TEXT NOT NULL,
    bundle_sha256 BLOB CHECK (bundle_sha256 IS NULL OR
                              (typeof(bundle_sha256) = 'blob' AND length(bundle_sha256) = 32)),
    retained_until_us INTEGER NOT NULL CHECK (retained_until_us >= 0),
    PRIMARY KEY (spacecraft_instance_id, origin_boot_id, product_id)
);

CREATE TABLE audit_log (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    principal TEXT NOT NULL,
    action TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_identity_json TEXT NOT NULL,
    old_value_json TEXT,
    new_value_json TEXT,
    created_at_us INTEGER NOT NULL CHECK (created_at_us >= 0)
);

CREATE INDEX ix_audit_created ON audit_log(created_at_us, audit_id);
