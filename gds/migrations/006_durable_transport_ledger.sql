-- Persist the exact TC wire contract used by each attempt.  Existing rows
-- retain a clearly marked legacy profile rather than being silently relabeled.
ALTER TABLE command_attempts ADD COLUMN tc_profile_id TEXT NOT NULL DEFAULT 'legacy';
ALTER TABLE command_attempts ADD COLUMN tc_profile_sha256 TEXT NOT NULL DEFAULT 'legacy';
ALTER TABLE command_attempts ADD COLUMN space_packet_type INTEGER NOT NULL DEFAULT 1
    CHECK (space_packet_type IN (0, 1));
ALTER TABLE command_attempts ADD COLUMN space_packet_sequence_flags INTEGER NOT NULL DEFAULT 3
    CHECK (space_packet_sequence_flags BETWEEN 0 AND 3);

-- A product transfer has its own GDS RequestKey.  The table is deliberately
-- independent from product publication because a transfer can be admitted
-- before APID 3 data arrives at the ground station.
CREATE TABLE product_downlink_ledger (
    downlink_ground_instance_id BLOB NOT NULL
        CHECK (typeof(downlink_ground_instance_id) = 'blob' AND length(downlink_ground_instance_id) = 8),
    downlink_request_id INTEGER NOT NULL CHECK (downlink_request_id BETWEEN 0 AND 4294967295),
    origin_ground_instance_id BLOB NOT NULL
        CHECK (typeof(origin_ground_instance_id) = 'blob' AND length(origin_ground_instance_id) = 8),
    origin_request_id INTEGER NOT NULL CHECK (origin_request_id BETWEEN 0 AND 4294967295),
    product_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(product_spacecraft_instance_id) = 'blob' AND length(product_spacecraft_instance_id) = 8),
    origin_boot_id INTEGER NOT NULL CHECK (origin_boot_id BETWEEN 0 AND 4294967295),
    product_id INTEGER NOT NULL CHECK (product_id BETWEEN 0 AND 4294967295),
    admission_ordinal INTEGER NOT NULL CHECK (admission_ordinal >= 1),
    transfer_id INTEGER CHECK (transfer_id BETWEEN 0 AND 4294967295),
    transfer_state TEXT NOT NULL CHECK (
        transfer_state IN ('ADMITTED', 'DISPATCHED', 'RECEIVING', 'VERIFIED', 'FAILED', 'CANCELED')
    ),
    created_at_us INTEGER NOT NULL CHECK (created_at_us >= 0),
    updated_at_us INTEGER NOT NULL CHECK (updated_at_us >= created_at_us),
    PRIMARY KEY (downlink_ground_instance_id, downlink_request_id),
    UNIQUE (
        origin_ground_instance_id, origin_request_id,
        product_spacecraft_instance_id, origin_boot_id, product_id, admission_ordinal
    ),
    FOREIGN KEY (downlink_ground_instance_id, downlink_request_id)
        REFERENCES commands(ground_instance_id, request_id)
        ON DELETE CASCADE
);

CREATE INDEX ix_product_downlink_transfer
    ON product_downlink_ledger(product_spacecraft_instance_id, transfer_id, transfer_state);

-- The active link identity is a durable watermark.  A lower generation or a
-- replaced session at the same generation is stale and must never drive state.
CREATE TABLE tm_source_generations (
    source_spacecraft_instance_id BLOB PRIMARY KEY
        CHECK (typeof(source_spacecraft_instance_id) = 'blob' AND length(source_spacecraft_instance_id) = 8),
    active_link_generation BLOB NOT NULL
        CHECK (typeof(active_link_generation) = 'blob' AND length(active_link_generation) = 8),
    active_link_session_id BLOB NOT NULL
        CHECK (typeof(active_link_session_id) = 'blob' AND length(active_link_session_id) = 8),
    updated_at_us INTEGER NOT NULL CHECK (updated_at_us >= 0)
);

-- APID 3 is scoped by the file epoch so every independently admitted product
-- transfer has a fresh FilePacket sequence baseline while TC/TM packet counts
-- remain durable across restart.
CREATE TABLE tm_counter_states (
    source_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(source_spacecraft_instance_id) = 'blob' AND length(source_spacecraft_instance_id) = 8),
    link_generation BLOB NOT NULL
        CHECK (typeof(link_generation) = 'blob' AND length(link_generation) = 8),
    link_session_id BLOB NOT NULL
        CHECK (typeof(link_session_id) = 'blob' AND length(link_session_id) = 8),
    sender_boot_id INTEGER NOT NULL CHECK (sender_boot_id BETWEEN 0 AND 4294967295),
    virtual_channel_id INTEGER NOT NULL CHECK (virtual_channel_id BETWEEN 0 AND 7),
    apid INTEGER NOT NULL CHECK (apid BETWEEN 0 AND 2047),
    file_epoch_id BLOB NOT NULL
        CHECK (typeof(file_epoch_id) = 'blob' AND length(file_epoch_id) = 8),
    last_master_channel_count INTEGER NOT NULL CHECK (last_master_channel_count BETWEEN 0 AND 255),
    last_virtual_channel_count INTEGER NOT NULL CHECK (last_virtual_channel_count BETWEEN 0 AND 255),
    last_packet_sequence INTEGER NOT NULL CHECK (last_packet_sequence BETWEEN 0 AND 16383),
    last_file_sequence INTEGER,
    master_epoch INTEGER NOT NULL CHECK (master_epoch BETWEEN 0 AND 4294967295),
    virtual_epoch INTEGER NOT NULL CHECK (virtual_epoch BETWEEN 0 AND 4294967295),
    packet_epoch INTEGER NOT NULL CHECK (packet_epoch BETWEEN 0 AND 4294967295),
    file_epoch INTEGER NOT NULL CHECK (file_epoch BETWEEN 0 AND 4294967295),
    last_link_frame_id BLOB NOT NULL
        CHECK (typeof(last_link_frame_id) = 'blob' AND length(last_link_frame_id) = 8),
    updated_at_us INTEGER NOT NULL CHECK (updated_at_us >= 0),
    PRIMARY KEY (
        source_spacecraft_instance_id, link_generation, link_session_id,
        sender_boot_id, virtual_channel_id, apid, file_epoch_id
    )
);

CREATE TABLE tm_counter_observations (
    source_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(source_spacecraft_instance_id) = 'blob' AND length(source_spacecraft_instance_id) = 8),
    simulation_run_id BLOB NOT NULL
        CHECK (typeof(simulation_run_id) = 'blob' AND length(simulation_run_id) = 8),
    link_generation BLOB NOT NULL
        CHECK (typeof(link_generation) = 'blob' AND length(link_generation) = 8),
    link_session_id BLOB NOT NULL
        CHECK (typeof(link_session_id) = 'blob' AND length(link_session_id) = 8),
    link_frame_id BLOB NOT NULL
        CHECK (typeof(link_frame_id) = 'blob' AND length(link_frame_id) = 8),
    copy_index INTEGER NOT NULL CHECK (copy_index BETWEEN 0 AND 4294967295),
    apid INTEGER NOT NULL CHECK (apid BETWEEN 0 AND 2047),
    frame_sha256 BLOB NOT NULL
        CHECK (typeof(frame_sha256) = 'blob' AND length(frame_sha256) = 32),
    status TEXT NOT NULL CHECK (
        status IN ('BASELINE', 'IN_ORDER', 'ROLLOVER', 'GAP', 'DUPLICATE',
                   'STALE_GENERATION', 'STALE_SESSION', 'STALE_COUNTER', 'COUNTER_CONFLICT')
    ),
    master_gap INTEGER NOT NULL DEFAULT 0 CHECK (master_gap >= 0),
    virtual_gap INTEGER NOT NULL DEFAULT 0 CHECK (virtual_gap >= 0),
    packet_gap INTEGER NOT NULL DEFAULT 0 CHECK (packet_gap >= 0),
    file_gap INTEGER NOT NULL DEFAULT 0 CHECK (file_gap >= 0),
    received_at_us INTEGER NOT NULL CHECK (received_at_us >= 0),
    PRIMARY KEY (
        source_spacecraft_instance_id, simulation_run_id, link_frame_id, copy_index
    )
);

CREATE INDEX ix_tm_counter_state_source
    ON tm_counter_states(source_spacecraft_instance_id, link_generation, link_session_id);
