ALTER TABLE command_outbox ADD COLUMN ack_deadline_at_us INTEGER;
ALTER TABLE command_outbox ADD COLUMN last_delivery_reason TEXT;

ALTER TABLE command_attempts ADD COLUMN acked_at_us INTEGER;
ALTER TABLE command_attempts ADD COLUMN sequence_epoch INTEGER NOT NULL DEFAULT 0;

ALTER TABLE spacecraft_instances ADD COLUMN contact_state TEXT NOT NULL DEFAULT 'NO_CONTACT';
ALTER TABLE spacecraft_instances ADD COLUMN contact_changed_at_us INTEGER NOT NULL DEFAULT 0;

ALTER TABLE gds_metadata ADD COLUMN bound_spacecraft_instance_id BLOB;
ALTER TABLE gds_metadata ADD COLUMN bound_link_generation BLOB;
ALTER TABLE gds_metadata ADD COLUMN bound_link_session_id BLOB;

CREATE TABLE tc_sequence_allocators (
    spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(spacecraft_instance_id) = 'blob' AND length(spacecraft_instance_id) = 8),
    apid INTEGER NOT NULL CHECK (apid BETWEEN 0 AND 2047),
    next_sequence INTEGER NOT NULL CHECK (next_sequence BETWEEN 0 AND 16384),
    sequence_epoch INTEGER NOT NULL CHECK (sequence_epoch BETWEEN 0 AND 4294967295),
    last_reset_reason TEXT,
    updated_at_us INTEGER NOT NULL CHECK (updated_at_us >= 0),
    PRIMARY KEY (spacecraft_instance_id, apid)
);

CREATE TABLE event_sequences (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    next_event_id BLOB NOT NULL
        CHECK (typeof(next_event_id) = 'blob' AND length(next_event_id) = 8),
    updated_at_us INTEGER NOT NULL CHECK (updated_at_us >= 0)
);

CREATE TABLE storage_reservations (
    reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    volume TEXT NOT NULL,
    owner TEXT NOT NULL,
    reserved_bytes INTEGER NOT NULL CHECK (reserved_bytes > 0),
    state TEXT NOT NULL CHECK (state IN ('ACTIVE', 'RELEASED', 'EXPIRED')),
    expires_at_us INTEGER NOT NULL CHECK (expires_at_us >= 0),
    created_at_us INTEGER NOT NULL CHECK (created_at_us >= 0),
    released_at_us INTEGER
);

CREATE INDEX ix_outbox_ack_deadline
    ON command_outbox(state, ack_deadline_at_us);
CREATE INDEX ix_outbox_reason
    ON command_outbox(last_delivery_reason);
CREATE INDEX ix_spacecraft_contact
    ON spacecraft_instances(contact_state, last_seen_at_us);
CREATE INDEX ix_storage_reservations_active
    ON storage_reservations(volume, state, expires_at_us);
