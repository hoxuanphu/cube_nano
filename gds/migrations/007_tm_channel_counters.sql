-- MCFC and VCFC belong to a transfer-frame channel, not an APID.  Keep this
-- state separate from the APID-scoped Space Packet and FilePacket state.
CREATE TABLE tm_channel_counter_states (
    source_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(source_spacecraft_instance_id) = 'blob' AND length(source_spacecraft_instance_id) = 8),
    link_generation BLOB NOT NULL
        CHECK (typeof(link_generation) = 'blob' AND length(link_generation) = 8),
    link_session_id BLOB NOT NULL
        CHECK (typeof(link_session_id) = 'blob' AND length(link_session_id) = 8),
    sender_boot_id INTEGER NOT NULL CHECK (sender_boot_id BETWEEN 0 AND 4294967295),
    virtual_channel_id INTEGER NOT NULL CHECK (virtual_channel_id BETWEEN 0 AND 7),
    last_master_channel_count INTEGER NOT NULL CHECK (last_master_channel_count BETWEEN 0 AND 255),
    last_virtual_channel_count INTEGER NOT NULL CHECK (last_virtual_channel_count BETWEEN 0 AND 255),
    master_epoch INTEGER NOT NULL CHECK (master_epoch BETWEEN 0 AND 4294967295),
    virtual_epoch INTEGER NOT NULL CHECK (virtual_epoch BETWEEN 0 AND 4294967295),
    last_link_frame_id BLOB NOT NULL
        CHECK (typeof(last_link_frame_id) = 'blob' AND length(last_link_frame_id) = 8),
    updated_at_us INTEGER NOT NULL CHECK (updated_at_us >= 0),
    PRIMARY KEY (
        source_spacecraft_instance_id, link_generation, link_session_id,
        sender_boot_id, virtual_channel_id
    )
);

CREATE INDEX ix_tm_channel_counter_source
    ON tm_channel_counter_states(source_spacecraft_instance_id, link_generation, link_session_id);

CREATE TABLE tm_packet_counter_states (
    source_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(source_spacecraft_instance_id) = 'blob' AND length(source_spacecraft_instance_id) = 8),
    link_generation BLOB NOT NULL
        CHECK (typeof(link_generation) = 'blob' AND length(link_generation) = 8),
    link_session_id BLOB NOT NULL
        CHECK (typeof(link_session_id) = 'blob' AND length(link_session_id) = 8),
    sender_boot_id INTEGER NOT NULL CHECK (sender_boot_id BETWEEN 0 AND 4294967295),
    virtual_channel_id INTEGER NOT NULL CHECK (virtual_channel_id BETWEEN 0 AND 7),
    apid INTEGER NOT NULL CHECK (apid BETWEEN 0 AND 2047),
    last_packet_sequence INTEGER NOT NULL CHECK (last_packet_sequence BETWEEN 0 AND 16383),
    packet_epoch INTEGER NOT NULL CHECK (packet_epoch BETWEEN 0 AND 4294967295),
    last_link_frame_id BLOB NOT NULL
        CHECK (typeof(last_link_frame_id) = 'blob' AND length(last_link_frame_id) = 8),
    updated_at_us INTEGER NOT NULL CHECK (updated_at_us >= 0),
    PRIMARY KEY (
        source_spacecraft_instance_id, link_generation, link_session_id,
        sender_boot_id, virtual_channel_id, apid
    )
);

CREATE INDEX ix_tm_packet_counter_source
    ON tm_packet_counter_states(source_spacecraft_instance_id, link_generation, link_session_id, apid);
