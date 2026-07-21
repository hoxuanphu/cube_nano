-- Reassembly metadata used to be a growing JSON sequence map and a full JSON
-- rewrite per DATA packet. Keep packet identity and coalesced coverage in
-- indexed rows so the receiver only reads the small overlap window it needs.
ALTER TABLE file_reassemblies ADD COLUMN received_bytes INTEGER NOT NULL DEFAULT 0
    CHECK (received_bytes >= 0);
ALTER TABLE file_reassemblies ADD COLUMN terminal_packet_type INTEGER;
ALTER TABLE file_reassemblies ADD COLUMN terminal_sequence_index INTEGER;
ALTER TABLE file_reassemblies ADD COLUMN terminal_payload_sha256 TEXT;

CREATE TABLE file_reassembly_packets (
    source_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(source_spacecraft_instance_id) = 'blob' AND length(source_spacecraft_instance_id) = 8),
    link_session_id BLOB NOT NULL
        CHECK (typeof(link_session_id) = 'blob' AND length(link_session_id) = 8),
    file_epoch_id BLOB NOT NULL
        CHECK (typeof(file_epoch_id) = 'blob' AND length(file_epoch_id) = 8),
    sequence_index INTEGER NOT NULL CHECK (sequence_index BETWEEN 1 AND 4294967295),
    offset INTEGER NOT NULL CHECK (offset BETWEEN 0 AND 4294967295),
    payload_length INTEGER NOT NULL CHECK (payload_length BETWEEN 1 AND 990),
    payload_sha256 TEXT NOT NULL
        CHECK (length(payload_sha256) = 64 AND payload_sha256 NOT GLOB '*[^0-9a-f]*'),
    PRIMARY KEY (
        source_spacecraft_instance_id, link_session_id, file_epoch_id, sequence_index
    ),
    FOREIGN KEY (source_spacecraft_instance_id, link_session_id, file_epoch_id)
        REFERENCES file_reassemblies(source_spacecraft_instance_id, link_session_id, file_epoch_id)
        ON DELETE CASCADE
);

CREATE TABLE file_reassembly_ranges (
    source_spacecraft_instance_id BLOB NOT NULL
        CHECK (typeof(source_spacecraft_instance_id) = 'blob' AND length(source_spacecraft_instance_id) = 8),
    link_session_id BLOB NOT NULL
        CHECK (typeof(link_session_id) = 'blob' AND length(link_session_id) = 8),
    file_epoch_id BLOB NOT NULL
        CHECK (typeof(file_epoch_id) = 'blob' AND length(file_epoch_id) = 8),
    start_offset INTEGER NOT NULL CHECK (start_offset BETWEEN 0 AND 4294967295),
    end_offset INTEGER NOT NULL CHECK (end_offset BETWEEN 1 AND 4294967296),
    CHECK (end_offset > start_offset),
    PRIMARY KEY (
        source_spacecraft_instance_id, link_session_id, file_epoch_id, start_offset
    ),
    FOREIGN KEY (source_spacecraft_instance_id, link_session_id, file_epoch_id)
        REFERENCES file_reassemblies(source_spacecraft_instance_id, link_session_id, file_epoch_id)
        ON DELETE CASCADE
);

CREATE INDEX ix_file_reassembly_ranges_overlap
    ON file_reassembly_ranges(
        source_spacecraft_instance_id, link_session_id, file_epoch_id,
        start_offset, end_offset
    );

-- Preserve active transfers across upgrade before compacting the old JSON
-- columns. JSON1 is part of SQLite in supported runtime builds.
INSERT OR IGNORE INTO file_reassembly_packets(
    source_spacecraft_instance_id, link_session_id, file_epoch_id,
    sequence_index, offset, payload_length, payload_sha256
)
SELECT
    r.source_spacecraft_instance_id,
    r.link_session_id,
    r.file_epoch_id,
    CAST(item.key AS INTEGER),
    CAST(json_extract(item.value, '$.offset') AS INTEGER),
    CAST(json_extract(item.value, '$.length') AS INTEGER),
    CAST(json_extract(item.value, '$.sha256') AS TEXT)
FROM file_reassemblies AS r
JOIN json_each(CASE WHEN json_valid(r.sequence_map_json) THEN r.sequence_map_json ELSE '{}' END) AS item
WHERE CAST(item.key AS INTEGER) > 0
  AND json_type(item.value, '$.offset') = 'integer'
  AND json_type(item.value, '$.length') = 'integer'
  AND length(CAST(json_extract(item.value, '$.sha256') AS TEXT)) = 64;

INSERT OR IGNORE INTO file_reassembly_ranges(
    source_spacecraft_instance_id, link_session_id, file_epoch_id,
    start_offset, end_offset
)
SELECT
    r.source_spacecraft_instance_id,
    r.link_session_id,
    r.file_epoch_id,
    CAST(json_extract(item.value, '$[0]') AS INTEGER),
    CAST(json_extract(item.value, '$[1]') AS INTEGER)
FROM file_reassemblies AS r
JOIN json_each(CASE WHEN json_valid(r.ranges_json) THEN r.ranges_json ELSE '[]' END) AS item
WHERE json_type(item.value) = 'array'
  AND CAST(json_extract(item.value, '$[1]') AS INTEGER) > CAST(json_extract(item.value, '$[0]') AS INTEGER);

UPDATE file_reassemblies
SET received_bytes = COALESCE((
    SELECT SUM(end_offset - start_offset)
    FROM file_reassembly_ranges AS coverage
    WHERE coverage.source_spacecraft_instance_id = file_reassemblies.source_spacecraft_instance_id
      AND coverage.link_session_id = file_reassemblies.link_session_id
      AND coverage.file_epoch_id = file_reassemblies.file_epoch_id
), 0);

UPDATE file_reassemblies
SET terminal_packet_type = CASE json_extract(sequence_map_json, '$._terminal.type')
        WHEN 'END' THEN 3
        WHEN 'CANCEL' THEN 4
        ELSE NULL
    END,
    terminal_sequence_index = CAST(json_extract(sequence_map_json, '$._terminal.sequence') AS INTEGER),
    terminal_payload_sha256 = CAST(json_extract(sequence_map_json, '$._terminal.sha256') AS TEXT)
WHERE json_valid(sequence_map_json);

UPDATE file_reassemblies
SET ranges_json = '[]', sequence_map_json = '{}';
