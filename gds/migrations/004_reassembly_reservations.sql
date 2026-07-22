ALTER TABLE file_reassemblies ADD COLUMN reservation_id INTEGER;

CREATE INDEX ix_file_reassemblies_reservation
    ON file_reassemblies(reservation_id, state);
