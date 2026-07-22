# CCSDS MVP conformance matrix

| Layer | Profile | Implemented boundary | Explicitly out of scope |
|---|---|---|---|
| Application | F Prime v4.1.0 dictionary, typed mission command schemas | Python reference schemas and stock descriptor mapping | ECSS PUS |
| Space Packet | CCSDS primary header, APID 0/1/2/3, sequence modulo 16384 | `protocol/ccsds.py` | Secondary header/time formats |
| TC link | Type-BD, VC0, CRC/FECF | Python Type-BD codec and validator | Type-AD, FOP, FARM, COP-1 |
| TM link | One VC, SCID 68, 1024 bytes, FECF, OCF absent | Fixed frame encoder/decoder | OCF/CLCW, channel coding |
| File | F Prime-style START/DATA/END/CANCEL, descriptor 0x0003 | 990-byte DATA boundary and checksum | CFDP acknowledged recovery |
| Scalar | Big-endian binary, 16-hex U64 JSON | `protocol/canonical.py` | JSON numbers for U64 |
| Input | TIFF, one series/level, memmap, RGB uint16 | `sat_ai/roi.py` | JP2/HDF/NetCDF/compressed full decode |
| Science | Scene-anchored patch/tile-area ratio, strict validity | `sat_ai/inference.py` | Pixel-level segmentation/calibration |

## Golden vector inventory

- `golden_vectors/threshold_lut.bin`: 10001 big-endian IEEE-754 float32 entries.
- Space Packet sequence vectors: 16382/16383/0/1.
- TC Type-BD frame sequence vectors: 254/255/0/1 with bypass flag and SCID 68.
- TM MCFC/VCFC vectors: 254/255/0/1 and 1024-byte FECF boundary.
- Descriptor/APID vectors: command/telemetry/event/file mapped to 0/1/2/3.
- File START/DATA/END/CANCEL vectors, DATA 990/991 boundary and CFDP checksums.
- U64 JSON/binary vectors: 0, `2^53-1`, `2^53`, `2^63-1`, `2^63`, `2^64-1`.

The Python test suite is the executable evidence for the vectors in the first
three phases. F Prime C++ build constants must continue to mirror this profile
when the native deployment is added.
