# Satellite AI adapter

`manifest.py`, `roi.py`, `threshold_lut.py`, `inference.py` and `products.py`
form the mission adapter. Runtime entry points must open a validated memmap
scene and pass an immutable `InputSpec`; CLI defaults are not a production
contract.

`worker_contract.py` and `worker_process.py` are the serialized process
boundary. The worker owns the singleton model, emits heartbeats while running,
checks per-job deadlines/cancellation and never mutates the durable journal.
