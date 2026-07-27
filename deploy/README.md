# Deployment profiles

`host_local_sil` is the first profile: loopback-only transport and a local
CPU/PyTorch reference worker. Compose/network and Jetson/TensorRT profiles are
separate profiles and cannot enter READY without target benchmark evidence.

## Phase 6 verification

From the repository root:

```text
python scripts/validate_deploy_profiles.py
python scripts/demo_scenario.py --timeout 90
python scripts/soak_test.py --iterations 100
```

The CPU image uses `deploy/compose_runtime_profile.yaml` and publishes only
`127.0.0.1:8000`; the Compose network is internal. `deploy/jetson-l4t-profile.yaml`
is deliberately fail-closed until a Jetson benchmark and TensorRT optimization
profile are committed.
