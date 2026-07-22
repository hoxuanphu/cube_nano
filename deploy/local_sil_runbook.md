# Local SIL quick runbook

Day la checklist ngan de chay nhanh. Huong dan van hanh day du, payload HTTP,
trang thai, recovery, Compose va gioi han release nam tai
[docs/gds_satellite_ccsds_operations_runbook.md](../docs/gds_satellite_ccsds_operations_runbook.md).

Tu repository root:

```
python scripts/validate_deploy_profiles.py --root .
python -m flight.satellite_simulator --root . --health-once
python -m flight.satellite_simulator --root . --roi-smoke
python -m flight.satellite_simulator --root . --status-interval 5
python scripts/demo_scenario.py --root . --timeout 90
python scripts/p4b_roundtrip.py --root .
python scripts/soak_test.py --iterations 100
```

Dieu kien smoke chinh:

- Health satellite va worker phai la READY.
- Lenh satellite daemon in log realtime ra `stderr`; `--status-interval 0` tat dong status dinh ky, `--log-level WARNING` chi hien canh bao/loi.
- Model phai hien demo_non_validated.
- roi-smoke phai co job_state=SUCCEEDED, downlink_frame_count > 0 va transfer_state=SEND_COMPLETED.
- Demo phai co product ground PUBLISHED, verified=true va SHA256_MATCH.

HTTP/backend va web:

```
python -m gds.http_app --root . --host 127.0.0.1 --port 8000
Set-Location gds/web
npm install
$env:VITE_API_BASE_URL="http://127.0.0.1:8000"
npm run dev -- --host 127.0.0.1 --port 4173
```

npm run test:e2e tu khoi dong backend/UI disposable, nen dung server thu cong
truoc khi chay. deploy/fault_profiles/*.yaml hien la fixture khai bao; CLI va
HTTP app chua load truc tiep cac file nay. Dung endpoint contact cho blackout va
cac test link_sim/recovery de kiem tra fault behavior.

Release evidence dung python scripts/generate_release_manifest.py --allow-dirty;
official release bat buoc worktree sach va chay khong co flag. Khong promote
Jetson hoac suy ra GPU SLO tu benchmark CPU.
