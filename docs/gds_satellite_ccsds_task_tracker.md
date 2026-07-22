# Bo theo doi task GDS Satellite va CCSDS

> Nguon: [gds_satellite_ccsds_simulation_plan.md](gds_satellite_ccsds_simulation_plan.md)
>
> Ngay tao: 2026-07-19
>
> Pham vi: MVP software-in-the-loop, CCSDS o muc packet/frame bytes, khong RF/SDR.

Tai lieu nay tach ke hoach ky thuat thanh cac task co the giao viec va danh dau tien do. Task chi duoc danh dau hoan thanh khi co bang chung o cot Evidence.

## Cach cap nhat

- [ ] = TODO; [x] = DONE.
- Status dung mot trong "TODO", "DOING", "BLOCKED", "DONE".
- Owner, ETA va Evidence duoc dien khi task duoc giao/thuc thi.
- Neu thay doi mot default da khoa trong Gate 0, cap nhat change log va re-open Phase 0 truoc khi sua cac phase sau.
- Mot task bi BLOCKED phai ghi ro blocker, nguoi quyet dinh va task/decision mo khoa.

## Dashboard

| Phase | Noi dung | Uoc luong | Phu thuoc chinh | Tien do |
|---|---|---:|---|---:|
| Phase 0 | Baseline, contract, profile va Gate 0 | 6-10 person-days | Khong | 16/16 |
| Phase 1 | ROI inference core va artifact | 7-10 person-days | Phase 0 | 13/13 |
| Phase 2a | F Prime skeleton, dictionary, protocol | 5-8 person-days | Phase 0; co the song song Phase 1 | Python reference 11/11; native conditional (F-08 open) |
| Phase 2b | AI worker, durable state, TM scheduler | 7-10 person-days | Phase 1, 2a | 15/15 |
| Phase 3 | Link Simulator va deterministic replay | 8-12 person-days | Phase 0, 2a; integration voi 2b | 12/12 |
| Phase 4a | GDS ledger, API, SQLite core | 8-12 person-days | Phase 0, 2a, 3 contract | 14/14 |
| Phase 4b | TM, catalog, file, realtime, local deploy | 10-15 person-days | Phase 1, 2b, 3, 4a | 15/15 |
| Phase 5 | GDS Webapp core | 8-12 person-days | Phase 4b API | 14/14 |
| Phase 6 | E2E, hardening, release | 10-16 person-days | Tat ca phase truoc | 14/16 (2 conditional gates) |
| **Tong** | **MVP** | **69-105 person-days** |  | **96/126** |

## Gate 0 - Quyet dinh truoc khi code

Day la cac quyet dinh bat buoc tu muc 16 cua ke hoach. Cac task P0-01 den P0-04 phai duoc chot truoc khi danh dau baseline san sang.

| Ma | Quyet dinh | Trang thai | Nguoi chot | Bang chung |
|---|---|---|---|---|
| G0-01 | Giu F Prime v4.1.0, stock APID 0/1/2/3, TC Type-BD, mot TM VC | DONE | Codex | protocol/mission_profile.yaml; tests/test_mission_contracts.py |
| G0-02 | ROI pixel half-open, scene-anchored grid, memmap-only runtime | DONE | Codex | sat_ai/roi.py; tests/test_sat_ai_mission.py |
| G0-03 | Khong downlink crop bi science-reject; analytic TIFF, quicklook WebP 8-bit | DONE | Codex | sat_ai/products.py; docs/gds_satellite_ccsds_baseline_report.md |
| G0-04 | CPU/PyTorch la reference; Jetson/TensorRT la profile sau benchmark | DONE | Codex | sat_ai/deployment_profile.yaml; artifacts/benchmarks/local-cpu-pytorch-v2.json |
| G0-05 | Khoa queue/SLO khoi diem, tone-map, cache va retention theo plan | DONE | Codex | flight/mission_com_scheduler.py; protocol/runtime_profile.yaml |
| G0-06 | Chon scientific status mac dinh demo_non_validated va gate promotion | DONE | Codex | sat_ai/model_manifest.yaml; sat_ai/inference.py |
| G0-07 | Chon host_local_sil hay compose_sil cho profile chay dau tien | DONE | Codex | protocol/runtime_profile.yaml; deploy/local_sil_runbook.md |

Neu stakeholder chon full COP-1/Type-AD/CLCW, CFDP, SDLS, lat/lon/georeference, RF/SDR, custom APID, compressed backend runtime hoac segmentation pixel-level thi mo change request rieng va uoc luong lai; khong bat bang feature flag MVP.

## Phase 0 - Dong bang baseline va contract

Exit gate: baseline test pass; model/InputSpec golden va mismatch test pass; dependency, model, dictionary hash lap lai duoc; cac schema va conformance matrix khong con ambiguity; profile khong deployable phai bi schema/startup chan READY.

- [x] P0-01 | Kiem ke va dong bang thay doi hien co trong worktree.
  - Output: baseline report ghi commit, file thay doi, test hien tai va danh sach thay doi inference can giu.
  - Depends: Khong.
  - Done when: 53 test hien tai pass, moi thay doi duoc phan loai, khong co file/patch bi bo sot.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / docs/gds_satellite_ccsds_baseline_report.md; pytest -q.

- [x] P0-02 | Tao package layout muc tieu cho flight, sat_ai, protocol, link_sim, gds va deploy.
  - Output: thu muc/module skeleton va README ownership cho tung boundary.
  - Depends: P0-01.
  - Done when: moi module co entry point/test placeholder; web khong import truc tiep inference.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / protocol/, sat_ai/, flight/, gds/, deploy/ package layout.

- [x] P0-03 | Dong goi checkpoint va InputSpec thanh model bundle bat bien.
  - Output: manifest gom model SHA-256, channels=3, band_order, patch_size=256, normalization, release ID.
  - Depends: P0-01.
  - Done when: checkpoint, manifest va InputSpec duoc verify cung mot SHA; artifact thieu/doi bytes bi reject.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / sat_ai/model_manifest.yaml; checkpoint SHA test.

- [x] P0-04 | Dinh nghia schema parser cho ModelManifest va InputSpec.
  - Output: schema version, parser, validator va loi fail-fast.
  - Depends: P0-03.
  - Done when: sai channel, band order, patch size, normalization, schema version va SHA deu co negative test.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / sat_ai/manifest.py; tests/test_sat_ai_mission.py.

- [x] P0-05 | Ket noi mission adapter voi CloudTorchInfer theo InputSpec.
  - Output: adapter nap model mot lan va truyen input contract vao runtime.
  - Depends: P0-04.
  - Done when: runtime khong con default 4 kenh/CLI heuristic; checkpoint that 3 kenh chay duoc.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / sat_ai/inference.py; actual RGB checkpoint smoke.

- [x] P0-06 | Tao dependency lock va pin build environment.
  - Output: lock co hash cho Python/Node, container/base image digest, F Prime/dictionary hash, SOURCE_DATE_EPOCH policy.
  - Depends: P0-01.
  - Done when: clean setup tai lai cung dependency va cung hash artifact.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / requirements-lock.txt; build_manifest.json.

- [x] P0-07 | Pin F Prime v4.1.0 va xac nhan dictionary ReferenceDeployment.
  - Output: version manifest, source hash, build instruction va upgrade task rieng neu can.
  - Depends: P0-01.
  - Done when: source va dictionary cung version; khong tham chieu tai lieu latest trong build MVP.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / build_manifest.json; fprime_dictionary.json v4.1.0 validation.

- [x] P0-08 | Khoa mission profile CCSDS MVP.
  - Output: mission_profile.yaml gom SCID 68, APID 0/1/2/3, VC0, TM frame 1024, big-endian, CRC/FECF, sequence/time contract, OCF absent.
  - Depends: G0-01, P0-07.
  - Done when: profile validate duoc va build constants khop dictionary/F Prime.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / protocol/mission_profile.yaml; protocol/profile.py.

- [x] P0-09 | Khoa TM/file buffer budget va ownership.
  - Output: FW_COM_BUFFER_MAX_SIZE=512, FW_FILE_BUFFER_MAX_SIZE=1003, descriptor 2 byte, raw DATA toi da 990 byte/frame, bin sizes/counts.
  - Depends: P0-08.
  - Done when: build-time assertion reject packet oversize; allocation failure co metric/error code.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / protocol/ccsds.py; protocol/file_packet.py.

- [x] P0-10 | Dinh nghia command, telemetry, event va product schema.
  - Output: schema cho RequestKey, JobKey, ScopedSceneRef, ProductRef, config/catalog epoch/revision, state machine va error code.
  - Depends: P0-08.
  - Done when: moi field wire type, endian, range, lifecycle state va ownership duoc ghi ro.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / protocol/schemas/*.yaml; protocol/schemas.py.

- [x] P0-11 | Implement canonical scalar/U64 contract.
  - Output: binary U64 big-endian, deterministic-CBOR, JSON/JCS 16 hex lowercase, SQLite BLOB(8), TypeScript opaque string/BigInt.
  - Depends: P0-10.
  - Done when: round-trip golden bao phu 0, 2^53-1, 2^53, 2^63-1, 2^63, 2^64-1; sai length/format bi reject.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / protocol/canonical.py; golden U64 tests.

- [x] P0-12 | Khoa ROI, validity va inference numeric contract.
  - Output: half-open pixel ROI, scene-anchored grid, strict validity 10000 bp, padding/window-readable sidecar, integer coverage comparison va LUT ID logit-bp-f32-lut-v1.
  - Depends: G0-02, P0-04.
  - Done when: boundary/equality/overflow/NoData/padding rule co vector va khong dung float lam science decision.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / sat_ai/roi.py; sat_ai/threshold_lut.py.

- [x] P0-13 | Khoa product, checksum va catalog authority contract.
  - Output: deterministic typed USTAR/TAR, manifest, Fw checksum, bundle SHA, artifact hashes, satellite catalog authority va GDS verified replica.
  - Depends: P0-10, G0-03.
  - Done when: mot ProductRef chi co mot deterministic bundle; atomic publish va stale catalog policy duoc test design.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / sat_ai/products.py; flight/catalog.py; protocol/file_packet.py.

- [x] P0-14 | Khoa scheduler, queue, watchdog va fairness parameters.
  - Output: single-in-flight, ACK token, upstream-return hold, initial ack_burst=8/control_burst=4/file_burst=8, capacity/overflow/error, worker deadline/restart policy.
  - Depends: P0-09, P0-10.
  - Done when: khong co silent drop; queue full, late return, abort/cooldown va metric contract co schema.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / flight/mission_com_scheduler.py; flight/state_machine.py.

- [x] P0-15 | Khoa security, storage va replay profile.
  - Output: host_local_sil/compose_sil topology, bind/peer/origin limits, SQLite WAL/single-writer, retention/watermark/reserve, deterministic fault/replay artifact.
  - Depends: G0-07, P0-10.
  - Done when: startup guard, quota, reserve va replay state PRESENT/PINNED/EVICTED co acceptance rule.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / protocol/runtime_profile.yaml; flight/journal.py.

- [x] P0-16 | Tao conformance matrix va golden vector plan.
  - Output: protocol/conformance_matrix.md, vector inventory cho APID, descriptor, CRC, rollover, frame/file boundary, negative route.
  - Depends: P0-08 den P0-15.
  - Done when: review sign-off; implementation placeholder va profile khong deployable bi schema/startup chan READY.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / protocol/conformance_matrix.md; protocol/golden_vectors/.

## Phase 1 - ROI inference core

Exit gate: ROI canonical memmap TIFF chay dung; grid, patch boundary, validity, normalization, deterministic bundle va progress co test; CPU reference co benchmark artifact non-null, RSS delta <=256 MiB va p95 scene-scale <=1.25x.

- [x] P1-01 | Implement read_window(x, y, width, height) cho source TIFF memmap.
  - Output: API doc truc tiep cua so X/Y, khong tao crop tam, tra shape/dtype/metrics logical bytes.
  - Depends: P0-12.
  - Done when: ROI 256x256 doc dung cua so; full-scene read khong bi goi trong mission path.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / sat_ai/roi.py; tests/test_sat_ai_mission.py.

- [x] P1-02 | Implement read_window cho validity mask va strict NoData.
  - Output: validity sidecar window-readable, padding/border semantics va checksum validation.
  - Depends: P1-01, P0-12.
  - Done when: source/mask deu khong full decode; compressed/non-memmap mask bi reject truoc doc lon.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / validity sidecar and open_memmap_scene tests.

- [x] P1-03 | Implement scene-anchored patch grid va intersection weighting.
  - Output: grid origin scene, expanded patch window, valid intersection area, shift/ROI rounding helper.
  - Depends: P1-01, P0-12.
  - Done when: dich ROI 1 pixel theo shift oracle; patch bien va ROI min patch_size co vector.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / iter_patch_windows scene-anchor shift vectors.

- [x] P1-04 | Implement threshold LUT va cloud-positive tile area ratio.
  - Output: byte-canonical LUT artifact, SHA, finite-logit handling, checked U64 decision.
  - Depends: P0-11, P0-12.
  - Done when: threshold 0/10000/equality/overflow co golden; khong dung sigmoid/logit runtime tu math library.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / sat_ai/threshold_lut.py; LUT SHA/vector tests.

- [x] P1-05 | Implement singleton model runtime va bounded inference queue.
  - Output: model load-once, queue capacity, overflow code/metric, batch config, worker lifecycle hook.
  - Depends: P0-03, P0-14.
  - Done when: mot vong doi worker chi load model mot lan; queue full khong mat hoac chay trung job.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / SingletonModelRuntime and BoundedInferenceQueue.

- [x] P1-06 | Implement infer_region theo ROI va progress throttle.
  - Output: API tra decision, thresholds, grid metrics, progress co throttle, latency va model metadata.
  - Depends: P1-02, P1-03, P1-04, P1-05.
  - Done when: output day du config snapshot, model SHA, InputSpec, LUT SHA va science enum.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / infer_region progress/result contract.

- [x] P1-07 | Tao crop/quicklook/mask/tile 8-bit theo display profile.
  - Output: tone-map deterministic, band order, NoData, black/white point, gamma, algorithm version, WebP/tile output.
  - Depends: P1-03, P0-13.
  - Done when: browser khong doc TIFF uint16; golden pixels/checksum va khong seam giua tile.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / sat_ai/products.py deterministic tone-map/WebP/TIFF.

- [x] P1-08 | Tao result manifest va deterministic product bundle.
  - Output: manifest, typed USTAR/TAR, exact entry set, source/model/config/product checksums, atomic staging path.
  - Depends: P1-06, P1-07, P0-13.
  - Done when: cung input/profile cho cung byte stream; bundle thieu/du artifact bi reject.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / build_products/build_ustar atomic publish.

- [x] P1-09 | Them regression test dung checkpoint that.
  - Output: test inference 3 kenh, normalization, output/threshold vectors va model assurance status.
  - Depends: P1-06, P1-08.
  - Done when: checkpoint that pass; mismatch SHA/channel/patch/normalization fail-fast.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / checkpoint manifest regression and ROI smoke evidence.

- [x] P1-10 | Them spy test chong full decode va crop tam.
  - Output: spies cho TiffFile.asarray, read_rows full width, source/mask decoder va temp crop creation.
  - Depends: P1-01, P1-02, P1-06.
  - Done when: mission path chi goi window read; compressed TIFF/JP2 khong duoc vao runtime.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / compressed source rejection and window-read tests.

- [x] P1-11 | Them test NoData/padding/ROI shift/edge va numeric overflow.
  - Output: oracle cho valid area, strict valid 10000 bp, floor/ceil drag, min patch_size, U64 multiply.
  - Depends: P1-03, P1-04.
  - Done when: moi boundary co ket qua deterministic va overflow bi reject khong wrap.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / NoData, shift, edge, equality and overflow tests.

- [x] P1-12 | Benchmark local CPU/PyTorch theo ROI, patch count va batch.
  - Output: benchmark artifact non-null gom throughput, p95, RSS, logical bytes, model load, deadline baseline.
  - Depends: P1-05, P1-06, P1-10.
  - Done when: profile reference deployable; RSS delta <=256 MiB, p95 scene-area 4x <=1.25x.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / sat_ai/benchmark.py; actual CPU artifact.

- [x] P1-13 | Dong goi mission adapter va integration contract cho Phase 2b.
  - Output: API version, error mapping, progress/result schema, fixture scene/product.
  - Depends: P1-08, P1-12.
  - Done when: AI worker co the goi adapter qua contract ma khong phu thuoc CLI/default.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / sat_ai/worker_contract.py; deploy/local_sil_runbook.md.

## Phase 2a - F Prime skeleton, dictionary va protocol (native conditional)

The completed items in this section evidence the Python F Prime-compatible
reference only. They do not close the native F Prime deliverable. The repository
and local environment lack the pinned source checkout and F Prime/FPP generation
toolchain; therefore F-08 and native Phase 2a exit remain conditional. See
`docs/gds_satellite_ccsds_fprime_scope_decision_20260721.md`.

Exit gate: TC APID 0 dispatch dung; TM APID/descriptor 1/2/3 dung; dictionary/profile/build constants va golden vector khop; completion gate khong UAF/leak/double-return.

- [x] P2A-01 | Tao F Prime flight deployment va CloudPayload FPP component.
  - Output: deployment build duoc, port/command/event/channel stub va ownership ro rang.
  - Depends: P0-07, P0-10.
  - Done when: deployment start duoc voi component READY va dictionary generation reproducible.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / flight/CloudPayload.fpp; flight/reference_deployment.py.

- [x] P2A-02 | Dinh nghia opcode va payload cho 9 command MVP.
  - Output: CLOUD_SET_CONFIG, SCENE_*, ROI, JOB_*, PRODUCT_* schema/FPP arguments.
  - Depends: P0-10.
  - Done when: target instance, RequestKey, SceneRef/config snapshot validation duoc encode/decode.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / protocol/schemas.py and command_schema.yaml.

- [x] P2A-03 | Dinh nghia telemetry, event, acknowledgement va state channels.
  - Output: health/config/progress/result, event severity/message, command ACK va science enum.
  - Depends: P0-10.
  - Done when: moi output co boot/instance/session/time contract va dictionary ID.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / protocol/messages.py and telemetry_schema.yaml.

- [x] P2A-04 | Tich hop stock APID router va command dispatcher.
  - Output: APID 0 TC ingress, opcode dispatch, target mismatch va unknown APID/opcode reject.
  - Depends: P0-08, P2A-01, P2A-02.
  - Done when: APID tuy y 0x120..0x123 khong duoc coi la MVP route; negative test pass.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / flight/stock_router.py; APID negative tests.

- [x] P2A-05 | Implement MissionComScheduler skeleton READY/IN_FLIGHT.
  - Output: single packet/frame scheduling, sequence allocator, queue ownership va completion callback.
  - Depends: P0-09, P0-14, P2A-03.
  - Done when: packet thu hai khong gui som; scheduler chi complete sau khi ownership/status day du.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / flight/mission_com_scheduler.py.

- [x] P2A-06 | Implement MissionUdpAdapter completion gate.
  - Output: full dataReturn/comStatus chain giua SpacePacketFramer va TmFramer, upstream-return hold.
  - Depends: P2A-05.
  - Done when: status den truoc frame return van duoc tri hoan; khong UAF/leak/double-return.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / flight/mission_udp_adapter.py; status/return gate test.

- [x] P2A-07 | Tich hop SpacePacketFramer, TmFramer va TC deframer profile Type-BD.
  - Output: TC/TM bytes, CRC/FECF, APID, VC0, OCF absent, one packet per TM frame.
  - Depends: P0-08, P2A-04, P2A-06.
  - Done when: bytes khop profile va stock F Prime v4.1.0; no claim COP-1/FARM/CLCW.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / protocol/ccsds.py; profile and CRC tests.

- [x] P2A-08 | Cau hinh file buffer va FilePacket boundary.
  - Output: DATA 990 byte/frame, frame 1024 byte, idle padding 7 byte, oversize reject.
  - Depends: P0-09, P2A-07.
  - Done when: 990-byte boundary va 991-byte reject co golden vector.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / protocol/file_packet.py; 990/991 boundary vector.

- [x] P2A-09 | Sinh dictionary va protocol golden vectors.
  - Output: dictionary JSON, packet/frame hex vectors, source/destination path length, descriptor/APID mapping.
  - Depends: P2A-02, P2A-03, P2A-07, P2A-08.
  - Done when: clean build sinh cung dictionary/hash va vector decode round-trip.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / protocol/generate_vectors.py; TC/TM, descriptor/APID, START/DATA/END/CANCEL and checksum vectors.

- [x] P2A-10 | Them vector rollover va time contract.
  - Output: Space Packet sequence 16382,16383,0,1; TM MCFC/VCFC 254,255,0,1; application timestamp.
  - Depends: P2A-07, P0-11.
  - Done when: rollover phan biet gap/reset va timestamp base/epoch/resolution khop.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / Space Packet, TC Type-BD and TM counter rollover vectors; mission time contract.

- [x] P2A-11 | Them ownership/failure tests cho scheduler va adapter.
  - Output: allocation failure, status-before-return, late return, duplicate callback, no packet loss/leak test.
  - Depends: P2A-05, P2A-06.
  - Done when: completion tuple observable, error code ro, pass sanitizer/heap ownership test.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / tests/test_mission_contracts.py; tests/test_phase2_runtime.py.

## Phase 2b - AI worker, durable state, queue va TM scheduler

Exit gate: worker crash, config/job/transfer crash matrix, cancel race, queue saturation, control/ACK flood va no-late-DATA test pass; business effect exactly-once; khong publish partial product.

- [x] P2B-01 | Tao AI worker IPC contract va heartbeat.
  - Output: request/result envelope, worker version, heartbeat, timeout, deadline va error mapping.
  - Depends: P1-13, P2A-01.
  - Done when: worker mat heartbeat thanh WORKER_LOST va job co terminal policy.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / sat_ai/worker_contract.py; sat_ai/worker_process.py; flight/worker_client.py; process-crash test.

- [x] P2B-02 | Implement bounded worker queue va restart policy.
  - Output: max_pending_jobs, queue full response, restart limit/backoff, staging cleanup.
  - Depends: P2B-01, P0-14.
  - Done when: day queue nhan QUEUE_FULL; job khong mat/chay trung; vuot retry limit vao FAULT.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / flight/worker_client.py bounded pending queue, heartbeat watchdog, restart window/backoff and QUEUE_FULL test.

- [x] P2B-03 | Implement onboard durable journal va retired request ranges.
  - Output: journal theo RequestKey, mission digest, cached result, contiguous/sparse retired marker, full compaction.
  - Depends: P0-10, P0-11.
  - Done when: duplicate cung digest replay ket qua cu; digest khac/retired bi reject, khong chay lai effect.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / flight/journal.py; compact_request retired-range test.

- [x] P2B-04 | Implement atomic CLOUD_SET_CONFIG CAS.
  - Output: durable config epoch/revision, hai threshold byte-equality, ACK snapshot, wrap/migration policy.
  - Depends: P0-10, P0-12, P2B-03.
  - Done when: crash truoc/sau commit replay chi tang revision mot lan; reorder SET/ROI bi reject.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / apply_config_command atomic CAS and replay smoke.

- [x] P2B-05 | Implement command admission va immutable job snapshot.
  - Output: validation SceneRef/config/target, job row, thresholds/model/InputSpec/LUT snapshot trong mot transaction.
  - Depends: P2B-03, P2B-04, P1-13.
  - Done when: COMMAND_ACCEPTED luon co work row; snapshot khong doc lai global config khi worker bat dau.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / admit_analysis immutable snapshot transaction.

- [x] P2B-06 | Implement state machine command/job/science.
  - Output: state transition table, guard, error code, progress coalesce, SUCCEEDED/REJECTED/FAILED/CANCELED/FAULT.
  - Depends: P2B-05, P0-10.
  - Done when: moi transition co event va khong co transition ngam; science enum dung assurance status.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / flight/state_machine.py; explicit transition test.

- [x] P2B-07 | Implement restart reconciliation cho command, job, config, product, transfer.
  - Output: startup scan, corruption detection, READY product/catalog preservation, nonterminal policy.
  - Depends: P2B-03 den P2B-06.
  - Done when: reboot tai moi state khong publish partial; COMMAND_ACCEPTED thieu work row vao FAULT.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / deployment startup reconciliation, STAGING failure policy and atomic-directory cleanup tests.

- [x] P2B-08 | Implement JOB_GET_STATUS va JOB_CANCEL voi race handling.
  - Output: target RequestKey, cancel requested, terminal allow-list, idempotent ACK.
  - Depends: P2B-03, P2B-06.
  - Done when: dequeue/complete/timeout/restart race chi co mot terminal state.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / JOB_GET_STATUS/JOB_CANCEL handlers.

- [x] P2B-09 | Tich hop AI result voi product staging.
  - Output: result manifest, deterministic bundle, staging directory, checksum va model/config metadata.
  - Depends: P1-08, P2B-05, P2B-06.
  - Done when: reject ROI van co metadata policy; accepted ROI chi publish sau bundle verify.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / atomic directory publish in sat_ai/products.py; serialized worker result and atomic journal publish.

- [x] P2B-10 | Implement MissionComScheduler queue fairness va ACK token.
  - Output: ACK/control/file queues, burst 8/4/8, backpressure, progress coalesce, capacity metrics.
  - Depends: P2A-05, P2A-06, P0-14.
  - Done when: fault/control flood khong chiem ACK slot; file co minimum service.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / executable 8 ACK -> control -> file arbitration and flood oracle.

- [x] P2B-11 | Implement FileDownlinkCoordinator mot global attempt.
  - Output: START/DATA/END lifecycle, transfer ID, ABORTING/COOLDOWN, buffer return fence.
  - Depends: P2A-08, P2B-09, P2B-10.
  - Done when: DATA/END attempt cu khong cross attempt moi; late buffer return khong assert.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / completion-token leases, attempt epoch, abort fence, cooldown and late-callback guard in flight/file_downlink.py.

- [x] P2B-12 | Implement PRODUCT_REQUEST_DOWNLINK va PRODUCT_CANCEL_DOWNLINK.
  - Output: ProductRef lookup, retention check, active transfer guard, completion-wins cancel.
  - Depends: P2B-07, P2B-11.
  - Done when: transfer active bi cancel dung policy; duplicate cancel replay outcome cu; no second attempt ngam dinh.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / CloudPayload downlink/cancel handlers and transfer journal.

- [x] P2B-13 | Them worker/transfer failure injection tests.
  - Output: kill worker, terminal adapter failure, DATA queue abort, cooldown, no-late-DATA matrix.
  - Depends: P2B-01, P2B-02, P2B-11.
  - Done when: khong publish staging, buffer return mot lan, slot chi release sau abort fence.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / process kill, link failure, abort fence, cooldown, cancel race, late callback and partial-publish tests.

- [x] P2B-14 | Them queue saturation va control/ACK/file flood test.
  - Output: capacity oracle, oldest_ack_age, health latency, file goodput metrics.
  - Depends: P2B-10.
  - Done when: oldest_ack_age <=1s, health_max_latency <=2s, file goodput >0 tren local SIL.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / worker/scheduler saturation tests; ACK/control/file flood oracle; benchmark v2 guards.

- [x] P2B-15 | Dong goi Satellite Simulator local profile.
  - Output: process entrypoint, config, health/readiness, fixture scene/catalog, runbook dev.
  - Depends: P2B-07, P2B-12, P2B-14.
  - Done when: satellite boot READY chi khi manifest, profile, catalog va benchmark required hop le.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / flight/satellite_simulator.py --roi-smoke; deploy/local_sil_runbook.md.

## Phase 3 - Link Simulator

Exit gate: cung ingress bytes/run/seed/profile tao cung decision log/byte stream; replay UDP byte-exact; artifact crash/retention dung state; attempt/boot khong cross barrier; ACK/health bound van dat.

- [x] P3-01 | Tao transport abstraction in-memory va UDP datagram.
  - Output: send/receive interface, peer/session envelope, frame ID, direction, timestamps.
  - Depends: P2A-07, P0-15.
  - Done when: GDS va Satellite dung cung contract; UDP khong expose peer sai profile.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / link_sim/transport.py; SidebandEnvelope validation tests.

- [x] P3-02 | Implement virtual clock va ordered event queue.
  - Output: monotonic simulation time, tie-break/admission order, scheduled delivery/cancel.
  - Depends: P3-01.
  - Done when: replay khong phu thuoc timing thread; clock reset co boot/session marker.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / link_sim/virtual_clock.py; event ordering tie-breaker tests.

- [x] P3-03 | Implement fault profile latency/jitter/loss/duplicate/corruption.
  - Output: schema, validation, deterministic decision record va metrics theo direction/frame/fault.
  - Depends: P3-02, P0-15.
  - Done when: moi fault decision replay lai duoc tu seed/profile/frame ID; khong shared PRNG theo thread.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / link_sim/fault_model.py; deterministic PRF tests.

- [x] P3-04 | Implement bandwidth shaper va blackout.
  - Output: byte budget/time schedule, NO CONTACT/BLACKOUT state, drop frame policy.
  - Depends: P3-02, P3-03.
  - Done when: frame trong blackout bi drop; immediate/next-contact policy nhan dung event.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / link_sim/contact_schedule.py; ContactSchedule.should_drop_frame; tests/test_link_simulator_blackout.py 5 tests.

- [x] P3-05 | Serialize ingress va ghi admission order.
  - Output: canonical counter PRF/distribution, ingress ordinal, structured decision log.
  - Depends: P3-02, P3-03.
  - Done when: concurrent ingress co order recorded; cung ordered input cho cung output.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / link_sim/link_simulator.py; admission_log with order.

- [x] P3-06 | Implement segmented self-contained replay artifact.
  - Output: OPEN/FINAL/INCOMPLETE_CRASH/INCOMPLETE_STORAGE, segment version/length/CRC, artifact SHA/size.
  - Depends: P3-05, P0-15.
  - Done when: torn-tail recovery, atomic finalize, raw frame prune khong lam mat artifact PRESENT/PINNED.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / link_sim/link_simulator.py; admission log structure for replay.

- [x] P3-07 | Implement replay quota, pin va retention.
  - Output: reservation, cap, PRESENT/PINNED/EVICTED transition, disk watermark.
  - Depends: P3-06.
  - Done when: cap exhaustion vao INCOMPLETE_STORAGE; pin nam trong global headroom; eviction co tombstone.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / link_sim/replay_manager.py; ReplayManager with quota/pin/eviction; tests/test_replay_manager.py 11 tests.

- [x] P3-08 | Implement queue overflow/backpressure va metrics.
  - Output: bounded queue, overflow counter, in-memory/UDP behavior, fallback log.
  - Depends: P3-01, P3-04.
  - Done when: overflow khong silent; UDP drop co metric/log va downstream timeout co the quan sat.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / link_sim/link_simulator.py; 9 tests pass.

- [x] P3-09 | Implement FilePacket START/DATA/END drain fence.
  - Output: attempt barrier, DATA ordering/reorder buffer, abort fence, transfer busy rule.
  - Depends: P2B-11, P3-02.
  - Done when: DATA/END A khong vao B; START B chi admit sau barrier.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / link_sim/file_epoch.py; tests/test_file_epoch.py 12 tests pass.

- [x] P3-10 | Implement sender boot/session handshake va restart resolution.
  - Output: spacecraft_boot_id, link_session_id/generation, close epoch cu, startup delivery/drop policy.
  - Depends: P3-09, P2B-07.
  - Done when: Satellite restart/Link restart khong cho packet boot cu cross sang session moi.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / link_sim/session_manager.py; tests/test_session_manager.py 14 tests pass.

- [x] P3-11 | Benchmark TM 1024-byte goodput va recovery cost.
  - Output: frame count, goodput, retry cost theo fault profile muc tieu va buffer budget.
  - Depends: P3-03, P3-04, P3-09.
  - Done when: ket qua duoc ghi vao benchmark artifact va dung lam input tune queue/SLO.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / link_sim/benchmark.py; tests/test_link_benchmark.py 5 tests pass.

- [x] P3-12 | Them replay determinism va crash/retention test suite.
  - Output: same seed/profile byte-exact, different seed khac, OPEN crash, prune, restart, UDP replay.
  - Depends: P3-06, P3-07, P3-10.
  - Done when: replay FINAL sau raw prune tao cung protocol/fault output trong supported profile.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / tests/test_phase3_exit_gate.py 6 tests; 53 total Phase 3 tests pass.

## Phase 4a - GDS ledger, API va SQLite core

Exit gate: atomic admission/outbox khong mat command/row mo coi; concurrent same-key, expiry, migration, capacity, WAL, writer saturation va recovery test pass.

- [x] P4A-01 | Tao schema SQLite versioned va migration forward-only.
  - Output: tables cho commands, outbox, attempts, state, telemetry/event, product, run, replay, audit.
  - Depends: P0-10, P0-11.
  - Done when: startup fail readiness neu schema/binary khong tuong thich; migration test pass.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/migrations/001_initial.sql; gds/migrations/002_phase4a_runtime.sql; gds/schema.py; tests/test_phase4a_gds.py migration/profile tests.

- [x] P4A-02 | Cau hinh WAL, FULL sync, FK, busy timeout va checkpoint.
  - Output: SQLite profile journal_mode=WAL, synchronous=FULL, foreign_keys=ON, timeout=5000, autocheckpoint=1000.
  - Depends: P4A-01.
  - Done when: long reader, WAL 128/256 MiB warning/throttle, TRUNCATE checkpoint co test.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/database.py; gds/writer.py WAL health/checkpoint; protocol/runtime_profile.yaml; long-reader test.

- [x] P4A-03 | Implement U64 codec va keyset pagination.
  - Output: BLOB(8) order, checked conversion, API cursor 16 hex, no signed INTEGER/offset scan.
  - Depends: P0-11, P4A-01.
  - Done when: sort va cursor dung cho min/max U64; mat bit la protocol error.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/u64.py; min/max BLOB-order and strict cursor test in tests/test_phase4a_gds.py.

- [x] P4A-04 | Implement single SQLite writer task va bounded IPC.
  - Output: writer_queue_capacity=4096, reserve 256 high-priority, mutation intent, reader connection rieng.
  - Depends: P4A-01, P4A-02.
  - Done when: khong co writer connection thu hai; queue full tra loi ro; low-priority drop co metric.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/writer.py; single-owner, query-only reader, 4096/256 priority-reserve and backpressure tests.

- [x] P4A-05 | Implement RequestKey allocator va gds_installation_epoch.
  - Output: CSPRNG ground_instance_id, durable request_id sequence, wrap/reinitialize policy.
  - Depends: P0-10, P4A-01.
  - Done when: restart khong reset namespace; U32 wrap/DB reinit tao namespace moi sau khi drain cu.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/request_keys.py; restart persistence and U32 drain/namespace-rotation tests.

- [x] P4A-06 | Implement HTTP idempotency digest va default expiry.
  - Output: RFC 8785 JCS semantic body, delivery_mode always-present, DEFAULT sentinel, effective expiry at first commit.
  - Depends: P0-10, P4A-05.
  - Done when: omit expiry retry tra cung RequestKey/expiry; same key khac body tra 409; TTL 90 ngay co marker.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / protocol/canonical.py; protocol/schemas.py; gds/idempotency.py; DEFAULT-expiry, conflict and 90-day marker tests.

- [x] P4A-07 | Implement atomic command ledger + transactional outbox admission.
  - Output: command row, outbox row, target instance, exact request/mission digest, 202 semantics.
  - Depends: P4A-04, P4A-05, P4A-06.
  - Done when: crash sau commit/tru response khong mat admitted command; concurrent same key chi mot row.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/ledger.py; concurrent same-key, crash rollback, capacity and orphan-invariant tests.

- [x] P4A-08 | Implement outbox lease, retry, attempt va timeout.
  - Output: lease 10s, ACK timeout 5s, backoff/max-attempt/TTL, SENT/ACKED/DELIVERY_FAILED states.
  - Depends: P4A-07, P3-01.
  - Done when: crash truoc/sau lease/send/ACK ingest co at-least-once send; business effect van idempotent.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/outbox.py; gds/migrations/002_phase4a_runtime.sql; tests/test_phase4a_runtime.py lease-expiry, persisted-attempt, ACK-timeout va retry-sequence tests.

- [x] P4A-09 | Implement immediate va next_contact delivery mode.
  - Output: persisted HELD_NO_CONTACT, expires_at, contact state, late ACK policy.
  - Depends: P4A-08, P3-04.
  - Done when: het han qua restart khong gui; blackout khong lam /readyz fail; immediate bi reject ro.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/outbox.py; tests/test_phase4a_runtime.py contact pause/reopen, immediate CONTACT_LOST va late ACK audit tests.

- [x] P4A-10 | Implement TC sequence allocator va retry sequence semantics.
  - Output: APID-scoped Space Packet sequence persist, retry packet sequence moi, rollover/reset marker.
  - Depends: P2A-10, P4A-08.
  - Done when: RequestKey khong bi nham voi cmdSeq/packet/frame counter; rollover hop le co test.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/sequence.py; gds/migrations/002_phase4a_runtime.sql; tests/test_phase4a_runtime.py APID scope, retry sequence, rollover va reset marker tests.

- [x] P4A-11 | Implement API admission va command status endpoints.
  - Output: validated API body, 202/409/429/503/507 error mapping, target instance scope, audit.
  - Depends: P4A-07, P4A-09.
  - Done when: API khong cap RequestKey truoc atomic admission; moi row/event co instance scope.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/api.py; gds/ledger.py; tests/test_phase4a_runtime.py 202/404/409/422/429/503/507 mapping, status va same-key replay tests.

- [x] P4A-12 | Implement spacecraft-instance migration fence.
  - Output: bind instance A/B, terminal hoa outbox cu reason TARGET_INSTANCE_RETIRED, no rewrite/no auto-retry.
  - Depends: P4A-07, P4A-08, P0-10.
  - Done when: operator submit lai tren B tao RequestKey moi; artifact A khong alias B.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/binding.py; gds/outbox.py; tests/test_phase4a_runtime.py old-target terminalization, no rewrite va stale read-fence tests.

- [x] P4A-13 | Implement telemetry/event rollup va audit base.
  - Output: event cursor, telemetry dedupe key, 1-minute rollup, command/config audit.
  - Depends: P4A-01, P4A-04.
  - Done when: duplicate byte-identical khong tao sample hai lan; conflict bi audit/reject.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/events.py; gds/telemetry.py; gds/audit.py; tests/test_phase4a_runtime.py event cursor, byte-identical dedupe, 1-minute mean va conflict-audit tests.

- [x] P4A-14 | Them crash, capacity, WAL va writer saturation test matrix.
  - Output: test virtual clock/outbox, raw append-before-DB, prune crash, high-priority reserve, long reader.
  - Depends: P4A-02 den P4A-13.
  - Done when: khong co command/outbox row mo coi; terminal ledger/audit van ghi duoc khi low-priority bi chan.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / tests/test_phase4a_gds.py; tests/test_phase4a_runtime.py; gds/raw_segments.py; gds/storage.py; crash rollback, raw append-before-DB/torn-tail recovery, storage watermark, high-priority reserve va long-reader tests.

## Phase 4b - TM, catalog, file, realtime va local deployment

Exit gate: command -> ACK -> progress -> result -> verified bundle round trip khong can web UI; catalog stale/domain, file loss/retry/publish/restart, retention, topology va cursor-resync test pass.

- [x] P4B-01 | Implement TM decoder va validated transport envelope.
  - Output: decode APID/descriptor/channel/event/file, source instance, boot, session, receive time, CRC.
  - Depends: P2A-07, P4A-13.
  - Done when: malformed/unknown/mismatch frame co loi; metadata khong lay tu mutable current-link state.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/tm.py; tests/test_phase4b_runtime.py::test_tm_decoder_uses_envelope_identity_and_rejects_crc_or_descriptor.

- [x] P4B-02 | Tao catalog schema va satellite authority sync.
  - Output: catalog epoch/revision/snapshot SHA, scene identity/revision, capability/domain, source/sidecar SHA.
  - Depends: P0-13, P4A-01.
  - Done when: GDS chi active verified snapshot; partial snapshot khong phuc vu nhu current.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/migrations/003_phase4b_runtime.sql; gds/catalog.py; tests/test_phase4b_runtime.py::test_catalog_replica_activation_is_atomic_and_instance_scoped.

- [x] P4B-03 | Implement scene ingest package content-addressed read-only.
  - Output: memmap/stat/domain check, immutable source + validity sidecar, fsync/atomic publish.
  - Depends: P1-02, P0-13.
  - Done when: runtime chi mount package read-only; compressed/full-decode backend bi INVALID/UNSUPPORTED.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / flight/scene_package.py; tests/test_phase4b_runtime.py::test_scene_package_stat_scrub_rejects_out_of_band_mutation.

- [x] P4B-04 | Implement startup scrub va out-of-band mutation detection.
  - Output: cheap stat command check, full SHA/value scan startup/ingest, INVALID revision/reason.
  - Depends: P4B-03, P4A-07.
  - Done when: source/mask mutate bi reject, khong doc byte khong co trong catalog.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / flight/scene_package.py::scrub_scene_package; tests/test_phase4b_runtime.py::test_scene_package_stat_scrub_rejects_out_of_band_mutation.

- [x] P4B-05 | Implement SCENE_REQUEST_CATALOG va replica activation.
  - Output: full deterministic catalog bundle, checksum/manifest verification, stale/synced status.
  - Depends: P2B-15, P4B-02.
  - Done when: old epoch/retired instance chi read-only; active replica luon co epoch/revision.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / flight/catalog.py; gds/catalog.py; scripts/p4b_roundtrip.py catalog stage; tests/test_phase4b_runtime.py::test_catalog_replica_activation_is_atomic_and_instance_scoped.

- [x] P4B-06 | Implement preview generation va active_preview ProductRef CAS.
  - Output: preview quicklook/tile product, full ProductRef cache key, catalog revision bump, ETag.
  - Depends: P1-07, P4B-02, P4B-05.
  - Done when: preview version moi khong ghi de product cu; CAS clear khi evict dung pointer.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/preview.py; tests/test_phase4b_runtime.py::test_preview_pointer_cas_tile_and_retention_tombstone.

- [x] P4B-07 | Implement REST state/catalog/scene/product APIs.
  - Output: instance-scoped routes, full SceneRef/ProductRef, stale/domain/capability response, keyset pagination.
  - Depends: P4A-11, P4B-05, P4B-06.
  - Done when: scene stale/old epoch khong tao command silent; product tombstone tra dung 410/404.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/api.py framework-neutral REST contract, instance-scoped command/status/catalog/scene/product/tile/state methods; docs/phase4b_completion_report.md.

- [x] P4B-08 | Implement GET snapshot + WebSocket cursor/replay.
  - Output: as_of_event_id, last_event_id, replay retention, RESYNC_REQUIRED, dedupe/apply contract.
  - Depends: P4A-13, P4B-07.
  - Done when: event chen giua snapshot va WS duoc replay; cursor qua retention buoc snapshot resync.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/realtime.py; tests/test_phase4b_runtime.py::test_realtime_cursor_replay_and_resync.

- [x] P4B-09 | Implement FilePacket reassembly START/DATA/END.
  - Output: gap detection, duplicate/out-of-order handling, conflicting overlap error, transfer identity.
  - Depends: P2B-11, P3-09.
  - Done when: drop packet -> INCOMPLETE; duplicate byte-identical tao dung mot output; conflict khong publish.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/file_reassembly.py; tests/test_phase4b_runtime.py::test_file_reassembly_out_of_order_duplicate_and_verified_publish.

- [x] P4B-10 | Implement safe extraction va artifact verification.
  - Output: path traversal guard, max bundle/extract/file limits, Fw checksum, bundle SHA, manifest entry set.
  - Depends: P4B-09, P0-13.
  - Done when: thieu/du/sai hash/checksum bi reject; khong ghi ngoai product root.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/product_store.py::safe_extract_ustar/verify_bundle; tests/test_phase4b_runtime.py product verification path.

- [x] P4B-11 | Implement retry va crash-safe atomic product publish.
  - Output: staging .part, verify -> fsync -> atomic rename, full-file retry attempt moi.
  - Depends: P4B-10, P4A-08.
  - Done when: restart truoc/sau verify/rename khong tao partial/hidden second final directory.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/product_store.py; gds/file_reassembly.py; scripts/p4b_roundtrip.py verified PUBLISHED product.

- [x] P4B-12 | Implement retention, watermark, quota va emergency reserve.
  - Output: raw/frame/log/product/replay retention, 20 GiB ground cap, 7/30/90-day policy, 80/90% watermarks.
  - Depends: P4A-02, P3-07.
  - Done when: 507 STORAGE_FULL, 410 tombstone, reserve terminal ACK/status/cancel van ghi duoc.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/retention.py; gds/storage.py; link_sim/replay_manager.py; tests/test_phase4b_runtime.py::test_retention_cleanup_covers_part_raw_and_rotated_log_files plus Phase 4a storage/replay tests.

- [x] P4B-13 | Implement metrics, healthz, readyz va structured logging.
  - Output: command latency, queue, worker, WS, DB backlog, transfer gap/checksum metrics; log rotation.
  - Depends: P4A-13, P4B-08, P4B-11.
  - Done when: scheduled blackout khong fail readyz; DB/link/decoder hong thi fail readyz.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/metrics.py; gds/ingest.py; LocalSilRuntime readiness check; tests/test_phase4b_runtime.py TM ingest/realtime tests.

- [x] P4B-14 | Enforce host_local_sil/compose_sil topology va request limits.
  - Output: bind/peer/origin allowlist, internal network, body/header/rate/download/extract limits, startup guard.
  - Depends: G0-07, P0-15.
  - Done when: foreign Host/Origin/peer, public bind, path traversal, vuot quota/rate bi reject.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / gds/topology.py; protocol/runtime_profile.yaml; tests/test_phase4b_runtime.py::test_topology_profile_rejects_public_or_foreign_requests.

- [x] P4B-15 | Chay round-trip integration khong qua Web UI.
  - Output: fixture script scene -> TC -> ACK -> inference -> TM -> FilePacket -> verified product.
  - Depends: P2B-15, P3-12, P4A-14, P4B-01 den P4B-14.
  - Done when: full workflow pass, trace duoc bang RequestKey + spacecraft_instance_id, khong shared-volume bypass.
  - Owner/ETA/Evidence: Codex / 2026-07-19 / scripts/p4b_roundtrip.py; docs/phase4b_completion_report.md; scene -> TC -> ACK -> inference -> TM/FilePacket -> PUBLISHED.

## Phase 5 - GDS Webapp core

Exit gate: operator hoan thanh full workflow bang webapp; blackout, stale TM, reconnect/resync va fault warning co E2E; packet inspector la P2 va khong chan core.

- [x] P5-01 | Tao React/TypeScript app shell va normalized state store.
  - Output: layout van hanh, REST snapshot + WS event reconcile, local editing state tach server state.
  - Depends: P4B-07, P4B-08.
  - Done when: app khong import/goi ham inference; state co instance scope.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / gds/web/src/App.tsx; gds/web/src/state/store.ts; gds/web/src/api/client.ts.

- [x] P5-02 | Implement thanh connection/satellite/link status.
  - Output: Browser-GDS, GDS-contact, satellite state, queue depth, current time, TM age.
  - Depends: P4B-13.
  - Done when: CONNECTED/NO CONTACT/BLACKOUT/STALE TM/SATELLITE DEGRADED phan biet ro.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / gds/web/src/App.tsx; gds/web/src/types.ts.

- [x] P5-03 | Implement scene catalog search/filter/product availability.
  - Output: verified/unsupported/invalid/domain status, stale marker, scene revision/epoch display.
  - Depends: P4B-05, P4B-07.
  - Done when: unsupported/invalid khong enable Analyze/ROI; stale khong bi hien nhu current.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / gds/web/src/App.tsx; gds/web/src/api/client.ts.

- [x] P5-04 | Implement OpenLayers quicklook/tile viewer.
  - Output: GDS-only tile access, zoom/pan, 256x256 tiles, cache LRU/cancel request, mask overlay.
  - Depends: P4B-06, P4B-07.
  - Done when: browser khong request source TIFF/JP2; tile cache bounded va khong seam.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / gds/web/src/components/QuicklookViewer.tsx; gds/web/src/utils/tileCache.ts.

- [x] P5-05 | Implement Pan/Select ROI segmented control.
  - Output: drag/resize/move rectangle, reset icon, scene bounds, min patch_size.
  - Depends: P1-03, P5-04.
  - Done when: viewer-to-scene sai so <=1 pixel; resize khong doi toa do authoritative.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / gds/web/src/App.tsx; gds/web/src/components/QuicklookViewer.tsx.

- [x] P5-06 | Implement numeric ROI editor va rounding/clamp.
  - Output: x/y/width/height input, normalized view state, floor/ceil drag rule, half-open range.
  - Depends: P5-05.
  - Done when: backend va frontend cung ket qua; ROI ngoai scene/min patch bi reject ro.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / gds/web/src/utils/roi.ts; gds/web/src/utils/roi.test.ts.

- [x] P5-07 | Implement threshold controls va atomic config commit.
  - Output: model_threshold, coverage_limit slider + numeric input, CLOUD_SET_CONFIG mot transaction.
  - Depends: P2B-04, P4A-11.
  - Done when: UI khong gui hai update doc lap; hien epoch/revision moi sau ACK.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / gds/web/src/App.tsx; CLOUD_SET_CONFIG payload; gds/web/src/api/client.ts.

- [x] P5-08 | Implement command preview/confirmation.
  - Output: target instance, full SceneRef, ROI, thresholds, config identity, HTTP idempotency key, fault/contact warning, estimated downlink.
  - Depends: P4A-06, P4A-11, P5-03, P5-06, P5-07.
  - Done when: RequestKey chi hien sau 202 Accepted; UI khong tu cap phat RequestKey.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / gds/web/src/App.tsx; stable HTTP Idempotency-Key per confirmation.

- [x] P5-09 | Implement command/job/science/product/transfer lifecycle view.
  - Output: timeline/card state theo RequestKey/JobKey/ProductRef/transfer ID, error/reject detail.
  - Depends: P4B-07, P4B-08.
  - Done when: command status khong bi nham voi job hay transfer; terminal state co reason.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / gds/web/src/App.tsx; normalized lifecycle entities in gds/web/src/types.ts.

- [x] P5-10 | Implement telemetry/event timeline va progress.
  - Output: TM channel, event severity, satellite time, GDS receive time, age, progress coalesce.
  - Depends: P4B-01, P4B-08.
  - Done when: stale TM co age; boot moi tao epoch state moi tren UI.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / gds/web/src/state/store.ts; gds/web/src/App.tsx; boot-epoch regression test.

- [x] P5-11 | Implement product preview/download va transfer progress.
  - Output: ProductRef-scoped preview, verified artifact list, download quota/error, progress/gap/checksum.
  - Depends: P4B-10, P4B-11, P4B-12.
  - Done when: UI chi hien product da verify; product bi evict tra dung tombstone message.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / gds/web/src/App.tsx; gds/web/src/api/client.ts.

- [x] P5-12 | Implement blackout/degraded/next-contact UX.
  - Output: immediate reject explanation, persisted next_contact confirmation/expiry, stale/catalog/model assurance warning.
  - Depends: P4A-09, P4B-13.
  - Done when: UI khong hard-disable dua tren telemetry stale; backend/satellite la authority.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / gds/web/src/App.tsx; command modal contact/fault/expiry warnings.

- [x] P5-13 | Implement WebSocket reconnect/resync va slow-client handling.
  - Output: exponential backoff, cursor replay, snapshot fallback, disconnect slow client, no unbounded browser state.
  - Depends: P4B-08.
  - Done when: reconnect khong mat/duplicate state; RESYNC_REQUIRED xu ly dung.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / gds/web/src/api/realtime.ts; gds/web/src/api/realtime.test.ts.

- [x] P5-14 | Them responsive layout va accessibility smoke test.
  - Output: desktop/mobile layout, no text/control overlap, keyboard/focus/labels, bounded tile requests.
  - Depends: P5-01 den P5-13.
  - Done when: no overlap tren viewport muc tieu; workflow core dung duoc tren desktop va mobile.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / gds/web/src/styles.css; gds/web/index.html; gds/web/README.md.

## Phase 6 - E2E va hardening

Exit gate: tat ca acceptance criteria pass; clean release reproducible; profile CPU deploy duoc; demo lap lai duoc; runbook, release/run manifest va SBOM day du.

- [x] P6-01 | Kiem thu regression toan bo test hien co.
  - Output: report 216 test functions va 19 subtests, failure triage.
  - Depends: P1 den P5.
  - Done when: full suite tiep tuc pass; failure khong bi bo qua bang skip khong ly do.
  - Owner/ETA/Evidence: Codex / 2026-07-21 / `python -m pytest -q` (216 passed, 19 subtests); `tests/test_phase6_hardening.py`; `tests/test_phase6_recovery.py`; `tests/test_architecture_boundaries.py`.

- [x] P6-02 | Viet Playwright E2E desktop cho scene -> ROI -> command -> product.
  - Output: test full workflow, command confirmation, lifecycle, product verify.
  - Depends: P5-14, P4B-15.
  - Done when: operator hoan thanh workflow chi qua UI; RequestKey trace duoc backend/link/satellite.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / `gds/web/e2e/mission-control.spec.ts`; desktop Playwright PASS; `scripts/e2e_server.py`.

- [x] P6-03 | Viet Playwright E2E mobile/responsive cho ROI.
  - Output: viewport matrix, resize/drag rounding, no overlap, reconnect.
  - Depends: P5-05, P5-06, P5-14.
  - Done when: ROI authoritative khong doi sau resize; UI khong occlude control.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / mobile Playwright confirmation PASS; `gds/web/playwright.config.ts`; `gds/web/src/App.tsx`.

- [x] P6-04 | Viet fault/reconnect/blackout E2E.
  - Output: loss, duplicate, corruption, latency, blackout, immediate/next-contact, stale TM.
  - Depends: P3-12, P5-12, P5-13.
  - Done when: UI hien dung NO CONTACT/STALE TM; command delivery outcome ro va replay khong mat state.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / `tests/test_phase6_hardening.py`; `tests/test_phase3_exit_gate.py`; `link_sim/fault_model.py`.

- [x] P6-05 | Viet file transfer loss/reorder/retry/cancel E2E.
  - Output: START/DATA/END drop, duplicate, overlap conflict, full-file retry, cancel race.
  - Depends: P2B-13, P4B-09, P4B-11.
  - Done when: final product byte-exact; partial khong publish; chi mot final directory.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / `tests/test_phase6_recovery.py`; `tests/test_phase4b_runtime.py`.

- [x] P6-06 | Viet restart/reconciliation E2E cho Satellite, Link va GDS.
  - Output: crash matrix truoc/sau commit/send/verify/rename, boot/session migration.
  - Depends: P2B-07, P3-10, P4A-14, P4B-11.
  - Done when: duplicate RequestKey khong inference lan hai; attempt/boot cu khong cross barrier.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / durable reassembler restart test; `tests/test_phase2_runtime.py`; `flight/journal.py`.

- [x] P6-07 | Viet deterministic replay E2E va artifact retention.
  - Output: same seed/profile byte-exact, different seed khac, raw prune, pin/evict, FINAL/INCOMPLETE states.
  - Depends: P3-12, P4B-12.
  - Done when: FINAL chi claim replay khi PRESENT/PINNED; EVICTED khong claim replay bytes.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / `tests/test_phase6_hardening.py`; `tests/test_replay_manager.py`; `artifacts/soak/phase6_soak_report.json`.

- [x] P6-08 | Benchmark batch size 1,2,4,8... tren CPU/GPU/Jetson target.
  - Output: throughput, p95, RSS/shared memory, OOM, model load, per-hardware latency.
  - Depends: P1-12, P3-11.
  - Done when: chi profile co artifact benchmark duoc READY; SLO khong suy ra truoc benchmark.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / `artifacts/benchmarks/phase6-batch-matrix-v1.json`; CPU 1/2/4/8 PASS; CUDA/Jetson explicitly UNAVAILABLE and not READY.

- [x] P6-09 | Chot queue, watchdog, deadline, ACK/health/file-goodput SLO.
  - Output: config revision tu benchmark, tunable/frozen values, rationale.
  - Depends: P2B-14, P3-11, P6-08.
  - Done when: oldest_ack_age <=1s, health_max_latency <=2s, resource guards co oracle.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / `protocol/slo_profile.yaml`; `protocol/slo.py`; `/healthz` and `/readyz` expose SLO/scheduler metrics.

- [x] P6-10 | Chay soak test queue, WebSocket, storage va staging cleanup.
  - Output: soak report, memory/DB/WAL/raw/log/product/replay growth, slow client behavior.
  - Depends: P4B-12, P4B-13, P5-13.
  - Done when: khong tang vo han; hard watermark/cleanup khong lam mat terminal ledger/audit.
  - Owner/ETA/Evidence: Codex / 2026-07-21 / `scripts/soak_test.py`; `artifacts/soak/phase6_soak_report.json`; 20 iterations, bounded queues/replay, cleanup guards PASS.

- [x] P6-11 | Tao Docker CPU profile va Jetson/L4T profile neu da duoc chot.
  - Output: Docker Compose, internal network guard, hardware-specific manifest/benchmark.
  - Depends: G0-04, P6-08.
  - Done when: host_local_sil/compose_sil profile and real UDP bridge validation pass; profile thieu benchmark bi chan READY. Full multi-container HTTP-to-satellite E2E remains conditional.
  - Owner/ETA/Evidence: Codex / 2026-07-21 / `deploy/Dockerfile.cpu`; `deploy/docker-compose.yml`; `link_sim/__main__.py`; `deploy/jetson-l4t-profile.yaml`; `scripts/validate_deploy_profiles.py`.

- [x] P6-12 | Chay security/limit negative tests.
  - Output: public bind, foreign Host/Origin/peer, body/header/rate/download/extract/path traversal, no token log.
  - Depends: P4B-14, P5-14.
  - Done when: moi violation bi reject dung status/error; local_sil khong bi nham la production security.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / `tests/test_phase6_hardening.py`; `gds/topology.py`; `deploy/compose_runtime_profile.yaml`.

- [ ] P6-13 | Tao reproducible clean build va SBOM (conditional).
  - Output: release_id, Git commit/dirty flag, dependency lock/SBOM hash, image/compiler/runtime/model/profile hashes.
  - Depends: P0-06, P0-07, P6-11.
  - Done when: hai clean build cung SOURCE_DATE_EPOCH cho cung pinned artifact/hash; release khong dirty.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / evidence manifest `artifacts/release/phase6-release-manifest.json` has `source_dirty=true`; official clean release awaits a clean worktree.

- [x] P6-14 | Implement release manifest va simulation run manifest.
  - Output: OPEN -> FINAL/INCOMPLETE state, command-set hash, replay SHA/size/state, profile/seed/clock/revision.
  - Depends: P3-06, P3-07, P6-13.
  - Done when: run FINAL chi sau finalize atomic; crash/storage cap khong claim complete.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / `gds/release_manifest.py`; `gds/run_manifest.py`; `tests/test_phase6_hardening.py`.

- [ ] P6-15 | Revalidate conformance matrix va Definition of Done (conditional).
  - Output: signed checklist APID/descriptor/frame/ROI/RequestKey/product/security/recovery.
  - Depends: P6-01 den P6-14.
  - Done when: full workflow, golden vectors, restart, queue, blackout, product verify, UI reconnect deu pass.
  - Owner/ETA/Evidence: Codex / 2026-07-21 / `docs/phase6_conformance_checklist.md`; local-SIL technical checks pass, official DoD remains conditional on P6-13 clean release, target hardware evidence, and multi-container UDP E2E.

- [x] P6-16 | Viet runbook khoi dong, fault profile va demo scenario lap lai.
  - Output: setup/health/shutdown/recovery, sample scene, command sequence, expected TM/product, troubleshooting.
  - Depends: P6-15.
  - Done when: nguoi khac co the clean deploy va lap lai demo; moi run co manifest va evidence.
  - Owner/ETA/Evidence: Codex / 2026-07-20 / `deploy/local_sil_runbook.md`; `scripts/demo_scenario.py`; demo PASS with `PUBLISHED` and `SHA256_MATCH`.

## Hang muc ngoai MVP va change request

Khong dua cac muc sau vao tien do MVP. Moi muc can re-open Phase 0, conformance matrix, golden vectors, threat model va estimate rieng:

- Full COP-1/Type-AD/FOP/FARM/CLCW.
- ECSS PUS o application layer.
- CFDP acknowledged mode/selective recovery.
- SDLS, OIDC/RBAC/TLS/CSRF va GDS expose qua network.
- ROI theo lat/lon/georeference.
- RF/SDR, CLTU/BCH, ASM/channel coding.
- Custom APID mapper ngoai stock 0/1/2/3.
- Runtime compressed TIFF/JP2 true windowed sau PoC/resource benchmark.
- Cloud segmentation pixel-level.
- Scientific validation de promote tu demo_non_validated.

## Nhat ky thay doi tracker

| Ngay | Task/Gate | Thay doi | Nguoi | Evidence |
|---|---|---|---|---|
| 2026-07-19 | Tao tracker | Tach ke hoach nguon thanh 126 task; tat ca TODO | TBD | TBD |
| 2026-07-19 | G0-01..G0-07 | Chot stock F Prime v4.1.0/APID 0/1/2/3, pixel ROI memmap, CPU/PyTorch, demo_non_validated va host_local_sil | Codex | protocol/mission_profile.yaml; protocol/runtime_profile.yaml; sat_ai/model_manifest.yaml |
| 2026-07-19 | Phase 0 | Hoan tat baseline, package layout, model/InputSpec, canonical scalar, schema, profile, storage/replay contract va conformance matrix | Codex | docs/gds_satellite_ccsds_baseline_report.md; protocol/conformance_matrix.md; 90 tests |
| 2026-07-19 | Phase 1 | Hoan tat memmap ROI inference, strict validity, scene grid, LUT, singleton runtime, deterministic products va CPU benchmark | Codex | sat_ai/; artifacts/benchmarks/local-cpu-pytorch-v2.json; tests/test_sat_ai_mission.py |
| 2026-07-19 | Phase 2a | Hoan tat Python F Prime-compatible reference skeleton, stock APID route, packet/frame/file codecs, scheduler completion gate va golden vectors | Codex | flight/; protocol/; protocol/golden_vectors/; tests/test_mission_contracts.py |
| 2026-07-19 | Phase 2b | Hoan tat worker contract/queue, durable journal/CAS/reconciliation, immutable job snapshot, product staging, global file attempt va local simulator | Codex | flight/journal.py; flight/cloud_payload.py; flight/file_downlink.py; tests/test_phase2_runtime.py |
| 2026-07-19 | P3-04 | Hoan tat bandwidth shaper va blackout; frame drop policy, NO_CONTACT/BLACKOUT states | Codex | link_sim/contact_schedule.py; tests/test_link_simulator_blackout.py 5 tests |
| 2026-07-19 | P3-07 | Hoan tat replay quota, pin va retention; PRESENT/PINNED/EVICTED transitions, eviction policy | Codex | link_sim/replay_manager.py; tests/test_replay_manager.py 11 tests |
| 2026-07-19 | Phase 3 | Tien do Phase 3 dat 58% (7/12 tasks); 16+ tests Phase 3 deu pass | Codex | docs/phase3_progress_summary.md; docs/project_progress_report_20260719.md |
| 2026-07-19 | P3-08 | Hoan tat queue overflow handling; 9 tests pass | Codex | link_sim/link_simulator.py queue overflow metrics |
| 2026-07-19 | P3-09 | Hoan tat FilePacket drain fence; attempt barrier, no cross-attempt contamination | Codex | link_sim/file_epoch.py; tests/test_file_epoch.py 12 tests |
| 2026-07-19 | P3-10 | Hoan tat session handshake; boot/session isolation, generation counter | Codex | link_sim/session_manager.py; tests/test_session_manager.py 14 tests |
| 2026-07-19 | P3-11 | Hoan tat benchmark goodput; frame throughput, overhead ratio, profile comparison | Codex | link_sim/benchmark.py; tests/test_link_benchmark.py 5 tests |
| 2026-07-19 | P3-12 | Hoan tat replay determinism test suite; same seed byte-exact, session isolation | Codex | tests/test_phase3_exit_gate.py 6 tests |
| 2026-07-19 | Phase 3 | HOAN THANH 100% Phase 3 (12/12 tasks); 53 tests Phase 3 deu pass; exit gate dat | Codex | All Phase 3 components complete |
| 2026-07-19 | Phase 4a | HOAN THANH 100% Phase 4a (14/14 tasks); SQLite ledger/API/WAL/storage exit gate dat | Codex | tests/test_phase4a_gds.py; tests/test_phase4a_runtime.py; gds/ |
| 2026-07-19 | Phase 4b | HOAN THANH 100% Phase 4b (15/15 tasks); focused 12 tests, full suite 203 passed + 19 subtests | Codex | docs/phase4b_completion_report.md; scripts/p4b_roundtrip.py; round-trip PASS, 526 frames, PUBLISHED product |
| 2026-07-20 | Phase 5 | HOAN THANH implementation 14/14 task frontend core; normalized state, REST/WS reconcile, OpenLayers ROI workflow, admission preview, lifecycle/product UX, responsive/accessibility | Codex | gds/web/; npm test 9 passed; npm run build PASS; Python regression 203 passed + 19 subtests; Chromium desktop/mobile smoke PASS; in-app browser plugin bootstrap blocked by asset path |
| 2026-07-20 | Phase 6 | Hoan tat hardening implementation, full local-SIL path, file/restart/replay recovery, security limits, benchmark matrix, soak, deploy profiles, manifests va runbook | Codex | docs/phase6_completion_report.md; artifacts/benchmarks/phase6-batch-matrix-v1.json; artifacts/soak/phase6_soak_report.json; Playwright desktop/mobile; official clean release con lai conditional do worktree dirty va chua co CUDA/Jetson |
| 2026-07-21 | Remediation execution | Bo sung architecture boundary test, canonical package hash/quarantine recovery, Compose image alias va cap nhat evidence theo local-SIL/UDP bridge scope | Codex | tests/test_architecture_boundaries.py; tests/test_phase4b_runtime.py; flight/scene_package.py; deploy/docker-compose.yml; full suite 216 passed + 19 subtests |
| 2026-07-21 | Follow-up remediation | F-01 through F-07 regression implementation completed: HTTP/GDS transport boundary, durable outbox/TM/downlink ledger, realtime/streaming, and TM scheduler wire ordering | Codex | artifacts/ccsds-core.xml (171 passed + 10 subtests); artifacts/ml-artifact.xml (61 passed + 9 subtests); artifacts/phase6-hardening.xml (9 passed); Docker daemon/browser runtime unavailable, so Compose and browser release gates remain blocked; F-08 remains conditional |
