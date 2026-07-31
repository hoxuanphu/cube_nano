# Task tracker: asymmetric penalty cho cloud segmentation

> Nguon: [segmentation_loss_asymmetric_penalty_plan.md](segmentation_loss_asymmetric_penalty_plan.md)
>
> Ngay tao: 2026-07-31
>
> Branch: `plan/asymmetric-cloud-loss`
>
> Pham vi: them class-weighted Cross Entropy vao baseline `CrossEntropy + SoftDice` de giam loi bo sot may trong semantic segmentation anh ve tinh.

Tai lieu nay tach plan thanh cac task co the giao viec. Task chi duoc danh dau `DONE` khi co bang chung trong cot Evidence.

## Cach cap nhat

- `[ ]` = TODO; `[>]` = DOING; `[x]` = DONE; `[!]` = BLOCKED; `[-]` = DEFERRED.
- Status dung mot trong `TODO`, `DOING`, `BLOCKED`, `DONE`, `DEFERRED`.
- Moi thay doi ablation phai giu nguyen dataset split, InputSpec, architecture, optimizer policy va `cross_entropy_weight=1.0`, `dice_weight=1.0`; chi thay doi `cloud_class_weight`.
- Multi-seed dung de do bien dong cua qua trinh train; scene bootstrap dung de uoc luong do bat dinh cua metric. Hai cach nay khong mac nhien thay the cho nhau.
- Neu task bi `BLOCKED`, ghi ro blocker, nguoi quyet dinh va dieu kien mo khoa.

## Dashboard

| Phase | Noi dung | Phu thuoc chinh | Tien do |
|---|---|---|---:|
| L0 | Contract, class distribution, acceptance criteria va baseline | Khong | 2/5 |
| L1 | Weighted loss, config, logging va unit test | L0 | 4/5 |
| L2 | Ablation, multi-seed, threshold calibration va uncertainty | L1 | 0/4 |
| L3 | Chon cau hinh, khoa release va test evaluation | L2 | 0/3 |
| L4 | Tversky fallback neu weighted CE khong dat muc tieu | L3 fail criteria | 0/2 |
| **Tong** |  |  | **6/19** |

## Gate L0 - contract va baseline

Exit gate: class mapping, invalid policy, class distribution, acceptance criteria va baseline report duoc chot; baseline co du metric, loss components, checkpoint, config va provenance.

- [x] **L0-01 | Khoa task va class contract.**
  - Output: class `0=surface/khong may`, class `1=may`, `255=invalid`; muc tieu uu tien la giam false negative/`false_clear_rate`.
  - Depends: Khong.
  - Done when: contract duoc ghi trong report va khop preprocessing, loss, evaluation.
  - Owner/ETA/Evidence: `Codex` / `2026-07-31` / `src/losses.py`, `src/data/segmentation_dataset.py`, `src/eval_segmentation.py`, `artifacts/segmentation_loss_asymmetric_penalty/l1_l2_smoke.json`.

- [!] **L0-02 | Khoa acceptance criteria truoc ablation.**
  - Output: gia tri chinh thuc cho `max_false_clear_rate`, floor cua `cloud_precision`, muc suy giam Dice/IoU cho phep, va thu tu uu tien metric.
  - Depends: L0-01.
  - Done when: khong con dung cac gia tri vi du nhu `5%`, `85%` hoac `2 diem phan tram` neu chua duoc stakeholder phe duyet.
  - Status: `BLOCKED` - `sat_ai/acceptance_profile.yaml` approves IoU/Dice/recall/false-clear/coverage gates, but no stakeholder-approved cloud precision floor or relative Dice-drop limit exists. The harness records these as pending instead of treating examples as official criteria.
  - Owner/ETA/Evidence: `Codex` / `2026-07-31` / `sat_ai/acceptance_profile.yaml`, `src/segmentation_experiment.py` (`acceptance_criteria.pending_stakeholder_approval`).

- [!] **L0-03 | Thong ke class distribution tren training set.**
  - Output: so pixel va ty le `surface:cloud`, chi dem pixel valid; bo qua `target=255` va `validity_mask=False`. Neu co the, bao cao them phan bo theo scene.
  - Depends: L0-01.
  - Done when: report co counts, ratios, dataset split ID va phuong phap dem; ket qua duoc dung de dien giai cloud weight.
  - Status: `BLOCKED` - frozen processed train/validation directories are absent; no valid class counts are fabricated from unrelated TIFF/JP2 files. The audit implementation is ready and counts only `validity & (mask != 255)`.
  - Owner/ETA/Evidence: `Codex` / `2026-07-31` / `artifacts/segmentation_loss_asymmetric_penalty/l2_ablation/ablation_manifest.json`, `src/segmentation_experiment.py:audit_dataset`.

- [!] **L0-04 | Chay va luu baseline.**
  - Output: baseline voi `cloud_class_weight=1.0`, seed `42`, checkpoint, training config, git commit hash, metric report, threshold va loss curves.
  - Depends: L0-01, L0-02, L0-03.
  - Done when: report co `cloud_recall`, `false_clear_rate`, `cloud_precision`, `cloud_dice`, `cloud_iou`, coverage error, epoch best validation loss va provenance day du.
  - Status: `BLOCKED` - baseline cannot be trained/evaluated without the frozen split and its lineage. Existing checkpoints under `chechpoint/` have no matching training/validation JSON report and are not accepted as evidence.
  - Owner/ETA/Evidence: `Codex` / `2026-07-31` / `artifacts/segmentation_loss_asymmetric_penalty/l2_ablation/ablation_manifest.json`.

- [x] **L0-05 | Xac nhan quy trinh threshold va uncertainty.**
  - Output: quy uoc calibrate pixel threshold tren validation va quy uoc multi-seed/scene bootstrap.
  - Depends: L0-02, L0-04.
  - Done when: threshold validation khong dung test; phan biet ro train variance voi scene sampling uncertainty.
  - Owner/ETA/Evidence: `Codex` / `2026-07-31` / `src/eval_segmentation.py:calibrate_validation_predictions`, `src/eval_segmentation.py:bootstrap_paired_scene_metric`, `artifacts/segmentation_loss_asymmetric_penalty/l1_l2_smoke.json`.

## Phase L1 - implementation va verification

Exit gate: weighted loss hoat dong voi default tuong thich baseline; invalid-mask/all-invalid behavior khong thay doi; config, report va test bao phu thay doi.

- [x] **L1-01 | Them `cloud_class_weight` vao loss.**
  - Output: `masked_segmentation_loss` nhan `cloud_class_weight`, tao weight tensor `[1.0, cloud_class_weight]` dung device/dtype voi logits va truyen vao `F.cross_entropy`.
  - Depends: L0-04.
  - Done when: weight duong duoc validate; `cloud_class_weight=1.0` tuong thich voi baseline trong tolerance; `SoftDice`, `ignore_index`, validity mask va resize logits van giu nguyen.
  - Owner/ETA/Evidence: `Codex` / `2026-07-31` / `src/losses.py`, `artifacts/segmentation_loss_asymmetric_penalty/l1_l2_smoke.json` (`loss_contract.status=PASS`).

- [x] **L1-02 | Cap nhat training config, CLI va artifact provenance.**
  - Output: `SegmentationTrainingConfig.cloud_class_weight`, CLI option neu can, checkpoint/report serialization va loss configuration.
  - Depends: L1-01.
  - Done when: mot run co the tai tao tu config/report; checkpoint ghi du `cloud_class_weight`, CE/Dice weights, epsilon, ignore index va seed.
  - Owner/ETA/Evidence: `Codex` / `2026-07-31` / `src/train_segmentation.py`, `src/segmentation_experiment.py`, `artifacts/segmentation_loss_asymmetric_penalty/l2_ablation/ablation_manifest.json`.

- [x] **L1-03 | Log loss components va convergence.**
  - Output: train/validation report co tong loss, `cross_entropy`, `soft_dice`, epoch best, learning rate va loss curve.
  - Depends: L1-01, L1-02.
  - Done when: training loop khong con discard `parts` neu report yeu cau components; aggregation va valid-pixel denominator duoc ghi ro.
  - Owner/ETA/Evidence: `Codex` / `2026-07-31` / `src/train_segmentation.py` (`cross_entropy`, `soft_dice`, `learning_rate`, `history`), `src/segmentation_experiment.py` aggregate schema.

- [x] **L1-04 | Viet unit test cho weighted loss.**
  - Output: test cho mixed batch co ca surface/cloud, cloud miss co dong gop tuong doi lon hon khi tang weight; khong test bang mot pixel cloud don le vi weighted mean co the normalize weight.
  - Depends: L1-01.
  - Done when: test pass cho invalid target, validity mask, all-invalid batch, finite loss, backward va default weight compatibility.
  - Owner/ETA/Evidence: `Codex` / `2026-07-31` / `tests/test_segformer_integration.py`, `scripts/verify_segmentation_loss_contract.py` (`status=PASS`).

- [!] **L1-05 | Chay regression test hien co.**
  - Output: test suite lien quan SegFormer/loss pass.
  - Depends: L1-01, L1-04.
  - Done when: khong co regression o native-size target, resize logits, train epoch va evaluation.
  - Status: `BLOCKED` - `pytest` is not installed; pip installation is blocked by Windows `_overlapped`/`WinError 10106` and PyPI certificate failure. Source compilation and the focused verifier pass, but the pytest regression suite is not claimed as passed.
  - Owner/ETA/Evidence: `Codex` / `2026-07-31` / `scripts/verify_segmentation_loss_contract.py`, `python -m py_compile ...` output; environment blocker recorded in this tracker.

## Phase L2 - ablation va danh gia

Exit gate: moi cau hinh co ket qua reproducible, threshold validation rieng, metric day du va uncertainty duoc bao cao.

- [!] **L2-01 | Chay ablation multi-seed.**
  - Output: cac weight `1.0`, `1.2`, `1.5`, `2.0`, `3.0`; moi weight chay seed `42`, `123`, `456` voi cung split va training policy.
  - Depends: L1-05, L0-05.
  - Done when: 15 run co checkpoint/report, loss curves, epoch convergence va provenance; run loi/non-finite duoc ghi nhan, khong bi bo qua im lang.
  - Status: `BLOCKED` - the required 5 weights x 3 seeds cannot run without the frozen processed train/validation split. The manifest contains all 15 planned combinations and records zero runs rather than synthetic results.
  - Owner/ETA/Evidence: `Codex` / `2026-07-31` / `artifacts/segmentation_loss_asymmetric_penalty/l2_ablation/ablation_manifest.json` (`status=BLOCKED`, weights/seeds, runs=[]).

- [!] **L2-02 | Re-calibrate pixel threshold cho tung run.**
  - Output: threshold duoc chon chi tu validation predictions cua tung checkpoint; metrics tinh lai tai threshold tuong ung.
  - Depends: L2-01.
  - Done when: moi run co threshold, constraint `false_clear_rate`, selected metrics va dataset role=validation trong report.
  - Status: `BLOCKED` for actual runs; implementation is verified on a synthetic in-memory validation fixture and requires a real checkpoint before producing per-run evidence.
  - Owner/ETA/Evidence: `Codex` / `2026-07-31` / `src/eval_segmentation.py:calibrate_validation_loader`, `artifacts/segmentation_loss_asymmetric_penalty/l1_l2_smoke.json`.

- [!] **L2-03 | Tong hop metric theo weight va seed.**
  - Output: mean/std hoac median/interval cua recall, false-clear, precision, Dice, IoU, coverage error, convergence epoch va loss components.
  - Depends: L2-01, L2-02.
  - Done when: ket qua duoc group theo `cloud_class_weight`; khong tron cac threshold hoac split khac nhau.
  - Status: `BLOCKED` - no successful runs exist to aggregate. The harness schema includes primary metrics, coverage error, loss components and convergence statistics once runs are available.
  - Owner/ETA/Evidence: `Codex` / `2026-07-31` / `src/segmentation_experiment.py:_aggregate`, `artifacts/segmentation_loss_asymmetric_penalty/l2_ablation/ablation_manifest.json`.

- [!] **L2-04 | Bootstrap uncertainty theo scene va paired comparison.**
  - Output: scene-level bootstrap CI cho tung cau hinh va, neu tuyen bo chenh lech, bootstrap chenh lech tren cung scene IDs.
  - Depends: L2-03.
  - Done when: khong dung `bootstrap_scene_metric` mot vector de ket luan truc tiep significance giua hai cau hinh; report neu chi la uncertainty estimate.
  - Status: `BLOCKED` for dataset evidence; paired bootstrap implementation and shared-scene validation pass offline, but no real scene-level CI can be reported without L2 run reports.
  - Owner/ETA/Evidence: `Codex` / `2026-07-31` / `src/eval_segmentation.py:bootstrap_paired_scene_metric`, `artifacts/segmentation_loss_asymmetric_penalty/l1_l2_smoke.json`.

## Phase L3 - selection va release gate

Exit gate: mot weight duoc chon tren validation theo criteria da khoa; test chi duoc chay sau khi config va threshold da freeze.

- [ ] **L3-01 | Loc cau hinh theo hard constraints.**
  - Output: danh sach cau hinh dat/khong dat `false_clear_rate`, precision va Dice/IoU constraints.
  - Depends: L2-03, L2-04, L0-02.
  - Done when: moi loai bo co ly do va evidence; khong cherry-pick theo metric don le.
  - Owner/ETA/Evidence: `TBD` / `TBD` / `TBD`.

- [ ] **L3-02 | Chon weight va khoa config tren validation.**
  - Output: selected `cloud_class_weight`, threshold, seed policy, loss config va decision record.
  - Depends: L3-01.
  - Done when: uu tien cloud recall trong tap dat constraint; neu gan nhau dung false-clear; neu khong khac biet chon weight nho hon.
  - Owner/ETA/Evidence: `TBD` / `TBD` / `TBD`.

- [ ] **L3-03 | Danh gia test va cap nhat report.**
  - Output: locked-config test report, model/checkpoint provenance va residual-risk note.
  - Depends: L3-02.
  - Done when: test khong tham gia chon weight/threshold; report phan biet ro validation selection va test evaluation.
  - Owner/ETA/Evidence: `TBD` / `TBD` / `TBD`.

## Phase L4 - Tversky fallback

Chi mo phase nay neu tat ca weighted CE configurations khong dat hard constraints hoac khong cai thien false-clear theo muc tieu da khoa.

- [ ] **L4-01 | Dinh nghia Tversky experiment.**
  - Output: cong thuc valid-mask-aware, xu ly all-invalid/empty-cloud va grid alpha/beta voi `beta > alpha`.
  - Depends: L3-01.
  - Done when: objective, alpha/beta, CE/Dice co ket hop hay khong va stopping criteria duoc phe duyet truoc khi chay.
  - Owner/ETA/Evidence: `TBD` / `TBD` / `TBD`.

- [ ] **L4-02 | Implement, test va so sanh Tversky.**
  - Output: implementation, unit tests, same validation/test protocol va comparative report voi weighted CE baseline.
  - Depends: L4-01.
  - Done when: Tversky chi duoc chon neu dat hard constraints va co loi ich ro rang so voi baseline; khong mo rong hyperparameter khong co decision record.
  - Owner/ETA/Evidence: `TBD` / `TBD` / `TBD`.

## Required evidence inventory

- Class distribution report tren training valid pixels.
- Baseline checkpoint, training/evaluation JSON, git commit hash va config.
- Loss implementation diff va unit-test result.
- Ablation manifest cho 5 weights x 3 seeds.
- Per-run validation threshold report.
- Aggregated metric table va convergence curves.
- Scene-level bootstrap/paired comparison artifact neu duoc su dung.
- Selection decision record va locked config.
- Final test report chi sau validation freeze.

## Decision log

| Date | ID | Decision | Rationale / impact | Owner |
|---|---|---|---|---|
| 2026-07-31 | D-01 | Giu `SoftDice` va them class-weighted CE lam experiment dau tien | Thay doi nho, giu overlap objective va de isolate tac dong | `TBD` |
| 2026-07-31 | D-02 | Multi-seed va scene bootstrap duoc xem la hai loai evidence khac nhau | Train variance khac scene sampling uncertainty | `TBD` |
| 2026-07-31 | D-03 | Weighted CE test phai dung mixed surface/cloud batch | Weighted mean co the normalize class weight trong batch don class | `TBD` |

## Update log

| Date | Change | Updated by |
|---|---|---|
| 2026-07-31 | Tao tracker tu plan asymmetric cloud-loss; bo sung dependency, gates, evidence va reliability requirements | `Codex` |
| 2026-07-31 | Hoan tat implementation L0-L2 contracts, them audit/ablation harness, paired bootstrap, focused verifier; ghi BLOCKED cho dataset frozen, approval criteria va pytest regression suite | `Codex` |
