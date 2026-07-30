# Task tracker: toi uu do chinh xac SegFormer 95-Cloud

Nguon: [segformer_accuracy_optimization_plan.md](segformer_accuracy_optimization_plan.md)

Tracker nay theo doi cong viec toi uu SegFormer-B0 RGB tren 95-Cloud. Task chi
duoc danh dau `DONE` khi co evidence link toi code, test, experiment report hoac
artifact da kiem tra. Viet code don thuan khong du de dong task.

## Quy uoc

- `TODO`: chua bat dau.
- `IN_PROGRESS`: dang thuc hien, chua qua exit gate.
- `BLOCKED`: can quyet dinh, du lieu, hardware hoac quyen truy cap ben ngoai.
- `DONE`: co evidence va dat exit gate.
- `DEFERRED`: co chu dich de sau release hien tai.

Khi cap nhat tracker, sua dong thoi Dashboard, Work items, Gate dashboard,
Blockers va Update log. Khong sua test split hoac dung test de chon config.

## Dashboard

| Truong | Gia tri |
|---|---|
| Scope | SegFormer-B0 RGB semantic cloud segmentation tren 95-Cloud |
| Overall status | `IN_PROGRESS` |
| Current phase | P0 baseline instrumentation va P1 training-loop implementation |
| Source notebook | `colab_train_segformer_95cloud_cache_only_p0_p1.ipynb` |
| Primary selection metric | Validation cloud Dice, constrained false-clear rate |
| Frozen test policy | Chi chay sau khi checkpoint va DecisionSpec da khoa |
| Tracker owner | Codex |
| Last reviewed | 2026-07-30 |

## Work items

| ID | Phase | Work item / exit gate | Status | Depends on | Evidence |
|---|---|---|---|---|---|
| P0-01 | P0 | Dong bang baseline: Git revision, cache/raw manifest ID, split lineage, InputSpec, environment va config hien tai | IN_PROGRESS | - | `colab_train_segformer_95cloud_cache_only_p0_p1.ipynb` yeu cau commit SHA bat bien va luu provenance/cache/split/InputSpec/environment theo seed; cho artifact run that. |
| P0-02 | P0 | Chay baseline seed 42, 43, 44 voi cache va split khong doi | TODO | P0-01 | Can GPU Colab va processed cache 95-Cloud; notebook luu artifact theo `experiment_id`/seed va co aggregate validation. |
| P0-03 | P0 | Them epoch metrics: micro/macro Dice/IoU, recall, precision, false-clear, boundary F1, coverage, LR, valid pixels va throughput | IN_PROGRESS | P0-01 | `src/train_segmentation.py`, `src/data/segmentation_dataset.py`, `tests/test_segformer_p0_p1.py` (padding/metric instrumentation PASS); cho history cua run that. |
| P0-04 | P0 | Mo rong validation threshold sweep `0..10000 bp`; luu calibration report va DecisionSpec | IN_PROGRESS | P0-02 | `src/eval_segmentation.py` co sweep inclusive coarse 100 bp + refine 25-50 bp; notebook luu calibration/DecisionSpec; regression PASS. |
| P0-05 | P0 | Tao error atlas cho false-clear, false-positive va boundary F1 thap theo scene | IN_PROGRESS | P0-02, P0-04 | `src/eval_segmentation.py` xuat top-k scene rankings; notebook luu `validation_error_atlas_*`; regression PASS. |
| P0-06 | P0 | Chot error diagnosis va baseline report co bootstrap interval | TODO | P0-02..P0-05 | - |
| P1-01 | P1 | Implement linear LR warm-up that; giu control config cho ablation | IN_PROGRESS | P0-06 | `src/train_segmentation.py` co linear warm-up thuc, log `warmup_factor`; notebook P0 control dung 0 epoch, E1 dung 3/5 epoch; regression PASS. |
| P1-02 | P1 | Implement parameter groups encoder/decoder va log tung LR | IN_PROGRESS | P1-01 | `src/train_segmentation.py` co encoder/decoder AdamW groups va `learning_rates` moi epoch; regression PASS. |
| P1-03 | P1 | Screen E1: warm-up 3 va 5 epoch voi batch 1 | TODO | P1-01 | - |
| P1-04 | P1 | Screen E2: encoder/decoder LR grid | TODO | P1-02, P1-03 | - |
| P1-05 | P1 | Do native padding ratio; screen batch 4 roi 8 neu VRAM cho phep | IN_PROGRESS | P0-06 | `src/data/segmentation_dataset.py` va `src/train_segmentation.py` tach collate padding khoi invalid pixel, log padding ratio; regression PASS; chua co screen VRAM. |
| P1-06 | P1 | Neu padding ratio cao, implement size bucketing hoac decoder GroupNorm va regression test | TODO | P1-05 | - |
| P1-07 | P1 | Xac nhan winning training recipe qua ba seed | TODO | P1-03..P1-06 | - |
| P2-01 | P2 | Tao InputSpec versioned cho ImageNet mean/std va train-only mean/std | TODO | P1-07 | - |
| P2-02 | P2 | Cap nhat train, inference, ONNX va golden parity cho InputSpec moi | TODO | P2-01 | - |
| P2-03 | P2 | Chay three-way normalization ablation va xac nhan winner qua ba seed | TODO | P2-02 | - |
| P3-01 | P3 | Tinh foreground ratio tren valid pixel va cloud-cover bin theo scene | TODO | P1-07 | - |
| P3-02 | P3 | Implement weighted CE + Dice; screen cloud weight 1, 2, 4, 8 | TODO | P3-01 | - |
| P3-03 | P3 | Screen focal CE/Tversky neu false-clear la failure mode chinh | TODO | P3-02, P0-06 | - |
| P3-04 | P3 | Implement balanced/hard-example sampler tu error atlas | TODO | P3-01, P0-05 | - |
| P3-05 | P3 | Implement versioned radiometric augmentation policy va test image-mask alignment | TODO | P0-06 | - |
| P3-06 | P3 | Xac nhan loss/sampling/augmentation winner qua ba seed | TODO | P3-02..P3-05 | - |
| P4-01 | P4 | So sanh native-size va tile training; them overlap/blending inference neu boundary F1 la bottleneck | TODO | P1-07, P0-06 | - |
| P4-02 | P4 | Screen SegFormer-B1/B2 chi sau khi B0 da bao hoa | TODO | P1-07, P3-06 | - |
| P4-03 | P4 | Danh gia TTA/three-seed ensemble voi latency va RAM budget | TODO | P4-01, P4-02 | - |
| P4-04 | P4 | Danh gia RGB+NIR release rieng neu runtime va sensor contract co NIR | DEFERRED | P5-03 | - |
| P5-01 | P5 | Audit raw label, invalid mask, boundary alignment va radiometry cho scene loi | TODO | P0-05 | - |
| P5-02 | P5 | Tao processed cache moi va split lineage moi neu data/contract can sua | TODO | P5-01 | - |
| P5-03 | P5 | Fine-tune/evaluate tren holdout dung sensor-product deploy co nhan | BLOCKED | External labeled target data | - |
| P6-01 | P6 | Khoa checkpoint, InputSpec, DecisionSpec va calibration tu validation | TODO | P1-07, P2-03, P3-06, P4-01 | - |
| P6-02 | P6 | Chay frozen test mot lan, bootstrap scene metrics va quality gates | TODO | P6-01 | - |
| P6-03 | P6 | Cap nhat ONNX/TensorRT parity va target runtime benchmark neu graph/contract doi | TODO | P6-01 | - |
| P6-04 | P6 | Tao evidence bundle va release decision | TODO | P6-02, P6-03 | - |

## Gate dashboard

| Gate | Dieu kien | Status | Evidence |
|---|---|---|---|
| G0 - Baseline trusted | P0-01..P0-06 DONE; baseline ba seed, calibration va error atlas day du | IN_PROGRESS | Instrumentation, calibration va atlas da co regression evidence; cho artifact P0 ba seed. |
| G1 - Training recipe | P1-01..P1-07 DONE; warm-up, LR groups va batch behavior da xac nhan | IN_PROGRESS | Warm-up/LR groups/padding telemetry da co regression evidence; cho E1-E4 va xac nhan ba seed. |
| G2 - Input parity | P2-01..P2-03 DONE; normalization winner co train/inference/ONNX parity | TODO | - |
| G3 - Hard-case robustness | P3-01..P3-06 DONE; loss/sampling/augmentation cai thien metric muc tieu | TODO | - |
| G4 - Spatial/capacity | P4 task lien quan DONE hoac DEFERRED co ly do; latency/RAM da danh gia | TODO | - |
| G5 - Data/domain evidence | P5-01 va cac task can thiet DONE; target-domain blocker duoc xu ly hoac ghi ro | TODO | - |
| G6 - Candidate release | P6-01..P6-04 DONE; test frozen, quality/parity/runtime gates dat | TODO | - |

## Blockers

| ID | Blocker | Anh huong | Can de mo khoa | Owner | Status |
|---|---|---|---|---|---|
| B-01 | Chua co baseline metric/test report trong workspace hien tai | Khong the chon failure mode dua tren evidence hoac dong G0/G1 | Chay P0-02 den P0-06 tren GPU Colab voi cache 95-Cloud co split lineage bat bien | Experiment owner | OPEN |
| B-02 | Mean/std normalization doi InputSpec va runtime contract | P2 khong duoc trien khai nhu mot training-only tweak | Versioned contract, inference/ONNX parity va tests | Unassigned | OPEN |
| B-03 | Chua co labeled holdout cua sensor/product deploy | Khong the ket luan cross-sensor accuracy | Target-domain labels da audit | External owner | OPEN |

## Update log

| Date | Change | Updated by |
|---|---|---|
| 2026-07-30 | Tao tracker tu ke hoach toi uu SegFormer; tat ca task bat dau o TODO, tru P4-04 DEFERRED va P5-03 BLOCKED do phu thuoc du lieu ngoai | Codex |
| 2026-07-30 | Implement P0/P1 instrumentation: epoch micro/macro metrics, CE/Soft Dice, valid/padding/throughput, calibration coarse+refine, validation error atlas, seed aggregate, true LR warm-up, encoder/decoder LR groups; them notebook P0/P1 va regression tests. Chua danh dau DONE vi chua co artifact ba-seed GPU/cache that. | Codex |
