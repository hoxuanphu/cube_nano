# Ke hoach toi uu do chinh xac SegFormer 95-Cloud

## 1. Muc tieu va pham vi

Ke hoach nay toi uu SegFormer-B0 RGB trong
[`colab_train_segformer_95cloud_cache_only.ipynb`](../colab_train_segformer_95cloud_cache_only.ipynb)
ma khong lam ro ri test split hoac pha vo input/deployment contract hien tai.

Metric chinh de chon ung vien la validation cloud Dice voi rang buoc
false-clear rate. Bao cao bat buoc gom ca pixel micro metric va scene macro
metric de scene lon khong chi phoi ket qua.

| Metric | Muc tieu release hien tai |
|---|---:|
| Cloud IoU | >= 0.75 |
| Cloud Dice | >= 0.85 |
| Cloud recall | >= 0.90 |
| False-clear rate | <= 0.05 |
| Boundary F1 | >= 0.75 |

Nguon gate: [`acceptance_profile.yaml`](../sat_ai/acceptance_profile.yaml).

## 2. Nguyen tac thi nghiem

- Giu nguyen train/validation/test scene split trong mot experiment family.
- Dung validation de chon checkpoint, hyperparameter va threshold; test chi chay
  mot lan cho ung vien da chot.
- Screen ung vien bang mot seed; xac nhan ung vien thang bang seed `42`, `43`
  va `44`.
- Promote mot thay doi khi median cloud Dice tang it nhat `0.01`, hoac giup vuot
  quality gate, ma khong lam vi pham false-clear rate hay suy giam dang ke
  recall/boundary F1.
- Moi run phai luu Git revision, cache/raw manifest ID, split lineage ID,
  InputSpec ID, seed, CUDA/Torch version, checkpoint SHA-256 va config day du.
- Neu thay doi normalization, kenh input, preprocessing hoac model graph, tao
  version contract/cache moi; khong dung lai artifact cu nhu la cung experiment.

## 3. P0 - Dong bang baseline va chan doan loi

### Cong viec

1. Chay lai cau hinh notebook hien tai voi ba seed, bao gom encoder pretrained,
   native-size training, batch size `1`, 50 epoch va cache hien tai.
2. Luu metric moi epoch: train/validation CE, Soft Dice, cloud IoU/Dice,
   precision, recall, false-clear rate, boundary F1, coverage error, LR,
   optimizer step, valid-pixel count, padding ratio va throughput.
3. Calibrate threshold chi tren validation trong day `0..10000 bp`. Dung buoc
   `100 bp` de screen va `25-50 bp` de tinh chinh quanh threshold tot nhat.
4. Tao error atlas theo scene: 20 false-clear cao nhat, 20 false-positive cao
   nhat va 20 scene co boundary F1 thap nhat.
5. Xuat bang so sanh micro/macro metric va 95% scene bootstrap interval giua
   cac seed.

### Chan doan va huong xu ly

| Dau hieu validation | Uu tien xu ly tiep theo |
|---|---|
| False-clear/FN cao | Threshold calibration, hard cloud sampling, weighted/focal/Tversky loss |
| False-positive cao | Normalization, radiometric augmentation, label audit va threshold |
| Boundary F1 thap | Boundary-centred crop, overlap/blending inference, label-edge audit |
| Train tot, validation kem | Normalization, augmentation, regularization va domain adaptation |
| Chenh lech micro/macro lon | Scene balancing, stratified split/audit va sampling theo cloud cover |

### Exit gate P0

- Ba run baseline co artifact day du va khong co split leakage.
- Threshold duoc khoa tu validation truoc test.
- Da co phan loai ro van de chinh: FN, FP, boundary, overfit hay domain shift.

## 4. P1 - On dinh fine-tuning va batch behavior

Hien tai `warmup_epochs` chi tri hoan `ReduceLROnPlateau`, khong thuc hien
learning-rate warm-up. [train_segmentation.py](../src/train_segmentation.py)
cung dung mot learning rate cho ca encoder pretrained va decoder moi.

### Thu tu ablation

| ID | Thay doi | Cau hinh screen |
|---|---|---|
| E1 | Warm-up LR that | Linear 3 va 5 epoch, giu batch 1 va LR hien tai |
| E2 | Parameter groups | Encoder `5e-6`, `1e-5`, `2e-5`; decoder `5e-5`, `1e-4` |
| E3 | Batch size | Batch `4`, sau do `8` neu VRAM cho phep |
| E4 | Thoi luong/scheduler | 50 so voi 80 epoch, cosine decay hoac plateau sau warm-up |

Neu batch native-size lon hon mot, do padded-pixel ratio. Padding zero di qua
`BatchNorm2d` trong decoder du loss bo qua pixel padding. Neu padding ratio cao,
uu tien size-bucket sampler hoac thay decoder BatchNorm bang GroupNorm truoc khi
ket luan batch lon lam giam chat luong.

### Exit gate P1

- Chon mot training recipe co ket qua on dinh qua ba seed.
- Khong co regression false-clear, macro Dice hoac boundary F1.
- Scheduler log the hien warm-up that va parameter-group LR dung nhu config.

## 5. P2 - Input normalization versioned

Input hien tai chi scale `uint16 / 65535`; notebook co tinh train-only RGB
statistics nhung khong ap dung vi InputSpec dang pin dtype-range.

### Ablation

1. Dtype-range hien tai lam control.
2. Dtype-range roi ImageNet mean/std de tuong thich hon voi MiT-B0 pretrained.
3. Dtype-range roi train-only per-band mean/std.

Moi phuong an phai co InputSpec ID moi, train/inference parity, ONNX golden
tensor moi va regression test cho mismatch contract. Khong duoc thay normalization
chi o training.

### Exit gate P2

- Phuong an thang cai thien metric validation qua ba seed.
- PyTorch, ONNX va inference runtime dung cung normalization va dat parity gate.

## 6. P3 - Imbalance, hard examples va augmentation

Loss hien tai la Cross-Entropy + Soft Dice, moi thanh phan trong so `1.0`.
DataLoader shuffle record nhung khong co weighted sampler ro rang.

### Cong viec

1. Tinh foreground ratio tren valid pixel theo scene va cloud-cover bin.
2. Thu weighted CE + Dice voi cloud weight `1`, `2`, `4`, `8`, co clip de tranh
   over-correction.
3. Neu false-clear van cao, screen focal CE (`gamma=1.5`, `2.0`) hoac Tversky
   co uu tien recall; giu false-positive trong acceptance target.
4. Them sampler theo cloud-cover bin va hard-example queue tu error atlas.
5. Neu chuyen sang tile training, dung mix crop 50% random, 25% cloud-centred,
   25% boundary-centred.
6. Giu flip/rotation hien co; them global/band-wise gain-offset nhe, noise nhe,
   blur/resampling xac suat thap. Khong dung hue jitter manh khong phu hop band
   ve tinh.

### Exit gate P3

- Cai thien metric cua nhom loi muc tieu ma khong chuyen loi sang metric khac.
- Augmentation policy, sampling policy va seed duoc luu trong checkpoint metadata.

## 7. P4 - Spatial strategy, nang cap model va deployment cost

Chi bat dau P4 sau khi P1-P3 da bao hoa.

1. So sanh native-size voi 256-tile random crop va overlap/blending khi inference.
2. Neu boundary F1 la bottleneck, uu tien overlap inference truoc khi tang capacity.
3. Screen SegFormer-B1/B2 neu B0 underfit va budget runtime cho phep.
4. Chi dung test-time augmentation hoac ensemble ba seed neu latency, memory va
   deployment SLO van dat.
5. Neu runtime co NIR va sensor contract xac nhan band nay, danh gia model RGB+NIR
   nhu release rieng. Khoi tao kenh NIR tu trung binh kenh RGB, version hoa
   InputSpec/model graph va khong tron voi baseline RGB.

### Exit gate P4

- Loi ich chat luong lon hon chi phi latency/RAM/export.
- ONNX/TensorRT parity, golden input va benchmark runtime duoc cap nhat cho
  contract moi.

## 8. P5 - Data quality va domain shift

Notebook cache-only khong the sua raw label, preprocessing hay split. Khi error
atlas chi ra label/domain la bottleneck, dung full pipeline de tao processed cache
moi thay vi ghi de cache hien tai.

1. Audit raw mask, invalid value, edge alignment va radiometry cua scene loi.
2. Tao scene-level split stratified theo cloud cover, sensor, thoi gian va khu vuc
   neu metadata co san; phat hanh split lineage moi.
3. Fine-tune tren du lieu co nhan cua sensor/product deploy neu co.
4. Luu domain evidence; khong coi external smoke image la quantitative holdout
   khi chua co ground truth duoc audit.

### Exit gate P5

- Cache moi co provenance, raw audit va split lineage ro rang.
- Cai thien tren holdout dung domain, khong chi tren validation noi bo.

## 9. P6 - Chot ung vien va release evidence

1. Chon ung vien tu validation, khoa checkpoint va DecisionSpec.
2. Chay frozen test mot lan va tao scene bootstrap intervals.
3. Kiem tra tat ca quality gate, ONNX parity, TensorRT parity va benchmark target.
4. Luu evidence bundle: checkpoint, config, training history, calibration, test
   report, cache/split manifests, normalization/InputSpec, environment va SHA-256.
5. Neu quality gate khong dat, quay lai phase co lien quan; khong tune tren test.

## 10. Thu tu trien khai de xuat

Bat dau bang P0 va P1. Day la nhom thay doi rui ro thap nhat, lam ro baseline va
sua hai van de da xac dinh trong training loop: warm-up khong that va cung LR
cho encoder/decoder. Chi sang P2-P5 khi error atlas va metric baseline chi ra
nhu cau cu the.
