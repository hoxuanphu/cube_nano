# Ke hoach them co che phat loi bo sot may

## 1. Muc tieu

Bo sung co che phat bat doi xung cho bai toan phan doan may trong anh ve tinh:

```text
class 0 = be mat Trai Dat / khong may
class 1 = may
255 = pixel khong hop le, bo qua
```

Muc tieu uu tien la giam loi bo sot may (`cloud -> surface`, false negative), vi loi nay co the de lai may trong anh sau khi loc.

Loss hien tai la baseline:

```text
Loss = CrossEntropy + SoftDice
```

Khong them reward rieng sau `argmax`; co che phat duoc dua vao loss de van giu gradient cho qua trinh hoc.

### Co che hoat dong cua weighted Cross Entropy

`F.cross_entropy(logits, target, weight=[1.0, W])` ap dung trong so theo **ground truth class** cua tung pixel:

- Khi `target = 1` (may): loss cua pixel do duoc nhan voi `W` -> truc tiep phat manh hon khi model du doan sai pixel may thanh be mat (false negative).
- Khi `target = 0` (be mat): loss van nhan voi `1.0` -> khong truc tiep tang loss cua false positive.

Tang `W` khong truc tiep phat false positive (surface -> cloud). Tuy nhien, vi model hoc cach uu tien doan dung pixel may hon, decision boundary co the dich chuyen theo huong du doan may nhieu hon, dan den false positive tang gian tiep. Day la hieu ung behavioral, khong phai loss-level.

## 2. Thong ke class distribution

Truoc khi chon gia tri `cloud_class_weight`, thong ke ty le pixel tren training set:

- So pixel `surface` (class 0) va `cloud` (class 1) tren toan bo training set (chi dem pixel valid, bo qua `target=255` va pixel bi danh dau invalid boi `validity_mask`).
- Bao cao ty le `surface:cloud`.
- Neu dataset da co nhieu cloud hon surface thi `cloud_class_weight > 1.0` co the gay double penalty (dataset da bias + loss cung bias). Can luu y dieu nay khi phan tich ket qua ablation.

Buoc nay giup co co so chon khoang gia tri `cloud_class_weight` hop ly va giai thich ket qua thu nghiem.

## 3. Baseline can ghi nhan

Truoc khi thay doi loss, train va danh gia voi cung dataset split, seed va training config hien tai. Ghi lai:

- `cloud_recall`;
- `false_clear_rate`;
- `cloud_precision`;
- `cloud_dice` va `cloud_iou`;
- sai so cloud coverage;
- gia tri rieng cua `cross_entropy` va `soft_dice`;
- epoch hoi tu (epoch co validation loss tot nhat);
- duong cong train loss va validation loss theo epoch.

Luu ket qua baseline thanh JSON file chuan (dung format cua `src/eval_segmentation.py`), bao gom model checkpoint path, git commit hash va training config day du.

Metric da co trong `src/eval_segmentation.py`. `false_clear_rate` hien duoc tinh tu so pixel may bi bo sot.

## 4. Thay doi loss de xuat

Them trong so cho Cross Entropy theo class:

```text
WeightedCE = CrossEntropy(logits, target, weight=[1.0, cloud_class_weight])
Loss = cross_entropy_weight * WeightedCE + dice_weight * SoftDice
```

Trong do `cloud_class_weight > 1.0` lam tang muc phat khi pixel may bi du doan thanh be mat (false negative). Trong moi thu nghiem ablation, giu co dinh `cross_entropy_weight = 1.0` va `dice_weight = 1.0`; chi thay doi `cloud_class_weight` de isolate tac dong cua bien duy nhat.

### Tham so can them

- Them `cloud_class_weight: float = 1.0` vao `SegmentationTrainingConfig`.
- Truyen tham so vao `masked_segmentation_loss` trong `src/losses.py`.
- Tao tensor trong so tren cung device va dtype phu hop voi logits.
- Validate trong so phai duong.
- Luu tham so vao checkpoint, JSON report va CLI neu can chay ablation tu command line.

Giu nguyen:

- `SoftDiceLoss` chi tap trung vao cloud class;
- `validity_mask`;
- `ignore_index=255`;
- xu ly batch toan pixel invalid;
- resize logits ve kich thuoc target.

## 5. Thu nghiem ablation

Chay tren cung split voi cac gia tri:

```text
cloud_class_weight = 1.0  # baseline
cloud_class_weight = 1.2
cloud_class_weight = 1.5
cloud_class_weight = 2.0
cloud_class_weight = 3.0
```

Khong chon truoc gia tri chinh thuc. Chon cau hinh dua tren validation, khong dua tren test.

### Yeu cau ve do tin cay

Moi cau hinh `cloud_class_weight` phai duoc chay voi it nhat 2-3 seed khac nhau (vi du: 42, 123, 456) hoac su dung bootstrap theo scene (ham `bootstrap_scene_metric` da co trong `src/eval_segmentation.py`) de xac nhan su khac biet giua cac cau hinh la co y nghia thong ke, khong nam trong noise range.

### Ghi nhan hoi tu

Voi moi lan chay, ghi nhan:

- Epoch hoi tu (epoch co validation loss tot nhat).
- Duong cong train loss va validation loss theo epoch.
- So sanh toc do hoi tu giua cac cau hinh de phat hien som bat on.

## 6. Unit test can bo sung

Bo sung test trong `tests/test_segformer_integration.py` cho cac truong hop:

- cung mot loi bo sot cloud co loss lon hon khi tang `cloud_class_weight`;
- pixel co `target=255` khong anh huong loss;
- pixel bi danh dau invalid khong anh huong loss;
- batch toan invalid tra ve loss bang `0` va khong tao gradient;
- loss luon huu han va backpropagation duoc;
- trong so mac dinh `1.0` cho ket qua tuong thich voi baseline hien tai.

## 7. Tieu chi danh gia

Xac dinh tieu chi chon **truoc khi** chay ablation de tranh cherry-pick ket qua:

### Rang buoc cung (loai bo cau hinh vi pham)

- `false_clear_rate` khong vuot qua nguong yeu cau cua du an (vi du: <= 5%).
- `cloud_precision` khong giam qua muc (vi du: >= 85%).
- `cloud_dice` khong suy giam dang ke so voi baseline (vi du: khong giam qua 2 diem phan tram).

### Tieu chi uu tien (chon cau hinh tot nhat trong cac cau hinh dat rang buoc)

1. Uu tien `cloud_recall` cao nhat (giam bo sot may).
2. Trong truong hop `cloud_recall` gan nhau, chon cau hinh co `false_clear_rate` thap hon.
3. Khi hai cau hinh khong khac biet co y nghia thong ke, chon `cloud_class_weight` nho hon (don gian hon).

### Trade-off can luu y

Tang `cloud_class_weight` co the giam bo sot may nhung lam tang false positive gian tiep (model du doan may nhieu hon). Can danh gia trade-off nay thay vi toi uu mot metric don le. Ket qua bootstrap hoac multi-seed giup xac nhan su khac biet la thuc su, khong phai noise.

## 8. Thu tu trien khai

1. Thong ke ty le pixel surface/cloud tren training set (muc 2).
2. Chay va luu baseline, bao gom epoch hoi tu va loss curve.
3. Them `cloud_class_weight` vao loss va training config.
4. Cap nhat checkpoint/report/CLI de dam bao tai lap duoc.
5. Viet va chay unit test cho weighted loss va invalid-mask behavior.
6. Chay ablation voi cac gia tri trong muc 5, moi cau hinh chay 2-3 seed.
7. Voi moi model da train xong, chay lai buoc hieu chinh pixel probability threshold tren validation (su dung `select_pixel_threshold` trong `src/eval_segmentation.py`). Buoc nay bat buoc vi phan phoi xac suat cua model co the thay doi khi doi loss.
8. Danh gia ket qua theo tieu chi da dinh truoc (muc 7), so sanh bootstrap CI giua cac cau hinh.
9. Chon cau hinh theo rang buoc `false_clear_rate` va `cloud_recall` tren validation.
10. Khoa cau hinh, sau do moi danh gia tren test.

## 9. Buoc mo rong neu can

Neu weighted Cross Entropy voi moi gia tri `cloud_class_weight` trong danh sach ablation deu khong cai thien `false_clear_rate` dang ke, chuyen sang Tversky Loss:

```text
Tversky(p, g) = TP / (TP + alpha * FP + beta * FN)
```

Voi `beta > alpha`, Tversky truc tiep phat false negative manh hon false positive. Day la buoc sau vi no thay doi hanh vi Dice nhieu hon, them 2 hyperparameter (`alpha`, `beta`), va kho so sanh truc tiep voi baseline.
