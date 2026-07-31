# Đánh giá chuyên gia: Phương án thưởng phạt bất đối xứng cho Segmentation Loss

> Tài liệu được đánh giá: [segmentation_loss_asymmetric_penalty_plan.md](file:///c:/Users/phuhx1/Documents/cube_nano/docs/segmentation_loss_asymmetric_penalty_plan.md)

---

## 1. Tổng quan nhận xét

**Kết luận chung: Phương án hợp lý về mặt kỹ thuật, bảo thủ đúng mức, và có quy trình kiểm soát tốt.** Tuy nhiên, có một số điểm cần cải thiện hoặc xem xét thêm.

---

## 2. Điểm mạnh

### 2.1. Đúng bài toán — Weighted Cross Entropy là lựa chọn hợp lý nhất cho bước đầu

| Tiêu chí | Đánh giá |
|---|---|
| Tính đơn giản | ✅ Chỉ thêm 1 tham số `cloud_class_weight` |
| Gradient flow | ✅ Không can thiệp sau `argmax`, giữ nguyên differentiability |
| Tương thích baseline | ✅ `weight=1.0` ⟺ baseline hiện tại |
| Phạm vi thay đổi code | ✅ Chỉ ảnh hưởng [losses.py](file:///c:/Users/phuhx1/Documents/cube_nano/src/losses.py) và [train_segmentation.py](file:///c:/Users/phuhx1/Documents/cube_nano/src/train_segmentation.py) |

Trong lý thuyết, Weighted CE với `weight[1] > 1.0` tương đương việc tăng gradient magnitude cho **mọi pixel cloud**, khiến model "sợ" việc dự đoán sai class 1 → class 0 (false negative / bỏ sót mây) nhiều hơn. Đây là phương pháp kinh điển, có nền tảng lý thuyết vững chắc.

### 2.2. Giữ nguyên SoftDiceLoss — Quyết định đúng

Plan **không sửa SoftDiceLoss**, chỉ thêm trọng số vào CE. Đây là quyết định khôn ngoan vì:

- SoftDice đã hoạt động tốt cho overlap metric (cloud class)
- Sửa cả hai component cùng lúc sẽ khó isolate nguyên nhân khi metric thay đổi
- Code hiện tại trong [soft_dice_loss](file:///c:/Users/phuhx1/Documents/cube_nano/src/losses.py#L31-L55) đã focus vào cloud class (lấy `[:, 1]`), nên tự nhiên đã "bias" về phía cloud

### 2.3. Quy trình ablation nghiêm ngặt

```text
cloud_class_weight = 1.0  # baseline
cloud_class_weight = 1.5
cloud_class_weight = 2.0
cloud_class_weight = 3.0
```

- ✅ **Không chọn trước giá trị** — Tránh confirmation bias
- ✅ **Cùng split, seed** — Đảm bảo so sánh công bằng
- ✅ **Chọn trên validation, không trên test** — Đúng phương pháp luận ML
- ✅ **Khóa cấu hình trước khi đánh giá test** — Ngăn data leakage

### 2.4. Nhận biết trade-off rõ ràng (Mục 6)

Plan đã nhận ra đúng rằng tăng `cloud_class_weight` sẽ:
- ↑ `cloud_recall` (giảm bỏ sót mây)
- ↑ `false positive rate` (loại nhầm bề mặt Trái Đất thành mây)
- Cần calibrate lại pixel probability threshold

Đây là hiểu biết quan trọng mà nhiều kế hoạch tương tự bỏ qua.

### 2.5. Unit test coverage tốt

6 test case trong mục 5 bao phủ các corner case quan trọng:
- All-invalid batch
- `target=255` isolation
- Gradient flow
- Backward compatibility (`weight=1.0`)

---

## 3. Điểm yếu và thiếu sót

### 3.1. ⚠️ Weighted CE tác động **cả hai chiều**, không chỉ false negative

> [!WARNING]
> `F.cross_entropy(logits, target, weight=[1.0, W])` với `W > 1.0` **tăng penalty cho mọi lỗi liên quan đến class 1**, bao gồm cả:
> - **Cloud → Surface** (false negative — đây là mục tiêu) ✅
> - **Surface → Cloud** (false positive — tác dụng phụ không mong muốn) ⚠️

Cụ thể, `weight` trong PyTorch CE là per-class weight cho **ground truth class**, không phải per-error-type weight. Khi `target = 1` (cloud), loss bị nhân với `W`:

$$\text{WeightedCE}_i = -w_{y_i} \log p(y_i | x_i)$$

Trong đó $w_1 = W > 1$ chỉ phạt khi **ground truth là cloud** và model dự đoán sai → đúng là phạt false negative. Tuy nhiên, vì tổng weight thay đổi, nó cũng gián tiếp ảnh hưởng cân bằng loss surface → có thể khiến model predict cloud nhiều hơn → tăng cả false positive.

**Khuyến nghị**: Plan nên giải thích rõ cơ chế này để team không nhầm lẫn "chỉ phạt false negative". Cân nhắc thêm phân tích per-class precision/recall khi ablation.

### 3.2. ⚠️ Thiếu khoảng giá trị ablation nhỏ hơn 1.5

Khoảng nhảy `[1.0, 1.5, 2.0, 3.0]` bỏ qua vùng `[1.1, 1.2, 1.3]`. Trong thực tế:

- Với dataset satellite cloud segmentation (thường class imbalance nhẹ đến trung bình), các giá trị **1.2–1.5** thường cho kết quả tốt nhất
- `3.0` có nguy cơ cao gây over-prediction cloud (model dự đoán quá nhiều mây)
- Nên thêm ít nhất `1.2` vào danh sách ablation

### 3.3. ⚠️ Chưa xét đến class imbalance thực tế trong dataset

Plan không đề cập đến **tỷ lệ cloud/surface pixels** trong dataset 95-Cloud. Đây là yếu tố quan trọng vì:

- Nếu dataset đã có **nhiều cloud hơn surface** (phổ biến trong satellite imagery), thì `cloud_class_weight > 1.0` có thể gây **double penalty** — dataset đã bias + loss cũng bias
- Nếu dataset **cân bằng**, thì approach này hợp lý hơn
- Lý tưởng nhất, `cloud_class_weight` nên được set dựa trên **inverse class frequency** làm điểm xuất phát, sau đó fine-tune

**Khuyến nghị**: Thêm bước 0 — thống kê class distribution trên training set, báo cáo tỷ lệ `surface:cloud` pixels.

### 3.4. ⚠️ Chưa có metric tổng hợp để so sánh cấu hình

Mục 6 liệt kê nhiều tiêu chí nhưng không nêu rõ **cách ra quyết định khi các metric xung đột**. Ví dụ:

| Config | `false_clear_rate` | `cloud_recall` | `cloud_dice` | Chọn? |
|---|---|---|---|---|
| W=1.5 | 3.5% ↓ | 96.5% ↑ | 0.91 = | ? |
| W=2.0 | 2.8% ↓↓ | 97.2% ↑↑ | 0.89 ↓ | ? |

**Khuyến nghị**: Xác định trước **ràng buộc cứng** (ví dụ: `cloud_dice ≥ 0.88`, `cloud_precision ≥ 0.85`) và **metric ưu tiên** (ví dụ: minimize `false_clear_rate` trong vùng thỏa ràng buộc).

### 3.5. ⚠️ Interaction giữa `cloud_class_weight` và `cross_entropy_weight` / `dice_weight`

Code hiện tại trong [masked_segmentation_loss](file:///c:/Users/phuhx1/Documents/cube_nano/src/losses.py#L58-L91) đã có `cross_entropy_weight` và `dice_weight`:

```python
total = cross_entropy_weight * cross_entropy + dice_weight * dice
```

Nếu thêm `cloud_class_weight` vào CE, thì tổng loss trở thành:

$$\text{Loss} = \alpha \cdot \text{WeightedCE}(W) + \beta \cdot \text{SoftDice}$$

Với 3 hyperparameter ($\alpha$, $\beta$, $W$) có thể **giao thoa**. Tăng $W$ tương đương một phần với tăng $\alpha$. Plan nên note rõ: **giữ nguyên** $\alpha = \beta = 1.0$ trong ablation, chỉ thay đổi $W$.

### 3.6. ❌ Thiếu đánh giá tốc độ hội tụ

Thêm class weight **có thể thay đổi tốc độ hội tụ** — model có thể cần nhiều epoch hơn hoặc learning rate khác. Plan nên:

- Ghi nhận **epoch hội tụ** cho mỗi cấu hình
- Xem xét tăng `epochs` hoặc `early_stopping_patience` nếu cần
- Plot loss curve để kiểm tra stability

### 3.7. ❌ Tversky Loss nên được xem xét song song, không phải "bước sau"

> [!IMPORTANT]
> Plan đặt Tversky Loss ở mục 8 "Bước mở rộng nếu cần", nhưng thực tế **Tversky Loss là công cụ trực tiếp hơn** cho bài toán này:
>
> $$\text{Tversky}(p, g) = \frac{TP}{TP + \alpha \cdot FP + \beta \cdot FN}$$
>
> Với $\beta > \alpha$, Tversky **trực tiếp** phạt false negative nhiều hơn false positive — đúng chính xác mục tiêu.

Weighted CE phạt theo class, Tversky phạt theo **loại lỗi**. Về mặt lý thuyết, Tversky cho kiểm soát chính xác hơn.

Tuy nhiên, plan cũng **có lý** khi để Tversky ở bước sau vì:
- Weighted CE dễ implement, dễ so sánh với baseline
- Tversky thay đổi behavior của cả Dice component → khó isolate
- Cần thêm 2 hyperparameter ($\alpha$, $\beta$) → không gian tìm kiếm lớn hơn

**Khuyến nghị**: Nếu weighted CE với mọi giá trị `W` đều không cải thiện `false_clear_rate` đáng kể, nên chuyển sang Tversky **ngay lập tức**, không thử thêm tricks trên CE.

---

## 4. Đánh giá khả năng triển khai trong codebase hiện tại

### 4.1. Code changes cần thiết — Impact thấp, rủi ro thấp

| File | Thay đổi | Độ phức tạp |
|---|---|---|
| [losses.py](file:///c:/Users/phuhx1/Documents/cube_nano/src/losses.py) | Thêm `cloud_class_weight` param, tạo `weight` tensor, truyền vào `F.cross_entropy` | Thấp |
| [train_segmentation.py](file:///c:/Users/phuhx1/Documents/cube_nano/src/train_segmentation.py) | Thêm field vào `SegmentationTrainingConfig`, truyền qua `masked_segmentation_loss`, thêm CLI arg | Thấp |
| [test_segformer_integration.py](file:///c:/Users/phuhx1/Documents/cube_nano/tests/test_segformer_integration.py) | Thêm 6 test case mới | Trung bình |

### 4.2. Tương thích với code hiện tại

Kiểm tra chi tiết code:

- ✅ `masked_segmentation_loss` đã dùng `F.cross_entropy(..., ignore_index=255)` — tương thích với `weight` parameter
- ✅ `SegmentationTrainingConfig` dùng `dataclass(frozen=True)` với `asdict()` → tự động serialize vào checkpoint/JSON
- ✅ `build_checkpoint` lưu `training_config` → `cloud_class_weight` sẽ tự động được lưu
- ✅ Test `test_loss_all_invalid_batch_is_zero_and_does_not_step` sẽ không bị ảnh hưởng vì weight chỉ tác động lên valid pixels

### 4.3. Pitfall kỹ thuật cần chú ý

> [!CAUTION]
> Khi tạo `weight` tensor cho `F.cross_entropy`:
> ```python
> weight = torch.tensor([1.0, cloud_class_weight], device=logits.device, dtype=logits.dtype)
> ```
> - **dtype phải match logits** — nếu dùng AMP (float16), weight cũng phải float16
> - **device phải match** — không để weight trên CPU khi logits trên GPU
> - **Tạo mới mỗi lần gọi** hoặc cache cẩn thận — avoid stale references

Plan đã đề cập đúng điểm này ("Tạo tensor trọng số trên cùng device và dtype phù hợp với logits") — rất tốt.

---

## 5. Đánh giá quy trình thực nghiệm

### 5.1. Baseline recording (Mục 2) — Thiếu chi tiết

Plan yêu cầu ghi lại baseline nhưng **chưa nói rõ format output**. Khuyến nghị:
- Lưu thành JSON file chuẩn (đã có template từ [eval_segmentation.py](file:///c:/Users/phuhx1/Documents/cube_nano/src/eval_segmentation.py))
- Bao gồm: model checkpoint path, git commit hash, training config, **full evaluation report**

### 5.2. Threshold re-calibration (Mục 6, câu cuối) — Đúng nhưng cần nhấn mạnh hơn

> "Sau khi thay đổi loss, phải chạy lại bước chọn pixel probability threshold trên validation"

Đây là **điểm cực kỳ quan trọng** mà plan nên đặt thành bước riêng trong mục 7 (Thứ tự triển khai), không chỉ là ghi chú ở mục 6. Code [select_pixel_threshold](file:///c:/Users/phuhx1/Documents/cube_nano/src/eval_segmentation.py#L119-L150) đã sẵn sàng cho bước này.

### 5.3. Thiếu statistical significance test

Với 4 cấu hình ablation, kết quả có thể nằm trong **noise range**. Khuyến nghị:
- Chạy mỗi cấu hình với **3 seeds khác nhau** (42, 123, 456)
- Hoặc dùng bootstrap CI từ [bootstrap_scene_metric](file:///c:/Users/phuhx1/Documents/cube_nano/src/eval_segmentation.py#L228-L241) (đã có sẵn!) để xác nhận improvement là có ý nghĩa thống kê

---

## 6. Tóm tắt đánh giá

| Khía cạnh | Điểm | Ghi chú |
|---|---|---|
| Lý thuyết ML | 8/10 | Đúng hướng, nhưng thiếu phân tích class imbalance |
| Thiết kế kỹ thuật | 9/10 | Thay đổi nhỏ, tương thích tốt, rủi ro thấp |
| Quy trình thực nghiệm | 7/10 | Thiếu multi-seed, decision criteria, convergence analysis |
| Code coverage | 8/10 | Unit test tốt, nên thêm integration test end-to-end |
| Khả năng mở rộng | 7/10 | Tversky nên có plan cụ thể hơn, không chỉ "nếu cần" |
| Tài liệu | 7/10 | Cần giải thích rõ hơn cơ chế weighted CE |

### Điểm tổng: **7.7/10** — Phương án tốt, khuyến nghị triển khai với các bổ sung ở mục 3

---

## 7. Khuyến nghị hành động

1. **Thêm bước 0**: Thống kê class distribution trên training set
2. **Mở rộng ablation**: Thêm `cloud_class_weight = 1.2` vào danh sách
3. **Chạy multi-seed**: Ít nhất 2–3 seeds cho mỗi cấu hình
4. **Xác định decision criteria cứng** trước khi chạy ablation
5. **Tách threshold re-calibration** thành bước riêng trong thứ tự triển khai
6. **Giữ nguyên** $\alpha = \beta = 1.0$ (CE weight và Dice weight) trong mọi ablation run, chỉ thay đổi $W$
7. **Ghi nhận epoch hội tụ** và plot training curve cho mỗi cấu hình
