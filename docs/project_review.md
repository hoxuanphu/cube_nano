# 🔍 Nhận xét Kỹ thuật — Dự án `cube_nano` Cloud Detection

> **Góc nhìn:** Senior AI/CV Engineer — chuyên xử lý ảnh vệ tinh & triển khai Edge AI
>
> **Phiên bản:** v2 — đã cập nhật sau phản hồi, bổ sung ngữ cảnh từ [kaggle_train_cloud_model.ipynb](file:///d:/AI20K/cube_nano/kaggle_train_cloud_model.ipynb)

---

## 1. Đánh giá Tổng quan

Dự án được thiết kế **rõ ràng, logic**, pipeline từ tiền xử lý → train → export → inference hoàn chỉnh. Việc chọn MobileNetV3-Small + TensorRT trên Jetson Nano cho bài toán cloud detection trên vệ tinh nhỏ là **hợp lý về mặt chiến lược tổng thể**. Tuy nhiên, có một số điểm cần trao đổi nghiêm túc.

> [!NOTE]
> Dự án có hai "môi trường" code: thư mục `src/` (production scripts) và `kaggle_train_cloud_model.ipynb` (training notebook). Notebook đã giải quyết một số vấn đề mà `src/` chưa có (scene-level split, pos_weight, seed, AMP). Các nhận xét dưới đây phân biệt rõ trạng thái từng phần.

---

## 2. Vấn đề Lớn Nhất: Classification vs Segmentation

> [!CAUTION]
> **Đây là quyết định kiến trúc quan trọng nhất của dự án, và cần được cân nhắc kỹ.**

Dự án hiện đang đóng khung bài toán cloud detection thành **Image-level Classification** (cloud hay clear trên từng patch 256×256). Điều này có nghĩa:

- Mỗi patch 256×256 chỉ nhận được **1 nhãn duy nhất** (cloud/clear)
- Ngưỡng 5% (`cloud_threshold=0.05`) quyết định nhãn → patch chỉ có 5% pixel mây vẫn bị gán "cloud"
- Kết quả inference trên ảnh lớn là một **cloud mask theo khối** (block artifact), không phải pixel-level

**Ưu điểm thực tế của cách tiếp cận này:**
- Model cực nhẹ (MobileNetV3-Small ~2.5M params), phù hợp tài nguyên hạn chế trên CubeSat
- Inference nhanh, đơn giản, dễ triển khai
- Đủ chính xác cho mục đích **sàng lọc nhanh**: lọc bỏ ảnh/vùng ảnh bị mây trước khi downlink

**Hạn chế cần nhận thức rõ:**
- Không có khả năng phát hiện ranh giới mây chính xác (pixel-level)
- Cloud mask đầu ra có độ phân giải thô (mỗi block 256×256 = 1 giá trị)
- Không phù hợp nếu yêu cầu downstream là cloud removal hay composite ảnh quang học

**Nhận xét:** Nếu mục tiêu là "on-board screening" — chụp xong → phát hiện mây → quyết định có downlink ảnh không — thì classification patch-level là **hoàn toàn đủ và thậm chí tối ưu hơn segmentation** về mặt tính toán. Đây là một trade-off hợp lý nếu đã được cân nhắc có chủ đích.

---

## 3. Nhận xét về Kiến trúc Model

### ✅ Điểm tốt
- **MobileNetV3-Small** là lựa chọn hợp lý cho edge device, TensorRT hỗ trợ tốt tất cả các op
- **Kế thừa pretrained ImageNet**: Tận dụng được feature extraction đã học, giảm thời gian train
- **Khởi tạo kênh NIR bằng copy trọng số Red**: Thông minh — Red và NIR có tương quan cao trong ảnh viễn thám (cùng phản xạ mạnh ở thực vật)

### ⚠️ Điểm cần lưu ý
- **Không có lớp Dropout trước classifier cuối**: MobileNetV3-Small gốc có `Dropout(p=0.2)` ở classifier, nhưng cần đảm bảo nó vẫn còn sau khi thay `classifier[3]`. Đoạn code hiện tại **chỉ thay `classifier[3]`** → Dropout ở `classifier[2]` vẫn giữ nguyên → **OK**, nhưng nên verify lại khi debug.
- **Thiếu cơ chế điều chỉnh confidence threshold**: Hiện tại hardcode `0.5`. Trong production trên vệ tinh, bạn có thể muốn threshold thấp hơn (ví dụ 0.3) để "thà báo nhầm mây còn hơn bỏ sót" → giảm lãng phí bandwidth downlink. ✅ **Đã sửa**: thêm `--threshold` arg vào tất cả scripts.

---

## 4. Nhận xét về Data Pipeline

### ✅ Điểm tốt
- **38-Cloud dataset** là benchmark phổ biến, chất lượng annotation tốt
- Logic tiền xử lý rõ ràng: đọc từng kênh TIF → stack → cắt patch → gán nhãn
- **Channel Dropout** (p=0.3) là kỹ thuật đúng đắn, được document rõ trong [multichannel_training_strategy.md](./multichannel_training_strategy.md)

### ⚠️ Vấn đề cần quan tâm

**a) Mất cân bằng dữ liệu (Class Imbalance)**
- 38-Cloud dataset thường có tỷ lệ clear >> cloud patches
- ~~Dự án **không có xử lý class imbalance**~~ → **Đính chính:** `src/train.py` ban đầu không có, nhưng notebook đã có `use_pos_weight` và `BCEWithLogitsLoss(pos_weight=...)`. ✅ **Đã backport** `pos_weight` về `src/train.py`.

**b) Augmentation quá cơ bản**
- Chỉ dùng D4 (flip + rotation) — đây là augmentation tối thiểu cho ảnh vệ tinh
- **Thiếu**: color jitter, random brightness/contrast, random scale/crop, mixup/cutmix
- Ảnh vệ tinh có đặc thù: góc mặt trời khác nhau, mùa khác nhau, sensor khác nhau → augmentation mạnh hơn sẽ giúp generalization đáng kể
- Tuy nhiên, với resource hạn chế trên CubeSat training pipeline (train trên Kaggle), đây cũng có thể là trade-off có chủ đích

**c) Normalization strategy**
- **Đính chính nhận xét gốc:** Reviewer ban đầu nói logic dựa vào `img.max() > 255` cho cả integer, nhưng thực tế code tách rõ hai nhánh:
  - **Integer dtype** → chia theo `np.iinfo(dtype).max` (ví dụ 65535 cho uint16, 255 cho uint8) — **đúng và ổn định**
  - **Float dtype** → dùng `img.max() > 1.0` rồi `img.max() > 255.0` để chọn scale — vẫn có rủi ro nếu ảnh float 16-bit đã calibrate nhưng giá trị max < 255 (ví dụ vùng biển tối)
- Khuyến nghị: với nhánh float, nên có option để chỉ định fixed scale thay vì hoàn toàn dựa vào max()

**d) Data leakage tiềm ẩn**
- ~~Script `split_dataset.py` chia theo file-level random~~ → **Đính chính:** notebook đã có scene-level split. `src/data/split_dataset.py` ban đầu random theo file. ✅ **Đã backport** scene-level split về `src/data/split_dataset.py`.

---

## 5. Nhận xét về Training Pipeline

### ✅ Điểm tốt
- `AdamW` + `CosineAnnealingLR`: Combo chuẩn, ổn định
- Lưu cả `best_model.pth` (theo F1) và `last_model.pth`: Thực hành tốt
- Metrics đầy đủ: Accuracy, Precision, Recall, F1

### ⚠️ Điểm cần cải thiện
- ~~**Thiếu seed**~~ → **Đính chính:** notebook có `set_seed()`. ✅ **Đã backport** về `src/train.py`.
- **Thiếu Early Stopping**: 20 epochs cứng → có thể overfitting hoặc lãng phí compute
- ~~**Thiếu logging**: Không TensorBoard/WandB → khó theo dõi learning curve~~ → ✅ **Đã bổ sung** W&B logging tùy chọn trong `src/train.py` và cấu hình Kaggle Secret trong notebook.
- **LR = 1e-3 cho pretrained model**: Hơi cao. Thường fine-tune pretrained nên dùng 1e-4 ~ 3e-4, hoặc dùng discriminative learning rates (backbone lr thấp hơn classifier lr)
- **Không có gradient clipping**: Với BCEWithLogitsLoss + AdamW thường ổn, nhưng nên có để an toàn

---

## 6. Nhận xét về Deployment Pipeline

### ✅ Điểm tốt rõ ràng
- **ONNX opset 11** + **TensorRT FP16**: Lựa chọn chuẩn cho Jetson Nano
- Code inference TensorRT viết cẩn thận: handle padding batch, validate input shape
- Hỗ trợ nhiều format ảnh đầu vào (TIFF, HDF5, NetCDF, numpy) — rất thực tế cho ảnh vệ tinh
- Comment song ngữ Việt-Anh, dễ hiểu

### ⚠️ Điểm đáng bàn

**a) Sliding Window không overlap**
- Hiện tại patch liền kề nhau, không overlap → ranh giới giữa các patch có thể tạo artifact
- Với classification (không phải segmentation) thì đây không phải vấn đề lớn, nhưng nếu muốn kết quả mượt hơn, có thể dùng overlap + voting

**b) Không có quantization**
- Chỉ dùng FP16 trên TensorRT. Jetson Nano (Maxwell GPU) **không hỗ trợ INT8 native** → FP16 là tối ưu rồi
- Nhưng nếu tương lai chuyển sang Jetson Orin/Xavier, nên cân nhắc INT8 calibration

**c) Memory management trên ảnh lớn**
- ~~`_read_image()` đọc **toàn bộ ảnh vào RAM**~~ → ✅ **Đã sửa**: thêm `_get_image_shape()` và `_read_image_strip()` để đọc theo dải hàng (row strip). Mỗi strip chỉ chiếm `patch_size × W × C` bytes thay vì toàn bộ ảnh.

---

## 7. Đánh giá Chiến lược Multi-channel

Tài liệu [multichannel_training_strategy.md](./multichannel_training_strategy.md) phân tích 3 phương án rất chi tiết. Việc chọn **Phương án 1 (Zero-Padding + Channel Dropout)** là **đúng đắn** cho mục tiêu deploy TensorRT:

- Đồ thị tính toán cố định → TensorRT optimize tốt nhất
- Không cần branching logic → ONNX đơn giản, không bug dynamic shape
- Channel Dropout đảm bảo model vẫn hoạt động tốt khi chỉ có RGB

> [!TIP]
> Đây là phần được suy nghĩ kỹ nhất trong dự án. Rõ ràng người thiết kế có hiểu biết tốt về constraints của TensorRT.

---

## 8. Tổng kết & Xếp hạng

| Tiêu chí | Đánh giá | Ghi chú |
|----------|----------|---------|
| Kiến trúc model | ⭐⭐⭐⭐ | Phù hợp target, pretrained tốt |
| Framing bài toán | ⭐⭐⭐⭐ | Classification patch-level hợp lý cho on-board screening |
| Data pipeline | ⭐⭐⭐⭐ | Scene-level split + pos_weight đã có (notebook → backported) |
| Training pipeline | ⭐⭐⭐⭐ | Seed + AMP backported, đã có W&B logging. Vẫn thiếu early stopping |
| Deployment pipeline | ⭐⭐⭐⭐½ | TensorRT pipeline chuẩn, tile-based reading đã thêm |
| Documentation | ⭐⭐⭐⭐⭐ | Multichannel strategy doc rất tốt |
| Production readiness | ⭐⭐⭐½ | Configurable threshold đã thêm. Cần thêm monitoring |

### Điểm mạnh nổi bật
1. Pipeline end-to-end hoàn chỉnh, từ raw data → deploy
2. Chiến lược multi-channel được suy nghĩ kỹ và document rõ
3. Lựa chọn model/framework phù hợp constraint phần cứng

### Các cải thiện đã thực hiện (backport từ notebook + mới)
1. ✅ **Scene-level split**: `src/data/split_dataset.py` giờ split theo scene thay vì random file
2. ✅ **pos_weight**: `src/train.py` tự tính `pos_weight = clear/cloud` và truyền vào loss
3. ✅ **Seed**: `src/train.py` có `set_seed()` đảm bảo reproducibility
4. ✅ **AMP**: `src/train.py` hỗ trợ Automatic Mixed Precision
5. ✅ **Configurable threshold**: tất cả scripts hỗ trợ `--threshold` arg
6. ✅ **Tile-based reading**: `inference_large_image_trt.py` đọc ảnh theo row strip, tránh OOM

### Còn lại nên cải thiện
1. **Early stopping** theo val F1 patience
2. **Discriminative learning rates** (backbone lr < classifier lr)

---

> [!IMPORTANT]
> **Kết luận:** Dự án có nền tảng tốt và đi đúng hướng. Sau các cải thiện backport + bổ sung, các rủi ro production chính (data leakage, class imbalance, OOM, hardcoded threshold) đã được xử lý. Dự án sẵn sàng hơn cho deploy thực tế trên vệ tinh nhỏ.
