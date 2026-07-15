# Đánh giá Pipeline Training RGB

## 1. Phạm vi

Đánh giá này tập trung vào pipeline huấn luyện mô hình phát hiện mây với ảnh quang học ba kênh RGB. Các band NIR, SWIR, thermal, hyperspectral và SAR không thuộc phạm vi.

Repository hiện có hai môi trường training:

- Pipeline production trong `src/`.
- Notebook Kaggle `kaggle_train_cloud_model.ipynb`.

Hai pipeline này hiện chưa đồng nhất. Kết luận dưới đây phải được xem xét riêng cho từng môi trường.

## 2. Kết luận tổng quát

`src/` có thể train model RGB khi truyền `--channels 3`, nhưng các giá trị mặc định vẫn là bốn kênh và preprocessing chưa đáp ứng input contract cross-satellite. Đây chỉ nên được coi là baseline RGB trên 95-Cloud.

Notebook Kaggle còn một lỗi nghiêm trọng về nhãn: nhãn được lấy từ patch nguồn trước khi random crop, nên nội dung crop có thể không khớp với nhãn. Nếu notebook là nơi tạo checkpoint chính, checkpoint đó chưa đáng tin cậy cho đánh giá cuối cùng.

Đánh giá hiện tại: **pipeline `src/` đạt mức thử nghiệm có kiểm soát; chưa production-ready cho cross-satellite RGB**.

## 3. Phát hiện theo mức độ

### 3.1. Nghiêm trọng: Notebook tạo noisy label sau random crop

Trong [kaggle_train_cloud_model.ipynb](D:/AI20K/cube_nano/kaggle_train_cloud_model.ipynb:493), nhãn được tạo từ số lượng file trong thư mục `cloud` và `clear`:

```python
self.labels = [1.0] * len(self.cloud_files) + [0.0] * len(self.clear_files)
```

Sau đó notebook random crop ảnh tại dòng 527 nhưng vẫn trả về `self.labels[idx]` tại dòng 547. Một source patch có mây có thể sinh crop không có mây nhưng vẫn nhận nhãn `cloud`.

Hệ quả:

- Loss học trên nhãn không khớp nội dung ảnh.
- F1 và threshold bị lệch.
- `pos_weight` cũng được tính theo source patch tại dòng 754, không theo phân phối nhãn crop thực tế.

Pipeline `src/` đã cải thiện điểm này bằng cách ghép image-mask và tính lại cloud ratio của crop tại [cloud_dataset.py](D:/AI20K/cube_nano/src/data/cloud_dataset.py:180). Notebook cần được đồng bộ hoặc không dùng để tạo checkpoint cuối.

### 3.2. Cao: RGB chưa là mặc định end-to-end

Các thành phần vẫn mặc định `4` kênh:

- Preprocessing: [preprocess_95cloud.py](D:/AI20K/cube_nano/src/data/preprocess_95cloud.py:142)
- Training: [train.py](D:/AI20K/cube_nano/src/train.py:141)
- Evaluation: [eval.py](D:/AI20K/cube_nano/src/eval.py:18)
- ONNX export: [export_onnx.py](D:/AI20K/cube_nano/src/export_onnx.py:12)

Model RGB chỉ train đúng khi tất cả các bước đều dùng `channels=3`. Nếu train RGB nhưng eval hoặc export vẫn dùng mặc định bốn kênh, checkpoint sẽ không tương thích hoặc pipeline sẽ chạy sai cấu hình.

`channel_dropout` được thiết kế cho kênh thứ tư và không có ý nghĩa trong model RGB. Với RGB, cần đặt về `0` và lưu cấu hình này cùng model.

### 3.3. Cao: Normalization chưa product-aware

[cloud_dataset.py](D:/AI20K/cube_nano/src/data/cloud_dataset.py:126) chuẩn hóa float bằng `max()` và integer bằng cực đại của dtype. Cách này có thể dùng cho baseline 95-Cloud, nhưng không đủ cho cross-satellite vì `uint16` không cho biết pixel là DN, TOA reflectance hay surface reflectance.

Pipeline training hiện chưa lưu hoặc sử dụng thống nhất:

- Sensor và product.
- Processing level.
- Scale/offset radiometric.
- NoData và validity mask.
- GSD, CRS và resampling.
- Clip range, mean và std cố định theo band.

Nếu chuyển sang reflectance calibration, phải tái tạo dữ liệu train và retrain/fine-tune. Không được chỉ thay preprocessing ở inference.

### 3.4. Cao: Validation và test chưa đại diện đầy đủ cho scene

Train dùng random crop, nhưng validation và test chỉ dùng một center crop tại [cloud_dataset.py](D:/AI20K/cube_nano/src/data/cloud_dataset.py:141). Các vùng mây nằm ngoài trung tâm source patch không được đánh giá.

Nên dùng deterministic tiling hoặc một tập nhiều crop cố định cho mỗi scene, sau đó báo cáo metric ở cả mức crop và mức scene.

### 3.5. Trung bình: Checkpoint thiếu input contract

[train.py](D:/AI20K/cube_nano/src/train.py:266) chỉ lưu `state_dict`; training config được lưu thành file JSON riêng. Checkpoint không tự mang theo:

- Số kênh.
- Band order.
- Normalization.
- Product/processing level.
- Threshold.
- Epoch và optimizer state.

Điều này dễ dẫn đến việc nạp checkpoint RGB bằng model bốn kênh hoặc dùng preprocessing khác với lúc train. Nên lưu checkpoint dạng bundle gồm model state, optimizer/scheduler state và `InputSpec`.

### 3.6. Trung bình: Pretrained weights chưa có normalization được kiểm chứng

Training nạp MobileNetV3 pretrained tại [train.py](D:/AI20K/cube_nano/src/train.py:195), nhưng input hiện chỉ được scale về `[0, 1]`. Chưa có thí nghiệm xác định nên dùng ImageNet mean/std hay mean/std cố định của RGB 95-Cloud.

Đây không nhất thiết là bug làm training thất bại, nhưng là một lựa chọn ảnh hưởng đến hiệu quả fine-tuning và phải được ghi rõ trong input contract.

### 3.7. Trung bình: Metric và threshold còn hạn chế

Training hiện báo accuracy, precision, recall và F1 ở threshold cố định. Với mục tiêu sàng lọc ảnh trước downlink, cần bổ sung:

- False-clear rate.
- False-cloud rate.
- AUROC và PR-AUC.
- Calibration error hoặc Brier score.
- Threshold sweep theo chi phí giữ/loại ảnh.

Threshold `0.5` không nên được coi là tối ưu cho sensor mới.

### 3.8. Trung bình: Notebook chưa deterministic hoàn toàn

Notebook bật `torch.backends.cudnn.benchmark = True` tại [kaggle_train_cloud_model.ipynb](D:/AI20K/cube_nano/kaggle_train_cloud_model.ipynb:169). Vì vậy seed không bảo đảm kết quả tái lập hoàn toàn. `src/train.py` đặt `benchmark=False` và `deterministic=True`, phù hợp hơn cho reproducibility.

## 4. Điểm đã làm tốt trong `src/`

- Scene-level split được triển khai tại [split_dataset.py](D:/AI20K/cube_nano/src/data/split_dataset.py:67), giảm leakage giữa các patch cùng scene.
- Image và ground-truth mask được ghép theo filename.
- Nhãn được tính lại từ crop mask thay vì tin hoàn toàn vào thư mục `cloud/clear`.
- `pos_weight` được ước lượng từ các crop tại [train.py](D:/AI20K/cube_nano/src/train.py:203).
- Có seed, AMP, AdamW và cosine scheduler.
- Kiến trúc MobileNetV3 hỗ trợ trực tiếp `num_channels=3`.

## 5. Cách chạy baseline RGB hiện tại

Toàn bộ chuỗi phải dùng cùng cấu hình ba kênh:

```powershell
python src/data/preprocess_95cloud.py --channels 3
python src/data/split_dataset.py
python src/train.py --channels 3 --channel_dropout_p 0
python src/eval.py --channels 3
python src/export_onnx.py --channels 3
```

Đây chỉ là baseline hiện tại. Nó chưa phải pipeline cross-satellite vì normalization vẫn dựa trên dtype/max và chưa có calibration product-aware.

## 6. Thứ tự khắc phục đề xuất

### P0 - Chọn một pipeline training duy nhất

- Dùng `src/` làm pipeline chuẩn hoặc backport đầy đủ logic crop-mask vào notebook.
- Không tạo checkpoint chính từ notebook với logic nhãn hiện tại.
- Chốt model RGB `num_channels=3`.

### P1 - Đồng bộ cấu hình RGB

- Đổi mặc định preprocessing, train, eval và export thành ba kênh.
- Tự đọc channels từ checkpoint/config thay vì nhập lại thủ công.
- Loại bỏ hoặc vô hiệu hóa channel dropout trong RGB.

### P2 - Tạo preprocessor dùng chung

- Dùng cùng module cho train, eval và inference.
- Lưu sensor, product, units, scale/offset, GSD và normalization trong `InputSpec`.
- Không dùng `max()` hoặc cực đại dtype để suy đoán radiometry.

### P3 - Cải thiện đánh giá

- Validation/test bằng nhiều crop hoặc tiling theo scene.
- Bổ sung false-clear, PR-AUC và calibration.
- Hiệu chỉnh threshold theo mục tiêu downlink và sensor đích.

### P4 - Retrain cross-satellite

- Tái tạo dữ liệu Landsat bằng preprocessor mới.
- Fine-tune trên tập RGB của Sentinel-2/PlanetScope có nhãn.
- Đánh giá theo scene độc lập, vùng địa lý và mùa khác nhau.

## 7. Trạng thái kiểm tra

- Các file Python training chính đã qua `py_compile` thành công.
- Unit test chưa chạy được trong môi trường hiện tại vì thiếu `torch` và `tifffile`; hai package này đã được khai báo trong [requirements.txt](D:/AI20K/cube_nano/requirements.txt:1).
- Chưa chạy training thực tế vì workspace không có dataset 95-Cloud đã xử lý và môi trường chưa có các dependency runtime.

## 8. Tiêu chí hoàn thành training RGB

Pipeline chỉ được coi là sẵn sàng khi:

- Notebook và `src/` tạo cùng một loại nhãn.
- Train/eval/export đều tự nhận diện và dùng `channels=3`.
- Checkpoint lưu kèm `InputSpec` và normalization.
- Validation không chỉ dựa trên một center crop.
- NoData không bị gán thành clear.
- Có metric và threshold phù hợp với mục tiêu cross-satellite RGB.
- Model đã được kiểm chứng trên scene RGB của sensor đích chưa xuất hiện trong training.
