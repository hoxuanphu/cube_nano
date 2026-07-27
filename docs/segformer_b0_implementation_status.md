# Trạng thái triển khai SegFormer-B0

Nhánh triển khai: `feature/segformer-b0-integration`.

## Đã triển khai

- Contract task-aware cho `InputSpec`, `ModelOutputSpec`, `PostprocessSpec`, `ProductSpec`, `DecisionSpec`, `AcceptanceProfile` và `TargetDeploymentSpec`; SegFormer khóa cả pixel threshold lẫn coverage limit theo `DecisionSpec` ở admission, worker và inference.
- Migration manifest schema 1 cho MobileNetV3 và schema 2 cho `semantic_cloud_segmentation`.
- MiT-B0/SegFormer-B0 reference implementation hỗ trợ H×W linh hoạt khi huấn luyện; contract runtime/export vẫn cố định input `256x256`, output `[1, 2, 64, 64]`.
- Preprocessor 95-Cloud fail-closed với raw GT, cloud mask và validity mask riêng; split scene-level có lineage hash.
- Dataset segmentation có chế độ native-size giữ nguyên H×W 95-Cloud cho train/validation batch `1`, đồng thời giữ deterministic tiling cho runtime; synchronized augmentation, `ignore_index=255`, CE + masked Soft Dice và training/evaluation entry points.
- Reference postprocess softmax + bilinear + threshold theo DecisionSpec; stitching edge padding không tham gia coverage. Evaluation tạo calibration report từ validation với macro-scene, coverage error và bootstrap CI; không cho chọn threshold ở chế độ test.
- Worker/deployment profile routing theo một `model_task` tại startup; job snapshot và health telemetry ghi release/task/contract IDs.
- Product bundle có `cloud_mask.tif` và `validity_mask.tif`; GDS validator phân biệt coverage coarse của MobileNet và pixel coverage của SegFormer.
- Fixed-shape ONNX export và TensorRT logits adapter cho vertical slice.
- Profile SegFormer chỉ có thể `deployable` khi manifest `validated`, `calibration_id` không còn `none`, AcceptanceProfile đã `approved`, và acceptance/target/batch binding khớp manifest/profile. MobileNet hiện hành không đổi.

## Gate còn chặn release

- Chưa có trained SegFormer checkpoint đã hash và pretrained artifact được bàn giao.
- Chưa có ONNX/TensorRT engine build trên target cùng parity report.
- Chưa có frozen 95-Cloud raw manifest/split thực tế, validation threshold report và test metrics.
- `segformer_deployment_profile.yaml` cố ý giữ `deployable: false` cho đến khi các artifact trên tồn tại.

Profile mặc định vẫn là MobileNetV3 và rollback bằng cách kích hoạt lại `sat_ai/deployment_profile.yaml` hiện hành.

## Danh sách file

### File tạo mới

| File | Vai trò |
| --- | --- |
| `sat_ai/contracts.py` | Đọc và kiểm tra các contract task-aware của model, product và deployment. |
| `sat_ai/acceptance_profile.yaml` | Profile acceptance dùng chung cho các task inference. |
| `sat_ai/target_deployment_spec.yaml` | Ràng buộc target deployment cho vertical slice SegFormer. |
| `sat_ai/segformer_model_manifest.yaml` | Manifest model và contract của SegFormer-B0. |
| `sat_ai/segformer_deployment_profile.yaml` | Profile deployment SegFormer-B0, hiện vẫn chưa bật release. |
| `sat_ai/segmentation.py` | Postprocess logits, stitching và coverage cho semantic segmentation. |
| `src/models/segformer_b0.py` | Reference implementation MiT-B0/SegFormer-B0 nhận H×W linh hoạt khi train; export/runtime giữ shape đã pin. |
| `src/data/segmentation_dataset.py` | Dataset, tiling và augmentation đồng bộ cho segmentation. |
| `src/losses.py` | Masked Cross-Entropy và Soft Dice loss. |
| `src/train_segmentation.py` | Entry point huấn luyện SegFormer-B0. |
| `src/eval_segmentation.py` | Entry point đánh giá segmentation và threshold. |
| `src/export_segformer_onnx.py` | Export model SegFormer-B0 sang ONNX fixed-shape. |
| `src/segformer_engine_contract.py` | Adapter contract logits cho engine TensorRT. |
| `tests/test_segformer_integration.py` | Test tích hợp contract, preprocessing, inference và product validation. |
| `docs/segformer_b0_implementation_status.md` | Tài liệu trạng thái và inventory của implementation. |

### Artifact tạo trong quá trình kiểm tra

| File | Vai trò |
| --- | --- |
| `artifacts/benchmarks/local-cpu-pytorch-v2.json` | Kết quả benchmark tham chiếu MobileNetV3 trên local CPU, dùng làm baseline/rollback reference. |

### File hiện hữu đã cập nhật

| Nhóm | File |
| --- | --- |
| Runtime và deployment | `build_manifest.json`, `flight/cloud_payload.py`, `flight/deployment.py`, `flight/worker_client.py`, `sat_ai/deployment_profile.yaml`, `sat_ai/inference.py`, `sat_ai/manifest.py`, `sat_ai/model_runtime.py`, `sat_ai/products.py`, `sat_ai/worker_process.py` |
| GDS và protocol | `gds/product_store.py`, `protocol/slo_profile.yaml` |
| Data và export | `src/data/preprocess_95cloud.py`, `src/data/split_dataset.py`, `src/inference_tensorrt.py` |
| Benchmark và package | `sat_ai/benchmark.py`, `sat_ai/__init__.py` |

`docs/segformer_b0_integration_plan.md` và `docs/segformer_b0_integration_plan_review.md` là tài liệu đã có trong workspace trước khi triển khai, không thuộc danh sách file tạo mới.
