# Phản hồi phản biện: Đánh giá lại các nhận xét chuyên gia

> Tài liệu phản biện: [`expert_review_remediation_plan_rebuttal.md`](./expert_review_remediation_plan_rebuttal.md)  
> Bản nhận xét gốc: [`expert_review_remediation_plan.md`](./expert_review_remediation_plan.md)  
> Kế hoạch gốc: [`training_pipeline_remediation_plan.md`](./training_pipeline_remediation_plan.md)  
> Ngày phản hồi: 2026-07-14

---

## 1. Đánh giá tổng quan về bài phản biện

Bài phản biện **có chất lượng cao và đúng đắn ở nhiều điểm cốt lõi**. Đặc biệt, nó thể hiện sự cẩn trọng kỹ thuật đáng ghi nhận: yêu cầu citation cho các con số, phân biệt rõ giữa nhận định có bằng chứng và suy luận, và từ chối các kết luận tuyệt đối khi chưa có dữ liệu thực nghiệm.

Tuy nhiên, một số phản biện đang đánh đồng **đề xuất khởi đầu (starting point recommendation)** với **kết luận kỹ thuật cuối cùng (engineering specification)**, dẫn đến việc bác bỏ những gợi ý vốn không nhằm mục đích trở thành specification cứng.

Dưới đây tôi phản hồi từng điểm.

---

## 2. Phản hồi theo từng điểm

### 2.1. OOD bằng `[μ ± kσ]` — **Chấp nhận phần lớn**

**Phản biện đúng.** Tôi đồng ý trên các điểm:

- Reflectance không phải Gaussian, phân phối đa mode → `μ ± kσ` có false positive cao trên scene hợp lệ nhưng extreme (tuyết, sa mạc, cloud-heavy)
- Kiểm tra per-band độc lập bỏ sót inter-band correlation shift (ví dụ: band swap vẫn có thể pass)
- Semantic error (đảo band, sai processing level) có thể nằm trong valid range

**Phần tôi giữ lại:** Đề xuất gốc của tôi dùng từ "bắt đầu bằng" (start with), không phải "production-ready". Ý định là gợi ý **iteration path**: rule-based trước → learned sau, phù hợp với resource constraints ban đầu. Tuy nhiên, tôi thừa nhận diễn đạt "đủ cho edge deployment" là **quá mạnh** và không nên viết như vậy.

**Consensus:** Sử dụng hệ thống OOD phân tầng mà phản biện đề xuất:

| Tầng | Phương pháp | Action |
|---|---|---|
| 1 | Contract check (sensor, product, band identity, units) | Reject |
| 2 | Validity check (NoData ratio, physical bounds) | `invalid_input` |
| 3 | Robust statistics (quantile/MAD, không phải μ±kσ) | Warning flag |
| 4 | Learned methods (nếu cần, sau khi có đủ dữ liệu OOD) | Future |

Đây là kiến trúc tốt hơn đề xuất ban đầu của tôi.

---

### 2.2. Bảng single-model vs. per-sensor — **Chấp nhận phần lớn**

**Phản biện đúng ở ba điểm cốt lõi:**

1. Multi-sensor training có thể **cải thiện** generalization thay vì luôn compromise — đặc biệt khi per-sensor data ít, shared features giúp regularization
2. Per-sensor model không nhất thiết cần N pipeline riêng nếu chia sẻ adapter framework, InputSpec schema, và training code
3. Jetson có thể lazy-load engine theo sensor metadata, nên tổng RAM không phải constraint đúng

**Phần tôi điều chỉnh:** Bảng so sánh trong review gốc quá binary (✅/❌). Thực tế nằm trên một phổ liên tục và phụ thuộc vào dữ liệu. Bảng nên được viết lại với "depends on data" thay vì các kết luận tuyệt đối.

**Phần tôi giữ lại:** Kế hoạch gốc để mở quyết định này là đúng. Tôi đồng ý **không nên chốt sớm** trước khi có ablation trên sensor đích. Đề xuất 3 phương án ablation (single model / shared backbone + per-sensor head / per-sensor model) trong phản biện là đầy đủ và nên được thêm vào P6.

---

### 2.3. Label noise của 95-Cloud — **Chấp nhận**

**Phản biện đúng.** Tôi đã viết:

> *"95-Cloud có một số vấn đề đã biết: nhãn sai ở boundary pixels, thin cirrus annotation không nhất quán"*

Đây là kiến thức tôi tổng hợp từ kinh nghiệm chung với các cloud detection dataset dựa trên Landsat, nhưng **không có citation trực tiếp** cho 95-Cloud cụ thể. Phản biện đúng khi chỉ ra:

- 95-Cloud không phải Landsat 8 Biome mà là extension của 38-Cloud, ground truth tạo thủ công
- "Vấn đề đã biết" cần audit data, không phải suy luận từ dataset khác
- Boundary pixel error có impact khác nhau giữa pixel segmentation và patch classification

**Correction cần thiết:** Thay "vấn đề đã biết" bằng "rủi ro tiềm ẩn cần audit". Cách tiếp cận đúng:

1. Lấy mẫu crop gần cloud-ratio threshold (ví dụ: ratio ∈ [0.05, 0.15] nếu T = 0.10)
2. Review thủ công để đo disagreement rate
3. Báo cáo kết quả audit trước khi kết luận label noise level

---

### 2.4. Cloud-ratio threshold khác nhau giữa train và eval — **Chấp nhận**

**Phản biện đúng tuyệt đối.** Đây là lỗi diễn đạt trong review gốc của tôi.

Khi tôi viết "T có nên khác nhau giữa train và eval", ý định là gợi ý **ablation nhiều giá trị T** — nhưng cách diễn đạt tạo ấn tượng rằng có thể train với T₁ và eval với T₂, điều này **sai về mặt methodology**.

Phản biện mô tả chính xác quy trình đúng:

```
Với mỗi candidate T:
  → Tạo label nhất quán cho train/val/test
  → Train + validate
  → So sánh kết quả giữa các T
→ Chọn T tốt nhất
→ Khóa T trước final test
```

Cloud-ratio threshold T (label definition) và probability threshold τ (decision boundary) là hai tham số độc lập — đúng.

---

### 2.5. Data versioning — **Chấp nhận một phần**

**Phản biện đúng khi nói** kế hoạch gốc đã có manifest hash, preprocessor version, và provenance. Nói "không có data versioning" là quá mức.

**Phần tôi giữ lại:** Kế hoạch có **traceability** nhưng chưa có **lifecycle management**. Cụ thể, thiếu:

- Dataset ID content-addressed (phản biện đề xuất hash composition rất tốt)
- Retention policy
- Rollback/rebuild procedure
- Dependency graph giữa dataset → split → InputSpec → checkpoint

Diễn đạt chính xác hơn: *"Kế hoạch có data provenance cơ bản nhưng chưa có data lifecycle management đầy đủ."*

Công thức dataset ID mà phản biện đề xuất:
```
raw_manifest_hash + preprocessor_version + preprocessing_params
+ split_manifest_hash + normalization_stats_hash
```
Là rất tốt và nên được adopt.

---

### 2.6. Augmentation — **Chấp nhận phần lớn**

**Phản biện đúng trên hai điểm quan trọng:**

1. **Channel dropout và augmentation giải quyết mục tiêu khác nhau.** Tắt channel dropout cho RGB không tạo "lỗ hổng" cần lấp bằng augmentation. Review gốc của tôi viết "augmentation strategy thay thế" — đây là diễn đạt sai vì ngụ ý hai thứ cùng vai trò.

2. **Augmentation phải phù hợp vật lý sensor, không phải ImageNet distribution.** Fine-tuning thay đổi toàn bộ feature distribution; ImageNet pretraining chỉ là initialization. Augmentation policy nên tuân theo physical constraints của product, không phải distribution mà backbone từng thấy.

**Phần tôi giữ lại:** Codebase đã có flip và rotation — tốt. Nhưng augmentation audit vẫn cần thiết:

- Liệu các augmentation hiện tại có được version hóa và lưu cùng checkpoint?
- Band-wise gain/noise, blur/resampling đã được xem xét chưa?
- Augmentation có được áp dụng đúng thứ tự so với normalization?

---

### 2.7. Physical units không bắt buộc cho mọi pipeline — **Chấp nhận**

**Phản biện đúng.** Tôi đồng ý:

- Rendered RGB (sau gamma, tone mapping) không thể khôi phục physical reflectance
- Cross-satellite vẫn khả thi nếu product representation nhất quán + train/inference parity
- Điều kiện bắt buộc là **versioned, product-aware input contract**, không phải physical units

Cách viết lại chính xác:

> Physical reflectance là representation ưu tiên cho analytic products. Product-aware input contract có version là điều kiện bắt buộc chung.

---

### 2.8. TensorRT compatibility — **Chấp nhận**

**Phản biện đúng.** JetPack version match chỉ là heuristic, không phải sufficient condition. Compatibility còn phụ thuộc TensorRT/CUDA version, GPU compute capability, plugin hashes, precision mode, builder flags, v.v.

Phương án phản biện đề xuất (ONNX portable → build on target → sidecar fingerprint → smoke test on device) là chính xác hơn và đã được kế hoạch gốc cover phần lớn.

**Tự phê:** Review gốc của tôi viết "JetPack version phải match" — đây là oversimplification.

---

### 2.9. Các con số không có nguồn — **Chấp nhận**

**Phản biện đúng tuyệt đối.** Các con số tôi dùng:

| Con số | Status |
|---|---|
| ~40% lỗi inference từ preprocessing mismatch | Không có citation. Đây là heuristic cá nhân từ kinh nghiệm, không phải fact |
| 2-4 tuần annotation per sensor | Ước lượng thô, phụ thuộc annotation level (patch vs pixel) và team size |
| 30-50 scene per sensor | Không có statistical power calculation |
| "đáng tin cậy hơn 90% pipeline" | Subjective, không có methodology đo |

**Correction:** Các con số này không nên được dùng làm engineering requirement. Nếu cần scene count, phải tính từ:
- Target error rate + confidence interval
- Cloud prevalence + strata coverage
- Correlation giữa patch trong cùng scene

---

## 3. Bảng tổng hợp: Consensus

| # | Điểm phản biện | Verdict | Hành động |
|---|---|---|---|
| 2.1 | OOD μ±kσ không đủ | ✅ Chấp nhận phần lớn | Dùng OOD phân tầng: contract → validity → robust stats |
| 2.2 | Single/per-sensor quá binary | ✅ Chấp nhận phần lớn | Không chốt sớm, thêm 3 ablation options vào P6 |
| 2.3 | Label noise chưa có nguồn | ✅ Chấp nhận | Thay "đã biết" → "cần audit", thêm audit procedure |
| 2.4 | T khác nhau train/eval | ✅ Chấp nhận | Sửa lại: mỗi T dùng nhất quán train/val/test |
| 2.5 | Data versioning đã có | ⚠️ Chấp nhận một phần | Thừa nhận có provenance, bổ sung lifecycle management |
| 2.6 | Augmentation ≠ thay thế dropout | ✅ Chấp nhận phần lớn | Sửa diễn đạt; giữ yêu cầu augmentation audit |
| 2.7 | Physical units không bắt buộc | ✅ Chấp nhận | Physical reflectance = ưu tiên, product contract = bắt buộc |
| 2.8 | TensorRT ≠ JetPack match | ✅ Chấp nhận | Sửa lại: sidecar fingerprint đầy đủ |
| 2.9 | Con số không có nguồn | ✅ Chấp nhận | Xóa hoặc gắn caveat rõ ràng |

**Tổng: 7/9 chấp nhận, 1 chấp nhận một phần, 1 chấp nhận phần lớn nhưng giữ augmentation audit.**

---

## 4. Các item nên đưa vào kế hoạch khắc phục (từ cả review + rebuttal)

Sau vòng phản biện, đây là các bổ sung **đã đạt consensus** nên được thêm vào remediation plan:

### Bổ sung vào P2 (Preprocessor)
1. Dataset ID content-addressed theo công thức hash composition
2. Retention policy cho raw/intermediate/release dataset
3. Rollback/rebuild procedure khi phát hiện lỗi preprocessor

### Bổ sung vào P5 (Retrain & Ablation)
4. Augmentation audit: version hóa policy, ablation band-wise gain/noise/blur, thứ tự so với normalization
5. Label-quality audit: lấy mẫu crop gần cloud-ratio threshold, đo disagreement rate
6. Ablation cloud-ratio threshold T: chạy nhiều T nhất quán cho train/val/test

### Bổ sung vào P6 (Cross-satellite)
7. Tách P6a (annotation + data curation) và P6b (training + evaluation)
8. Ba phương án ablation: single model / shared backbone + per-sensor head / per-sensor model
9. OOD detection phân tầng: contract check → validity check → robust statistics → learned (future)
10. Tính scene count từ target error rate + statistical power, không dùng con số cứng

### Bổ sung phase mới: Post-deployment Monitoring
11. Input contract violation rate + invalid/OOD rate
12. Band statistics + score distribution per sensor/product
13. Telemetry: latency, RAM, thermal, inference errors
14. Ground truth sampling mechanism cho performance drift detection

### Bổ sung vào TensorRT sidecar (P4)
15. Target device, OS/L4T, JetPack, TensorRT, CUDA, GPU capability, plugin hashes, builder flags, precision, ONNX checksum

---

## 5. Lời kết

Vòng phản biện này đã cải thiện đáng kể chất lượng các đề xuất bổ sung. Bài học chính:

1. **Đề xuất gợi ý (suggestion) phải tách rõ khỏi kết luận kỹ thuật (specification).** Review gốc của tôi trộn lẫn hai dạng này.
2. **Con số không có nguồn không nên xuất hiện trong tài liệu kỹ thuật**, dù chỉ mang tính minh họa.
3. **Quyết định kiến trúc lớn (single vs. multi-model, OOD method) cần ablation data, không nên chốt bằng suy luận.**

Kế hoạch gốc vẫn đúng hướng. Các bổ sung sau vòng phản biện nâng nó từ "kế hoạch training pipeline" lên "kế hoạch ML system lifecycle" — đây là mức cần thiết cho production satellite AI.
