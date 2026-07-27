# Nhận xét chuyên gia: GDS Satellite CCSDS Simulation Plan

> Reviewer: AI/ML, Web Development, Satellite Communications  
> File reviewed: [gds_satellite_ccsds_simulation_plan.md](file:///d:/AI20K/cube_nano/docs/gds_satellite_ccsds_simulation_plan.md)  
> Date: 2026-07-19

---

## Tổng quan đánh giá

**Đây là một bản kế hoạch kỹ thuật có chất lượng cao**, thể hiện sự hiểu biết sâu sắc về cả ba domain: AI inference pipeline, CCSDS protocol stack, và GDS web architecture. Tài liệu hiếm thấy ở mức này khi kết hợp satellite comms với AI payload một cách có hệ thống.

**Điểm mạnh nổi bật:**
- Nguyên tắc thiết kế rất đúng (separation of concerns, idempotency, fail-fast)
- Rất trung thực về giới hạn hiện tại (Type-BD only, coarse tile coverage, mất georeference)
- Luồng TC/TM end-to-end rõ ràng, không có đường tắt UI → inference

**Tuy nhiên, vẫn có một số điểm cần bổ sung hoặc điều chỉnh:**

---

## I. GÓC ĐỘ TRUYỀN THÔNG VỆ TINH (CCSDS/Protocol)

### ✅ Điểm đúng và tốt

1. **Phân biệt rõ TC Type-BD vs Type-AD/COP-1** (Section 5.1, line 147-149): Đây là điểm mà rất nhiều dự án mô phỏng bỏ qua. Việc ghi rõ F Prime `TcDeframer` chỉ hỗ trợ Expedited Service là cực kỳ quan trọng — tránh tuyên bố sai về mức tuân thủ CCSDS.

2. **APID/VCID separation** (Section 5.2): Tách biệt APID cho TC commands (`0x120`), ACK/events (`0x121`), cloud results (`0x122`), và data products (`0x123`) là hợp lý, phù hợp với thực tế phân loại traffic trong mission operations.

3. **Timeout/retry thay vì COP-1** (line 320): Với Type-BD, frame mất là mất vĩnh viễn. Quyết định dùng idempotent retry ở tầng GDS là phương án thực tiễn đúng.

### ⚠️ Cần bổ sung / điều chỉnh

4. **Thiếu CLTU (Communications Link Transmission Unit) cho uplink**: Tài liệu mô tả TC Transfer Frame nhưng không đề cập CLTU framing — lớp bên dưới TC frame trong CCSDS uplink stack thực tế. Dù là simulation, nên ghi rõ:
   - MVP bỏ qua CLTU/BCH encoding
   - Nếu sau này nối SDR, CLTU là bắt buộc

5. **Thiếu ASM (Attached Sync Marker) cho TM downlink**: TM Transfer Frame thực tế cần ASM (`0x1ACFFC1D`) để đồng bộ frame. Trong simulation byte-level, điều này ảnh hưởng đến frame synchronization logic. Nên ghi rõ trong conformance matrix.

6. **TM frame 1024 byte — cần xác nhận lại**: CCSDS Blue Book 132.0-B cho phép frame size từ 7 đến 2048 bytes. Giá trị 1024 là hợp lệ nhưng khá nhỏ cho data product downlink (ảnh crop có thể hàng MB). Nên tính toán:
   - Overhead ratio: với primary header 6 bytes + trailer 2-4 bytes, 1024 byte frame cho ~1014 bytes payload effective
   - Thời gian downlink ảnh 1MB ≈ ~1000 frames — cần cân nhắc tăng frame size hoặc document rõ đây là design choice

7. **Sequence counter overflow**: Line 170 nói "sequence counter riêng theo APID" nhưng không ghi rõ hành vi khi counter wrap (14-bit = max 16383). Cần specify:
   - Counter resets sau 16383 hay rollover?
   - GDS reassembly phải xử lý wrap-around

8. **Thiếu mô tả cấu trúc Space Packet secondary header**: Line 138 ghi "Secondary header" là nâng cấp tương lai, nhưng nên define sớm ít nhất timestamp format (CCSDS Day Segmented Time Code hay epoch-based) vì nó ảnh hưởng đến telemetry correlation.

### 🔴 Rủi ro đáng chú ý

9. **Single VC cho TM trong MVP** (line 168): Khi satellite vừa gửi health TM vừa downlink data product trên cùng 1 VC, health frames sẽ bị starve nếu data product lớn. Dù đã ghi ưu tiên health/ACK (line 257), cần mechanism cụ thể: **interleave ratio** (ví dụ: 1 health frame cho mỗi N data frames) hoặc preemptive insertion.

---

## II. GÓC ĐỘ AI/ML

### ✅ Điểm đúng và tốt

10. **Nhận diện đúng giới hạn model** (line 524): MobileNetV3-Small là patch-level classifier, không phải semantic segmentation. `cloud_coverage` là coarse tile ratio. Ghi rõ điều này tránh hiểu nhầm nghiêm trọng.

11. **Tách biệt 2 ngưỡng runtime** (Section 6.1): `model_threshold` (patch classification) vs `coverage_limit` (area decision) — đây là thiết kế đúng về mặt ML operations. Nhiều hệ thống trộn lẫn hai khái niệm này.

12. **Fail-fast model manifest** (Section 8.2): Validate checkpoint SHA-256, channels, patch size trước inference — rất quan trọng cho deployment vệ tinh thực tế, nơi debug từ xa gần như không thể.

13. **Basis point encoding** (line 77): Dùng U16 `0..10000` thay vì float cho wire protocol — tránh hoàn toàn vấn đề float precision trên cross-platform serialization.

### ⚠️ Cần bổ sung / điều chỉnh

14. **Cross-reference model code cho thấy lỗ hổng 3 vs 4 channels**: File [mobilenetv3.py](file:///d:/AI20K/cube_nano/src/models/mobilenetv3.py) default `num_channels=4` (line 5), nhưng checkpoint thực tế train 3 channels (line 27 của plan). Tài liệu đã ghi nhận đúng (line 27, line 525) nhưng **chưa có action item cụ thể**:
    - Nên thêm vào Phase 0: sửa default `get_cloud_model()` về 3 channels, hoặc ít nhất đảm bảo manifest enforce channel count
    - Thêm unit test: load checkpoint với wrong channel count → phải raise error rõ ràng

15. **Thiếu normalization specification trong manifest**: Line 271 ghi `normalization_id: "..."` nhưng đây là critical:
    - Ảnh Sentinel-2 là `uint16` (line 36), cần divide `10000` hoặc percentile stretch?
    - Pipeline hiện tại normalize thế nào? Nếu model train với normalize khác inference runtime → sai kết quả silent
    - Phải fix `normalization_id` thành giá trị cụ thể trong Phase 0

16. **Batch size cho ROI inference**: Section 8.3 không specify batch size. Với Jetson Nano (4GB VRAM), batch MobileNetV3-Small 256×256×3 float32:
    - Mỗi patch ≈ 768KB input tensor
    - Batch 8 ≈ 6MB + model weights + activations
    - Nên thêm `max_batch_size` vào config, default conservative (4-8), và cho benchmark điều chỉnh

17. **Progress callback granularity**: Line 283 `progress_callback` — nên specify:
    - Report mỗi N patches hay mỗi % completion?
    - Progress TM frame rate có bị throttle không? (tránh flood TM channel khi ROI lớn với hàng nghìn patches)

---

## III. GÓC ĐỘ WEB DEVELOPMENT (GDS Webapp)

### ✅ Điểm đúng và tốt

18. **Layout kiểu operations console** (Section 11.1): "đậm, dễ scan, tối ưu cho thao tác lặp lại; không dùng bố cục landing page" — hoàn toàn đúng cho mission control UI. Đây là dấu hiệu người viết hiểu ops UX.

19. **Tile pyramid cho ảnh lớn** (Section 11.3): 10980×10980 uint16 ≈ 350MB raw — bắt buộc phải dùng tile service. Quyết định browser chỉ truy cập GDS storage (không đọc satellite volume) là đúng về mặt fidelity.

20. **ROI normalized coordinates** (line 382): Lưu normalized → không bị sai khi resize viewer → convert về pixel integer phía backend — pattern chuẩn cho map-based ROI tools.

21. **WebSocket cho realtime TM** (line 328, 344): Đúng choice cho telemetry streaming. REST cho commands, WS cho events/progress — separation of concerns tốt.

### ⚠️ Cần bổ sung / điều chỉnh

22. **React + OpenLayers nhưng thiếu performance budget**: Section 11 nêu OpenLayers viewer nhưng không đề cập:
    - Tile cache strategy (memory limit, LRU eviction)
    - Maximum concurrent tile requests
    - WebGL rendering hay Canvas 2D? (OpenLayers hỗ trợ cả hai — WebGL nên là default cho overlay mask)
    - Ảnh uint16 cần tone-mapping trước khi render 8-bit canvas — ai chịu trách nhiệm? Server-side tile gen hay client-side?

23. **Thiếu offline/degraded mode cho webapp**: Khi link simulator blackout, webapp nên:
    - Hiển thị rõ trạng thái "NO LINK" / "BLACKOUT"
    - Queue commands locally hay block?
    - Cached telemetry data có hiển thị stale indicator không?
    - Reconnect strategy cho WebSocket?

24. **Command confirmation UX** (line 380): "hiện command preview gồm scene, ROI, thresholds và request ID để operator xác nhận" — tốt, nhưng nên thêm:
    - Estimated downlink time dựa trên product size và link bandwidth
    - Warning nếu link đang có fault profile active (loss rate > 0)
    - Disable Send khi satellite state ≠ READY

25. **Packet inspector** (line 367): Rất hữu ích cho debug nhưng cần specify:
    - Hex dump hay parsed view hay cả hai?
    - Filter theo APID, direction, time range?
    - Buffer bao nhiêu frames? (tránh memory leak khi chạy lâu)
    - Export capability (PCAP-like format cho offline analysis)?

26. **Thiếu đề cập đến state management frontend**: React app với:
    - WebSocket streaming TM
    - REST polling commands
    - Map viewer state
    - ROI editing state
    
    → Cần state management rõ ràng (Context + useReducer, Zustand, hoặc Redux). Không specify sẽ dẫn đến prop drilling hoặc inconsistent state giữa panels.

---

## IV. GÓC ĐỘ KIẾN TRÚC HỆ THỐNG

### ✅ Điểm đúng

27. **Docker Compose cho SIL** (line 59): Hợp lý cho single-machine development.

28. **Correlation ID xuyên suốt** (line 76): `request_id` từ UI → TC → satellite → TM → product — pattern chuẩn cho distributed tracing.

29. **Idempotency design** (Section 6.2): `request_id` + TTL window + reject duplicate conflict — pattern production-grade cho unreliable link.

### ⚠️ Cần bổ sung

30. **Thiếu error recovery / watchdog cho satellite simulator**: Section 8.1 có state machine nhưng:
    - Transition từ `DEGRADED` về `READY` qua mechanism nào? Operator command? Auto-heal?
    - `FAULT` state có recoverable không hay phải restart process?
    - Worker process crash → state machine transition gì?
    - Timeout cho `PROCESSING` state → tự chuyển `DEGRADED` sau bao lâu?

31. **Thiếu back-pressure mechanism**: Khi GDS gửi nhiều commands liên tiếp:
    - Satellite queue depth maximum là bao nhiêu?
    - Khi queue đầy, reject TC hay drop silently?
    - GDS có biết queue depth để throttle không? (cần TM channel cho queue depth)

32. **Data product reassembly** (line 441): File downlink qua CCSDS TM frames cần:
    - Sequence numbering cho reassembly
    - Handling missing/out-of-order frames (nhất là khi link simulator inject loss)
    - Checksum verification after reassembly
    - Timeout nếu transfer bị gián đoạn giữa chừng
    
    Đây là phần phức tạp nhất và nên được detail hơn trong Phase 4.

33. **Thiếu observability/metrics cho system**: Ngoài telemetry channel, hệ thống cần:
    - GDS backend metrics (request latency, WS connection count, TC encode time)
    - Link simulator metrics (effective throughput, fault injection count)
    - Centralized logging format (structured JSON) để correlate events across components

---

## V. GÓC ĐỘ LỘ TRÌNH & ƯỚC LƯỢNG

### ✅ Hợp lý

34. **Phase ordering đúng**: Protocol foundation → Link → GDS Backend → Webapp. Không build UI trước khi có protocol layer.

35. **Exit criteria mỗi phase rõ ràng** — đây là điểm hiếm thấy trong planning docs.

### ⚠️ Concern

36. **Phase 2 (F Prime/CCSDS) 8-12 ngày có thể underestimate**: Nếu cần viết FPP component từ đầu (`CloudPayload`), build F Prime deployment, integrate IPC với AI worker, VÀ sinh golden vectors — đây là phase có rủi ro cao nhất. Suggest:
    - Tách Phase 2a: FPP component + dictionary (4-5 ngày)
    - Phase 2b: Integration với AI worker + golden vectors (5-7 ngày)

37. **Phase 5 (Webapp) 7-10 ngày cho OpenLayers + ROI + realtime TM + packet inspector** — aggressive nếu chỉ 1 dev. Suggest MVP webapp focus:
    - P0: Scene viewer + ROI + command panel
    - P1: Realtime TM timeline
    - P2: Packet inspector (có thể defer)

---

## VI. ĐÁNH GIÁ TỔNG KẾT

| Tiêu chí | Điểm (1-10) | Ghi chú |
|---|:---:|---|
| Độ chính xác CCSDS | **8/10** | Rất tốt cho simulation level; thiếu CLTU/ASM/secondary header detail |
| Thiết kế AI pipeline | **9/10** | Fail-fast, manifest, basis point encoding — production-grade thinking |
| Kiến trúc web/GDS | **7.5/10** | Solid foundation nhưng thiếu frontend state mgmt, performance budget, degraded mode |
| Completeness | **8/10** | Rất thorough; thiếu error recovery detail và data reassembly spec |
| Feasibility/Timeline | **7/10** | Phase 2 và 5 có thể cần thêm buffer |
| Trung thực về giới hạn | **10/10** | Hiếm thấy — ghi rõ mọi limitation thay vì overclaim |

### Khuyến nghị ưu tiên cao:

1. **Phase 0 — Fix `num_channels` default** trong model code và thêm manifest validation test
2. **Specify TM interleave ratio** cho health vs data product trên single VC
3. **Detail data product reassembly protocol** trước khi vào Phase 4
4. **Thêm error recovery transitions** cho satellite state machine
5. **Define normalization pipeline** cụ thể trong model manifest

> [!IMPORTANT]
> Tài liệu này đủ chất lượng để bắt đầu Gate 0 review. Các điểm thiếu có thể bổ sung trong quá trình review mà không cần viết lại kiến trúc.
