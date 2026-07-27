# Phản biện và đánh giá bản review kế hoạch mô phỏng GDS - vệ tinh

> Tài liệu được đánh giá: [review_gds_simulation_plan.md](review_gds_simulation_plan.md)  
> Kế hoạch gốc: [gds_satellite_ccsds_simulation_plan.md](gds_satellite_ccsds_simulation_plan.md)  
> Phạm vi kiểm tra: AI inference, web GDS, CCSDS TC/TM, F´ và khả năng vận hành  
> Ngày đánh giá: 2026-07-19

---

## 1. Kết luận điều hành

Bản review là một đầu vào tốt cho Gate 0, đặc biệt ở các vấn đề phục hồi truyền file, watchdog, back-pressure, điều phối telemetry và trải nghiệm degraded mode. Tuy nhiên, bản review chưa đủ chính xác để được áp dụng nguyên trạng.

Đánh giá tổng thể của tôi đối với bản review là khoảng **7/10**:

- Có khả năng phát hiện rủi ro vận hành tốt.
- Phân tích đúng nhiều vấn đề xuyên suốt AI, GDS và vệ tinh.
- Một số kết luận CCSDS/F´ chưa đúng với boundary của hệ thống hiện tại.
- Một số đề xuất AI và tài nguyên Jetson chưa dựa trên benchmark hoặc model contract.
- Bỏ sót một số lỗi có thể làm kết quả cloud filtering sai về mặt nghiệp vụ.

Kết luận sử dụng: **chấp nhận có điều kiện làm đầu vào chỉnh sửa kế hoạch, không chấp nhận như kết luận kỹ thuật cuối cùng**.

---

## 2. Phạm vi và phương pháp kiểm tra

Đánh giá này được thực hiện bằng cách:

1. Đối chiếu toàn bộ bản review với kế hoạch gốc.
2. Kiểm tra checkpoint và kiến trúc model trong repository.
3. Kiểm tra preprocessing, patch labeling và cách tính cloud coverage hiện tại.
4. Kiểm tra F´ dictionary, cấu hình khung TM và hành vi sequence counter.
5. Đối chiếu với implementation F´ v4.1.0 và tài liệu CCSDS hiện hành.
6. Kiểm tra trạng thái test hiện có trong repository.

Điểm quan trọng là phải phân biệt ba lớp:

- CCSDS packet và transfer frame.
- Sync/channel coding như CLTU, ASM, BCH, LDPC hoặc Turbo.
- Transport mô phỏng hiện tại qua UDP.

Nếu không giữ đúng boundary này, rất dễ coi một thành phần vật lý/link-layer là bắt buộc trong MVP software-in-the-loop.

---

## 3. Các phát hiện mức P0

### 3.1. Hợp đồng đầu vào AI chưa được khóa

Checkpoint hiện tại có convolution đầu tiên nhận **3 kênh**, trong khi helper model mặc định dùng 4 kênh tại:

- src/models/mobilenetv3.py:5
- src/inference_large_image.py

Bản review đúng khi phát hiện sự không thống nhất này, nhưng đề xuất đổi mặc định toàn hệ thống từ 4 xuống 3 là chưa đủ an toàn. Cách đó có thể làm hỏng luồng RGB+NIR trong tương lai và tiếp tục để model phụ thuộc vào implicit defaults.

Giải pháp đúng là đóng gói model cùng một InputSpec bất biến, tối thiểu gồm:

- Số kênh và thứ tự band.
- Kiểu dữ liệu và miền giá trị.
- Quy tắc scale/normalize chính xác.
- Patch size, stride và padding policy.
- Sensor, product level và phiên bản preprocessing.
- Model version, checksum và calibration version.

Satellite phải fail fast nếu scene hoặc request không phù hợp với InputSpec.

### 3.2. Normalization là rủi ro khoa học, không chỉ là cấu hình

Pipeline hiện tại đang normalize integer raster theo cực đại của kiểu dữ liệu:

- src/data/cloud_dataset.py:126 và 135
- src/inference.py:13 và 26
- src/input_contract.py:260, 417 và 424

Do đó, không thể tùy ý thay bằng chia 10000 hoặc percentile stretch chỉ vì đây là dữ liệu viễn thám. Cả hai lựa chọn đều có thể làm lệch phân phối đầu vào so với lúc huấn luyện.

Yêu cầu Gate 0:

1. Khôi phục chính xác preprocessing đã dùng khi tạo checkpoint.
2. Ghi preprocessing đó vào model manifest có version.
3. Tạo golden input/output và probability tolerance.
4. Nếu không thể khôi phục, phải huấn luyện hoặc hiệu chỉnh lại model.

### 3.3. Cloud coverage hiện không phải phần trăm pixel mây

Model hiện tại là patch classifier. Dataset gán cả patch là cloud khi tỷ lệ mask thật đạt từ 10%, tại:

- src/data/cloud_dataset.py:153
- src/data/cloud_dataset.py:187

Inference sau đó tô toàn bộ patch dương tính thành vùng mây. Vì vậy, một patch có 10% pixel mây có thể đóng góp 100% diện tích patch vào cloud coverage.

Chỉ số hiện tại nên được gọi là:

**cloud-positive tile area ratio**

Không nên hiển thị hoặc truyền trong TM như phần trăm pixel mây nếu chưa đổi sang segmentation.

Gate 0 phải lựa chọn một trong hai hướng:

- Chấp nhận chỉ số proxy, đổi tên và ghi rõ sai số/ngữ nghĩa.
- Dùng model segmentation hoặc cloud mask pixel-level.

Bản review có nhắc độ thô của tỷ lệ nhưng chưa đánh giá đúng mức độ ảnh hưởng tới quyết định giữ/loại ảnh.

### 3.4. Kết quả ROI chưa ổn định theo tọa độ

Kế hoạch hiện cho tiling bắt đầu từ gốc ROI và zero-pad ở mép. Khi người dùng dịch ROI chỉ một pixel, toàn bộ lưới patch có thể thay đổi, dẫn đến cloud score thay đổi lớn dù vùng địa lý gần như giống nhau.

Việc trừ padding khỏi mẫu số không giải quyết vấn đề vì padding vẫn ảnh hưởng đến đầu vào model.

Cần xác định rõ:

- Lưới patch neo theo gốc toàn scene, hoặc dùng overlap/aggregation.
- Minimum valid pixel fraction cho patch biên.
- NoData/mask policy.
- Half-open pixel bounds dạng [x, x+w) và [y, y+h).
- Quy tắc rounding duy nhất giữa trình duyệt, GDS và satellite.

Đây là rủi ro quan trọng mà bản review bỏ sót.

### 3.5. Request chưa chụp lại cấu hình xử lý

Kế hoạch có lệnh thay đổi model threshold và coverage limit, nhưng ROI request không mang đầy đủ config revision. Nếu threshold thay đổi trong lúc job đang nằm trong queue, kết quả sẽ phụ thuộc vào thời điểm worker bắt đầu chạy thay vì nội dung lệnh đã được chấp nhận.

Mỗi job cần lưu bất biến:

- Request ID.
- Scene ID và ROI pixel bounds.
- Model threshold.
- Coverage limit.
- Model/input-spec version.
- Config revision.
- Thuật toán tiling và coverage version.

Duplicate request chỉ được coi là idempotent khi toàn bộ payload và config snapshot giống nhau.

### 3.6. Phục hồi truyền sản phẩm chưa được thiết kế đầy đủ

Bản review đúng khi yêu cầu xử lý:

- Mất frame hoặc packet.
- Packet đến sai thứ tự.
- Timeout giữa chừng.
- Checksum thất bại.
- Restart GDS hoặc satellite trong lúc truyền.

Tuy nhiên, không nên thiết kế lại file protocol từ đầu. F´ Fw::FilePacket đã có START, DATA, END, CANCEL, sequence index, byte offset, length và checksum.

Phần còn thiếu là mission policy:

- Retry toàn file hay partial resend.
- Resume theo offset hoặc theo sequence.
- Retention của nguồn sản phẩm trên satellite.
- Giới hạn retry và trạng thái thất bại cuối.
- Có dùng CFDP acknowledged mode trong phase sau hay không.

COP-1 bảo vệ TC link; nó không tự giải quyết mất dữ liệu TM file.

### 3.7. Watchdog, back-pressure và điều phối telemetry phải có tiêu chí đo được

Bản review đúng về nhu cầu watchdog và queue control. Kế hoạch đã có bounded queue và queue-depth telemetry, nhưng chưa đủ để vận hành.

Cần bổ sung:

- Maximum queue depth cụ thể.
- Admission response QUEUE_FULL hoặc BUSY; không silent drop.
- Processing deadline suy ra từ kích thước ROI.
- Heartbeat của AI worker và supervisor restart.
- Trạng thái FAILED_RETRYABLE cho job bị crash.
- DEGRADED khi worker chưa sẵn sàng.
- Latched FAULT cho lỗi model contract hoặc storage integrity.
- Priority và maximum burst giữa command, event và file telemetry.
- Starvation bound có thể test được.

---

## 4. Các nhận định trong review cần hiệu chỉnh

| Nhận định của review | Kết quả kiểm tra |
|---|---|
| Kế hoạch thiếu CLTU | Chưa chính xác. Kế hoạch đã nêu CLTU/BCH ở phần mở rộng. Với UDP nhận TC Transfer Frame, CLTU không bắt buộc trong MVP. Nó cần khi mô phỏng sync/channel coding hoặc nối SDR. |
| Kế hoạch thiếu ASM | Chưa chính xác theo boundary hiện tại. ASM thuộc sync/channel coding, không phải trường của TM Transfer Frame. Marker còn phụ thuộc coding profile; không nên áp dụng cứng 0x1ACFFC1D cho mọi cấu hình. |
| Khung TM 1024 byte còn khoảng 1014 byte payload | Sai với cấu hình F´ v4.1.0 đang dùng. TM primary header là 6 byte, FECF là 2 byte và OCF không hiện diện, nên TM Data Field là 1016 byte. File goodput thực tế còn phải trừ Space Packet và FilePacket headers. |
| Nên tăng frame lên 2048 vì file ảnh lớn | Chưa đủ căn cứ. Frame size phải được quyết định theo channel coding, BER/fault model, buffer, latency và link budget. Frame lớn giảm overhead nhưng tăng lượng dữ liệu phải retransmit khi lỗi. |
| Cần quyết định sequence counter reset hay rollover | CCSDS Packet Sequence Count là 14 bit; F´ ApidManager rollover modulo 16384. Kế hoạch đã có test rollover. |
| Bắt buộc dùng CCSDS secondary header cho timestamp | Không bắt buộc. Timestamp có thể nằm trong application payload. Điều bắt buộc là định nghĩa time source, epoch, resolution, synchronization state và stale-time behavior. |
| Một TM VC chắc chắn gây starvation | Chưa đủ. Starvation phụ thuộc scheduler và queue policy. Một VC vẫn có thể dùng được nếu có priority, burst limit và starvation bound. Nhiều VC là một lựa chọn kiến trúc, không phải sửa lỗi duy nhất. |
| Đổi mặc định num_channels từ 4 thành 3 | Chỉ giải quyết triệu chứng. Phải dùng model manifest/InputSpec và validate checkpoint contract. |
| Jetson Nano có 4 GB VRAM, batch 4-8 là hợp lý | Sai về mô hình tài nguyên. Nano dùng 4 GB bộ nhớ hệ thống chia sẻ. Batch phải benchmark cùng activation, TensorRT/CUDA workspace, image buffers và OS; batch 1 là mặc định bảo thủ phù hợp hơn. |
| Scene khoảng 350 MB | Sai khoảng hai lần. Scene đã kiểm tra có kích thước 723,362,624 byte, xấp xỉ 690 MiB. |
| WebGL nên là giải pháp mặc định | Chưa có benchmark. Canvas, WebGL hoặc tile viewer đều có thể phù hợp tùy quicklook size và interaction model. |
| Normalized coordinates bảo đảm crop chính xác | Không đủ. Tọa độ authoritative trên wire phải là pixel integer với origin và rounding rõ ràng. Normalized coordinates chỉ nên là representation của UI. |
| Observability, bounded queue và reconnect hoàn toàn bị thiếu | Kế hoạch đã có một phần các mục này. Review đúng rằng chúng chưa đo được, nhưng nên mô tả là chưa hoàn chỉnh thay vì hoàn toàn chưa có. |

---

## 5. Các điểm bản review đánh giá đúng

Các nhận định sau nên được tiếp thu:

1. Type-BD không cung cấp độ tin cậy tương đương Type-AD/COP-1.
2. Cần mô tả rõ phạm vi tuân thủ CCSDS thay vì dùng tuyên bố chung chung.
3. Cần cơ chế reassembly, timeout và integrity check cho sản phẩm downlink.
4. AI worker cần watchdog, deadline và restart policy.
5. Queue overflow phải có policy rõ ràng.
6. Command/event telemetry không được bị file transfer làm starvation.
7. Quicklook 16-bit cần tone mapping có version và metadata.
8. Webapp cần degraded mode, reconnect và khả năng resynchronize state.
9. Timeline hiện tại lạc quan, đặc biệt ở integration F´, raster windowing và product transfer.
10. Packet inspector cần retention, pagination và giới hạn dữ liệu thay vì giữ vô hạn trong browser.

---

## 6. Các vấn đề quan trọng bản review bỏ sót

### 6.1. Pin phiên bản F´

Dictionary trong repository ghi F´ v4.1.0, nhưng kế hoạch tham chiếu tài liệu dạng latest. Phiên bản chính thức mới hơn đã tồn tại, nên hành vi có thể drift.

Gate 0 phải:

- Pin F´ v4.1.0 và tài liệu/source tương ứng; hoặc
- Nâng cấp có chủ đích lên phiên bản mới và chạy lại compatibility tests.

Tham khảo: [F´ v4.2.2 release](https://github.com/nasa/fprime/releases/tag/v4.2.2).

### 6.2. Khả năng đọc window của raster

Kế hoạch đang đánh giá thấp độ khó của read_window:

- JP2 qua PIL có thể giải mã lại toàn ảnh cho từng window/strip.
- Compressed TIFF true streaming hiện chưa được hỗ trợ đầy đủ.
- Scene gần 690 MiB tạo áp lực lớn lên RAM và thời gian decode.

MVP cần giới hạn input ở uncompressed tiled/memmap TIFF, hoặc tích hợp backend windowed I/O thực sự như GDAL/rasterio với codec được kiểm chứng.

### 6.3. Hiệu lực khoa học của model

Test pass không chứng minh model phù hợp với sensor mục tiêu. Còn thiếu:

- Model card.
- Dataset/sensor provenance.
- Held-out metrics theo scene.
- PR-AUC và false-clear rate.
- Calibration report.
- Threshold selection rationale.
- Golden probability outputs.
- Domain-shift validation.

Đặc biệt, false clear có thể nguy hiểm hơn false reject trong bài toán quyết định downlink.

### 6.4. Giao dịch GDS

Để tránh mất hoặc gửi trùng lệnh khi process crash, GDS cần:

- Persisted outbox.
- Atomic DB admission.
- Unique request key.
- Trạng thái SENT/ACKED/FAILED có transition hợp lệ.
- Retry policy sau restart.
- API trả 202 Accepted cho command bất đồng bộ.

### 6.5. Đồng bộ WebSocket

Reconnect đơn thuần không bảo đảm state đúng. Cần:

- Event sequence/cursor.
- Gap detection.
- REST snapshot.
- Resubscribe từ cursor hoặc full resync.
- Quy tắc xử lý event trùng và event đến sai thứ tự.

### 6.6. Bảo mật và boundary triển khai

Kế hoạch có nhắc Auth/RBAC nhưng chưa quyết định security profile.

Phải chọn rõ:

- Local SIL chỉ bind localhost và ghi rõ không production-ready; hoặc
- Có authentication, TLS, CORS/CSRF/WS-origin policy, secret handling, rate limit, body-size limit, download authorization và audit retention.

### 6.7. SQLite và telemetry volume

SQLite có thể phù hợp với MVP nhưng cần:

- WAL mode.
- Single writer hoặc batch writes.
- Retention/downsampling.
- Rolling raw-frame files thay vì ghi mọi byte/frame vào bảng không giới hạn.
- Index và pagination cho packet inspector.

### 6.8. Fault injection phải tái lập được

Fixed random seed chưa đủ khi có concurrency. Cần xác định:

- Simulation clock.
- Thứ tự áp dụng drop, corruption, delay và reorder.
- Probability distribution.
- Queue overflow behavior.
- Deterministic scheduling hoặc event log để replay.

### 6.9. Scene catalog authority

Chưa rõ catalog do GDS hay satellite làm nguồn dữ liệu chuẩn. Cần định nghĩa:

- Ai sở hữu scene metadata.
- Cách đồng bộ catalog.
- Hành vi khi GDS biết scene nhưng satellite không có file.
- Version/hash của scene và quicklook.

---

## 7. Thứ tự cập nhật kế hoạch được đề xuất

### P0 - phải đóng trước implementation

1. Khóa model InputSpec, preprocessing và model manifest.
2. Quyết định ngữ nghĩa cloud coverage: tile proxy hay pixel segmentation.
3. Chốt ROI coordinate, grid anchor, padding và NoData policy.
4. Pin phiên bản F´ và lập conformance matrix theo đúng boundary.
5. Hoàn thiện request ID, config snapshot và idempotency persistence.
6. Chốt product reassembly/retry/integrity policy.
7. Định nghĩa watchdog, deadlines, queue limits và scheduling bounds.
8. Giới hạn format raster MVP hoặc chọn backend true windowed I/O.

### P1 - hoàn thiện trước demo end-to-end

1. Persisted outbox và restart recovery cho GDS.
2. WebSocket cursor, snapshot và resync.
3. Tone mapping và tile/quicklook strategy.
4. SQLite retention, batching và packet-inspector pagination.
5. Security profile cho local simulation hoặc deployment.
6. Deterministic fault injection và replay.
7. Scene catalog authority và synchronization contract.
8. Benchmark Jetson với memory watermark và batch auto-tuning.

---

## 8. Đánh giá trạng thái kiểm thử

Bộ test hiện tại chạy đạt:

- 53 tests passed.
- 9 subtests passed.

Đây là baseline phần mềm tốt, nhưng chưa bao phủ đầy đủ:

- Golden inference của checkpoint thật.
- Calibration/threshold correctness.
- ROI grid-shift invariance.
- Out-of-order hoặc missing product packets.
- Restart giữa command và downlink.
- WebSocket gap/resync.
- Queue saturation và starvation bound.
- Large-scene memory watermark trên Jetson.
- Security negative tests.

---

## 9. Phán quyết cuối

Bản review nên được dùng để cải thiện kế hoạch, đặc biệt về reliability và vận hành. Tuy nhiên, cần sửa các kết luận về CLTU/ASM, TM frame overhead, secondary header, sequence rollover, kích thước scene, Jetson memory và batch size.

Quan trọng hơn, bản kế hoạch cập nhật phải bổ sung những vấn đề mà review chưa phát hiện: ngữ nghĩa thật của cloud coverage, độ ổn định của lưới ROI, model/preprocessing provenance, config race, F´ version pinning, khả năng đọc raster theo window, transactional outbox, state resynchronization và security boundary.

Chỉ nên thông qua Gate 0 sau khi các mục P0 ở trên được chuyển thành quyết định kỹ thuật có thể kiểm thử, không chỉ là ghi chú hoặc định hướng.
