import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # Automatically initializes CUDA driver
import time

# Create a TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class CloudTRTInfer:
    def __init__(self, engine_path, channels=4, patch_size=256, threshold=0.5):
        """
        Khởi tạo bộ suy luận TensorRT từ file .engine

        Args:
            engine_path: Path to the serialized TensorRT engine.
            channels: Number of input channels (3 or 4).
            patch_size: Spatial size of each input patch.
            threshold: Probability threshold for cloud classification (default 0.5).
        """
        self.engine_path = engine_path
        self.channels = channels
        self.patch_size = patch_size
        self.threshold = threshold
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.load_engine()
        self.context = self.engine.create_execution_context()
        
        # Cấp phát bộ nhớ (Memory Allocation)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def load_engine(self):
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"Không tìm thấy file {self.engine_path}. Vui lòng chạy lệnh trtexec để build engine.")
            
        print(f"Đang nạp TensorRT Engine: {self.engine_path}")
        with open(self.engine_path, "rb") as f:
            return self.runtime.deserialize_cuda_engine(f.read())

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            shape = tuple(self.engine.get_binding_shape(binding))
            if any(dim < 0 for dim in shape):
                raise ValueError(
                    f"Dynamic TensorRT binding shape {shape} is not supported by this runtime. "
                    "Export/build a fixed-batch engine or add explicit optimization-profile handling."
                )
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Cấp phát bộ nhớ RAM (Host)
            host_mem = cuda.pagelocked_empty(size, dtype)
            # Cấp phát bộ nhớ VRAM (Device - GPU)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
                
        return inputs, outputs, bindings, stream

    def _prepare_input(self, img_patch):
        img_patch = np.asarray(img_patch, dtype=np.float32)
        if img_patch.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {img_patch.shape}")
        if img_patch.shape[2:] != (self.patch_size, self.patch_size):
            raise ValueError(
                f"Expected spatial shape ({self.patch_size}, {self.patch_size}), got {img_patch.shape[2:]}"
            )
        if img_patch.shape[1] not in (3, 4):
            raise ValueError(f"Expected 3 or 4 input channels, got {img_patch.shape[1]}")

        if img_patch.shape[1] < self.channels:
            pad = np.zeros(
                (img_patch.shape[0], self.channels - img_patch.shape[1], self.patch_size, self.patch_size),
                dtype=np.float32,
            )
            img_patch = np.concatenate([img_patch, pad], axis=1)
        elif img_patch.shape[1] > self.channels:
            img_patch = img_patch[:, :self.channels, :, :]

        return np.ascontiguousarray(img_patch)

    def infer(self, img_patch):
        """
        Thực hiện suy luận trên 1 patch ảnh 256x256xC
        Args:
            img_patch: numpy array (1, C, 256, 256), đã được chuẩn hóa [0, 1]
        Returns:
            bool: True nếu là mây, False nếu là clear
        """
        img_patch = self._prepare_input(img_patch)
        preds, probs = self.infer_batch(img_patch)
        return bool(preds[0]), float(probs[0])

    def infer_batch(self, img_batch):
        img_batch = self._prepare_input(img_batch)
        original_batch_size = img_batch.shape[0]
        values_per_patch = self.channels * self.patch_size * self.patch_size
        buffer_size = self.inputs[0]['host'].size

        if buffer_size % values_per_patch != 0:
            raise ValueError(
                f"TensorRT input buffer has {buffer_size} values, which is not divisible by one patch "
                f"({values_per_patch} values). Check engine input shape."
            )

        engine_batch_size = buffer_size // values_per_patch
        if original_batch_size > engine_batch_size:
            raise ValueError(
                f"Batch size {original_batch_size} exceeds TensorRT engine batch capacity {engine_batch_size}."
            )

        if original_batch_size < engine_batch_size:
            padding = np.zeros(
                (engine_batch_size - original_batch_size, self.channels, self.patch_size, self.patch_size),
                dtype=np.float32,
            )
            img_batch = np.concatenate([img_batch, padding], axis=0)

        np.copyto(self.inputs[0]['host'], img_batch.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        logits = self.outputs[0]['host'][:original_batch_size]
        logits = np.clip(logits, -500, 500)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs > self.threshold, probs

if __name__ == '__main__':
    # ------------------ TEST ------------------
    ENGINE_PATH = "cloud_model.engine"
    CHANNELS = 4
    
    # Tạo dữ liệu giả lập với kích thước input 256x256.
    dummy_input = np.random.rand(1, CHANNELS, 256, 256).astype(np.float32)
    
    try:
        # Khởi tạo mô hình
        trt_infer = CloudTRTInfer(ENGINE_PATH, channels=CHANNELS)
        
        # Warmup (chạy nháp 1 lần để khởi động GPU)
        trt_infer.infer(dummy_input)
        
        # Đo tốc độ
        start_time = time.time()
        for i in range(100):
            is_cloud, probability = trt_infer.infer(dummy_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        fps = 1.0 / avg_time
        
        print(f"--- Kết quả Test TensorRT ---")
        print(f"Dự đoán là mây: {is_cloud} (Xác suất: {probability:.4f})")
        print(f"Thời gian trung bình: {avg_time*1000:.2f} ms/patch")
        print(f"Tốc độ: {fps:.2f} FPS")
        print("Tốc độ thực tế trên Jetson Nano (Maxwell FP16) có thể lên tới hàng nghìn FPS.")
        
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Lỗi khi chạy TensorRT: {e}")
        print("\nĐể chạy được script này trên Jetson Nano, bạn cần cài đặt:")
        print("sudo apt-get install python3-libnvinfer python3-libnvinfer-dev")
        print("pip3 install pycuda")
