import os
import asyncio
import time
import logging
from typing import List, Tuple, Dict, Any
from contextlib import asynccontextmanager
import numpy as np
import cv2
import psutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Request
from paddleocr import PaddleOCR
from paddlex.inference import load_pipeline_config
import concurrent.futures
from paddlex.inference.models.text_recognition.predictor import TextRecRunnerPredictor

# Set up logging for performance telemetry
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(process)d - %(message)s"
)
logger = logging.getLogger("paddleocr_api")

# -----------------------------------------------------------------------------
# 1. CPU AFFINITY BINDING (Lock each worker to a single CPU core for cache efficiency)
# -----------------------------------------------------------------------------
try:
    pid = os.getpid()
    process = psutil.Process(pid)
    # Check allowed CPUs for this container process to avoid invalid core binding errors on Docker
    if hasattr(os, "sched_getaffinity"):
        allowed_cpus = list(os.sched_getaffinity(0))
        cpu_count = len(allowed_cpus)
        assigned_cpu = allowed_cpus[pid % cpu_count]
    else:
        cpu_count = psutil.cpu_count(logical=False) or 1
        assigned_cpu = pid % cpu_count
        
    process.cpu_affinity([assigned_cpu])
    logger.info(f"[CPU Affinity] Worker process {pid} successfully bound to core {assigned_cpu}")
except Exception as e:
    logger.warning(f"[CPU Affinity] Could not set process affinity: {e}")

# -----------------------------------------------------------------------------
# 2. PARALLEL TEXT RECOGNITION (GIL Bypass via ThreadPoolExecutor)
# -----------------------------------------------------------------------------
original_rec_apply = TextRecRunnerPredictor.apply

def make_parallel_apply(original_apply, max_workers=2, chunk_size=12):
    def parallel_apply(self, input_list, **kwargs):
        if not isinstance(input_list, list) or len(input_list) <= chunk_size:
            yield from original_apply(self, input_list, **kwargs)
            return

        chunks = [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]
        
        def run_chunk(chunk):
            return list(original_apply(self, chunk, **kwargs))
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_chunk, chunk) for chunk in chunks]
            results = []
            for future in futures:
                results.extend(future.result())
            yield from results
                
    return parallel_apply

# WARNING: Do not increase max_workers beyond 2 in production if multiple Gunicorn
# processes are running. 4 workers * 2 threads = 8 total threads, which matches the
# physical CPU core count. Exceeding this will cause heavy context switching.
TextRecRunnerPredictor.apply = make_parallel_apply(original_rec_apply, max_workers=2, chunk_size=12)

# -----------------------------------------------------------------------------
# 3. ENVIRONMENT CONFIGURATION & THREAD TUNING
# -----------------------------------------------------------------------------
# Since each Gunicorn worker is bound to exactly 1 CPU core, we enforce 1 thread
# per process across all math/deep learning libraries to eliminate CPU context switching overhead.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PADDLE_PDX_CPU_NUM_THREADS"] = "1"

# -----------------------------------------------------------------------------
# 4. MODEL INITIALIZATION (ONNX Runtime configuration)
# -----------------------------------------------------------------------------
CPU_THREADS = int(os.environ.get("CPU_THREADS", "1"))

paddlex_config = load_pipeline_config("OCR")
paddlex_config["hpi_config"] = {"backend": "onnxruntime"}
paddlex_config['SubModules']['TextRecognition']['batch_size'] = 12

# Control thread count inside ONNX Runtime to prevent context switching
engine_config = {
    "intra_op_num_threads": CPU_THREADS,
    "inter_op_num_threads": CPU_THREADS
}

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="th_PP-OCRv5_mobile_rec",
    device="cpu",
    engine="onnxruntime",
    engine_config=engine_config,
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    det_limit_side_len=640,
    paddlex_config=paddlex_config
)

# -----------------------------------------------------------------------------
# 5. DYNAMIC BATCHING QUEUE SYSTEM
# -----------------------------------------------------------------------------
request_queue = asyncio.Queue()
BATCH_SIZE = 4
MAX_WAIT_TIME = 0.5  # seconds

def ocr_predict_batch(images: List[np.ndarray]) -> List[Dict[str, Any]]:
    """Executes CPU batch prediction synchronously inside the thread executor."""
    return ocr.predict(images)

async def ocr_batch_worker():
    """Asynchronous worker loop that builds batches dynamically from the queue with zero latency penalty."""
    logger.info("Background dynamic batch worker loop started.")
    while True:
        try:
            first_item = await request_queue.get()
        except asyncio.CancelledError:
            break
            
        batch = [first_item]
        
        # Non-blocking batch accumulation (GIL-friendly, zero delay for single requests)
        while len(batch) < BATCH_SIZE:
            if request_queue.empty():
                break
            try:
                item = request_queue.get_nowait()
                batch.append(item)
            except asyncio.QueueEmpty:
                break

        batch_size = len(batch)
        
        # Extract batch inputs
        images = [item[0] for item in batch]
        scales = [item[1] for item in batch]
        resized_flags = [item[2] for item in batch]
        formats = [item[3] for item in batch]
        futures = [item[4] for item in batch]
        
        try:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, ocr_predict_batch, images)
            
            # Map batch predictions back to their respective request futures
            for i, (future, result) in enumerate(zip(futures, results)):
                if future.cancelled():
                    continue
                
                scale = scales[i]
                resized = resized_flags[i]
                fmt = formats[i]
                
                formatted_result = []
                if result:
                    texts = result.get("rec_texts", [])
                    scores = result.get("rec_scores", [])
                    boxes = result.get("rec_boxes", [])
                    
                    for idx in range(len(texts)):
                        text = texts[idx]
                        score = scores[idx]
                        raw_box = boxes[idx]
                        
                        box_list = raw_box.tolist() if hasattr(raw_box, 'tolist') else list(raw_box)
                        
                        # Rescale coordinates to match original image dimensions
                        if resized:
                            if len(box_list) == 4:
                                xmin, ymin, xmax, ymax = [int(val / scale) for val in box_list]
                                polygon_box = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                            else:
                                polygon_box = [[int(pt[0] / scale), int(pt[1] / scale)] for pt in box_list]
                        else:
                            if len(box_list) == 4:
                                xmin, ymin, xmax, ymax = box_list
                                polygon_box = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                            else:
                                polygon_box = box_list
                        
                        formatted_result.append({
                            "text": text,
                            "confidence": round(float(score), 4),
                            "box": polygon_box
                        })
                
                if fmt == "text":
                    text_lines = [item["text"] for item in formatted_result]
                    full_text = "\n".join(text_lines)
                    response_obj = Response(content=full_text, media_type="text/plain; charset=utf-8")
                else:
                    response_obj = {"data": formatted_result}
                
                future.set_result(response_obj)
                
        except Exception as e:
            logger.error(f"Inference exception during batch execution: {e}", exc_info=True)
            for future in futures:
                if not future.cancelled():
                    future.set_exception(e)
        finally:
            for _ in range(batch_size):
                request_queue.task_done()

# -----------------------------------------------------------------------------
# 6. LIFESPAN MANAGEMENT
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the model on startup to compile the graph and warm up memory allocator
    try:
        logger.info("[Warmup] Starting eager OCR model warmup run...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        loop = asyncio.get_running_loop()
        # Run warm-up in executor to avoid blocking the event loop during server startup
        await loop.run_in_executor(None, ocr_predict_batch, [dummy_img])
        logger.info("[Warmup] OCR model warmup completed successfully.")
    except Exception as e:
        logger.warning(f"[Warmup] OCR model warmup failed: {e}")

    batch_worker_task = asyncio.create_task(ocr_batch_worker())
    app.state.batch_worker = batch_worker_task
    logger.info("Production dynamic batch worker activated.")
    yield
    batch_worker_task.cancel()
    try:
        await batch_worker_task
    except asyncio.CancelledError:
        pass
    logger.info("Production dynamic batch worker deactivated.")

app = FastAPI(
    title="PaddleOCR v6 Production API (CPU-Optimized)",
    lifespan=lifespan
)

from PIL import Image
import io

def verify_image_dimensions(contents: bytes, max_pixels: int = 16777216): # 4096x4096 px limit
    """Verify image dimensions using Pillow without decoding pixels into memory (Image Bomb mitigation)."""
    if not contents or len(contents) < 100:
        raise ValueError("ไฟล์รูปภาพว่างเปล่าหรือสั้นเกินกว่าจะตรวจเช็กขนาดได้")
    try:
        with Image.open(io.BytesIO(contents)) as img:
            w, h = img.size
            if w * h > max_pixels:
                raise ValueError(f"ขนาดพิกเซลของภาพใหญ่เกินไป: {w}x{h} ({w*h} px) เกินเกณฑ์สูงสุด {max_pixels} px")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError("ไฟล์รูปภาพชำรุดเสียหายหรือไม่สามารถอ่านมิติตัวอักษรของรูปภาพได้")

# -----------------------------------------------------------------------------
# 7. IMAGE PREPROCESSING & RESIZING (Robust Edge-case Handling)
# -----------------------------------------------------------------------------
def decode_and_resize_image(contents: bytes, max_target_side: int = 1024) -> Tuple[np.ndarray, float, bool]:
    """Decodes image and dynamically resizes it to the specified limit to preserve RAM."""
    verify_image_dimensions(contents)
    
    if not contents or len(contents) < 100:
        raise ValueError("ไฟล์รูปภาพว่างเปล่าหรือสั้นเกินกว่าจะถอดรหัสได้")
        
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError("ไฟล์ภาพชำรุด เสียหาย หรือถอดรหัสรูปภาพล้มเหลว")
    
    h_orig, w_orig = img.shape[:2]
    max_side = max(h_orig, w_orig)
    resized = False
    scale = 1.0
    
    # Resize keeping aspect ratio if any dimension exceeds max_target_side
    if max_side > max_target_side:
        scale = float(max_target_side) / max_side
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized = True
        
    return img, scale, resized

# -----------------------------------------------------------------------------
# 8. ENDPOINTS (Inputs Verification)
# -----------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "engine": "PaddleOCR-v6-ONNX", 
        "device": "CPU",
        "dynamic_batch_limit": BATCH_SIZE
    }

@app.post("/ocr")
async def run_ocr(request: Request, file: UploadFile = File(...), format: str = "json"):
    # 1. Validate size from Content-Length header first (zero-I/O check)
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > 5 * 1024 * 1024:
                logger.warning(f"[Size Limit] Rejected payload size of {content_length} bytes via header.")
                raise HTTPException(
                    status_code=413,
                    detail="ขนาดไฟล์รูปภาพห้ามเกิน 5MB (Request payload size exceeds 5MB limit)"
                )
        except ValueError:
            pass

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="ไฟล์ที่อัปโหลดไม่ใช่รูปภาพ")
    
    # Fail-fast: Return HTTP 429 Too Many Requests if queue is overloaded (>= 20)
    current_queue_size = request_queue.qsize()
    if current_queue_size >= 20:
        logger.warning(f"[Rate Limiting] Queue size is {current_queue_size}. Rejecting request with HTTP 429.")
        raise HTTPException(
            status_code=429,
            detail="เซิร์ฟเวอร์ประมวลผลงานหนาแน่นเกินไป กรุณารอและส่งคำขอใหม่อีกครั้งในภายหลัง (Server too busy. Please try again later.)"
        )
    
    try:
        # Read from file stream in chunks to limit memory allocation (Mitigation for header spoofing)
        contents_accumulator = bytearray()
        max_size = 5 * 1024 * 1024
        while True:
            chunk = await file.read(65536) # 64KB chunks
            if not chunk:
                break
            contents_accumulator.extend(chunk)
            if len(contents_accumulator) > max_size:
                logger.warning(f"[Size Limit] Payload size exceeded limit of {max_size} bytes during stream read.")
                raise HTTPException(
                    status_code=413,
                    detail="ขนาดไฟล์รูปภาพห้ามเกิน 5MB (File size exceeds 5MB limit)"
                )
        contents = bytes(contents_accumulator)
        
        loop = asyncio.get_running_loop()
        
        # Adaptive Proactive Resizing: if the queue is getting congested (>10),
        # automatically downscale the image aggressively to 800px instead of 1024px to speed up inference.
        max_target_side = 1024
        if current_queue_size > 10:
            max_target_side = 800
            logger.info(f"[Adaptive Resize] Queue size is {current_queue_size}. Scaling image down to {max_target_side}px to speed up processing.")
            
        # Offload CPU-heavy decoding and resizing to thread executor
        img, scale, resized = await loop.run_in_executor(None, decode_and_resize_image, contents, max_target_side)
        
        future = loop.create_future()
        await request_queue.put((img, scale, resized, format, future))
        
        try:
            response = await future
            return response
        except asyncio.CancelledError:
            future.cancel()
            raise

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"OCR request handling error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="เกิดข้อผิดพลาดภายในระบบในการประมวลผลรูปภาพ")