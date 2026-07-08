FROM python:3.10-slim

# 1. Prevent python bytecode generation, lock CPU mode, and optimize thread configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    USE_GPU=False \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PADDLE_PDX_CPU_NUM_THREADS=1

# 2. Install essential system dependencies and clear apt cache in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 3. Create non-root application user and group for container security compliance
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -d /home/appuser appuser

WORKDIR /app

# 4. Copy requirements and install python packages with pip layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ && \
    pip install --no-cache-dir "paddleocr>=2.9.1"

# 5. Create app directory and set ownership to appuser
RUN chown -R appuser:appgroup /app

# 6. Switch to non-root execution
USER appuser

# Copy application source code
COPY --chown=appuser:appgroup . .

# 7. Pre-download and compile native model structures to ONNX during build stage
RUN python -c "from paddleocr import PaddleOCR; from paddlex.inference import load_pipeline_config; paddlex_config=load_pipeline_config('OCR'); paddlex_config['hpi_config']={'backend': 'onnxruntime'}; PaddleOCR(text_detection_model_name='PP-OCRv5_mobile_det', text_recognition_model_name='th_PP-OCRv5_mobile_rec', device='cpu', engine='onnxruntime', enable_mkldnn=False, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, cpu_threads=1, paddlex_config=paddlex_config)"

# 8. Apply dynamic INT8 quantization to the recognition model at build-time
RUN python quantize_onnx_int8.py -i /home/appuser/.paddlex/official_models/th_PP-OCRv5_mobile_rec_onnx/inference.onnx -o /home/appuser/.paddlex/official_models/th_PP-OCRv5_mobile_rec_onnx/inference.onnx

EXPOSE 8000

# 9. Launch production server with 2 Workers utilizing the multi-process architecture to bypass GIL
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "app:app"]