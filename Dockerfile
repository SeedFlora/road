# The report target regenerates figures without a GPU or the original dataset.
FROM python:3.13.5-slim-bookworm@sha256:4c2cf9917bd1cbacc5e9b07320025bdb7cdf2df7b0ceaccb55e9dd7e30987419 AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-report.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-report.txt

FROM runtime AS report
COPY . .
CMD ["python", "reproduce.py", "report"]

# Match the original CUDA/PyTorch training environment.
FROM runtime AS experiments
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
    --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV YOLO_CONFIG_DIR=/tmp/ultralytics ROAD_DEVICE=0
CMD ["python", "reproduce.py", "check", "--data"]
