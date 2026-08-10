# GPU training image (CUDA 12.1, PyTorch 2.2.2).
# For CPU-only / smoke test: docker run --rm liver-lesion-poc bash scripts/smoke_test.sh
# For GPU: docker run --rm --gpus all liver-lesion-poc bash scripts/run_train.sh configs/ablation_all_8.yaml

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
        torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        nibabel==5.2.1 \
        scipy==1.13.0 \
        scikit-learn==1.4.2 \
        pyyaml==6.0.1 \
        matplotlib==3.8.4 \
        tqdm==4.66.2

# Copy project (data is excluded via .dockerignore / .gitignore)
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

RUN chmod +x scripts/*.sh

# Smoke test by default; override with docker run ... bash scripts/run_train.sh <config>
CMD ["bash", "scripts/smoke_test.sh"]
