
# ==================================================
# BASE IMAGE
# ==================================================

FROM python:3.11-slim


# ==================================================
# ENVIRONMENT VARIABLES
# ==================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=-1
ENV PIP_NO_CACHE_DIR=1


# ==================================================
# SYSTEM DEPENDENCIES
# ==================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*


# ==================================================
# NON-ROOT USER
# ==================================================

RUN useradd -m -u 1000 user

ENV HOME=/home/user
ENV HF_HOME=/home/user/.cache/huggingface


# ==================================================
# APPLICATION DIRECTORY
# ==================================================

WORKDIR /app


# ==================================================
# PYTHON DEPENDENCIES
# ==================================================

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt


# ==================================================
# APPLICATION SOURCE CODE
# ==================================================

COPY --chown=user:user backend/ ./backend/
COPY --chown=user:user frontend/ ./frontend/
COPY --chown=user:user download_models.py .


# ==================================================
# MODEL STORAGE
# ==================================================

RUN mkdir -p /app/ml-model/saved_models && \
    chown -R user:user /app


# ==================================================
# MODEL CONFIGURATION
# ==================================================

ENV MODEL_PATH=/app/ml-model/saved_models/best_resnet50.keras

ENV VALIDATOR_MODEL_PATH=/app/ml-model/saved_models/retinal_validator.keras

ENV VALIDATOR_THRESHOLD=0.99


# ==================================================
# RUNTIME USER
# ==================================================

USER user


# ==================================================
# APPLICATION PORT
# ==================================================

EXPOSE 7860


# ==================================================
# START APPLICATION
# ==================================================

CMD ["sh", "-c", "python download_models.py && exec python -m uvicorn backend.app:app --host 0.0.0.0 --port 7860 --workers 1"]
