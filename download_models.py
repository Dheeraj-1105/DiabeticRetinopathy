
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_DIR = (
    PROJECT_ROOT
    / "ml-model"
    / "saved_models"
)

MODEL_REPO_ID = "dheeruoleti/retinaai-models"

MODEL_REVISION = (
    "03810c6eb73d35744450dc862efda6faf808b42d"
)

MODEL_FILES = [
    "best_resnet50.keras",
    "retinal_validator.keras"
]


# ==================================================
# DOWNLOAD MODELS
# ==================================================

def download_models():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("RETINAAI MODEL DOWNLOAD")
    print("=" * 60)

    for filename in MODEL_FILES:

        model_path = MODEL_DIR / filename

        print(f"\nChecking model: {filename}")

        downloaded_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=filename,
            repo_type="model",
            revision=MODEL_REVISION,
            local_dir=str(MODEL_DIR),
            token=os.environ.get("HF_TOKEN") or None
        )

        downloaded_path = Path(downloaded_path)

        if not downloaded_path.is_file():
            raise FileNotFoundError(
                f"Model download failed: {filename}"
            )

        size_mb = (
            downloaded_path.stat().st_size
            / (1024 * 1024)
        )

        print(f"Model ready: {model_path}")
        print(f"Model size: {size_mb:.2f} MiB")

    print("\n" + "=" * 60)
    print("ALL RETINAAI MODELS ARE READY")
    print("=" * 60)


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    download_models()
