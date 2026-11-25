
import os
from pathlib import Path

import numpy as np
import tensorflow as tf


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_SIZE = (224, 224)

VALIDATOR_THRESHOLD = float(
    os.environ.get(
        "VALIDATOR_THRESHOLD",
        "0.99"
    )
)

if not 0.0 < VALIDATOR_THRESHOLD <= 1.0:
    raise ValueError("Invalid validator threshold.")


# Model location
VALIDATOR_MODEL_PATH = Path(
    os.environ.get(
        "VALIDATOR_MODEL_PATH",
        str(
            PROJECT_ROOT
            / "ml-model"
            / "saved_models"
            / "retinal_validator.keras"
        )
    )
)


# Verify model availability
if not VALIDATOR_MODEL_PATH.is_file():

    raise FileNotFoundError(
        f"Validator model not found: "
        f"{VALIDATOR_MODEL_PATH}"
    )


# Load model once during application startup
validator_model = tf.keras.models.load_model(
    VALIDATOR_MODEL_PATH,
    compile=False
)


def preprocess_validator_image(image_path):

    image = tf.io.read_file(
        str(image_path)
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


def validate_retinal_image(image_path):

    if not Path(image_path).is_file():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = preprocess_validator_image(
        image_path
    )

    score = float(
        validator_model(
            image,
            training=False
        ).numpy()[0][0]
    )

    if not np.isfinite(score):
        raise RuntimeError(
            "Invalid validator model output."
        )

    is_retinal = score >= VALIDATOR_THRESHOLD

    return {
        "is_retinal": bool(is_retinal),
        "retinal_score": score,
        "threshold": VALIDATOR_THRESHOLD,
        "message": (
            "Retinal fundus image accepted."
            if is_retinal
            else
            "Image could not be validated as a "
            "retinal fundus photograph."
        )
    }
