import os
import numpy as np
import tensorflow as tf
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(
        PROJECT_ROOT
        / "ml-model"
        / "saved_models"
        / "best_resnet50.keras"
    )
)

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]


# --------------------------------------------------
# Load the trained model
# --------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("ResNet50 model loaded successfully.")


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_dr(image_path):
    """
    Predict diabetic retinopathy severity
    from a retinal fundus image.
    """

    # Check whether image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Read image
    image = tf.io.read_file(image_path)

    # Decode image
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    # Resize to model input size
    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    # Convert to float32
    image = tf.cast(
        image,
        tf.float32
    )

    # Add batch dimension
    image = tf.expand_dims(
        image,
        axis=0
    )

    # Model prediction
    predictions = model.predict(
        image,
        verbose=0
    )[0]

    # Find class with highest probability
    predicted_class = int(
        np.argmax(predictions)
    )

    # Get confidence
    confidence = float(
        predictions[predicted_class]
    )

    # Convert class number to name
    predicted_label = CLASS_NAMES[
        predicted_class
    ]

    return {
        "class_id": predicted_class,
        "severity": predicted_label,
        "confidence": confidence,
        "probabilities": predictions.tolist()
    }
# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    test_image = str(
    PROJECT_ROOT
    / "ml-model"
    / "dataset"
    / "raw"
    / "train_images"
    / "3c1efa38d0da.png"
)

    result = predict_dr(test_image)

    print("\nPrediction Result")
    print("-----------------")

    print("Class ID:", result["class_id"])
    print("Severity:", result["severity"])
    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )

    print("\nClass Probabilities:")

    for class_name, probability in zip(
        CLASS_NAMES,
        result["probabilities"]
    ):
        print(
            f"{class_name}: "
            f"{probability * 100:.2f}%"
        )