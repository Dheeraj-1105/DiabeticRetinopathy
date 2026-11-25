import os
import numpy as np
import tensorflow as tf
import cv2

from backend.prediction import (
    model,
    IMAGE_SIZE,
    PROJECT_ROOT
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = r"D:\AI-Powered-Diabetic-Retinopathy-System"

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ml-model",
    "saved_models",
    "best_resnet50.keras"
)

IMAGE_SIZE = (224, 224)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("ResNet50 model loaded for Grad-CAM.")


# --------------------------------------------------
# Access ResNet50 backbone
# --------------------------------------------------

base_model = model.get_layer("resnet50")

# Target convolutional layer
last_conv_layer = base_model.get_layer(
    "conv5_block3_out"
)


# --------------------------------------------------
# Feature extraction model
# --------------------------------------------------

feature_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=last_conv_layer.output
)


# --------------------------------------------------
# Access classification layers
# --------------------------------------------------

gap_layer = model.get_layer(
    "global_average_pooling"
)

dropout_layer = model.get_layer(
    "dropout"
)

classifier_layer = model.get_layer(
    "dr_severity"
)


# --------------------------------------------------
# Load and preprocess image
# --------------------------------------------------

def load_image(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = tf.io.read_file(
        image_path
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

    image_batch = tf.expand_dims(
        image,
        axis=0
    )

    return image, image_batch


# --------------------------------------------------
# Generate Grad-CAM heatmap
# --------------------------------------------------

def make_gradcam_heatmap(
    image_batch,
    target_class=None
):

    with tf.GradientTape() as tape:

        # Get feature maps
        feature_maps = feature_model(
            image_batch,
            training=False
        )

        # Apply the same classification pathway
        x = gap_layer(
            feature_maps
        )

        x = dropout_layer(
            x,
            training=False
        )

        predictions = classifier_layer(
            x
        )

        # If target class isn't provided,
        # use the predicted class
        if target_class is None:

            target_class = tf.argmax(
                predictions[0]
            )

        class_output = predictions[
            :,
            target_class
        ]

    # Calculate gradients
    gradients = tape.gradient(
        class_output,
        feature_maps
    )

    # Average gradients spatially
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    feature_maps = feature_maps[0]

    pooled_gradients = pooled_gradients[0]

    # Weight feature maps
    weighted_feature_maps = (
        feature_maps *
        pooled_gradients
    )

    # Combine channels
    heatmap = tf.reduce_sum(
        weighted_feature_maps,
        axis=-1
    )

    # Keep positive activations
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize to 0–1
    max_value = tf.reduce_max(
        heatmap
    )

    if max_value != 0:

        heatmap /= max_value

    return (
        heatmap.numpy(),
        predictions.numpy()
    )


# --------------------------------------------------
# Create heatmap overlay
# --------------------------------------------------

def create_gradcam_overlay(
    image,
    heatmap,
    alpha=0.4
):

    # Convert TensorFlow image to NumPy
    image = image.numpy().astype(
        np.uint8
    )

    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(
        heatmap,
        (
            image.shape[1],
            image.shape[0]
        )
    )

    # Convert heatmap to 0–255
    heatmap_uint8 = np.uint8(
        255 * heatmap_resized
    )

    # Apply OpenCV color map
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # OpenCV uses BGR, convert to RGB
    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # Create overlay
    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        heatmap_color,
        alpha,
        0
    )

    return overlay


# --------------------------------------------------
# Complete Grad-CAM prediction
# --------------------------------------------------

def generate_gradcam(
    image_path
):

    # Load image
    image, image_batch = load_image(
        image_path
    )

    # Generate heatmap
    heatmap, predictions = (
        make_gradcam_heatmap(
            image_batch
        )
    )

    # Predicted class
    predicted_class = int(
        np.argmax(predictions[0])
    )

    # Confidence
    confidence = float(
        predictions[0][predicted_class]
    )

    # Create overlay
    overlay = create_gradcam_overlay(
        image,
        heatmap
    )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "heatmap": heatmap,
        "overlay": overlay,
        "probabilities": predictions[0]
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

    result = generate_gradcam(test_image)

    print("\nGrad-CAM Result")
    print("-----------------")

    print(
        "Predicted Class:",
        result["predicted_class"]
    )

    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        "Heatmap Shape:",
        result["heatmap"].shape
    )

    output_path = PROJECT_ROOT / "backend" / "outputs"

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    success = cv2.imwrite(
        str(output_path / "gradcam_test.jpg"),
        cv2.cvtColor(
            result["overlay"],
            cv2.COLOR_RGB2BGR
        )
    )

    if success:
        print("Grad-CAM image saved successfully.")
    else:
        raise RuntimeError("Failed to save Grad-CAM image.")