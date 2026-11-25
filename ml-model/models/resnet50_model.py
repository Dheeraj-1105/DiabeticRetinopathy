import tensorflow as tf
from keras import layers, models


NUM_CLASSES = 5
IMAGE_SIZE = (224, 224)


def build_resnet50_model(trainable=False, fine_tune_at=None):
    """
    Build a ResNet50 transfer-learning model
    for 5-class diabetic retinopathy classification.

    Classes:
        0 - No DR
        1 - Mild DR
        2 - Moderate DR
        3 - Severe DR
        4 - Proliferative DR

    Parameters:
        trainable:
            Whether the ResNet50 backbone should be trainable.

        fine_tune_at:
            Layer index from which fine-tuning begins.
    """

    # Load pretrained ResNet50
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    # Freeze the complete pretrained backbone
    base_model.trainable = False

    # Optional fine-tuning configuration
    if trainable:

        base_model.trainable = True

        if fine_tune_at is not None:

            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False

            for layer in base_model.layers[fine_tune_at:]:
                layer.trainable = True

    # Input
    inputs = layers.Input(
        shape=(224, 224, 3),
        name="retinal_image"
    )

    # ResNet50 feature extraction
    x = base_model(
        inputs,
        training=False
    )

    # Convert feature maps into a single vector
    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    # Reduce overfitting
    x = layers.Dropout(
        0.3,
        name="dropout"
    )(x)

    # Five DR severity classes
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="dr_severity"
    )(x)

    # Final model
    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="ResNet50_DR_Classifier"
    )

    return model