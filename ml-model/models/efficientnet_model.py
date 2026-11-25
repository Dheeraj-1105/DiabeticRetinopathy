import tensorflow as tf
from tensorflow.keras import layers, models


NUM_CLASSES = 5
IMAGE_SIZE = (224, 224)


def build_efficientnet_model(
    trainable=False,
    fine_tune_at=None
):
    """
    Build EfficientNetB0 model for 5-class
    diabetic retinopathy classification.

    Parameters:
        trainable:
            Whether the EfficientNet backbone should
            contain trainable layers.

        fine_tune_at:
            Layer index from which layers become trainable.
            Example: 200 means layers 200 onward are trainable.
    """

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    # Initially freeze the complete backbone
    base_model.trainable = False

    if trainable:

        base_model.trainable = True

        if fine_tune_at is not None:

            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False

            for layer in base_model.layers[fine_tune_at:]:
                layer.trainable = True

    inputs = tensorflow.keras.layers.Input(
        shape=(224, 224, 3)
    )

    x = base_model(
        inputs,
        training=False
    )

    x = tensorflow.keras.layers.GlobalAveragePooling2D()(x)

    x = tensorflow.keras.layers.Dropout(0.3)(x)

    outputs = tensorflow.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = tensorflow.keras.models.Model(
        inputs,
        outputs
    )

    return model