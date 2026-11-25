
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)


def build_retinal_validator(
    input_shape=INPUT_SHAPE,
    learning_rate=0.001
):

    # Pretrained feature extraction backbone
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Input layer
    inputs = layers.Input(
        shape=input_shape,
        name="input_image"
    )

    # Training-only data augmentation
    x = layers.RandomFlip(
        "horizontal"
    )(inputs)

    x = layers.RandomRotation(
        0.05
    )(x)

    x = layers.RandomZoom(
        0.1
    )(x)

    # MobileNetV2 preprocessing
    # Convert pixel values from [0, 255] to [-1, 1]
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1
    )(x)

    # Feature extraction
    x = base_model(
        x,
        training=False
    )

    # Classification head
    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.Dense(
        128,
        activation="relu",
        name="dense_features"
    )(x)

    x = layers.Dropout(
        0.3,
        name="dropout"
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="retinal_probability"
    )(x)

    # Complete model
    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="Retinal_Image_Validator"
    )

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
            tf.keras.metrics.AUC(
                name="auc"
            )
        ]
    )

    return model
