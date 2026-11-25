import tensorflow as tf


def create_augmentation_pipeline():
    """
    Create the data augmentation pipeline used
    during model training.
    """

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                mode="horizontal"
            ),

            tf.keras.layers.RandomRotation(
                factor=0.05
            ),

            tf.keras.layers.RandomZoom(
                height_factor=0.10,
                width_factor=0.10
            ),

            tf.keras.layers.RandomContrast(
                factor=0.10
            ),
        ],
        name="retinal_augmentation"
    )

    return augmentation