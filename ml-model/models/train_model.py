import tensorflow as tf


def compile_model(model, learning_rate):
    """
    Compile the DR classification model.
    """

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            "accuracy"
        ]
    )

    return model