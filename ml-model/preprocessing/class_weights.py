import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def calculate_class_weights(labels):
    """
    Calculate balanced class weights.

    Parameters:
        labels (array-like): Integer class labels.

    Returns:
        dict: Mapping from class index to class weight.
    """

    classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }

    return class_weights