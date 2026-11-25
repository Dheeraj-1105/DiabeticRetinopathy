import os

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_augmentation import create_augmentation_pipeline

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 5


def load_metadata(csv_path):
    """
    Load the APTOS metadata CSV.

    Parameters:
        csv_path (str): Path to train.csv.

    Returns:
        pd.DataFrame: Dataset metadata.
    """

    df = pd.read_csv(csv_path)

    required_columns = {"id_code", "diagnosis"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {required_columns}"
        )

    return df

def create_image_paths(df, image_dir):
    """
    Create complete image paths from image IDs.

    Parameters:
        df (pd.DataFrame): Dataset metadata.
        image_dir (str): Directory containing retinal images.

    Returns:
        pd.DataFrame: DataFrame with image paths.
    """

    df = df.copy()

    df["image_path"] = df["id_code"].apply(
        lambda x: os.path.join(image_dir, f"{x}.png")
    )

    return df

def validate_image_paths(df):
    """
    Remove records whose image files do not exist.
    """

    exists = df["image_path"].apply(os.path.exists)

    missing_count = (~exists).sum()

    if missing_count > 0:
        print(
            f"Warning: {missing_count} image files were not found."
        )

    return df[exists].reset_index(drop=True)

def split_dataset(
    df,
    test_size=0.15,
    validation_size=0.15,
    random_state=42
):
    """
    Create stratified train, validation and test splits.
    """

    train_df, temp_df = train_test_split(
        df,
        test_size=test_size + validation_size,
        stratify=df["diagnosis"],
        random_state=random_state
    )

    relative_validation_size = (
        validation_size /
        (test_size + validation_size)
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_validation_size,
        stratify=temp_df["diagnosis"],
        random_state=random_state
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )

def load_image_tf(image_path, label):
    """
    Load an image and convert it into a TensorFlow tensor.
    """

    image = tf.io.read_file(image_path)

    image = tf.image.decode_png(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label

def create_tf_dataset(
    df,
    batch_size=16,
    shuffle=False,
    training=False
):
    """
    Create a TensorFlow Dataset.

    Parameters:
        df: DataFrame containing image paths and labels.
        batch_size: Number of images per batch.
        shuffle: Whether to shuffle the dataset.
        training: Whether to apply data augmentation.

    Returns:
        TensorFlow Dataset.
    """

    image_paths = df["image_path"].values

    labels = df["diagnosis"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image_tf,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        augmentation = create_augmentation_pipeline()

        dataset = dataset.map(
            lambda image, label: (
                augmentation(image, training=True),
                label
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset