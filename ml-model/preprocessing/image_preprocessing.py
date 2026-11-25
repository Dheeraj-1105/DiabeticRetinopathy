import cv2
import numpy as np
import matplotlib.pyplot as plt
def load_image(image_path):
    """
    Load an image from disk.

    Parameters:
        image_path (str): Path to image.

    Returns:
        image (numpy.ndarray)
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def resize_image(image, target_size=(224, 224)):
    """
    Resize image to target dimensions.
    """

    return cv2.resize(image, target_size)



def display_image(image, title="Image"):

    plt.figure(figsize=(5,5))

    plt.imshow(image)

    plt.title(title)

    plt.axis("off")

    plt.show()

def crop_black_border(image: np.ndarray) -> np.ndarray:
    """
    Crop unnecessary black borders from a retinal image.

    Parameters:
        image (np.ndarray): RGB image

    Returns:
        np.ndarray: Cropped image
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create a binary mask of non-black pixels
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find coordinates of non-black pixels
    coords = cv2.findNonZero(thresh)

    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)

    cropped = image[y:y+h, x:x+w]

    return cropped    

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance local contrast using CLAHE.

    Parameters:
        image (np.ndarray): RGB image.

    Returns:
        np.ndarray: Contrast-enhanced RGB image.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2RGB
    )

    return enhanced_image

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values from [0, 255] to [0, 1].

    Parameters:
        image (np.ndarray): RGB image.

    Returns:
        np.ndarray: Normalized image.
    """

    return image.astype(np.float32) / 255.0