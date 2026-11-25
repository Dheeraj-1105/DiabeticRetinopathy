from backend.prediction import (
    predict_dr,
    PROJECT_ROOT
)
from backend.services.retinal_validator import (
    validate_retinal_image
)
from backend.services.gradcam import generate_gradcam

from backend.services.recommendations import get_recommendations
# --------------------------------------------------
# Class names
# --------------------------------------------------

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

class InvalidRetinalImageError(Exception):

    def __init__(self, message, retinal_score):

        super().__init__(message)

        self.retinal_score = retinal_score
# --------------------------------------------------
# Complete image analysis
# --------------------------------------------------

def analyze_image(image_path):

    # ----------------------------------------------
    # 1. Prediction
    # ----------------------------------------------
    # Step 1: Validate retinal fundus image
    validation = validate_retinal_image(image_path)

    if not validation["is_retinal"]:

        raise InvalidRetinalImageError(
            message=(
                "The uploaded image could not be "
                "validated as a retinal fundus photograph. "
                "Please upload a clear retinal fundus image."
            ),
            retinal_score=validation["retinal_score"]
        )

    # Continue existing DR analysis belowg
    prediction = predict_dr(
        image_path
    )

    class_id = prediction[
        "class_id"
    ]

    severity = prediction[
        "severity"
    ]

    confidence = prediction[
        "confidence"
    ]

    probabilities = prediction[
        "probabilities"
    ]


    # ----------------------------------------------
    # 2. Grad-CAM
    # ----------------------------------------------

    gradcam_result = generate_gradcam(
        image_path
    )

    heatmap = gradcam_result[
        "heatmap"
    ]

    overlay = gradcam_result[
        "overlay"
    ]


    # ----------------------------------------------
    # 3. Recommendations
    # ----------------------------------------------

    recommendation_result = (
        get_recommendations(
            class_id
        )
    )


    # ----------------------------------------------
    # 4. Combine everything
    # ----------------------------------------------

    result = {

        "class_id": class_id,

        "severity": severity,

        "confidence": confidence,

        "probabilities": probabilities,

        "recommendations":
            recommendation_result[
                "recommendations"
            ],

        "disclaimer":
            recommendation_result[
                "disclaimer"
            ],

        "heatmap": heatmap,

        "gradcam_overlay": overlay
    }


    return result


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

    result = analyze_image(
        test_image
    )


    print("\n")
    print("=" * 50)
    print("COMPLETE DR ANALYSIS")
    print("=" * 50)


    print(
        "\nSeverity:",
        result["severity"]
    )

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


    print("\nRecommendations:")

    for index, recommendation in enumerate(
        result["recommendations"],
        start=1
    ):

        print(
            f"{index}. {recommendation}"
        )


    print("\nDisclaimer:")

    print(
        result["disclaimer"]
    )


    print("\nGrad-CAM:")

    print(
        "Heatmap shape:",
        result["heatmap"].shape
    )

    print(
        "Grad-CAM overlay generated successfully."
    )