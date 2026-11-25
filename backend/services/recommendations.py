# --------------------------------------------------
# Diabetic Retinopathy Recommendation Engine
# --------------------------------------------------

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]


RECOMMENDATIONS = {

    0: [
        "Continue regular eye examinations as recommended by your eye-care professional.",
        "Maintain good blood-glucose control and follow your diabetes management plan.",
        "Monitor blood pressure and other health risk factors regularly.",
        "Maintain a healthy lifestyle with appropriate diet and physical activity.",
    ],

    1: [
        "Arrange regular follow-up eye examinations with an eye-care professional.",
        "Maintain good blood-glucose control according to your healthcare plan.",
        "Monitor blood pressure and other diabetes-related risk factors.",
        "Follow prescribed medications and diabetes-management guidance.",
        "Seek professional evaluation if you notice any changes in vision.",
    ],

    2: [
        "Schedule an eye-care professional evaluation and follow-up.",
        "Maintain careful blood-glucose control according to your healthcare plan.",
        "Monitor blood pressure and other diabetes-related risk factors.",
        "Continue prescribed diabetes-management and medication plans.",
        "Report new or worsening vision changes promptly to a healthcare professional.",
    ],

    3: [
        "Arrange a prompt evaluation by an eye-care professional.",
        "Follow your diabetes and blood-pressure management plan carefully.",
        "Do not delay recommended retinal examinations or follow-up appointments.",
        "Report new or worsening vision changes promptly.",
        "Follow treatment or monitoring recommendations provided by your healthcare professional.",
    ],

    4: [
        "Arrange prompt evaluation by an ophthalmologist or other qualified eye-care professional.",
        "Follow recommended retinal examinations and professional follow-up closely.",
        "Maintain careful diabetes and blood-pressure management.",
        "Report sudden or significant changes in vision immediately to a healthcare professional.",
        "Follow any treatment plan prescribed by your eye-care professional.",
    ]
}


def get_recommendations(class_id):
    """
    Return educational recommendations based
    on the predicted diabetic retinopathy class.
    """

    if class_id not in RECOMMENDATIONS:
        raise ValueError(
            f"Invalid DR class: {class_id}"
        )

    return {
        "severity": CLASS_NAMES[class_id],
        "recommendations": RECOMMENDATIONS[class_id],
        "disclaimer": (
            "This result is AI-assisted educational information "
            "and is not a medical diagnosis. Please consult a "
            "qualified eye-care professional for clinical evaluation "
            "and treatment decisions."
        )
    }


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    test_class = 2

    result = get_recommendations(
        test_class
    )

    print("\nRecommendation Result")
    print("----------------------")

    print(
        "Severity:",
        result["severity"]
    )

    print("\nRecommendations:")

    for i, recommendation in enumerate(
        result["recommendations"],
        start=1
    ):
        print(
            f"{i}. {recommendation}"
        )

    print("\nDisclaimer:")
    print(result["disclaimer"])