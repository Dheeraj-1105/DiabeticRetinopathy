from pathlib import Path

from backend.services.retinal_validator import (
    validate_retinal_image
)

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_DIR = (
    PROJECT_ROOT
    / "ml-model"
    / "dataset"
    / "retinal_validation"
    / "manual_test"
)

extensions = {".jpg", ".jpeg", ".png"}

for class_name in ["retinal", "non_retinal"]:

    print(f"\nTesting: {class_name}")
    print("-" * 50)

    class_dir = TEST_DIR / class_name

    for image_path in sorted(class_dir.rglob("*")):

        if (
            not image_path.is_file()
            or image_path.suffix.lower()
            not in extensions
        ):
            continue

        result = validate_retinal_image(image_path)

        print("Image:", image_path.name)

        print(
            "Retinal Score:",
            f"{result['retinal_score']:.10f}"
        )

        print(
            "Accepted:",
            result["is_retinal"]
        )

        print("-" * 50)