import sys
from pathlib import Path
from PIL import Image
import tensorflow as tf

from src.config import MODEL_PATH, CLASS_NAMES_PATH
from src.model_utils import load_class_names, predict_emotion


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py path/to/image.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_class_names(CLASS_NAMES_PATH)

    image = Image.open(image_path)
    label, probabilities = predict_emotion(model, class_names, image)

    print(f"Predicted emotion: {label}")
    print("Probabilities:")
    for class_name, score in zip(class_names, probabilities):
        print(f"  {class_name}: {score:.4f}")


if __name__ == "__main__":
    main()
    