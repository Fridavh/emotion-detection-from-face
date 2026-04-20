from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import TRAIN_DIR, TEST_DIR, MODEL_PATH, CLASS_NAMES_PATH, PLOT_PATH, EPOCHS
from src.data_utils import get_class_names, load_train_dataset, load_test_dataset
from src.model_utils import build_model, save_class_names


def plot_history(history, save_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    print("Loading datasets...")
    class_names = get_class_names(TRAIN_DIR)
    train_dataset = load_train_dataset(TRAIN_DIR)
    test_dataset = load_test_dataset(TEST_DIR)

    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=autotune)
    test_dataset = test_dataset.cache().prefetch(buffer_size=autotune)

    print(f"Classes found: {class_names}")
    model = build_model(num_classes=len(class_names))
    model.summary()

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=EPOCHS,
    )

    print("Evaluating model...")
    loss, accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    save_class_names(class_names, CLASS_NAMES_PATH)
    plot_history(history, PLOT_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Class names saved to: {CLASS_NAMES_PATH}")
    print(f"Training plot saved to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
