import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf

# Resolve base directories dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

CACHE_DIR = os.path.join(BASE_DIR, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("KERAS_HOME", os.path.join(CACHE_DIR, "keras"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(CACHE_DIR, "matplotlib"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from models.builder import build_mobilenetv2_model, MODEL_H5_PATH, MODEL_KERAS_PATH

def create_dataset_splits(data_dir, img_size=(224, 224), batch_size=64, val_split=0.2, seed=42):
    """
    Creates balanced tf.data.Dataset loaders from CIFAKE_Train50K_Test20K:
    - Target class mapping:
        Index 0: REAL
        Index 1: FAKE (AI-Generated)
    - Train split (80% of 50K): 40,000 images (20,000 REAL + 20,000 FAKE)
    - Validation split (20% of 50K): 10,000 images (5,000 REAL + 5,000 FAKE)
    - Test split: 20,000 images (10,000 REAL + 10,000 FAKE) - completely separate
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at: {test_dir}")

    # Enforce exact class ordering: 0 -> REAL, 1 -> FAKE
    class_names = ["REAL", "FAKE"]

    print("=" * 70)
    print("AuthentiVision CIFAKE Dataset Setup")
    print(f"Data Root: {data_dir}")
    print(f"Class Mapping: 0 -> '{class_names[0]}' (Authentic), 1 -> '{class_names[1]}' (AI-Generated)")
    print(f"Image Dimensions: {img_size} | Batch Size: {batch_size}")
    print("=" * 70)

    # 1. Training Subset (40,000 images)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        class_names=class_names,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True,
        validation_split=val_split,
        subset="training",
        seed=seed
    )

    # 2. Validation Subset (10,000 images)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        class_names=class_names,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True,
        validation_split=val_split,
        subset="validation",
        seed=seed
    )

    # 3. Test Set (20,000 untouched images)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        class_names=class_names,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False
    )

    # MobileNetV2 standard normalization to [-1, 1]
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=autotune)
    val_ds = val_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=autotune)
    test_ds = test_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=autotune)

    # Prefetch optimization for maximum pipeline throughput
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds

def train_mobilenetv2(
    data_dir,
    output_h5_path=MODEL_H5_PATH,
    output_keras_path=MODEL_KERAS_PATH,
    epochs=5,
    batch_size=64,
    learning_rate=0.001
):
    """
    Executes MobileNetV2 transfer learning with frozen backbone:
    - Base: MobileNetV2 (ImageNet weights, frozen)
    - Head: GlobalAveragePooling2D -> Dropout(0.3) -> Dense(1, sigmoid)
    - Optimizer: Adam(0.001)
    - Loss: BinaryCrossentropy
    - Callbacks: EarlyStopping(patience=2), ModelCheckpoint, ReduceLROnPlateau
    - Saves final complete .h5 model
    """
    train_ds, val_ds, test_ds = create_dataset_splits(data_dir, batch_size=batch_size)

    print("\n[Architecture] Initializing MobileNetV2 with frozen ImageNet backbone...")
    model = build_mobilenetv2_model(
        input_shape=(224, 224, 3),
        weights="imagenet",
        learning_rate=learning_rate
    )
    model.summary()

    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)

    # Callbacks configuration per Section 3
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_h5_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        verbose=1
    )

    callbacks = [checkpoint_cb, early_stopping_cb, reduce_lr_cb]

    print("\n" + "=" * 70)
    print(f"Starting Training: {epochs} Epochs Max | Batch Size: {batch_size} | Adam lr: {learning_rate}")
    print("=" * 70 + "\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Ensure best model is saved to required .h5 location
    model.save(output_h5_path)
    print(f"\n[✓] Saved final complete HDF5 model to: {output_h5_path}")

    # Also save .keras backup
    if output_keras_path:
        try:
            model.save(output_keras_path)
            print(f"[✓] Saved .keras backup model to: {output_keras_path}")
        except Exception as e:
            print(f"[!] Note: Could not save .keras backup: {e}")

    # Verify reloading
    print("\n[Verification] Reloading saved model via tf.keras.models.load_model(..., compile=False)...")
    reloaded_model = tf.keras.models.load_model(output_h5_path, compile=False)
    dummy_input = tf.zeros((1, 224, 224, 3))
    test_pred = reloaded_model.predict(dummy_input, verbose=0)
    print(f"[✓] Reload verification successful! Output shape: {test_pred.shape}, sample pred: {float(test_pred[0][0]):.4f}")

    # Evaluate on untouched 20,000 test set
    print("\n" + "=" * 70)
    print("Evaluating Model on Untouched 20,000-Image CIFAKE Test Dataset")
    print("=" * 70 + "\n")
    evaluation_metrics = evaluate_model_on_test_set(reloaded_model, test_ds)

    return reloaded_model, evaluation_metrics

def evaluate_model_on_test_set(model, test_ds, threshold=0.50):
    """
    Evaluates model performance across all 20,000 test images:
    - Overall Accuracy
    - REAL Class Accuracy (Specificity)
    - FAKE Class Accuracy (Recall of AI-Generated)
    - Precision, Recall, F1-Score
    - Confusion Matrix
    """
    all_y_true = []
    all_y_pred_probs = []

    print("[Evaluation] Computing predictions over 20,000 test images...")
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        all_y_true.extend(y_batch.numpy().flatten())
        all_y_pred_probs.extend(preds.flatten())

    y_true = np.array(all_y_true, dtype=int)
    y_probs = np.array(all_y_pred_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)

    # Metrics calculation
    # Class 0: REAL, Class 1: FAKE
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))  # Correctly identified FAKE
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))  # Correctly identified REAL
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))  # REAL falsely flagged as FAKE
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))  # FAKE missed as REAL

    total = len(y_true)
    total_real = int(np.sum(y_true == 0))
    total_fake = int(np.sum(y_true == 1))

    overall_accuracy = (tp + tn) / total if total > 0 else 0.0
    real_accuracy = tn / total_real if total_real > 0 else 0.0
    fake_accuracy = tp / total_fake if total_fake > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("=" * 70)
    print("CIFAKE 20,000 TEST DATASET EVALUATION REPORT")
    print("=" * 70)
    print(f"Total Test Images:       {total:,} (REAL: {total_real:,}, FAKE: {total_fake:,})")
    print(f"Decision Threshold:      {threshold:.2f}")
    print("-" * 70)
    print(f"Overall Accuracy:        {overall_accuracy * 100:.2f}%")
    print(f"REAL Class Accuracy:     {real_accuracy * 100:.2f}% ({tn:,}/{total_real:,})")
    print(f"FAKE Class Accuracy:     {fake_accuracy * 100:.2f}% ({tp:,}/{total_fake:,})")
    print("-" * 70)
    print(f"Precision (AI-Generated): {precision * 100:.2f}%")
    print(f"Recall (AI-Generated):    {recall * 100:.2f}%")
    print(f"F1-Score (AI-Generated):  {f1_score * 100:.2f}%")
    print("-" * 70)
    print("Confusion Matrix:")
    print(f"  True REAL (TN):  {tn:,} | False FAKE (FP): {fp:,}")
    print(f"  False REAL (FN): {fn:,} | True FAKE (TP):  {tp:,}")
    print("=" * 70 + "\n")

    report = {
        "total_test_images": total,
        "total_real": total_real,
        "total_fake": total_fake,
        "decision_threshold": threshold,
        "overall_accuracy": round(overall_accuracy * 100, 2),
        "real_accuracy": round(real_accuracy * 100, 2),
        "fake_accuracy": round(fake_accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1_score * 100, 2),
        "confusion_matrix": {
            "true_negatives_real": tn,
            "false_positives_fake": fp,
            "false_negatives_missed_fake": fn,
            "true_positives_fake": tp
        }
    }

    report_path = os.path.join(BASE_DIR, "models", "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[✓] Evaluation report saved to: {report_path}")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on CIFAKE dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(BASE_DIR, "dataset"),
        help="Path to unzipped CIFAKE dataset root containing train/ and test/."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Maximum training epochs (default: 5)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate for Adam (default: 0.001)."
    )
    parser.add_argument(
        "--output_h5",
        type=str,
        default=MODEL_H5_PATH,
        help=f"Target .h5 file path (default: {MODEL_H5_PATH})."
    )

    args = parser.parse_args()

    train_mobilenetv2(
        data_dir=args.data_dir,
        output_h5_path=args.output_h5,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
