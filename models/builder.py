import os
import sys
import tensorflow as tf

# Resolve base directories dynamically without hardcoded machine paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_H5_PATH = os.path.join(MODELS_DIR, "authentivision_mobilenetv2.h5")
MODEL_KERAS_PATH = os.path.join(MODELS_DIR, "authentivision_mobilenetv2.keras")

# Default model path pointing to the required .h5 model
MODEL_PATH = MODEL_H5_PATH
CLASS_NAMES = ["Real", "AI-Generated"]

def build_mobilenetv2_model(input_shape=(224, 224, 3), weights="imagenet", learning_rate=0.001):
    """
    Constructs MobileNetV2 transfer learning architecture as specified:
    - Base: MobileNetV2 pretrained on ImageNet (backbone frozen)
    - Input size: 224 x 224 x 3
    - GlobalAveragePooling2D
    - Dropout: 0.3
    - Output: Dense(1, activation='sigmoid')
    - Loss: BinaryCrossentropy
    - Optimizer: Adam (initial lr=0.001)
    
    Target layer for Grad-CAM is 'out_relu' (the final 1280-channel conv activation).
    """
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights
        )
    except Exception as e:
        print(f"[AuthentiVision] Note: Online ImageNet download failed ({e}). Loading without pretrained weights.")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=None
        )

    # Freeze pretrained MobileNetV2 backbone
    base_model.trainable = False

    # Classification Head
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(0.3, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs, name="AuthentiVision_MobileNetV2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def get_or_create_model(model_path=MODEL_PATH):
    """
    Loads saved model if available; otherwise builds, initializes, and saves a baseline model.
    Prioritizes .h5 format per project requirements.
    """
    # 1. Try loading specified model path (.h5 preferred)
    if os.path.exists(model_path):
        try:
            print(f"[AuthentiVision] Loading model from {model_path}...")
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )
            return model
        except Exception as e:
            print(f"[AuthentiVision] Failed to load model at {model_path}: {e}")

    # 2. Try loading .keras backup if .h5 is not yet present
    if model_path != MODEL_KERAS_PATH and os.path.exists(MODEL_KERAS_PATH):
        try:
            print(f"[AuthentiVision] Loading fallback model from {MODEL_KERAS_PATH}...")
            model = tf.keras.models.load_model(MODEL_KERAS_PATH, compile=False)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )
            return model
        except Exception as e:
            print(f"[AuthentiVision] Fallback .keras loading failed: {e}")

    # 3. Build and initialize baseline model
        raise RuntimeError(
        "Trained AuthentiVision model could not be loaded. "
        "Check models/authentivision_mobilenetv2.h5 "
        "and TensorFlow compatibility."
    )

if __name__ == "__main__":
    m = get_or_create_model()
    m.summary()
