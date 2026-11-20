import os
import cv2
import numpy as np
import tensorflow as tf

def find_target_conv_layer(model, preferred_layer="out_relu"):
    """
    Finds the target convolutional/activation layer for Grad-CAM.
    Defaults to 'out_relu' (MobileNetV2 final 1280-channel conv activation),
    or falls back to 'Conv_1' or the last 4D feature map layer.
    """
    try:
        layer = model.get_layer(preferred_layer)
        return layer.name
    except (ValueError, AttributeError):
        pass

    try:
        layer = model.get_layer("Conv_1")
        return layer.name
    except (ValueError, AttributeError):
        pass

    # Search backwards for the last 4-dimensional layer (batch, h, w, channels)
    for layer in reversed(model.layers):
        try:
            if len(layer.output_shape) == 4:
                return layer.name
        except Exception:
            continue
            
    raise ValueError("Could not find any suitable 4D convolutional layer for Grad-CAM.")

def generate_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu", pred_index=None):
    """
    Computes Grad-CAM heatmap for a given input array and model.
    Supports both single sigmoid output Dense(1) and multi-class softmax Dense(num_classes).
    
    Args:
        img_array: Preprocessed image tensor of shape (1, 224, 224, 3)
        model: tf.keras.Model
        last_conv_layer_name: Target layer name ('out_relu' in MobileNetV2)
        pred_index: Target class index to explain (0 for Real, 1 for AI-Generated)
        
    Returns:
        numpy 2D array of normalized heatmap values in range [0, 1]
    """
    target_layer_name = find_target_conv_layer(model, last_conv_layer_name)
    target_layer = model.get_layer(target_layer_name)

    # Multi-output model mapping input to [last_conv_output, predictions]
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        predictions = tf.convert_to_tensor(predictions)
        
        # Check if output is single sigmoid unit or multi-class softmax
        if predictions.shape[-1] == 1:
            # Binary sigmoid output: predictions[0][0] is P(AI-Generated)
            # If pred_index is 0 (Real), target is (1.0 - predictions[:, 0])
            # If pred_index is 1 (AI-Generated) or None, target is predictions[:, 0]
            if pred_index == 0:
                class_channel = 1.0 - predictions[:, 0]
            else:
                class_channel = predictions[:, 0]
        else:
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

    # Gradient of target concept with respect to the feature map activations
    grads = tape.gradient(class_channel, conv_outputs)

    # Spatial average of gradients across feature channels
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each feature map channel by its importance
    conv_outputs_val = conv_outputs[0]
    heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to keep only features that positively contributed to the target prediction
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()

def save_and_overlay_gradcam(original_img_path, heatmap, output_dir, file_prefix="scan", alpha=0.45):
    """
    Overlays Grad-CAM heatmap onto the original image and saves both
    the pure heatmap and the composite visualization.
    
    Returns:
        dict: Relative file paths for the composite and pure heatmap images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original image
    orig_img = cv2.imread(original_img_path)
    if orig_img is None:
        raise ValueError(f"Unable to read image at {original_img_path}")
        
    orig_h, orig_w = orig_img.shape[:2]

    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))

    # Convert to 0-255 uint8 and apply JET colormap (Red = high influence, Blue = low influence)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Superimpose heatmap onto original image
    superimposed = cv2.addWeighted(orig_img, 1.0 - alpha, heatmap_colored, alpha, 0)

    # File names
    heatmap_filename = f"{file_prefix}_heatmap.jpg"
    overlay_filename = f"{file_prefix}_gradcam_overlay.jpg"

    heatmap_path = os.path.join(output_dir, heatmap_filename)
    overlay_path = os.path.join(output_dir, overlay_filename)

    cv2.imwrite(heatmap_path, heatmap_colored)
    cv2.imwrite(overlay_path, superimposed)

    return {
        "heatmap_file": heatmap_filename,
        "overlay_file": overlay_filename,
        "heatmap_full_path": heatmap_path,
        "overlay_full_path": overlay_path
    }
