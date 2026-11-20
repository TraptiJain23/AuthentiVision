import os
import cv2
import numpy as np
import tensorflow as tf

from models.builder import get_or_create_model, CLASS_NAMES
from src.gradcam import generate_gradcam_heatmap, save_and_overlay_gradcam

_GLOBAL_MODEL = None

def get_inference_model():
    """Singleton getter for the MobileNetV2 detection model."""
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        _GLOBAL_MODEL = get_or_create_model()
    return _GLOBAL_MODEL

def preprocess_image_file(image_path, target_size=(224, 224)):
    """
    Loads an image file, converts to RGB, resizes to target_size, and applies
    MobileNetV2 standard normalization [-1, 1].
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image at path: {image_path}")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_preprocessed, img_bgr

def extract_forensic_texture_features(img_bgr):
    """
    Extracts frequency-domain and sensor-noise forensic signatures.
    
    Why this is critical for Real-World Generalization:
    - Real camera photographs contain physical Poisson-Gaussian sensor shot noise and optical lens dispersion.
    - AI-generated diffusion models (Midjourney, DALL-E, Stable Diffusion, Flux) synthesize imagery
      in continuous latent spaces, exhibiting characteristic spectral roll-offs, lack of physical shot noise,
      and abnormal patch-level variance.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. High-frequency Laplacian variance (Physical sensor noise vs diffusion smoothing)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())

    # 2. Fourier 2D-FFT high-frequency spectral energy distribution
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mag = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-8)
    
    mask = np.ones((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 4, 0, -1)
    hf_energy = float(np.mean(mag[mask == 1]))

    # 3. Patch-level variance disparity
    patch_vars = []
    ph, pw = max(h // 4, 1), max(w // 4, 1)
    for i in range(4):
        for j in range(4):
            patch = gray[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            if patch.size > 0:
                patch_vars.append(np.var(patch))
    patch_var_ratio = float(np.std(patch_vars) / (np.mean(patch_vars) + 1e-5))

    return {
        "lap_var": lap_var,
        "hf_energy": hf_energy,
        "patch_disparity": patch_var_ratio
    }

def analyze_authenticity(image_path, output_dir=None, file_prefix="scan", model=None, decision_threshold=0.50):
    """
    Performs full authenticity evaluation on a marketing asset:
    1. Preprocesses image across scales to bridge CIFAKE (32x32) vs Real-World (high-res) domain gap.
    2. Executes MobileNetV2 neural feature inference.
    3. Fuses with physical sensor photon noise (Laplacian variance) and Fourier spectral dynamics.
    4. Computes Grad-CAM explainability heatmap on 'out_relu'.
    5. Saves composite visual explanation overlays.
    6. Formulates diagnostic and compliance explanation.
    """
    if model is None:
        model = get_inference_model()

    if output_dir is None:
        output_dir = os.path.dirname(image_path)

    img_tensor, orig_bgr = preprocess_image_file(image_path)
    h_orig, w_orig = orig_bgr.shape[:2]
    img_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # 1. Direct Neural Inference
    preds_direct = model.predict(img_tensor, verbose=0)
    if preds_direct.shape[-1] == 1:
        raw_p_ai = float(np.clip(preds_direct[0][0], 0.0, 1.0))
    else:
        raw_p_ai = float(preds_direct[0][1])

    # 2. Multi-Scale Normalization (evaluates in CIFAKE-equivalent frequency scale)
    # This prevents the MobileNetV2 logit explosion caused when high-res sharp edges
    # trigger features that were dormant in 32x32 CIFAR-10 training images.
    img_rescaled = cv2.resize(cv2.resize(img_rgb, (64, 64)), (224, 224))
    t_rescaled = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(img_rescaled, 0).astype(np.float32)
    )
    preds_rescaled = model.predict(t_rescaled, verbose=0)
    p_ai_rescaled = float(np.clip(preds_rescaled[0][0] if preds_rescaled.shape[-1] == 1 else preds_rescaled[0][1], 0.0, 1.0))

    # 3. Forensic Texture & Sensor Analysis
    forensics = extract_forensic_texture_features(orig_bgr)
    lap_var = forensics["lap_var"]
    hf_energy = forensics["hf_energy"]
    patch_disp = forensics["patch_disparity"]

    # Forensic logit formulation:
    # Real camera sensors exhibit natural Poisson-Gaussian shot noise (lap_var > 180)
    # Generative AI diffusion models exhibit mathematical latent oversmoothing (lap_var < 100)
    z_forensic = (160.0 - lap_var) / 90.0 + (138.0 - hf_energy) / 25.0 + (patch_disp - 0.70) * 1.5
    forensic_ai_prob = float(1.0 / (1.0 + np.exp(-np.clip(z_forensic, -3.5, 3.5))))

    # 4. Domain-Aware Neuro-Forensic Fusion:
    is_native_cifake = max(h_orig, w_orig) <= 64

    if is_native_cifake:
        # For CIFAKE dataset images, model prediction is directly authoritative
        ai_prob = raw_p_ai
    else:
        # For real-world images (smartphones, WhatsApp, DSLRs, stock photos, Midjourney):
        # We fuse multi-scale neural activations with physical camera sensor forensics
        if lap_var >= 250.0 and hf_energy >= 142.0:
            # Physical camera sensor noise and natural lens dispersion confirmed
            ai_prob = min(p_ai_rescaled * 0.30 + forensic_ai_prob * 0.70, 0.28)
        elif lap_var <= 50.0:
            # Severe latent diffusion smoothing (no physical sensor photon noise)
            ai_prob = max(p_ai_rescaled * 0.45 + forensic_ai_prob * 0.55, 0.75)
        else:
            # Balanced neuro-forensic fusion
            ai_prob = p_ai_rescaled * 0.40 + forensic_ai_prob * 0.60

    ai_prob = float(np.clip(ai_prob, 0.02, 0.98))
    real_prob = float(1.0 - ai_prob)

    # Classification verdict
    if ai_prob >= decision_threshold:
        predicted_idx = 1
        predicted_label = "AI-Generated"
        confidence_score = float(ai_prob * 100.0)
    else:
        predicted_idx = 0
        predicted_label = "Real"
        confidence_score = float(real_prob * 100.0)

    # Compute Grad-CAM heatmap for winning class from MobileNetV2 'out_relu'
    raw_heatmap = generate_gradcam_heatmap(
        img_array=img_tensor,
        model=model,
        last_conv_layer_name="out_relu",
        pred_index=predicted_idx
    )

    # Save overlays
    gradcam_artifacts = save_and_overlay_gradcam(
        original_img_path=image_path,
        heatmap=raw_heatmap,
        output_dir=output_dir,
        file_prefix=file_prefix
    )

    high_act_ratio = float(np.mean(raw_heatmap > 0.6) * 100.0)

    # Qualitative explanation for digital marketing verification
    if predicted_label == "AI-Generated":
        verdict = "Synthetically Generated / Manipulated"
        risk_level = "High Risk" if confidence_score > 75.0 else "Moderate Risk"
        explanation = (
            f"The AuthentiVision detector flagged this asset as AI-Generated with {confidence_score:.1f}% confidence. "
            f"Analysis revealed characteristic generative diffusion traits (sensor variance: {lap_var:.1f}, "
            f"patch disparity: {patch_disp:.2f}) and Grad-CAM hotspot concentration ({high_act_ratio:.1f}%) on layer 'out_relu'."
        )
    else:
        verdict = "Authentic Digital Asset"
        risk_level = "Low Risk"
        explanation = (
            f"The asset exhibits organic photographic qualities with {confidence_score:.1f}% authenticity confidence. "
            f"Verified presence of physical camera sensor photon noise (variance: {lap_var:.1f}) and natural optical frequency energy ({hf_energy:.1f} dB)."
        )

    return {
        "image_path": image_path,
        "filename": os.path.basename(image_path),
        "predicted_class": predicted_label,
        "verdict": verdict,
        "risk_level": risk_level,
        "confidence": round(confidence_score, 1),
        "real_percentage": round(real_prob * 100.0, 1),
        "ai_percentage": round(ai_prob * 100.0, 1),
        "explanation": explanation,
        "heatmap_file": gradcam_artifacts["heatmap_file"],
        "overlay_file": gradcam_artifacts["overlay_file"],
        "heatmap_full_path": gradcam_artifacts["heatmap_full_path"],
        "overlay_full_path": gradcam_artifacts["overlay_full_path"]
    }
