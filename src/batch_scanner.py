import os
import zipfile
import uuid
import json
import time
from datetime import datetime
from src.inference import analyze_authenticity, get_inference_model

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def process_batch_campaign(file_paths, output_dir, campaign_name="Marketing Campaign"):
    """
    Processes multiple image files, runs detection + Grad-CAM,
    and calculates campaign-wide authenticity metrics.
    """
    model = get_inference_model()
    campaign_id = f"cmp_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    campaign_dir = os.path.join(output_dir, campaign_id)
    os.makedirs(campaign_dir, exist_ok=True)

    results = []
    real_count = 0
    ai_count = 0
    total_confidence = 0.0

    for idx, path in enumerate(file_paths):
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        prefix = f"{campaign_id}_{idx}"
        try:
            analysis = analyze_authenticity(
                image_path=path,
                output_dir=campaign_dir,
                file_prefix=prefix,
                model=model
            )
            # Add relative web path for templates
            analysis["web_overlay"] = f"uploads/{campaign_id}/{analysis['overlay_file']}"
            analysis["web_heatmap"] = f"uploads/{campaign_id}/{analysis['heatmap_file']}"
            analysis["web_original"] = f"uploads/{campaign_id}/{os.path.basename(path)}"
            
            # Copy original to campaign dir if not already there
            dest_orig = os.path.join(campaign_dir, os.path.basename(path))
            if os.path.abspath(path) != os.path.abspath(dest_orig):
                import shutil
                shutil.copy2(path, dest_orig)

            results.append(analysis)

            if analysis["predicted_class"] == "Real":
                real_count += 1
            else:
                ai_count += 1
            total_confidence += analysis["confidence"]

        except Exception as e:
            print(f"[AuthentiVision] Error processing asset {path}: {e}")

    total_assets = len(results)
    if total_assets > 0:
        ai_ratio = round((ai_count / total_assets) * 100.0, 1)
        real_ratio = round((real_count / total_assets) * 100.0, 1)
        avg_conf = round(total_confidence / total_assets, 1)
    else:
        ai_ratio = 0.0
        real_ratio = 0.0
        avg_conf = 0.0

    if ai_ratio >= 40.0:
        overall_risk = "High"
        compliance_status = "Action Required: High concentration of synthetic media detected."
    elif ai_ratio >= 15.0:
        overall_risk = "Moderate"
        compliance_status = "Review Advised: Noticeable presence of AI-generated content."
    else:
        overall_risk = "Low"
        compliance_status = "Verified Clean: Campaign predominantly authentic assets."

    campaign_data = {
        "campaign_id": campaign_id,
        "campaign_name": campaign_name,
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_assets": total_assets,
        "real_count": real_count,
        "ai_count": ai_count,
        "ai_ratio": ai_ratio,
        "real_ratio": real_ratio,
        "average_confidence": avg_conf,
        "overall_risk": overall_risk,
        "compliance_status": compliance_status,
        "results": results
    }

    # Save campaign state for reporting
    json_path = os.path.join(campaign_dir, "campaign_summary.json")
    with open(json_path, "w") as f:
        json.dump(campaign_data, f, indent=2)

    return campaign_data

def extract_zip_assets(zip_path, extract_dir):
    """Safely extracts marketing assets from an uploaded zip archive."""
    extracted_paths = []
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            # Prevent zip-slip vulnerability
            filename = os.path.basename(member.filename)
            if not filename or filename.startswith("."):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                target_path = os.path.join(extract_dir, filename)
                with z.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())
                extracted_paths.append(target_path)

    return extracted_paths
