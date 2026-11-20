
import os
import sys
import uuid
import json
import shutil
import traceback
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ.setdefault("KERAS_HOME", os.path.join(CACHE_DIR, "keras"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(CACHE_DIR, "matplotlib"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, jsonify
)
from werkzeug.utils import secure_filename

from src.inference import analyze_authenticity, get_inference_model
from src.batch_scanner import (
    process_batch_campaign,
    extract_zip_assets,
    SUPPORTED_EXTENSIONS
)
from src.report_generator import generate_pdf_report

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
REPORTS_FOLDER = os.path.join(BASE_DIR, "static", "reports")
SAMPLE_FOLDER = os.path.join(BASE_DIR, "sample_data")

for folder in (UPLOAD_FOLDER, REPORTS_FOLDER, SAMPLE_FOLDER):
    os.makedirs(folder, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "FLASK_SECRET_KEY", "authentivision-local-dev"
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

CAMPAIGN_CACHE = {}


def save_single_report(scan_id, result):
    """Save single-image results for PDF generation."""

    result["scan_id"] = scan_id

    is_ai = result.get("predicted_class") == "AI-Generated"

    report_data = {
        "campaign_id": scan_id,
        "campaign_name": "Single Image Scan",
        "scanned_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "total_assets": 1,
        "real_count": 0 if is_ai else 1,
        "ai_count": 1 if is_ai else 0,
        "ai_ratio": 100.0 if is_ai else 0.0,
        "real_ratio": 0.0 if is_ai else 100.0,
        "average_confidence": result.get("confidence", 0),
        "overall_risk": result.get("risk_level", "Unknown"),
        "compliance_status": (
            "AI-Generated predicted"
            if is_ai else "Authentic predicted"
        ),
        "results": [result]
    }

    report_dir = os.path.join(UPLOAD_FOLDER, scan_id)
    os.makedirs(report_dir, exist_ok=True)

    summary_path = os.path.join(
        report_dir, "campaign_summary.json"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    CAMPAIGN_CACHE[scan_id] = report_data


def get_campaign_data(campaign_id):
    """Load stored scan results."""

    if campaign_id in CAMPAIGN_CACHE:
        return CAMPAIGN_CACHE[campaign_id]

    summary_path = os.path.join(
        UPLOAD_FOLDER,
        campaign_id,
        "campaign_summary.json"
    )

    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        CAMPAIGN_CACHE[campaign_id] = data
        return data

    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scan", methods=["GET", "POST"])
def single_scan():

    if request.method == "POST":

        if "image" not in request.files:
            flash("Please upload an image.", "error")
            return redirect(request.url)

        file = request.files["image"]

        if not file.filename:
            flash("Please select an image.", "error")
            return redirect(request.url)

        ext = os.path.splitext(file.filename)[1].lower()

        if ext not in SUPPORTED_EXTENSIONS:
            flash("Unsupported image format.", "error")
            return redirect(request.url)

        scan_id = (
            f"single_{int(datetime.now().timestamp())}_"
            f"{uuid.uuid4().hex[:6]}"
        )

        safe_name = (
            f"{scan_id}_{secure_filename(file.filename)}"
        )

        saved_path = os.path.join(
            UPLOAD_FOLDER, safe_name
        )

        file.save(saved_path)

        try:
            result = analyze_authenticity(
                image_path=saved_path,
                output_dir=UPLOAD_FOLDER,
                file_prefix=scan_id
            )

            result["original_web_path"] = safe_name

            save_single_report(scan_id, result)

            return render_template(
                "single_scan.html",
                result=result
            )

        except Exception as e:
            traceback.print_exc()
            flash(
                f"Inference error during scan: {str(e)}",
                "error"
            )
            return redirect(request.url)

    return render_template(
        "single_scan.html",
        result=None
    )


@app.route("/scan-sample", methods=["POST"])
def scan_sample():

    sample_name = secure_filename(
        request.form.get("sample_name", "")
    )

    sample_path = os.path.join(
        SAMPLE_FOLDER, sample_name
    )

    if not sample_name or not os.path.isfile(sample_path):
        flash("Sample image not found.", "error")
        return redirect(url_for("index"))

    scan_id = (
        f"sample_{int(datetime.now().timestamp())}_"
        f"{uuid.uuid4().hex[:6]}"
    )

    safe_name = f"{scan_id}_{sample_name}"

    dest_path = os.path.join(
        UPLOAD_FOLDER, safe_name
    )

    shutil.copy2(sample_path, dest_path)

    try:
        result = analyze_authenticity(
            image_path=dest_path,
            output_dir=UPLOAD_FOLDER,
            file_prefix=scan_id
        )

        result["original_web_path"] = safe_name

        save_single_report(scan_id, result)

        return render_template(
            "single_scan.html",
            result=result
        )

    except Exception as e:
        traceback.print_exc()
        flash(
            f"Error scanning sample: {str(e)}",
            "error"
        )
        return redirect(url_for("index"))


@app.route("/batch", methods=["GET", "POST"])
def batch_scan():

    if request.method == "POST":

        campaign_name = request.form.get(
            "campaign_name", "Image Scan Report"
        ).strip()

        files = request.files.getlist("assets")

        if not files or not any(f.filename for f in files):
            flash(
                "Please select images or a ZIP file.",
                "error"
            )
            return redirect(request.url)

        temp_batch_id = (
            f"raw_{int(datetime.now().timestamp())}_"
            f"{uuid.uuid4().hex[:6]}"
        )

        temp_dir = os.path.join(
            UPLOAD_FOLDER, temp_batch_id
        )

        os.makedirs(temp_dir, exist_ok=True)

        input_paths = []

        try:
            for file in files:

                if not file.filename:
                    continue

                ext = os.path.splitext(
                    file.filename
                )[1].lower()

                safe_name = secure_filename(
                    file.filename
                )

                saved_path = os.path.join(
                    temp_dir, safe_name
                )

                file.save(saved_path)

                if ext == ".zip":
                    extracted = extract_zip_assets(
                        saved_path,
                        os.path.join(
                            temp_dir, "unzipped"
                        )
                    )
                    input_paths.extend(extracted)

                elif ext in SUPPORTED_EXTENSIONS:
                    input_paths.append(saved_path)

            if not input_paths:
                flash(
                    "No valid images found.",
                    "error"
                )
                return redirect(request.url)

            campaign_data = process_batch_campaign(
                file_paths=input_paths,
                output_dir=UPLOAD_FOLDER,
                campaign_name=campaign_name
            )

            campaign_id = campaign_data["campaign_id"]

            CAMPAIGN_CACHE[campaign_id] = campaign_data

            return render_template(
                "batch_scan.html",
                campaign=campaign_data
            )

        except Exception as e:
            traceback.print_exc()
            flash(
                f"Batch scan failed: {str(e)}",
                "error"
            )
            return redirect(request.url)

    return render_template(
        "batch_scan.html",
        campaign=None
    )


@app.route("/report/<campaign_id>")
def campaign_report(campaign_id):

    campaign = get_campaign_data(campaign_id)

    if not campaign:
        flash("Report not found.", "error")
        return redirect(url_for("index"))

    return render_template(
        "report.html",
        campaign=campaign
    )


@app.route("/download/pdf/<campaign_id>")
def download_pdf(campaign_id):

    campaign = get_campaign_data(campaign_id)

    if not campaign:
        flash("Report not found.", "error")
        return redirect(url_for("index"))

    pdf_filename = (
        f"AuthentiVision_Report_{campaign_id}.pdf"
    )

    pdf_path = os.path.join(
        REPORTS_FOLDER,
        pdf_filename
    )

    try:
        generate_pdf_report(
            campaign,
            pdf_path
        )

        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=pdf_filename,
            mimetype="application/pdf"
        )

    except Exception as e:
        traceback.print_exc()
        flash(
            f"PDF generation failed: {str(e)}",
            "error"
        )
        return redirect(url_for("index"))


@app.route("/model-info")
def model_info():
    return render_template("model_info.html")


@app.route("/api/scan", methods=["POST"])
def api_scan():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({
            "error": "No image selected"
        }), 400

    ext = os.path.splitext(
        file.filename
    )[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({
            "error": "Unsupported image format"
        }), 400

    scan_id = (
        f"api_{int(datetime.now().timestamp())}_"
        f"{uuid.uuid4().hex[:6]}"
    )

    safe_name = (
        f"{scan_id}_{secure_filename(file.filename)}"
    )

    saved_path = os.path.join(
        UPLOAD_FOLDER, safe_name
    )

    file.save(saved_path)

    try:
        result = analyze_authenticity(
            image_path=saved_path,
            output_dir=UPLOAD_FOLDER,
            file_prefix=scan_id
        )

        return jsonify({
            "status": "success",
            "prediction": result["predicted_class"],
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "real_probability": result["real_percentage"],
            "ai_probability": result["ai_percentage"],
            "risk_level": result["risk_level"],
            "explanation": result["explanation"],
            "overlay_url": (
                "/static/uploads/" +
                result["overlay_file"]
            ),
            "heatmap_url": (
                "/static/uploads/" +
                result["heatmap_file"]
            )
        })

    except Exception as e:
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":

    print(
        "[AuthentiVision] Loading MobileNetV2 model..."
    )

    get_inference_model()

    print(
        "[AuthentiVision] Running at "
        "http://127.0.0.1:5000"
    )

    app.run(
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 7860)),
    debug=False
)
