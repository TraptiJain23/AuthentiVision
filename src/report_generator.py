import os
import csv
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_csv_report(campaign_data, output_csv_path):
    """Exports campaign scan results to a structured CSV audit log."""
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Campaign ID", campaign_data.get("campaign_id")])
        writer.writerow(["Campaign Name", campaign_data.get("campaign_name")])
        writer.writerow(["Scanned At", campaign_data.get("scanned_at")])
        writer.writerow(["Total Assets", campaign_data.get("total_assets")])
        writer.writerow(["AI-Generated Count", campaign_data.get("ai_count")])
        writer.writerow(["Authentic Count", campaign_data.get("real_count")])
        writer.writerow(["Synthetic Media %", f"{campaign_data.get('ai_ratio')}%"])
        writer.writerow(["Overall Risk Assessment", campaign_data.get("overall_risk")])
        writer.writerow([])
        writer.writerow(["Filename", "Classification", "Confidence (%)", "Real Prob (%)", "AI Prob (%)", "Risk Level", "Explanation"])
        
        for item in campaign_data.get("results", []):
            writer.writerow([
                item.get("filename"),
                item.get("predicted_class"),
                f"{item.get('confidence'):.2f}",
                f"{item.get('real_percentage'):.2f}",
                f"{item.get('ai_percentage'):.2f}",
                item.get("risk_level"),
                item.get("explanation")
            ])
            
    return output_csv_path

def generate_pdf_report(campaign_data, output_pdf_path):
    """
    Generates a formal, executive PDF Authenticity Audit Report using ReportLab.
    """
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=15
    )

    section_header = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=14,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )

    badge_ai = ParagraphStyle(
        "BadgeAI",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=colors.HexColor("#dc2626")
    )

    badge_real = ParagraphStyle(
        "BadgeReal",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=colors.HexColor("#16a34a")
    )

    story = []

    # Title Banner
    story.append(Paragraph("AuthentiVision™ Content Audit Report", title_style))
    story.append(Paragraph("Deep Learning & Explainable AI Verification for Digital Marketing Assets", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3b82f6"), spaceAfter=15))

    # Campaign Meta & KPI Table
    campaign_name = campaign_data.get("campaign_name", "Marketing Campaign")
    campaign_id = campaign_data.get("campaign_id", "N/A")
    scanned_at = campaign_data.get("scanned_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    total_assets = campaign_data.get("total_assets", 0)
    ai_count = campaign_data.get("ai_count", 0)
    real_count = campaign_data.get("real_count", 0)
    ai_ratio = campaign_data.get("ai_ratio", 0.0)
    overall_risk = campaign_data.get("overall_risk", "Low")
    compliance_status = campaign_data.get("compliance_status", "Compliant")

    kpi_data = [
        [
            Paragraph("<b>Campaign Name:</b>", body_style),
            Paragraph(str(campaign_name), body_style),
            Paragraph("<b>Scanned Date:</b>", body_style),
            Paragraph(str(scanned_at), body_style)
        ],
        [
            Paragraph("<b>Audit ID:</b>", body_style),
            Paragraph(str(campaign_id), body_style),
            Paragraph("<b>Architecture:</b>", body_style),
            Paragraph("MobileNetV2 + Grad-CAM", body_style)
        ],
        [
            Paragraph("<b>Total Assets:</b>", body_style),
            Paragraph(f"<b>{total_assets}</b>", body_style),
            Paragraph("<b>Synthetic Ratio:</b>", body_style),
            Paragraph(f"<b>{ai_ratio}%</b> ({ai_count} AI / {real_count} Authentic)", body_style)
        ],
        [
            Paragraph("<b>Risk Level:</b>", body_style),
            Paragraph(f"<b>{overall_risk}</b>", body_style),
            Paragraph("<b>Status:</b>", body_style),
            Paragraph(str(compliance_status), body_style)
        ]
    ]

    kpi_table = Table(kpi_data, colWidths=[90, 175, 95, 170])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 15))

    # Executive Summary Paragraph
    story.append(Paragraph("Executive Summary", section_header))
    summary_text = (
        f"This automated authenticity audit was performed using the AuthentiVision framework utilizing a fine-tuned "
        f"MobileNetV2 convolutional neural network and Grad-CAM visual interpretability. "
        f"A total of <b>{total_assets}</b> digital marketing assets were scanned. "
        f"<b>{ai_count}</b> asset(s) ({ai_ratio}%) were classified as AI-generated or synthetically manipulated, "
        f"while <b>{real_count}</b> asset(s) were verified as authentic photographic media. "
        f"Overall campaign threat assessment: <b>{overall_risk} Risk</b>."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 15))

    # Asset Audit Breakdown Table
    story.append(Paragraph("Detailed Asset Audit Findings", section_header))

    table_headers = [
        Paragraph("<b>#</b>", body_style),
        Paragraph("<b>Filename</b>", body_style),
        Paragraph("<b>Detection Verdict</b>", body_style),
        Paragraph("<b>Confidence</b>", body_style),
        Paragraph("<b>AI Probability</b>", body_style),
        Paragraph("<b>Risk</b>", body_style)
    ]

    asset_rows = [table_headers]
    for idx, item in enumerate(campaign_data.get("results", []), start=1):
        is_ai = item.get("predicted_class") == "AI-Generated"
        badge = Paragraph("<b>AI-GENERATED</b>", badge_ai) if is_ai else Paragraph("<b>AUTHENTIC</b>", badge_real)
        
        # Truncate filename if long
        fn = item.get("filename", "")
        if len(fn) > 24:
            fn = fn[:21] + "..."

        asset_rows.append([
            Paragraph(str(idx), body_style),
            Paragraph(fn, body_style),
            badge,
            Paragraph(f"{item.get('confidence'):.1f}%", body_style),
            Paragraph(f"{item.get('ai_percentage'):.1f}%", body_style),
            Paragraph(item.get("risk_level", "Low"), body_style)
        ])

    assets_table = Table(asset_rows, colWidths=[25, 160, 110, 75, 80, 80])
    assets_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(assets_table)
    story.append(Spacer(1, 20))

    # Explainability & Methodology Note
    story.append(Paragraph("Explainable AI (Grad-CAM) Methodology", section_header))
    method_note = (
        "Visual explanations are generated using Gradient-weighted Class Activation Mapping (Grad-CAM) "
        "calculated at the final convolutional activation layer (<code>out_relu</code>) of the MobileNetV2 backbone. "
        "Heatmaps isolate spatial artifact clusters such as diffusion boundary anomalies, frequency warping, "
        "and generative smoothing textures. This audit provides evidentiary support for brand safety and legal verification."
    )
    story.append(Paragraph(method_note, body_style))
    story.append(Spacer(1, 25))

    # Footer Disclaimer
    disclaimer = Paragraph(
        "<i>CONFIDENTIAL & PROPRIETARY. Generated by AuthentiVision Deep Learning Content Verification Engine. "
        "Results reflect probabilistic model inference.</i>",
        subtitle_style
    )
    story.append(disclaimer)

    doc.build(story)
    return output_pdf_path
