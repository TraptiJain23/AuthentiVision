#!/usr/bin/env bash
# AuthentiVision - AI Content Detection Framework Launcher
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "====================================================================="
echo " AuthentiVision: Deep Learning AI Content Detection in Digital Marketing"
echo " Powered by TensorFlow, MobileNetV2 (.h5), and Grad-CAM Explainable AI"
echo "====================================================================="
echo ""

if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "Starting Flask web dashboard on http://127.0.0.1:5000 ..."
python app.py
