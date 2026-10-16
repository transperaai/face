#!/usr/bin/env bash
set -e

apt-get update && apt-get install -y ffmpeg libgl1-mesa-glx libglib2.0-0 python3-tk

pip install -r requirements.txt

echo "Setup complete. Run with: python app.py"
