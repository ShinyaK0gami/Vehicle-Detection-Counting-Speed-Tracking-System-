#!/bin/bash
echo "[+] Initializing Jetson Nano Performance Profiles..."
sudo nvpmodel -m 0 2>/dev/null || true
sudo jetson_clocks 2>/dev/null || true

echo "[+] Launching Docker Inference Container..."
sudo docker run --rm --runtime nvidia --network host \
    -v ~/Desktop/Traffic_Analysis_System:/workspace \
    -w /workspace \
    nvcr.io/nvidia/l4t-ml:r32.7.1-py3 \
    python3 run_web_stream.py
