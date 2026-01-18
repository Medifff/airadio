#!/bin/bash

# Останавливаем скрипт при любой ошибке
set -e

echo "⚙️  [1/5] Configuring Environment..."
export HF_HOME="/workspace/hf_cache"
mkdir -p $HF_HOME

echo "📦 [2/5] Installing System Dependencies..."
apt-get update -qq && apt-get install -y ffmpeg -qq

echo "⚡ [3/5] Installing UV (Fast Pip)..."
pip install uv

echo "🔥 [4/5] Installing Python Libraries..."
# Сначала torch
uv pip install torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --system
# Потом остальное
uv pip install -r requirements.txt --system

echo "🚀 [5/5] Launching AI Radio (radio.py)..."

# ПРЯМОЙ ЗАПУСК ФАЙЛА (Без проверок "если/или")
python radio.py
