#!/bin/bash

# Останавливаем скрипт, если любая команда выдаст ошибку
set -e

echo "⚙️  [1/5] Configuring Environment..."
# Перенос кэша на большой диск (чтобы не забить систему)
export HF_HOME="/workspace/hf_cache"
mkdir -p $HF_HOME

echo "📦 [2/5] Installing System Dependencies..."
# Ставим FFmpeg (тихий режим -qq, чтобы не спамил логами)
apt-get update -qq && apt-get install -y ffmpeg -qq

echo "⚡ [3/5] Installing UV (Fast Pip)..."
# Ставим uv - спаситель от зависаний pip
pip install uv

echo "🔥 [4/5] Installing Python Libraries..."
# 1. Сначала принудительно ставим PyTorch 2.6+ (критично для безопасности HuggingFace)
# Флаг --system нужен, так как в RunPod мы работаем от root без venv
uv pip install torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --system

# 2. Теперь ставим всё остальное (uv решит конфликт crewai/litellm за секунду)
uv pip install -r requirements.txt --system

echo "🚀 [5/5] Launching AI Radio..."

# Автоматически определяем имя файла (main.py или radio.py)
if [ -f "main.py" ]; then
    python main.py
elif [ -f "radio.py" ]; then
    python radio.py
else
    echo "❌ Error: Could not find main.py or radio.py!"
    exit 1
fi
