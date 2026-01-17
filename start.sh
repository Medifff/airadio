#!/bin/bash

# 1. Говорим системе, где хранить нейросети (на большом диске)
export HF_HOME="/workspace/hf_cache"
mkdir -p $HF_HOME

# 2. Обновляем системные пакеты и ставим FFmpeg (обязательно!)
echo "📦 Installing System Deps..."
apt-get update && apt-get install -y ffmpeg

# 3. Ставим Python библиотеки из твоего файла
echo "🐍 Installing Python Deps..."
pip install -r requirements.txt

# 4. Запускаем радио (замени radio.py на имя твоего файла)
echo "📻 Starting AI Radio..."
python radio.py
