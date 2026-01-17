import os
import time
import random
import subprocess
import gc
import torch
import soundfile as sf
import numpy as np
import threading
import queue
import asyncio
from transformers import MusicgenForConditionalGeneration, MusicgenProcessor
from diffusers import StableDiffusionPipeline
import edge_tts

# =========================
# CONFIG
# =========================
# Убедись, что переменная окружения задана в RunPod, или вставь ключ сюда
STREAM_KEY = os.environ.get("TWITCH_STREAM_KEY") 
if not STREAM_KEY:
    print("⚠️ WARNING: TWITCH_STREAM_KEY not found. Stream will fail.")
    # STREAM_KEY = "live_xxxx_....." # Раскомментируй и вставь, если лень через env

RTMP_URL = f"rtmp://live.twitch.tv/app/{STREAM_KEY}"
WORKDIR = "/workspace/airadio/data"
os.makedirs(WORKDIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Device: {DEVICE} (RTX A4000 Power!)")

# Очередь для готовых видео-сегментов
video_queue = queue.Queue(maxsize=5)

# =========================
# LOAD MODELS
# =========================
print("⏳ Loading MusicGen Medium (High Quality)...")
# Используем MEDIUM модель, так как A4000 это тянет легко
processor = MusicgenProcessor.from_pretrained("facebook/musicgen-medium")
music_model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-medium",
    torch_dtype=torch.float16 # FP16 для скорости и экономии памяти
).to(DEVICE)
music_model.eval()

print("⏳ Loading Stable Diffusion...")
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to(DEVICE)
sd_pipe.safety_checker = None

# =========================
# THE "BRAIN" (Smart DJ Logic)
# =========================
class DJBrain:
    def __init__(self):
        self.locations = ["Cyber-Tokyo", "Neo-Seoul", "Mars Colony 4", "Digital Void", "Sector 7"]
        self.weather = ["Acid Rain", "Neon Fog", "Solar Flares", "Data Storms", "Clear Skies"]
        self.topics = [
            "AI Rights", "The simulation theory", "Why humans love coffee", 
            "The update to Neural Link v5.0", "Old school internet archives"
        ]
    
    def get_script(self):
        # Здесь можно подключить реальный API к GPT-4/Claude (CrewAI)
        # Пока имитируем умную генерацию
        mode = random.choice(["weather", "news", "vibe"])
        
        if mode == "weather":
            loc = random.choice(self.locations)
            weath = random.choice(self.weather)
            return f"Current status in {loc}: {weath}. Stay safe, net-runners. Here is the next track."
        
        elif mode == "news":
            topic = random.choice(self.topics)
            return f"Trending now on the neural net: {topic}. Think about it while you listen to this beat."
        
        else:
            return "System optimal. Vitals stable. Injecting dopamine through audio waves. Enjoy."

    def get_music_prompt(self):
        genres = [
            "lo-fi hip hop, vinyl crackle, chill", 
            "synthwave, retrowave, 80s drums, driving", 
            "cyberpunk, industrial, dark bass, cinematic", 
            "deep house, melodic, vocal chops, summer", 
            "ambient, space drone, meditation, relaxing"
        ]
        return random.choice(genres)

brain = DJBrain()

# =========================
# HELPERS
# =========================
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def save_audio_normalized(audio_tensor, filename, sr):
    """
    ВАЖНО: Нормализация и конвертация в 16-bit PCM.
    Это решает проблему 'шума' вместо звука.
    """
    audio_np = audio_tensor[0, 0].cpu().float().numpy()
    
    # Нормализация (чтобы не было клиппинга)
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        audio_np = audio_np / max_val * 0.9  # 90% громкости
    
    sf.write(filename, audio_np, sr, subtype='PCM_16')

# =========================
# GENERATION WORKER
# =========================
def generate_segment(idx):
    print(f"\n🔨 [Worker] Processing segment {idx}...")
    t0 = time.time()
    
    # 1. Получаем задание от "Мозга"
    music_prompt = brain.get_music_prompt()
    dj_text = brain.get_script()
    
    # Пути
    music_path = os.path.join(WORKDIR, f"temp_music_{idx}.wav")
    voice_path = os.path.join(WORKDIR, f"temp_voice_{idx}.wav")
    cover_path = os.path.join(WORKDIR, f"temp_cover_{idx}.png")
    final_video = os.path.join(WORKDIR, f"segment_{idx}.ts") # .ts лучше клеится

    # 2. Генерация музыки (MusicGen)
    inputs = processor(text=[music_prompt], padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        # 1000 токенов ~= 20 секунд. Medium модель дает хорошее качество.
        audio_values = music_model.generate(**inputs, max_new_tokens=1000, guidance_scale=3.5)
    
    save_audio_normalized(audio_values, music_path, music_model.config.audio_encoder.sampling_rate)

    # 3. Генерация обложки (SD)
    with torch.no_grad():
        image = sd_pipe(f"{music_prompt}, masterpiece, 8k, digital art", num_inference_steps=20).images[0]
    image.save(cover_path)

    # 4. Генерация голоса (TTS)
    asyncio.run(edge_tts.Communicate(dj_text, "en-US-ChristopherNeural").save(voice_path))

    # 5. Сборка видео (FFmpeg)
    # Определяем длительность музыки для лупа
    f = sf.SoundFile(music_path)
    music_dur = len(f) / f.samplerate
    # Делаем сегмент ~60 секунд (лупим музыку 3 раза)
    total_dur = music_dur * 3 

    # Сложная команда FFmpeg:
    # - Лупим картинку
    # - Лупим музыку
    # - Накладываем голос в начале
    # - Нормализуем аудио при миксе
    # - Кодируем аудио в AAC сразу, чтобы стримеру было легче
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", cover_path,              # 0: Картинка
        "-i", voice_path,                            # 1: Голос
        "-stream_loop", "-1", "-i", music_path,      # 2: Музыка (бесконечный луп, обрежем по -t)
        "-t", str(total_dur),                        # Длительность сегмента
        "-filter_complex",
        "[1:a]volume=1.5[v];[2:a]volume=0.7[m];[v][m]amix=inputs=2:duration=first:dropout_transition=3[a_mix];[a_mix]acompressor=ratio=4[a_out]",
        "-map", "0:v", "-map", "[a_out]",
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-g", "60",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-f", "mpegts", final_video
    ]
    
    subprocess.run(cmd, check=True)
    
    # Чистим временные файлы
    for f in [music_path, voice_path, cover_path]:
        if os.path.exists(f): os.remove(f)
    
    cleanup()
    print(f"✅ [Worker] Segment {idx} ready ({round(time.time()-t0)}s)")
    return final_video

# =========================
# THREADS
# =========================
def worker_thread():
    """Постоянно создает новые сегменты в фон"""
    idx = 0
    while True:
        if video_queue.full():
            time.sleep(1)
            continue
        
        try:
            seg_path = generate_segment(idx)
            video_queue.put(seg_path)
            idx += 1
        except Exception as e:
            print(f"❌ Worker Error: {e}")
            time.sleep(5)

def streamer_thread():
    """Отправляет готовые сегменты в Twitch"""
    print("📡 Streamer started. Waiting for buffer...")
    
    # Ждем заполнения буфера (минимум 2 видео)
    while video_queue.qsize() < 2:
        time.sleep(2)
        print(f"   Buffering: {video_queue.qsize()}/2...")

    print("🔴 GOING LIVE!")

    # Запускаем FFmpeg в режиме чтения из pipe
    stream_cmd = [
        "ffmpeg", "-re", 
        "-f", "concat", "-safe", "0", "-i", "pipe:0",
        "-c", "copy", # Просто копируем, так как Worker уже всё закодировал
        "-f", "flv", RTMP_URL
    ]
    
    process = subprocess.Popen(stream_cmd, stdin=subprocess.PIPE)

    while True:
        seg_path = video_queue.get()
        print(f"▶️ Playing: {seg_path} (Queue: {video_queue.qsize()})")
        
        # Формируем строку для concat протокола FFmpeg
        # file '/path/to/file.ts'
        line = f"file '{seg_path}'\n".encode('utf-8')
        
        try:
            process.stdin.write(line)
            process.stdin.flush()
        except BrokenPipeError:
            print("❌ Stream pipe broken. Restarting...")
            break
            
        # Важно: В режиме concat через pipe мы не можем просто удалить файл сразу,
        # так как ffmpeg его читает.
        # В идеале нужно удалять старые файлы с задержкой. 
        # Для простоты в этом скрипте мы оставим их копиться (на A4000 места много),
        # либо можно реализовать "мусорщик" в отдельном потоке.

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Запускаем Worker в отдельном потоке
    t_worker = threading.Thread(target=worker_thread, daemon=True)
    t_worker.start()

    # Стример запускаем в главном потоке (или тоже в отдельном)
    streamer_thread()
