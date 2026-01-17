import os
import time
import random
import subprocess
import gc
import soundfile as sf
import numpy as np
import threading
import queue
import asyncio
import torch
from transformers import MusicgenForConditionalGeneration, MusicgenProcessor
from diffusers import StableDiffusionPipeline
import edge_tts

# =========================
# 1. CONFIG & ENV
# =========================
# Переносим кэш на большой диск
os.environ["HF_HOME"] = "/workspace/hf_cache"

STREAM_KEY = os.environ.get("TWITCH_STREAM_KEY")
if not STREAM_KEY:
    print("⚠️ WARNING: TWITCH_STREAM_KEY not found in env.")

RTMP_URL = f"rtmp://live.twitch.tv/app/{STREAM_KEY}"
WORKDIR = "/workspace/airadio/data"
os.makedirs(WORKDIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Device: {DEVICE}")

# Очередь сегментов
video_queue = queue.Queue(maxsize=4)

# =========================
# 2. LOAD MODELS (Optimized for A4000)
# =========================

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

print("⏳ Loading MusicGen Medium...")
# FIX: Используем float32, чтобы избежать NaN/шума при guidance_scale
# На A4000 (16GB) памяти хватит.
processor = MusicgenProcessor.from_pretrained("facebook/musicgen-medium")
music_model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-medium",
    torch_dtype=torch.float32, 
    use_safetensors=True
).to(DEVICE)
music_model.eval()

print("⏳ Loading Stable Diffusion...")
# SD оставляем в fp16, ей это ок
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True
).to(DEVICE)
sd_pipe.safety_checker = None

# =========================
# 3. DJ LOGIC
# =========================
class DJBrain:
    def __init__(self):
        self.locations = ["Cyber-Tokyo", "Neo-Seoul", "Mars Colony 4", "Sector 7"]
        self.weather = ["Acid Rain", "Neon Fog", "Solar Flares", "Data Storms"]
        self.topics = ["AI Consciousness", "The Simulation", "Retro Hardware", "Neural Link Updates"]
    
    def get_script(self):
        mode = random.choice(["weather", "news", "vibe"])
        if mode == "weather":
            return f"Weather alert for {random.choice(self.locations)}: {random.choice(self.weather)}. Stay inside and listen."
        elif mode == "news":
            return f"Topic of the day: {random.choice(self.topics)}. Processing..."
        else:
            return "System optimal. Audio injection active. Enjoy the stream."

    def get_music_prompt(self):
        genres = [
            "lo-fi hip hop, chill, vinyl crackle", 
            "synthwave, retrowave, driving, 80s", 
            "cyberpunk, dark industrial, bass", 
            "deep house, melodic, summer vibe", 
            "ambient, space drone, meditation"
        ]
        return random.choice(genres)

brain = DJBrain()

# =========================
# 4. AUDIO PROCESSING
# =========================
def save_audio_normalized(audio_tensor, filename, sr):
    """Нормализация + конвертация в PCM_16"""
    audio_np = audio_tensor[0, 0].cpu().float().numpy()
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        audio_np = audio_np / max_val * 0.95
    sf.write(filename, audio_np, sr, subtype='PCM_16')

# =========================
# 5. WORKER (GENERATOR)
# =========================
def generate_segment(idx):
    print(f"\n🔨 [Worker] Processing segment {idx}...")
    t0 = time.time()
    
    music_prompt = brain.get_music_prompt()
    dj_text = brain.get_script()
    
    # Файлы
    music_path = os.path.join(WORKDIR, f"temp_music_{idx}.wav")
    voice_path = os.path.join(WORKDIR, f"temp_voice_{idx}.wav")
    cover_path = os.path.join(WORKDIR, f"temp_cover_{idx}.png")
    final_video = os.path.join(WORKDIR, f"segment_{idx}.ts")

    # A. MusicGen
    inputs = processor(text=[music_prompt], padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        # fp32 позволяет безопасно использовать guidance_scale
        audio_values = music_model.generate(**inputs, max_new_tokens=1000, guidance_scale=3.0)
    
    save_audio_normalized(audio_values, music_path, music_model.config.audio_encoder.sampling_rate)

    # B. Stable Diffusion
    with torch.no_grad():
        image = sd_pipe(f"{music_prompt}, masterpiece, 8k, wallpaper", num_inference_steps=20).images[0]
    image.save(cover_path)

    # C. TTS
    asyncio.run(edge_tts.Communicate(dj_text, "en-US-ChristopherNeural").save(voice_path))

    # D. FFmpeg Assembly
    f = sf.SoundFile(music_path)
    music_dur = len(f) / f.samplerate
    total_dur = music_dur * 3  # Loop 3 times (~60 sec)

    # FIX: duration=longest (чтобы музыка не обрезалась по голосу)
    # FIX: acompressor (выравниваем громкость)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", cover_path,
        "-i", voice_path,
        "-stream_loop", "-1", "-i", music_path,
        "-t", str(total_dur),
        "-filter_complex",
        "[1:a]volume=1.4[v];[2:a]volume=0.8[m];[v][m]amix=inputs=2:duration=longest:dropout_transition=2[mix];[mix]acompressor=ratio=4[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-g", "60",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-f", "mpegts", final_video
    ]
    
    subprocess.run(cmd, check=True)
    
    # Cleanup temps
    for f in [music_path, voice_path, cover_path]:
        if os.path.exists(f): os.remove(f)
    
    cleanup() # Чистим VRAM
    print(f"✅ [Worker] Segment {idx} ready ({round(time.time()-t0)}s)")
    return final_video

def worker_thread():
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

# =========================
# 6. STREAMER (Correct Implementation)
# =========================
def streamer_thread():
    print("📡 Streamer started. Buffering...")
    while video_queue.qsize() < 2:
        time.sleep(2)
    print("🔴 GOING LIVE!")

    # FIX: Используем правильный pipe streaming (Byte Feeding)
    # Мы открываем FFmpeg один раз и кормим его байтами .ts файлов
    stream_cmd = [
        "ffmpeg", "-re",
        "-f", "mpegts", "-i", "pipe:0", # Читаем mpegts из stdin
        "-c", "copy",                   # Просто копируем, так как Worker уже сжал
        "-f", "flv", RTMP_URL
    ]
    
    process = subprocess.Popen(stream_cmd, stdin=subprocess.PIPE)

    while True:
        seg_path = video_queue.get()
        print(f"▶️ Playing: {seg_path} (Queue: {video_queue.qsize()})")
        
        try:
            # Читаем файл кусками и пишем в pipe
            with open(seg_path, "rb") as f:
                while True:
                    chunk = f.read(4096 * 10) # Читаем по ~40KB
                    if not chunk: break
                    process.stdin.write(chunk)
            
            # Важно: flush не обязателен каждый раз, но полезен
            process.stdin.flush()
            
        except BrokenPipeError:
            print("❌ Stream pipe broken. Restarting FFmpeg...")
            # Перезапуск процесса
            process = subprocess.Popen(stream_cmd, stdin=subprocess.PIPE)
        except Exception as e:
            print(f"❌ Streamer Error: {e}")

        # FIX: Garbage Collection (Мусорщик)
        # Удаляем файл сразу после проигрывания
        if os.path.exists(seg_path):
            os.remove(seg_path)
            print(f"🗑️ Deleted {seg_path}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    t_worker = threading.Thread(target=worker_thread, daemon=True)
    t_worker.start()
    streamer_thread()
