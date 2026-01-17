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
import requests
import torch
from diffusers import StableDiffusionPipeline
import edge_tts
from crewai import Agent, Task, Crew

# === НОВЫЕ ИМПОРТЫ ДЛЯ STABLE AUDIO ===
from huggingface_hub import login
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference import generate_diffusion_cond

# =========================
# 1. CONFIG & ENV
# =========================
os.environ["HF_HOME"] = "/workspace/hf_cache"

# КЛЮЧИ
STREAM_KEY = os.environ.get("TWITCH_STREAM_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN") # <--- ОБЯЗАТЕЛЬНО НУЖЕН!

if not STREAM_KEY:
    print("⚠️ WARNING: TWITCH_STREAM_KEY not found.")
if not HF_TOKEN:
    print("❌ CRITICAL: HF_TOKEN not found! Stable Audio won't download.")
    # Если токена нет в ENV, попробуем войти интерактивно или упадем
else:
    print("🔑 Logging into HuggingFace...")
    login(token=HF_TOKEN)

RTMP_URL = f"rtmp://live.twitch.tv/app/{STREAM_KEY}"
WORKDIR = "/workspace/airadio/data"
os.makedirs(WORKDIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Device: {DEVICE}")

# Очередь сегментов
video_queue = queue.Queue(maxsize=3) # Уменьшил буфер до 3, т.к. генерация дольше
TRACKS_BEFORE_DJ = 3 

# =========================
# 2. LOAD MODELS
# =========================
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

print("⏳ Loading Stable Audio Open 1.0...")
# Загружаем модель Stable Audio
# Она тяжелая, поэтому сразу чистим память перед загрузкой
cleanup()
audio_model, audio_cfg = get_pretrained_model("stabilityai/stable-audio-open-1.0", device=DEVICE)
audio_model.to(DEVICE).eval()

print("⏳ Loading Stable Diffusion...")
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True
).to(DEVICE)
sd_pipe.safety_checker = None

# =========================
# 3. CREW AI & LOGIC
# =========================
TECH_KEYWORDS = ["AI", "ML", "OpenAI", "LLM", "NVIDIA", "Robotics", "SpaceX", "Python", "Cyberpunk", "Neural"]

def fetch_tech_news():
    try:
        r = requests.get(
            "https://hn.algolia.com/api/v1/search_by_date",
            params={"query": "AI", "tags": "story", "hitsPerPage": 15},
            timeout=10
        )
        hits = r.json().get("hits", [])
        news = [h["title"] for h in hits if any(k.lower() in h.get("title", "").lower() for k in TECH_KEYWORDS)]
        return news[:3] if news else ["Neural networks are evolving rapidly."]
    except Exception as e:
        print(f"⚠️ News fetch failed: {e}")
        return ["Systems operating normally."]

class CrewAIDJ:
    def __init__(self):
        if not OPENAI_API_KEY:
            self.agent = None
            return

        self.agent = Agent(
            role="Cyberpunk Radio Host",
            goal="Deliver short, cool updates about technology between music tracks.",
            backstory="You are 'Nexus', an AI host on a futuristic radio station playing Punk and Electronic Rock.",
            verbose=False,
            allow_delegation=False
        )

    def generate_script(self, mood="high energy"):
        if not self.agent:
            return "System status nominal. Crank up the volume."

        news_items = fetch_tech_news()
        news_str = "\n- ".join(news_items)

        task = Task(
            description=f"""
            Live on air. Mood: {mood}.
            Tech Headlines: {news_str}
            Instructions:
            1. Mention one headline briefly.
            2. Be cool, concise, energetic.
            3. Under 3 sentences.
            """,
            agent=self.agent,
            expected_output="Short DJ script."
        )

        crew = Crew(agents=[self.agent], tasks=[task])
        try:
            return str(crew.kickoff())
        except Exception as e:
            print(f"⚠️ CrewAI Error: {e}")
            return "Data stream synchronized. Listen to this."

ai_dj = CrewAIDJ()

# === НОВЫЕ ПРОМПТЫ ДЛЯ STABLE AUDIO (ЛЮБИТ СПИСКИ ЧЕРЕЗ ЗАПЯТУЮ) ===
def get_vibes():
    genres = [
        (
            "punk rock, fast tempo, distorted guitars, aggressive drums, high fidelity, studio recording, heavy bass", 
            "punk rock poster, anarchy symbol, graffiti, red and black, grunge texture"
        ),
        (
            "post-punk, dark wave, chorus guitar, driving bassline, melancholic, 80s goth vibe, reverb, atmospheric", 
            "post-punk album cover, monochrome, brutalist architecture, dark fog"
        ),
        (
            "happy hardcore, 170bpm, energetic piano, heavy kick drum, rave, dance, synthesizer, uplifting", 
            "colorful rave party, lasers, neon rainbows, high energy"
        ),
        (
            "electronic rock, industrial metal, distorted synths, powerful drums, cyberpunk action, cinematic", 
            "cyberpunk rocker, neon guitar, futuristic city, glitch art"
        ),
        (
            "drum and bass, liquid dnb, fast breakbeats, deep sub bass, atmospheric pads, soulful, melodic", 
            "futuristic tunnel, speed lines, neon blue and orange, liquid fluid abstract"
        )
    ]
    return random.choice(genres)

# =========================
# 4. AUDIO GENERATION (STABLE AUDIO)
# =========================
def gen_music_stable_audio(prompt, out_wav, duration_sec=45):
    print(f"🎧 StableAudio Generating: {prompt}...")
    sample_rate = 44100
    
    # Очищаем память перед тяжелой генерацией
    cleanup()
    
    with torch.no_grad():
        audio = generate_diffusion_cond(
            model=audio_model,
            conditioning=[{
                "prompt": prompt,
                "seconds_start": 0,
                "seconds_total": duration_sec
            }],
            steps=100,           # 100 шагов - баланс скорости/качества (можно 150, но будет медленнее)
            cfg_scale=7.0,       # 7.0 дает хорошее следование промпту
            sample_rate=sample_rate,
            device=DEVICE
        )

    # Нормализация
    audio = audio / audio.abs().max().clamp(min=1e-6)
    audio = audio * 0.95 

    # Сохранение (Транспонируем тензор, если нужно)
    # Stable Audio выдает [1, 2, samples], torchaudio ожидает [channels, samples] - обычно ок
    audio = audio.squeeze(0) # убираем batch dimension -> [2, samples]
    
    # Сохраняем через soundfile для надежности (torchaudio иногда капризничает с форматами на RunPod)
    audio_np = audio.cpu().numpy().T # -> [samples, 2]
    sf.write(out_wav, audio_np, sample_rate)
    
    return sample_rate

# =========================
# 5. WORKER
# =========================
def generate_segment(idx, is_dj_turn):
    print(f"\n🔨 [Worker] Processing segment {idx} (DJ Turn: {is_dj_turn})...")
    t0 = time.time()
    
    music_prompt, visual_prompt = get_vibes()
    
    # Файлы
    music_path = os.path.join(WORKDIR, f"temp_music_{idx}.wav")
    voice_path = os.path.join(WORKDIR, f"temp_voice_{idx}.wav") if is_dj_turn else None
    cover_path = os.path.join(WORKDIR, f"temp_cover_{idx}.png")
    final_video = os.path.join(WORKDIR, f"segment_{idx}.ts")

    # A. Generate Music (Stable Audio)
    # Генерируем ~45-50 секунд (Stable Audio Open умеет до 47с)
    gen_music_stable_audio(music_prompt, music_path, duration_sec=45)

    # B. Generate Cover (SD)
    # Чтобы не вылететь по памяти, можно выгрузить SD в CPU, если будет OOM,
    # Но на 16GB должно влезть параллельно.
    with torch.no_grad():
        image = sd_pipe(f"{visual_prompt}, masterpiece, 8k", num_inference_steps=20).images[0]
    image.save(cover_path)

    # C. TTS
    if is_dj_turn:
        mood = music_prompt.split(",")[0]
        dj_text = ai_dj.generate_script(mood=mood)
        print(f"🗣️ DJ Says: {dj_text}")
        asyncio.run(edge_tts.Communicate(dj_text, "en-US-ChristopherNeural").save(voice_path))

    # D. Assembly
    f = sf.SoundFile(music_path)
    music_dur = len(f) / f.samplerate
    # Stable Audio делает полноценный трек, его не обязательно лупить 3 раза
    # Но для надежности залупим 2 раза, чтобы получить ~1.5 минуты эфира
    total_dur = music_dur * 2 

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    cmd += ["-loop", "1", "-i", cover_path]
    if is_dj_turn:
        cmd += ["-i", voice_path]
    
    cmd += ["-stream_loop", "1", "-i", music_path] # loop 1 раз (играет 2 раза)
    cmd += ["-t", str(total_dur)]

    if is_dj_turn:
        # Crossfade
        filter_str = "[1:a]volume=1.5[v];[2:a]volume=0.9[m];[v][m]amix=inputs=2:duration=longest:dropout_transition=2[mix];[mix]acompressor=ratio=4[aout]"
    else:
        filter_str = "[1:a]volume=1.0,acompressor=ratio=4[aout]"

    cmd += ["-filter_complex", filter_str]
    cmd += [
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-g", "60",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-f", "mpegts", final_video
    ]
    
    subprocess.run(cmd, check=True)
    
    for f in [music_path, voice_path, cover_path]:
        if f and os.path.exists(f): os.remove(f)
    
    cleanup()
    print(f"✅ [Worker] Segment {idx} ready ({round(time.time()-t0)}s)")
    return final_video

def worker_thread():
    idx = 0
    tracks_since_dj = 0
    while True:
        if video_queue.full():
            time.sleep(1)
            continue
        try:
            is_dj_turn = (tracks_since_dj >= TRACKS_BEFORE_DJ)
            seg_path = generate_segment(idx, is_dj_turn)
            video_queue.put(seg_path)
            idx += 1
            if is_dj_turn:
                tracks_since_dj = 0
            else:
                tracks_since_dj += 1
        except Exception as e:
            print(f"❌ Worker Error: {e}")
            time.sleep(5)

# =========================
# 6. STREAMER
# =========================
def streamer_thread():
    print("📡 Streamer started. Buffering...")
    # Ждем 1 готовый сегмент. Stable Audio медленный, поэтому буфер набирается дольше.
    while video_queue.qsize() < 1:
        time.sleep(5)
    print("🔴 GOING LIVE!")

    stream_cmd = [
        "ffmpeg", "-re",
        "-f", "mpegts", "-i", "pipe:0",
        "-c", "copy",
        "-f", "flv", RTMP_URL
    ]
    
    process = subprocess.Popen(stream_cmd, stdin=subprocess.PIPE)

    while True:
        seg_path = video_queue.get()
        print(f"▶️ Playing: {seg_path} (Queue: {video_queue.qsize()})")
        
        try:
            with open(seg_path, "rb") as f:
                while True:
                    chunk = f.read(4096 * 10)
                    if not chunk: break
                    process.stdin.write(chunk)
            process.stdin.flush()
        except BrokenPipeError:
            print("❌ Stream pipe broken. Restarting FFmpeg...")
            process = subprocess.Popen(stream_cmd, stdin=subprocess.PIPE)
        except Exception as e:
            print(f"❌ Streamer Error: {e}")

        if os.path.exists(seg_path):
            os.remove(seg_path)

if __name__ == "__main__":
    t_worker = threading.Thread(target=worker_thread, daemon=True)
    t_worker.start()
    streamer_thread()
