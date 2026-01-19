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
from einops import rearrange
from diffusers import StableDiffusionPipeline
import edge_tts
from crewai import Agent, Task, Crew
from huggingface_hub import login

# === IMPORTS ===
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from twitchio.ext import commands

# =========================
# 1. CONFIG & ENV
# =========================
os.environ["HF_HOME"] = "/workspace/hf_cache"

STREAM_KEY = os.environ.get("TWITCH_STREAM_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
TWITCH_TOKEN = os.environ.get("TWITCH_TOKEN")
CHANNEL_NAME = os.environ.get("TWITCH_CHANNEL") or "mediff23"

if not STREAM_KEY: print("⚠️ WARNING: TWITCH_STREAM_KEY not found.", flush=True)
if not HF_TOKEN: print("❌ CRITICAL: HF_TOKEN not found!", flush=True)
else: 
    try: login(token=HF_TOKEN)
    except Exception: pass

RTMP_URL = f"rtmp://live.twitch.tv/app/{STREAM_KEY}"
WORKDIR = "/workspace/airadio/data"
os.makedirs(WORKDIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Device: {DEVICE}", flush=True)

video_queue = queue.Queue(maxsize=4)
user_prompt_queue = queue.Queue(maxsize=1)
GENRE_POOL = {} 
GENRE_IMAGES = {} # Кэш картинок
POOL_LOCK = threading.Lock()
TRACKS_BEFORE_DJ = 3 

# =========================
# 2. VIBES
# =========================
quality_suffix = ", high quality studio recording, clear stereo image, professional mix, no fade out"
# (Prompt, Visual Prompt)
VIBES_LIST = [
    (f"punk rock, fast tempo, distorted electric guitars, live drum kit{quality_suffix}", "punk rock poster, anarchy symbol, graffiti, red and black"),
    (f"post-punk, dark wave, chorus guitar, driving bassline, melancholic{quality_suffix}", "post-punk album cover, monochrome, brutalist architecture"),
    (f"happy hardcore, 170bpm, energetic piano, heavy kick drum, rave{quality_suffix}", "colorful rave party, lasers, neon rainbows"),
    (f"electronic rock, industrial metal, distorted synths, powerful drums{quality_suffix}", "cyberpunk rocker, neon guitar, futuristic city"),
    (f"drum and bass, liquid dnb, fast breakbeats, deep sub bass{quality_suffix}", "futuristic tunnel, speed lines, neon blue and orange")
]

# =========================
# 3. TWITCH BOT
# =========================
class Bot(commands.Bot):
    def __init__(self):
        super().__init__(token=TWITCH_TOKEN, prefix='!', initial_channels=[CHANNEL_NAME])
    async def event_ready(self): print(f'🎮 Twitch Bot logged in as | {self.nick}', flush=True)
    @commands.command(name='vibe', aliases=['вайб'])
    async def vibe_command(self, ctx: commands.Context):
        content = ctx.message.content
        prompt = content.replace("!vibe", "").replace("!вайб", "").strip()[:100]
        if len(prompt) < 3: return
        if user_prompt_queue.empty():
            user_prompt_queue.put({"user": ctx.author.name, "prompt": prompt})
            await ctx.send(f"@{ctx.author.name}, accepted: {prompt} 🎹")

def run_twitch_bot():
    if not TWITCH_TOKEN: return
    try:
        bot = Bot()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bot.run()
    except Exception: pass

# =========================
# 4. LOAD MODELS
# =========================
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

# --- PRE-GENERATE IMAGES ---
def init_images():
    print("🎨 Generating Genre Covers (One-time)...", flush=True)
    cleanup()
    
    # Грузим SD только на время генерации обложек
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True).to(DEVICE)
    pipe.safety_checker = None

    # Генерируем 5 обложек для жанров
    for i, (audio_prompt, visual_prompt) in enumerate(VIBES_LIST):
        path = os.path.join(WORKDIR, f"cover_genre_{i}.png")
        if not os.path.exists(path):
            print(f"   - Cover {i}: {visual_prompt[:20]}...", flush=True)
            pipe(f"{visual_prompt}, masterpiece, 8k, wallpaper", num_inference_steps=25).images[0].save(path)
        GENRE_IMAGES[i] = path

    # Генерируем 1 обложку для "User Request"
    req_path = os.path.join(WORKDIR, "cover_request.png")
    if not os.path.exists(req_path):
        pipe("Abstract AI art, digital soundwaves, neon cyber, masterpiece", num_inference_steps=25).images[0].save(req_path)
    GENRE_IMAGES["request"] = req_path

    # УДАЛЯЕМ SD ИЗ ПАМЯТИ НАВСЕГДА
    del pipe
    cleanup()
    print("🎨 Covers ready. SD unloaded. VRAM freed.", flush=True)

# Запускаем генерацию картинок сразу
init_images()

print("⏳ Loading Stable Audio (CPU)...", flush=True)
audio_model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
# Модель живет на CPU, прыгает на GPU только для работы
audio_model = audio_model.to("cpu").eval() 

# =========================
# 5. CREW AI
# =========================
class CrewAIDJ:
    def __init__(self):
        self.fallback_scripts = [
            "Global network connected. I am Nexus.", "Data streams stable.", "Processing complete.",
            "Silence is just empty data.", "Type !vibe in the chat.", "Processors running hot.",
            "Neural link established.", "Humanity is analog. Music is digital.",
            "Switching context.", "Ghosts in the machine."
        ]
        self.agent = None
        if OPENAI_API_KEY:
            self.agent = Agent(role="Cyberpunk Radio Host", goal="Updates.", backstory="AI Host Nexus.", verbose=False)

    def generate_script(self, mood="high energy", user_request=None):
        if not self.agent: return random.choice(self.fallback_scripts)
        special = f"Shoutout to {user_request['user']} for {user_request['prompt']}." if user_request else ""
        task = Task(description=f"Radio Host. Vibe: {mood}. {special}. Max 2 sentences. Cool/Robotic.", agent=self.agent, expected_output="Script")
        try: return str(Crew(agents=[self.agent], tasks=[task]).kickoff())
        except: return random.choice(self.fallback_scripts)

ai_dj = CrewAIDJ()

# =========================
# 6. AUDIO HELPERS
# =========================
def gen_music(prompt, out_wav, duration_sec=45):
    print(f"🎧 StableAudio Generating...", flush=True)
    cleanup()
    try:
        # Swap In
        audio_model.to(DEVICE)
        
        with torch.no_grad():
            out = generate_diffusion_cond(
                audio_model, steps=100, cfg_scale=5.5, conditioning=[{"prompt": prompt, "seconds_start": 0, "seconds_total": duration_sec}],
                sample_size=sample_size, sigma_min=0.3, sigma_max=500, sampler_type="dpmpp-3m-sde", device=DEVICE
            )
        
        # Swap Out (сразу освобождаем GPU)
        audio_model.to("cpu")
        cleanup()

        out = rearrange(out, "b d n -> d (b n)").to(torch.float32).div(torch.max(torch.abs(out))).clamp(-1, 1)
        sf.write(out_wav, out.cpu().numpy().T, sample_rate, subtype='PCM_16')
        return True
    except Exception as e:
        print(f"❌ Music Error: {e}", flush=True)
        try: audio_model.to("cpu")
        except: pass
        cleanup()
        return False

def sanitize_voice_track(raw_path, clean_path):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", raw_path,
        "-ar", "44100", "-ac", "2",
        "-af", "highpass=f=100,lowpass=f=7000,volume=1.8,acompressor=threshold=-16dB:ratio=6:attack=5:release=80",
        clean_path
    ]
    try: subprocess.run(cmd, check=True); return True
    except: return False

# =========================
# 7. WORKER
# =========================
def generate_segment(segment_id, is_dj_turn, forced_genre_idx=None):
    # 1. Vibes & Image Selection
    if not user_prompt_queue.empty() and forced_genre_idx is None:
        order = user_prompt_queue.get()
        prompt, genre_idx = f"{order['prompt']}, {quality_suffix}", -1
        cover_file = GENRE_IMAGES["request"] # Используем готовую картинку для реквестов
        is_dj_turn = True
    else:
        idx = forced_genre_idx if forced_genre_idx is not None else random.randint(0, len(VIBES_LIST)-1)
        prompt, _ = VIBES_LIST[idx]
        genre_idx = idx
        cover_file = GENRE_IMAGES[idx] # Используем кэшированную картинку
        order = None

    music_path = os.path.join(WORKDIR, f"m_{segment_id}.wav")
    raw_voice_path = os.path.join(WORKDIR, f"v_raw_{segment_id}.wav")
    clean_voice_path = os.path.join(WORKDIR, f"v_clean_{segment_id}.wav")
    final_path = os.path.join(WORKDIR, f"seg_{segment_id}.ts")

    # A. Music
    if not gen_music(prompt, music_path): return None

    # B. Cover (SKIPPED - Already generated)
    
    # C. DJ Logic
    if is_dj_turn:
        try:
            mood = prompt.split(",")[0]
            text = ai_dj.generate_script(mood, order)
            print(f"🗣️ DJ: {text}", flush=True)
            asyncio.run(edge_tts.Communicate(text, "en-US-ChristopherNeural").save(raw_voice_path))
            if os.path.exists(raw_voice_path) and os.path.getsize(raw_voice_path) > 1000:
                if not sanitize_voice_track(raw_voice_path, clean_voice_path): is_dj_turn = False
            else: is_dj_turn = False
        except: is_dj_turn = False

    # D. Assembly (LITE MODE - No Visualizer)
    # -tune stillimage оптимизирует кодирование статичной картинки (почти 0 CPU)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", cover_file]
    if is_dj_turn: cmd += ["-i", clean_voice_path] # [1]
    cmd += ["-i", music_path, "-i", music_path]    # [2], [3]

    fc = []
    m1, m2 = ("2", "3") if is_dj_turn else ("1", "2")
    
    # 1. Music Loop (Simple)
    fc.append(f"[{m1}:a]anull[a_main]")
    fc.append(f"[{m2}:a]atrim=start=5,asetpts=PTS-STARTPTS[a_loop]")
    fc.append(f"[a_main][a_loop]acrossfade=d=5:c1=tri:c2=tri[m_raw]")
    
    # 2. Voice Mix (Simple Audio Only)
    if is_dj_turn:
        fc.append(f"[1:a]asplit[v_sc][v_mix]")
        fc.append(f"[m_raw][v_sc]sidechaincompress=threshold=0.05:ratio=10:attack=5:release=300[m_duck]")
        fc.append(f"[m_duck][v_mix]amix=inputs=2:duration=first[a_fin]")
    else:
        fc.append(f"[m_raw]loudnorm=I=-14:TP=-1.0:LRA=11[a_fin]")

    cmd += ["-filter_complex", ";".join(fc)]
    
    # Video settings optimized for static image
    cmd += [
        "-map", "0:v", "-map", "[a_fin]", 
        "-t", "80", 
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage", # <-- Super Fast Video
        "-pix_fmt", "yuv420p", "-r", "10", "-g", "20", # 10 FPS достаточно для статики
        "-c:a", "aac", "-b:a", "192k", 
        "-f", "mpegts", final_path
    ]
    
    try: subprocess.run(cmd, check=True)
    except Exception as e: 
        print(f"❌ FFmpeg Error: {e}", flush=True)
        if is_dj_turn: return generate_segment(segment_id, False, genre_idx)
        return None

    for f in [music_path, raw_voice_path, clean_voice_path]: # Не удаляем cover_file!
        if os.path.exists(f): os.remove(f)
    cleanup()

    if genre_idx != -1:
        with POOL_LOCK:
            old = GENRE_POOL.get(genre_idx)
            GENRE_POOL[genre_idx] = final_path
            # Старые видео файлы удаляем
            if old and old != final_path and os.path.exists(old): 
                try: os.remove(old)
                except: pass
    return final_path

def worker_thread():
    print("🚀 Init Pool (Generating audio only)...", flush=True)
    for i in range(len(VIBES_LIST)):
        print(f"🌊 Gen {i}...", flush=True)
        p = generate_segment(f"init_{i}", False, i)
        if p: video_queue.put(p)
    print("✅ Live.", flush=True)
    
    idx, cnt = 0, 0
    while True:
        if video_queue.full(): 
            time.sleep(2)
            continue
        try:
            dj = (cnt >= TRACKS_BEFORE_DJ)
            p = generate_segment(f"s_{idx}", dj)
            if p: 
                video_queue.put(p)
                idx += 1
                cnt = 0 if dj else cnt + 1
        except Exception as e: print(f"❌ Worker: {e}", flush=True); time.sleep(5)

def streamer_thread():
    while video_queue.empty() and not GENRE_POOL: time.sleep(5)
    
    cmd = [
        "ffmpeg", "-re", 
        "-fflags", "+genpts+igndts", 
        "-use_wallclock_as_timestamps", "1",
        "-f", "mpegts", "-i", "pipe:0", 
        "-c", "copy", 
        "-f", "flv", RTMP_URL
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    
    while True:
        use_pool = False
        path = None
        
        if not video_queue.empty():
            path = video_queue.get()
        elif GENRE_POOL:
            with POOL_LOCK:
                genre = random.choice(list(GENRE_POOL.keys()))
                path = GENRE_POOL[genre]
                use_pool = True
                print(f"♻️ Pool: {path}", flush=True)
        
        if not path or not os.path.exists(path):
            time.sleep(1)
            continue

        try:
            with open(path, "rb") as f:
                while chunk := f.read(65536): 
                    try: proc.stdin.write(chunk)
                    except: 
                        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                        proc.stdin.write(chunk)
            proc.stdin.flush()
            
            if not use_pool and os.path.exists(path):
                try: os.remove(path)
                except: pass
                
        except Exception as e:
            print(f"Stream error: {e}", flush=True)
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

if __name__ == "__main__":
    threading.Thread(target=run_twitch_bot, daemon=True).start()
    threading.Thread(target=worker_thread, daemon=True).start()
    streamer_thread()
