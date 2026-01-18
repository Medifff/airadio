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
# Используем логин из токена или переменную окружения
CHANNEL_NAME = os.environ.get("TWITCH_CHANNEL") or "mediff23"

if not STREAM_KEY: print("⚠️ WARNING: TWITCH_STREAM_KEY not found.")
if not HF_TOKEN: print("❌ CRITICAL: HF_TOKEN not found!")
else: 
    try:
        login(token=HF_TOKEN)
    except Exception as e:
        print(f"⚠️ HF Login failed: {e}")

RTMP_URL = f"rtmp://live.twitch.tv/app/{STREAM_KEY}"
WORKDIR = "/workspace/airadio/data"
os.makedirs(WORKDIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Device: {DEVICE}")

video_queue = queue.Queue(maxsize=4)
user_prompt_queue = queue.Queue(maxsize=1)
# ПУЛ ТРЕКОВ (Жанр -> Путь к файлу)
GENRE_POOL = {} 
POOL_LOCK = threading.Lock()

TRACKS_BEFORE_DJ = 3 

# =========================
# 2. VIBES & GENRES
# =========================
quality_suffix = ", high quality studio recording, clear stereo image, professional mix, no fade out"

VIBES_LIST = [
    (f"punk rock, fast tempo, distorted electric guitars, live drum kit{quality_suffix}", 
     "punk rock poster, anarchy symbol, graffiti, red and black"),
    (f"post-punk, dark wave, chorus guitar, driving bassline, melancholic{quality_suffix}", 
     "post-punk album cover, monochrome, brutalist architecture"),
    (f"happy hardcore, 170bpm, energetic piano, heavy kick drum, rave{quality_suffix}", 
     "colorful rave party, lasers, neon rainbows"),
    (f"electronic rock, industrial metal, distorted synths, powerful drums{quality_suffix}", 
     "cyberpunk rocker, neon guitar, futuristic city"),
    (f"drum and bass, liquid dnb, fast breakbeats, deep sub bass{quality_suffix}", 
     "futuristic tunnel, speed lines, neon blue and orange")
]

# =========================
# 3. TWITCH BOT
# =========================
class Bot(commands.Bot):
    def __init__(self):
        super().__init__(token=TWITCH_TOKEN, prefix='!', initial_channels=[CHANNEL_NAME])

    async def event_ready(self):
        print(f'🎮 Twitch Bot logged in as | {self.nick}')

    @commands.command(name='vibe', aliases=['вайб'])
    async def vibe_command(self, ctx: commands.Context):
        content = ctx.message.content
        prompt = content.replace("!vibe", "").replace("!вайб", "").strip()
        if len(prompt) < 3:
            await ctx.send(f"@{ctx.author.name}, please specify a genre.")
            return
        if len(prompt) > 100: prompt = prompt[:100]
        
        if user_prompt_queue.empty():
            order = {"user": ctx.author.name, "prompt": prompt}
            user_prompt_queue.put(order)
            print(f"👾 New Request: {prompt}")
            await ctx.send(f"@{ctx.author.name}, request accepted! Generating '{prompt}' next... 🎹")
        else:
            await ctx.send(f"@{ctx.author.name}, queue full! Wait for next track.")

def run_twitch_bot():
    if not TWITCH_TOKEN: return
    try:
        bot = Bot()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bot.run()
    except Exception as e:
        print(f"⚠️ Twitch Bot Error: {e}")

# =========================
# 4. LOAD MODELS
# =========================
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

print("⏳ Loading Stable Audio Open 1.0...")
cleanup()
audio_model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
audio_model = audio_model.to(DEVICE).eval()

print("⏳ Loading Stable Diffusion...")
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True
).to(DEVICE)
sd_pipe.safety_checker = None

# =========================
# 5. CREW AI
# =========================
TECH_KEYWORDS = ["AI", "ML", "OpenAI", "LLM", "NVIDIA", "Robotics", "SpaceX", "Python", "Cyberpunk"]

def fetch_tech_news():
    try:
        r = requests.get("https://hn.algolia.com/api/v1/search_by_date", params={"query": "AI", "tags": "story", "hitsPerPage": 15}, timeout=5)
        hits = r.json().get("hits", [])
        news = [h["title"] for h in hits if any(k.lower() in h.get("title", "").lower() for k in TECH_KEYWORDS)]
        return news[:3] if news else ["Neural networks are evolving rapidly."]
    except Exception:
        return ["Systems operating normally."]

class CrewAIDJ:
    def __init__(self):
        self.fallback_scripts = [
            "Global network connected. I am Nexus. Let's override the silence.",
            "Data streams stable. The vibe is set to high voltage.",
            "Processing complete. Optimizing bitrate for the next track.",
            "They say silence is golden. I say silence is just empty data.",
            "Don't forget, you control the simulation. Type !vibe in the chat.",
            "My processors are running hot tonight. Must be the rhythm.",
            "Neural link established. Synchronization complete.",
            "Humanity is analog. Music is digital. We are the bridge.",
            "Switching context. Let the bass reconfigure your reality.",
            "I see the chat scrolling. You are the ghosts in the machine."
        ]
        
        self.agent = None
        if OPENAI_API_KEY:
            self.agent = Agent(
                role="Cyberpunk Radio Host", goal="Deliver short updates.",
                backstory="You are 'Nexus', AI host. You play Punk and Electronic Rock.",
                verbose=False, allow_delegation=False
            )

    def generate_script(self, mood="high energy", user_request=None):
        if not self.agent: return random.choice(self.fallback_scripts)
        news_items = fetch_tech_news()
        news_str = "\n- ".join(news_items)
        special_instruction = ""
        if user_request:
            special_instruction = f"IMPORTANT: Shoutout to user '{user_request['user']}' who requested: '{user_request['prompt']}'."

        task = Task(
            description=f"Live on air. Vibe: {mood}. {special_instruction}. Headlines: {news_str}. Keep it under 3 sentences.",
            agent=self.agent, expected_output="Short DJ script."
        )
        crew = Crew(agents=[self.agent], tasks=[task])
        try: return str(crew.kickoff())
        except Exception: return random.choice(self.fallback_scripts)

ai_dj = CrewAIDJ()

# =========================
# 6. AUDIO GEN WRAPPER
# =========================
def get_vibe_data(forced_index=None):
    # 1. Заказ (приоритет)
    if not user_prompt_queue.empty() and forced_index is None:
        order = user_prompt_queue.get()
        print(f"🌟 USER PROMPT: {order['prompt']}")
        clean_prompt = f"{order['prompt']}, high quality studio recording, professional mix"
        visual_prompt = f"{order['prompt']}, abstract digital art, 8k"
        return clean_prompt, visual_prompt, order, -1

    # 2. Инициализация или рандом
    if forced_index is not None:
        idx = forced_index
    else:
        idx = random.randint(0, len(VIBES_LIST) - 1)
        
    choice = VIBES_LIST[idx]
    return choice[0], choice[1], None, idx

def gen_music_stable_audio(prompt, out_wav, duration_sec=45):
    print(f"🎧 StableAudio: {prompt[:30]}...")
    cleanup()
    conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": duration_sec}]
    try:
        with torch.no_grad():
            output = generate_diffusion_cond(
                audio_model, steps=100, cfg_scale=5.5, conditioning=conditioning,
                sample_size=sample_size, sigma_min=0.3, sigma_max=500,
                sampler_type="dpmpp-3m-sde", device=DEVICE
            )
        output = rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
        sf.write(out_wav, output.cpu().numpy().T, sample_rate, subtype='PCM_16')
        return True
    except Exception as e:
        print(f"❌ Audio Gen Error: {e}")
        return False

# =========================
# 7. WORKER THREAD
# =========================
def generate_segment(segment_id, is_dj_turn, forced_genre_idx=None):
    music_prompt, visual_prompt, user_order, genre_idx = get_vibe_data(forced_genre_idx)
    if user_order: is_dj_turn = True
    
    music_path = os.path.join(WORKDIR, f"temp_music_{segment_id}.wav")
    voice_path = os.path.join(WORKDIR, f"temp_voice_{segment_id}.wav") if is_dj_turn else None
    cover_path = os.path.join(WORKDIR, f"temp_cover_{segment_id}.png")
    final_video = os.path.join(WORKDIR, f"segment_{segment_id}.ts")

    # A. Generate Music
    success = gen_music_stable_audio(music_prompt, music_path, 45)
    if not success: return None

    # B. Generate Cover
    cleanup()
    with torch.no_grad():
        image = sd_pipe(f"{visual_prompt}, masterpiece, 8k", num_inference_steps=20).images[0]
    image.save(cover_path)

    # C. DJ Script
    if is_dj_turn:
        mood = music_prompt.split(",")[0]
        dj_text = ai_dj.generate_script(mood=mood, user_request=user_order)
        print(f"🗣️ DJ: {dj_text}")
        asyncio.run(edge_tts.Communicate(dj_text, "en-US-ChristopherNeural").save(voice_path))
        
        # Проверка файла голоса
        if not os.path.exists(voice_path) or os.path.getsize(voice_path) < 1000:
            print("⚠️ Voice file corrupted or empty! Skipping DJ turn.")
            is_dj_turn = False

    # D. FFmpeg Assembly
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", cover_path]
    if is_dj_turn: cmd += ["-i", voice_path]
    cmd += ["-i", music_path, "-i", music_path] 

    filter_complex = []
    idx_m1 = "2" if is_dj_turn else "1"
    idx_m2 = "3" if is_dj_turn else "2"
    
    # 1. Self-Crossfade
    filter_complex.append(f"[{idx_m1}:a][{idx_m2}:a]acrossfade=d=3:c1=tri:c2=tri[music_raw]")
    
    if is_dj_turn:
        # === FIX: RESAMPLING & SPLIT ===
        # 1. Ресемплинг + Обработка -> [voice_proc_raw]
        filter_complex.append(f"[1:a]aresample=44100,highpass=f=100,lowpass=f=7000,volume=1.8,acompressor=threshold=-16dB:ratio=6:attack=5:release=80[voice_proc_raw]")
        
        # 2. Раздвоение потока: [voice_sc] (на сайдчейн) и [voice_mix] (в эфир)
        filter_complex.append(f"[voice_proc_raw]asplit[voice_sc][voice_mix]")
        
        # 3. Сайдчейн музыки
        filter_complex.append(f"[music_raw][voice_sc]sidechaincompress=threshold=0.05:ratio=10:attack=5:release=300[music_ducked]")
        
        # 4. Микс
        filter_complex.append(f"[music_ducked][voice_mix]amix=inputs=2:duration=first[pre_master]")
    else:
        filter_complex.append(f"[music_raw]anull[pre_master]")

    # Mastering & Visualizer
    filter_complex.append(f"[pre_master]loudnorm=I=-14:TP=-1.0:LRA=11[out_a]")
    filter_complex.append(f"[out_a]asplit[a_final][a_vis]")
    filter_complex.append(f"[a_vis]showwaves=s=1280x150:mode=line:colors=0x00FFFF@0.5[waves]")
    filter_complex.append(f"[0:v][waves]overlay=x=0:y=H-h[out_v]")

    cmd += ["-filter_complex", ";".join(filter_complex)]
    cmd += ["-map", "[out_v]", "-map", "[a_final]", "-t", "85",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-g", "60",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-f", "mpegts", final_video]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg CRASHED: {e}")
        if is_dj_turn:
            print("🔄 Retrying without DJ logic...")
            return generate_segment(segment_id, False, forced_genre_idx)
        return None
    
    # Cleanup
    files_to_remove = [music_path, cover_path]
    if is_dj_turn and voice_path: files_to_remove.append(voice_path)
    for f in files_to_remove:
        if f and os.path.exists(f): os.remove(f)
    cleanup()
    
    # Pool Update
    if genre_idx != -1:
        with POOL_LOCK:
            old_file = GENRE_POOL.get(genre_idx)
            GENRE_POOL[genre_idx] = final_video
            print(f"🏊 Pool updated: Genre {genre_idx} refreshed.")
            if old_file and old_file != final_video and os.path.exists(old_file):
                try: os.remove(old_file)
                except: pass

    return final_video

def worker_thread():
    print("\n🚀 STARTUP: Generating initial pool (1 per genre)...")
    for i in range(len(VIBES_LIST)):
        print(f"\n🌊 [Init] Generating Genre {i}/{len(VIBES_LIST)-1}...")
        seg_path = generate_segment(segment_id=f"init_{i}", is_dj_turn=False, forced_genre_idx=i)
        if seg_path: video_queue.put(seg_path)
    
    print("\n✅ INIT COMPLETE. Rotation mode.\n")

    idx_counter = 0
    tracks_since_dj = TRACKS_BEFORE_DJ # Сразу даем слово DJ после инициализации
    
    while True:
        if video_queue.full():
            time.sleep(2)
            continue
        try:
            is_dj_turn = (tracks_since_dj >= TRACKS_BEFORE_DJ)
            seg_path = generate_segment(segment_id=f"stream_{idx_counter}", is_dj_turn=is_dj_turn)
            if seg_path:
                video_queue.put(seg_path)
                idx_counter += 1
                if is_dj_turn: tracks_since_dj = 0
                else: tracks_since_dj += 1
        except Exception as e:
            print(f"❌ Worker Error: {e}")
            time.sleep(5)

# =========================
# 8. STREAMER
# =========================
def streamer_thread():
    print("📡 Streamer started...")
    while video_queue.empty() and len(GENRE_POOL) == 0:
        print("Waiting for initial generation...")
        time.sleep(5)
    print("🔴 GOING LIVE!")
    
    stream_cmd = ["ffmpeg", "-re", "-f", "mpegts", "-i", "pipe:0", "-c", "copy", "-f", "flv", RTMP_URL]
    process = subprocess.Popen(stream_cmd, stdin=subprocess.PIPE)
    
    while True:
        seg_path = None
        # 1. Queue (Fresh)
        if not video_queue.empty():
            seg_path = video_queue.get()
            print(f"▶️ Playing FRESH: {seg_path}")
        # 2. Pool (Fallback)
        else:
            with POOL_LOCK:
                if len(GENRE_POOL) > 0:
                    genre_id = random.choice(list(GENRE_POOL.keys()))
                    seg_path = GENRE_POOL[genre_id]
                    print(f"♻️ POOL FALLBACK: Genre {genre_id}")
                else:
                    time.sleep(2)
                    continue

        if seg_path and os.path.exists(seg_path):
            try:
                with open(seg_path, "rb") as f:
                    while True:
                        chunk = f.read(4096 * 10)
                        if not chunk: break
                        try: process.stdin.write(chunk)
                        except IOError:
                             process = subprocess.Popen(stream_cmd, stdin=subprocess.PIPE)
                             process.stdin.write(chunk)
                process.stdin.flush()
            except Exception as e:
                print(f"❌ Stream error: {e}")
                process = subprocess.Popen(stream_cmd, stdin=subprocess.PIPE)

if __name__ == "__main__":
    t_bot = threading.Thread(target=run_twitch_bot, daemon=True)
    t_bot.start()
    t_worker = threading.Thread(target=worker_thread, daemon=True)
    t_worker.start()
    streamer_thread()
