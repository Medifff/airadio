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

if not STREAM_KEY: print("⚠️ WARNING: TWITCH_STREAM_KEY not found.")
if not HF_TOKEN: print("❌ CRITICAL: HF_TOKEN not found!")
else: login(token=HF_TOKEN)

RTMP_URL = f"rtmp://live.twitch.tv/app/{STREAM_KEY}"
WORKDIR = "/workspace/airadio/data"
os.makedirs(WORKDIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Device: {DEVICE}")

# Increase queue size slightly to buffer against generation delays
video_queue = queue.Queue(maxsize=4)
user_prompt_queue = queue.Queue(maxsize=1)
TRACKS_BEFORE_DJ = 3 

# =========================
# 2. TWITCH BOT
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
            await ctx.send(f"@{ctx.author.name}, please specify a genre. Example: !vibe cyberpunk techno")
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
    bot = Bot()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bot.run()

# =========================
# 3. LOAD MODELS (Lazy Loading / Cleanup)
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
# 4. CREW AI (DJ Logic)
# =========================
TECH_KEYWORDS = ["AI", "ML", "OpenAI", "LLM", "NVIDIA", "Robotics", "SpaceX", "Python", "Cyberpunk"]

def fetch_tech_news():
    try:
        r = requests.get(
            "https://hn.algolia.com/api/v1/search_by_date",
            params={"query": "AI", "tags": "story", "hitsPerPage": 15},
            timeout=5
        )
        hits = r.json().get("hits", [])
        news = [h["title"] for h in hits if any(k.lower() in h.get("title", "").lower() for k in TECH_KEYWORDS)]
        return news[:3] if news else ["Neural networks are evolving rapidly."]
    except Exception:
        return ["Systems operating normally."]

class CrewAIDJ:
    def __init__(self):
        if not OPENAI_API_KEY:
            self.agent = None
            return

        self.agent = Agent(
            role="Cyberpunk Radio Host",
            goal="Deliver short updates.",
            backstory="You are 'Nexus', AI host. You play Punk and Electronic Rock.",
            verbose=False, allow_delegation=False
        )

    def generate_script(self, mood="high energy", user_request=None):
        if not self.agent: return "System nominal."

        news_items = fetch_tech_news()
        news_str = "\n- ".join(news_items)
        
        special_instruction = ""
        if user_request:
            special_instruction = f"IMPORTANT: Shoutout to user '{user_request['user']}' who requested: '{user_request['prompt']}'."

        task = Task(
            description=f"""
            Live on air. Vibe: {mood}.
            {special_instruction}
            Headlines: {news_str}
            Keep it under 3 sentences. Cool, robotic, friendly.
            """,
            agent=self.agent,
            expected_output="Short DJ script."
        )

        crew = Crew(agents=[self.agent], tasks=[task])
        try:
            return str(crew.kickoff())
        except Exception:
            return "Request acknowledged. Playing track."

ai_dj = CrewAIDJ()

# =========================
# 5. AUDIO GEN
# =========================
def get_vibes():
    if not user_prompt_queue.empty():
        order = user_prompt_queue.get()
        print(f"🌟 USER PROMPT: {order['prompt']}")
        clean_prompt = f"{order['prompt']}, high quality studio recording, professional mix"
        visual_prompt = f"{order['prompt']}, abstract digital art, 8k"
        return clean_prompt, visual_prompt, order

    quality_suffix = ", high quality studio recording, clear stereo image, professional mix, no fade out"
    genres = [
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
    choice = random.choice(genres)
    return choice[0], choice[1], None

def gen_music_stable_audio(prompt, out_wav, duration_sec=45):
    print(f"🎧 StableAudio: {prompt[:30]}...")
    cleanup() # Clean VRAM before generation
    
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
# 6. WORKER (SINGLE PASS + LOOP)
# =========================
def generate_segment(idx, is_dj_turn):
    print(f"\n🔨 [Worker] Seg {idx} | DJ: {is_dj_turn}")
    t0 = time.time()
    
    music_prompt, visual_prompt, user_order = get_vibes()
    if user_order: is_dj_turn = True

    music_path = os.path.join(WORKDIR, f"temp_music_{idx}.wav")
    voice_path = os.path.join(WORKDIR, f"temp_voice_{idx}.wav") if is_dj_turn else None
    cover_path = os.path.join(WORKDIR, f"temp_cover_{idx}.png")
    final_video = os.path.join(WORKDIR, f"segment_{idx}.ts")

    # A. Generate Music (SINGLE PASS - 45s)
    success = gen_music_stable_audio(music_prompt, music_path, 45)
    if not success:
        print("⚠️ Audio gen failed, skipping segment")
        return None

    # B. Generate Cover
    cleanup() # Clean VRAM before SD
    with torch.no_grad():
        image = sd_pipe(f"{visual_prompt}, masterpiece, 8k", num_inference_steps=20).images[0]
    image.save(cover_path)

    # C. DJ Script
    if is_dj_turn:
        mood = music_prompt.split(",")[0]
        dj_text = ai_dj.generate_script(mood=mood, user_request=user_order)
        print(f"🗣️ DJ: {dj_text}")
        asyncio.run(edge_tts.Communicate(dj_text, "en-US-ChristopherNeural").save(voice_path))

    # D. FFmpeg - LOOP & CROSSFADE
    # Instead of generating 2 files, we loop the single file with a crossfade
    # This creates a ~85s track from a 45s source seamlessly
    total_dur = 85 

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    
    # Inputs
    cmd += ["-loop", "1", "-i", cover_path]      # [0] Image
    if is_dj_turn: cmd += ["-i", voice_path]     # [1] Voice
    
    # We input the SAME music file twice to crossfade it with itself
    cmd += ["-i", music_path]                    # [Music A]
    cmd += ["-i", music_path]                    # [Music B]

    filter_complex = []
    
    idx_m1 = "2" if is_dj_turn else "1"
    idx_m2 = "3" if is_dj_turn else "2"
    
    # Self-Crossfade logic
    filter_complex.append(f"[{idx_m1}:a][{idx_m2}:a]acrossfade=d=3:c1=tri:c2=tri[music_raw]")
    
    if is_dj_turn:
        # Voice processing & Sidechain
        filter_complex.append(f"[1:a]highpass=f=100,lowpass=f=7000,volume=1.8,acompressor=threshold=-16dB:ratio=6:attack=5:release=80[voice_proc]")
        filter_complex.append(f"[music_raw][voice_proc]sidechaincompress=threshold=0.05:ratio=10:attack=5:release=300[music_ducked]")
        filter_complex.append(f"[music_ducked][voice_proc]amix=inputs=2:duration=first[pre_master]")
    else:
        filter_complex.append(f"[music_raw]anull[pre_master]")

    # Visualizer & Mastering
    filter_complex.append(f"[pre_master]loudnorm=I=-14:TP=-1.0:LRA=11[out_a]")
    filter_complex.append(f"[out_a]asplit[a_final][a_vis]")
    # Reduced visualizer size for performance
    filter_complex.append(f"[a_vis]showwaves=s=1280x150:mode=line:colors=0x00FFFF@0.5[waves]")
    filter_complex.append(f"[0:v][waves]overlay=x=0:y=H-h[out_v]")

    cmd += ["-filter_complex", ";".join(filter_complex)]
    cmd += [
        "-map", "[out_v]", "-map", "[a_final]",
        "-t", str(total_dur),
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-g", "60", # UltraFast for CPU saving
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-f", "mpegts", final_video
    ]
    
    subprocess.run(cmd, check=True)
    
    files_to_remove = [music_path, cover_path]
    if is_dj_turn: files_to_remove.append(voice_path)
    for f in files_to_remove:
        if f and os.path.exists(f): os.remove(f)
    
    cleanup()
    print(f"✅ [Worker] Seg {idx} ready")
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
            
            if seg_path:
                video_queue.put(seg_path)
                idx += 1
                if is_dj_turn: tracks_since_dj = 0
                else: tracks_since_dj += 1
                
        except Exception as e:
            print(f"❌ Worker Error: {e}")
            time.sleep(5)

# =========================
# 7. STREAMER
# =========================
def streamer_thread():
    print("📡 Streamer started...")
    while video_queue.qsize() < 1: time.sleep(5)
    print("🔴 GOING LIVE!")
    
    # Increased buffer size for stability
    stream_cmd = ["ffmpeg", "-re", "-f", "mpegts", "-i", "pipe:0", "-c", "copy", "-f", "flv", RTMP_URL]
    
    process = subprocess.Popen(stream_cmd, stdin=subprocess.PIPE)
    while True:
        seg_path = video_queue.get()
        print(f"▶️ Playing: {seg_path}")
        try:
            with open(seg_path, "rb") as f:
                while True:
                    chunk = f.read(4096 * 10)
                    if not chunk: break
                    process.stdin.write(chunk)
            process.stdin.flush()
        except Exception:
            print("⚠️ Stream dropped, restarting ffmpeg...")
            process = subprocess.Popen(stream_cmd, stdin=subprocess.PIPE)
        
        if os.path.exists(seg_path): os.remove(seg_path)

if __name__ == "__main__":
    t_bot = threading.Thread(target=run_twitch_bot, daemon=True)
    t_bot.start()
    t_worker = threading.Thread(target=worker_thread, daemon=True)
    t_worker.start()
    streamer_thread()
