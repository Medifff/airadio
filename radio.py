import os
import time
import random
import subprocess
import threading
import queue
import asyncio
import gc

import torch
import soundfile as sf
from einops import rearrange

import edge_tts
from twitchio.ext import commands
from huggingface_hub import login

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

# =========================
# CONFIG
# =========================

os.environ["HF_HOME"] = "/workspace/hf_cache"

STREAM_KEY = os.environ.get("TWITCH_STREAM_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
TWITCH_TOKEN = os.environ.get("TWITCH_TOKEN")
CHANNEL = os.environ.get("TWITCH_CHANNEL", "mediff23")

RTMP_URL = f"rtmp://live.twitch.tv/app/{STREAM_KEY}"
WORKDIR = "/workspace/airadio/data"
os.makedirs(WORKDIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRACKS_BEFORE_DJ = 3
audio_queue = queue.Queue(maxsize=2)
prompt_queue = queue.Queue(maxsize=1)

# =========================
# VIBES
# =========================

QUALITY = ", high quality studio recording, clean mix"

VIBES = [
    "post-punk, dark wave, melancholic",
    "drum and bass, liquid dnb, deep bass",
    "electronic rock, industrial",
    "happy hardcore, 170 bpm rave",
]

# =========================
# TWITCH BOT
# =========================

class Bot(commands.Bot):
    def __init__(self):
        super().__init__(
            token=TWITCH_TOKEN,
            prefix="!",
            initial_channels=[CHANNEL],
        )

    async def event_ready(self):
        print(f"🎮 Twitch bot ready: {self.nick}", flush=True)

    @commands.command(name="vibe")
    async def vibe(self, ctx):
        text = ctx.message.content.replace("!vibe", "").strip()[:80]
        if len(text) < 3:
            return
        if prompt_queue.empty():
            prompt_queue.put(text)
            await ctx.send(f"🎧 Accepted: {text}")

def run_twitch_bot():
    if TWITCH_TOKEN:
        asyncio.run(Bot().start())

# =========================
# MODEL
# =========================

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

if HF_TOKEN:
    login(token=HF_TOKEN)

print("⏳ Loading Stable Audio…", flush=True)
audio_model, cfg = get_pretrained_model("stabilityai/stable-audio-open-1.0")
audio_model = audio_model.to("cpu").eval()

SAMPLE_RATE = cfg["sample_rate"]
SAMPLE_SIZE = cfg["sample_size"]

# =========================
# AUDIO GENERATION
# =========================

def generate_music(prompt, path, seconds=40):
    cleanup()
    try:
        audio_model.to(DEVICE)
        with torch.no_grad():
            audio = generate_diffusion_cond(
                audio_model,
                steps=80,
                cfg_scale=5.5,
                conditioning=[{
                    "prompt": prompt,
                    "seconds_start": 0,
                    "seconds_total": seconds
                }],
                sample_size=SAMPLE_SIZE,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=DEVICE
            )
        audio_model.to("cpu")
        cleanup()

        audio = rearrange(audio, "b d n -> d (b n)")
        audio = audio / audio.abs().max()
        sf.write(path, audio.cpu().numpy().T, SAMPLE_RATE)
        return True
    except Exception as e:
        print("❌ Music error:", e, flush=True)
        audio_model.to("cpu")
        cleanup()
        return False

# =========================
# WORKER
# =========================

def worker():
    idx = 0
    count = 0

    while True:
        if audio_queue.full():
            time.sleep(1)
            continue

        vibe = random.choice(VIBES)
        user_prompt = None

        if not prompt_queue.empty():
            user_prompt = prompt_queue.get()
            vibe = user_prompt

        with_dj = count >= TRACKS_BEFORE_DJ

        music_path = f"{WORKDIR}/music_{idx}.wav"
        voice_path = f"{WORKDIR}/voice_{idx}.wav"

        if not generate_music(vibe + QUALITY, music_path):
            continue

        if with_dj:
            text = f"Next track. {vibe}"
            asyncio.run(
                edge_tts.Communicate(
                    text, "en-US-ChristopherNeural"
                ).save(voice_path)
            )
        else:
            voice_path = None

        audio_queue.put((music_path, voice_path))
        idx += 1
        count = 0 if with_dj else count + 1

# =========================
# STREAMER (ONE FFMPEG)
# =========================

def streamer():
    print("📡 Starting stream…", flush=True)

    cmd = [
        "ffmpeg",
        "-loglevel", "warning",
        "-fflags", "+genpts",
        "-f", "lavfi",
        "-i", "color=c=black:s=512x512:r=30",
        "-f", "wav",
        "-i", "pipe:0",

        "-shortest",
        "-vsync", "cfr",
        "-r", "30",

        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-g", "60",
        "-keyint_min", "60",
        "-sc_threshold", "0",
        "-pix_fmt", "yuv420p",

        "-c:a", "aac",
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "160k",
        "-af", "aresample=async=1:first_pts=0",

        "-f", "flv",
        RTMP_URL
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    while True:
        music_path, voice_path = audio_queue.get()

        with open(music_path, "rb") as f:
            proc.stdin.write(f.read())

        proc.stdin.flush()
        os.remove(music_path)

        if voice_path and os.path.exists(voice_path):
            os.remove(voice_path)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    threading.Thread(target=run_twitch_bot, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()
    streamer()
