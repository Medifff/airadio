import os
import time
import random
import subprocess
import gc
import soundfile as sf
import threading
import queue
import asyncio
import torch

from einops import rearrange
import edge_tts
from huggingface_hub import login
from twitchio.ext import commands

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

# =========================
# CONFIG
# =========================

os.environ["HF_HOME"] = "/workspace/hf_cache"

STREAM_KEY = os.environ.get("TWITCH_STREAM_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
TWITCH_TOKEN = os.environ.get("TWITCH_TOKEN")
CHANNEL_NAME = os.environ.get("TWITCH_CHANNEL") or "mediff23"

RTMP_URL = f"rtmp://live.twitch.tv/app/{STREAM_KEY}"
WORKDIR = "/workspace/airadio/data"
os.makedirs(WORKDIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

video_queue = queue.Queue(maxsize=3)
user_prompt_queue = queue.Queue(maxsize=1)

TRACKS_BEFORE_DJ = 3

# =========================
# VIBES
# =========================

QUALITY = ", high quality studio recording, clear stereo image, professional mix"

VIBES = [
    "post-punk, dark wave, chorus guitar, melancholic",
    "drum and bass, liquid dnb, deep sub bass",
    "electronic rock, industrial metal",
    "happy hardcore, 170 bpm, rave",
]

# =========================
# TWITCH BOT
# =========================

class Bot(commands.Bot):
    def __init__(self):
        super().__init__(
            token=TWITCH_TOKEN,
            prefix="!",
            initial_channels=[CHANNEL_NAME],
        )

    async def event_ready(self):
        print(f"🎮 Twitch bot ready: {self.nick}", flush=True)

    @commands.command(name="vibe")
    async def vibe(self, ctx):
        prompt = ctx.message.content.replace("!vibe", "").strip()[:80]
        if len(prompt) < 3:
            return
        if user_prompt_queue.empty():
            user_prompt_queue.put(prompt)
            await ctx.send(f"🎧 Next vibe accepted: {prompt}")

def run_twitch_bot():
    if not TWITCH_TOKEN:
        return
    asyncio.run(Bot().start())

# =========================
# MODELS
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
# AUDIO
# =========================

def generate_music(prompt, out_wav, seconds=40):
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
        sf.write(out_wav, audio.cpu().numpy().T, SAMPLE_RATE)
        return True
    except Exception as e:
        print("❌ Music error:", e, flush=True)
        audio_model.to("cpu")
        cleanup()
        return False

# =========================
# SEGMENT
# =========================

def generate_segment(idx, with_dj):
    vibe = random.choice(VIBES)
    if not user_prompt_queue.empty():
        vibe = user_prompt_queue.get()

    music = f"{WORKDIR}/music_{idx}.wav"
    voice = f"{WORKDIR}/voice_{idx}.wav"
    out_ts = f"{WORKDIR}/seg_{idx}.ts"

    if not generate_music(vibe + QUALITY, music):
        return None

    inputs = [
        "-f", "lavfi",
        "-i", "color=c=black:s=512x512:r=30",
        "-i", music,
    ]

    fc = []

    if with_dj:
        text = f"Next track. {vibe}"
        asyncio.run(edge_tts.Communicate(text, "en-US-ChristopherNeural").save(voice))
        inputs += ["-i", voice]
        fc.append("[1:a][2:a]amix=inputs=2:duration=first[a]")
    else:
        fc.append("[1:a]anull[a]")

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(fc),
        "-map", "0:v",
        "-map", "[a]",
        "-shortest",

        "-vsync", "cfr",
        "-r", "30",

        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-g", "60",
        "-keyint_min", "60",
        "-sc_threshold", "0",
        "-x264-params", "repeat-headers=1",

        "-c:a", "aac",
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "160k",

        "-f", "mpegts",
        "-mpegts_flags", "+resend_headers",
        out_ts
    ]

    subprocess.run(cmd, check=True)

    os.remove(music)
    if os.path.exists(voice):
        os.remove(voice)

    return out_ts

# =========================
# WORKER
# =========================

def worker():
    idx = 0
    count = 0
    while True:
        if video_queue.full():
            time.sleep(1)
            continue
        with_dj = count >= TRACKS_BEFORE_DJ
        seg = generate_segment(idx, with_dj)
        if seg:
            video_queue.put(seg)
            idx += 1
            count = 0 if with_dj else count + 1

# =========================
# STREAMER
# =========================

def streamer():
    while video_queue.empty():
        time.sleep(2)

    cmd = [
        "ffmpeg",
        "-ignore_eof", "1",
        "-fflags", "+genpts+igndts",
        "-f", "mpegts",
        "-i", "pipe:0",

        "-vsync", "cfr",
        "-r", "30",

        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-g", "60",
        "-keyint_min", "60",
        "-sc_threshold", "0",
        "-b:v", "3000k",
        "-maxrate", "3000k",
        "-bufsize", "6000k",

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
        seg = video_queue.get()
        with open(seg, "rb") as f:
            proc.stdin.write(f.read())
        proc.stdin.flush()
        os.remove(seg)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    threading.Thread(target=run_twitch_bot, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()
    streamer()
