import os
import time
import random
import socket
import subprocess
import threading
import queue
import gc

import torch
import soundfile as sf
from einops import rearrange

import edge_tts
from huggingface_hub import login
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

# =========================
# CONFIG
# =========================

STREAM_KEY = os.environ["TWITCH_STREAM_KEY"]
TWITCH_OAUTH = os.environ["TWITCH_OAUTH"]      # oauth:xxxx
TWITCH_NICK = os.environ["TWITCH_NICK"]
CHANNEL = os.environ.get("TWITCH_CHANNEL", "mediff23")

HF_TOKEN = os.environ.get("HF_TOKEN")

RTMP_URL = f"rtmp://live.twitch.tv/app/{STREAM_KEY}"
WORKDIR = "/workspace/airadio/data"
os.makedirs(WORKDIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

audio_queue = queue.Queue(maxsize=2)
prompt_queue = queue.Queue(maxsize=1)

TRACKS_BEFORE_DJ = 3

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
# TWITCH IRC (NO CRASH)
# =========================

def twitch_irc():
    s = socket.socket()
    s.connect(("irc.chat.twitch.tv", 6667))
    s.send(f"PASS {TWITCH_OAUTH}\r\n".encode())
    s.send(f"NICK {TWITCH_NICK}\r\n".encode())
    s.send(f"JOIN #{CHANNEL}\r\n".encode())

    print("🎮 Twitch IRC connected", flush=True)

    while True:
        data = s.recv(2048).decode(errors="ignore")
        if data.startswith("PING"):
            s.send("PONG\r\n".encode())
            continue

        if "!vibe" in data:
            try:
                msg = data.split("!vibe", 1)[1].strip()
                if len(msg) > 3 and prompt_queue.empty():
                    prompt_queue.put(msg[:80])
                    print(f"🎧 Chat vibe: {msg}", flush=True)
            except:
                pass

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

        if not prompt_queue.empty():
            vibe = prompt_queue.get()

        with_dj = count >= TRACKS_BEFORE_DJ

        music = f"{WORKDIR}/music_{idx}.wav"
        voice = f"{WORKDIR}/voice_{idx}.wav"

        if not generate_music(vibe + QUALITY, music):
            continue

        if with_dj:
            text = f"Next track. {vibe}"
            edge_tts.Communicate(
                text, "en-US-ChristopherNeural"
            ).save_sync(voice)
        else:
            voice = None

        audio_queue.put((music, voice))
        idx += 1
        count = 0 if with_dj else count + 1

# =========================
# STREAMER (SINGLE FFMPEG)
# =========================

def streamer():
    print("📡 Streaming started", flush=True)

    cmd = [
        "ffmpeg",
        "-loglevel", "warning",
        "-fflags", "+genpts",
        "-f", "lavfi",
        "-i", "color=c=black:s=512x512:r=30",
        "-f", "wav",
        "-i", "pipe:0",
        "-shortest",

        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-g", "60",

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
        music, _ = audio_queue.get()
        with open(music, "rb") as f:
            proc.stdin.write(f.read())
        proc.stdin.flush()
        os.remove(music)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    threading.Thread(target=twitch_irc, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()
    streamer()
