# radio.py — Production (FFmpeg 4.4 safe, no EOF, no DTS hell)
# Key ideas:
# 1) Streamer NEVER sends EOF to ffmpeg: if queue is empty -> push a short "silence.ts"
# 2) No unsupported -ignore_eof (FFmpeg 4.4). Use +genpts+igndts and aresample async.
# 3) Segments are generated as clean MPEGTS with stable CFR 30fps + repeated headers.
# 4) Minimal GPU usage: ONLY Stable Audio on GPU during generation. Video is black frame.
# 5) DJ is optional and sanitized to 44.1kHz stereo.

import os
import time
import random
import subprocess
import gc
import threading
import queue
import asyncio

import soundfile as sf
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

if not STREAM_KEY:
    print("⚠️ WARNING: TWITCH_STREAM_KEY not found.", flush=True)

RTMP_URL = f"rtmp://live.twitch.tv/app/{STREAM_KEY}"
WORKDIR = "/workspace/airadio/data"
os.makedirs(WORKDIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Device: {DEVICE}", flush=True)

# Buffers
video_queue: "queue.Queue[str]" = queue.Queue(maxsize=3)
user_prompt_queue: "queue.Queue[str]" = queue.Queue(maxsize=1)

TRACKS_BEFORE_DJ = 3

# Vibes
QUALITY = ", high quality studio recording, clear stereo image, professional mix"
VIBES = [
    "post-punk, dark wave, chorus guitar, melancholic",
    "drum and bass, liquid dnb, deep sub bass",
    "electronic rock, industrial metal",
    "happy hardcore, 170 bpm, rave",
]

# Filler segment (never let ffmpeg see EOF)
SILENCE_TS = os.path.join(WORKDIR, "silence.ts")

# =========================
# UTILS
# =========================
def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_cmd(cmd: list[str], *, check: bool = True):
    # Helper to run and raise readable errors
    try:
        subprocess.run(cmd, check=check)
    except subprocess.CalledProcessError as e:
        print("❌ Command failed:", " ".join(cmd), flush=True)
        raise e

def ensure_silence_ts():
    """Create a small TS chunk that is valid H264/AAC and can be looped as filler."""
    if os.path.exists(SILENCE_TS) and os.path.getsize(SILENCE_TS) > 50_000:
        return

    print("🧱 Building silence filler TS…", flush=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=512x512:r=30",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", "5",
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
        "-b:a", "128k",
        "-f", "mpegts",
        "-mpegts_flags", "+resend_headers",
        SILENCE_TS
    ]
    run_cmd(cmd)
    print("🧱 Silence TS ready.", flush=True)

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
        print("⚠️ TWITCH_TOKEN not found. Chat control disabled.", flush=True)
        return
    # twitchio manages its own loop; keep it isolated
    try:
        bot = Bot()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bot.run()
    except Exception as e:
        print(f"⚠️ Twitch bot crashed: {e}", flush=True)

# =========================
# MODELS
# =========================
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
    except Exception:
        pass

print("⏳ Loading Stable Audio…", flush=True)
audio_model, cfg = get_pretrained_model("stabilityai/stable-audio-open-1.0")
audio_model = audio_model.to("cpu").eval()
SAMPLE_RATE = int(cfg["sample_rate"])
SAMPLE_SIZE = int(cfg["sample_size"])

# =========================
# AUDIO
# =========================
def generate_music(prompt: str, out_wav: str, seconds: int = 40) -> bool:
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
                    "seconds_total": seconds,
                }],
                sample_size=SAMPLE_SIZE,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=DEVICE,
            )

        audio_model.to("cpu")
        cleanup()

        audio = rearrange(audio, "b d n -> d (b n)")
        audio = audio / (audio.abs().max() + 1e-9)
        sf.write(out_wav, audio.cpu().numpy().T, SAMPLE_RATE, subtype="PCM_16")
        return True

    except Exception as e:
        print("❌ Music error:", e, flush=True)
        try:
            audio_model.to("cpu")
        except Exception:
            pass
        cleanup()
        return False

def sanitize_voice(raw_path: str, clean_path: str) -> bool:
    # Make TTS always match stream audio format (44100 stereo)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", raw_path,
        "-ar", "44100", "-ac", "2",
        "-af", "highpass=f=100,lowpass=f=7000,volume=1.6,acompressor=threshold=-18dB:ratio=5:attack=5:release=80",
        clean_path,
    ]
    try:
        run_cmd(cmd)
        return True
    except Exception:
        return False

# =========================
# SEGMENT
# =========================
def generate_segment(idx: int, with_dj: bool) -> str | None:
    # pick vibe (user override)
    vibe = random.choice(VIBES)
    if not user_prompt_queue.empty():
        vibe = user_prompt_queue.get()

    music_wav = os.path.join(WORKDIR, f"music_{idx}.wav")
    voice_raw = os.path.join(WORKDIR, f"voice_{idx}_raw.wav")
    voice_wav = os.path.join(WORKDIR, f"voice_{idx}.wav")
    out_ts = os.path.join(WORKDIR, f"seg_{idx}.ts")

    ok = generate_music(vibe + QUALITY, music_wav, seconds=40)
    if not ok:
        return None

    # Build a stable TS segment: black video + audio, constant CFR
    inputs = [
        "-f", "lavfi",
        "-i", "color=c=black:s=512x512:r=30",
        "-i", music_wav,
    ]

    fc: list[str] = []

    if with_dj:
        try:
            text = f"Next track. {vibe}"
            asyncio.run(edge_tts.Communicate(text, "en-US-ChristopherNeural").save(voice_raw))
            if os.path.exists(voice_raw) and os.path.getsize(voice_raw) > 1000 and sanitize_voice(voice_raw, voice_wav):
                inputs += ["-i", voice_wav]
                # music is [1:a], voice is [2:a]
                fc.append("[1:a][2:a]amix=inputs=2:duration=first[a]")
            else:
                fc.append("[1:a]anull[a]")
        except Exception:
            fc.append("[1:a]anull[a]")
    else:
        fc.append("[1:a]anull[a]")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        *inputs,
        "-filter_complex", ";".join(fc),
        "-map", "0:v",
        "-map", "[a]",
        "-t", "45",              # hard cap for consistent segments
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
        out_ts,
    ]

    try:
        run_cmd(cmd)
    except Exception as e:
        print(f"❌ Segment ffmpeg failed: {e}", flush=True)
        # cleanup temp files
        for p in (music_wav, voice_raw, voice_wav):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        return None

    # cleanup temp files
    for p in (music_wav, voice_raw, voice_wav):
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    return out_ts

# =========================
# WORKER
# =========================
def worker():
    idx = 0
    count = 0
    while True:
        if video_queue.full():
            time.sleep(0.5)
            continue

        with_dj = count >= TRACKS_BEFORE_DJ
        seg = generate_segment(idx, with_dj)

        if seg:
            video_queue.put(seg)
            idx += 1
            count = 0 if with_dj else (count + 1)
        else:
            time.sleep(1)

# =========================
# STREAMER
# =========================
def streamer():
    ensure_silence_ts()

    # Wait until we have at least one real segment
    t0 = time.time()
    while video_queue.empty() and (time.time() - t0) < 30:
        time.sleep(1)

    # Re-encode to Twitch (stable timestamps + CBR-ish)
    cmd = [
        "ffmpeg",
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
        RTMP_URL,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def restart_ffmpeg():
        nonlocal proc
        try:
            proc.kill()
        except Exception:
            pass
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    while True:
        try:
            # If we have no new segment, feed filler so ffmpeg never sees EOF
            if video_queue.empty():
                path = SILENCE_TS
            else:
                path = video_queue.get()

            with open(path, "rb") as f:
                data = f.read()

            try:
                proc.stdin.write(data)
                proc.stdin.flush()
            except BrokenPipeError:
                print("⚠️ ffmpeg pipe broken, restarting…", flush=True)
                restart_ffmpeg()
                proc.stdin.write(data)
                proc.stdin.flush()

            if path != SILENCE_TS:
                try:
                    os.remove(path)
                except Exception:
                    pass

        except Exception as e:
            print(f"⚠️ Streamer error: {e}", flush=True)
            restart_ffmpeg()
            time.sleep(1)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ensure_silence_ts()

    threading.Thread(target=run_twitch_bot, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()

    streamer()
