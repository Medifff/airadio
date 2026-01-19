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
from einops import rearrange
from diffusers import StableDiffusionPipeline
import edge_tts
from crewai import Agent, Task, Crew
from huggingface_hub import login
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from twitchio.ext import commands

# =========================
# 0. ENV FIXES (CRITICAL)
# =========================
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"

# =========================
# 1. CONFIG
# =========================
STREAM_KEY = os.environ.get("TWITCH_STREAM_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
TWITCH_TOKEN = os.environ.get("TWITCH_TOKEN")
CHANNEL_NAME = os.environ.get("TWITCH_CHANNEL") or "mediff23"

RTMP_URL = f"rtmp://live.twitch.tv/app/{STREAM_KEY}"
WORKDIR = "/workspace/airadio/data"
os.makedirs(WORKDIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Device: {DEVICE}", flush=True)

video_queue = queue.Queue(maxsize=4)
user_prompt_queue = queue.Queue(maxsize=1)
GENRE_IMAGES = {}
POOL_LOCK = threading.Lock()

# =========================
# 2. VIBES
# =========================
quality_suffix = ", high quality studio recording, clear stereo image, professional mix"
VIBES_LIST = [
    (f"punk rock, fast tempo, distorted electric guitars{quality_suffix}", "punk rock poster graffiti red black"),
    (f"post-punk, dark wave, melancholic bassline{quality_suffix}", "post punk brutalist monochrome"),
    (f"happy hardcore, 170bpm rave energy{quality_suffix}", "neon rave lasers"),
    (f"industrial rock, cyberpunk metal{quality_suffix}", "cyberpunk guitarist neon"),
    (f"liquid drum and bass, deep sub bass{quality_suffix}", "futuristic tunnel speed")
]

# =========================
# 3. INIT IMAGES (ONCE)
# =========================
def init_images():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to(DEVICE)
    pipe.safety_checker = None

    for i, (_, prompt) in enumerate(VIBES_LIST):
        path = f"{WORKDIR}/cover_{i}.png"
        if not os.path.exists(path):
            pipe(prompt + ", masterpiece", num_inference_steps=20).images[0].save(path)
        GENRE_IMAGES[i] = path

    req = f"{WORKDIR}/cover_request.png"
    if not os.path.exists(req):
        pipe("abstract ai radio cyberpunk", num_inference_steps=20).images[0].save(req)
    GENRE_IMAGES["request"] = req

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

init_images()

# =========================
# 4. AUDIO MODEL
# =========================
audio_model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
audio_model = audio_model.to("cpu").eval()

# =========================
# 5. DJ AI
# =========================
class CrewAIDJ:
    def __init__(self):
        self.fallback = [
            "Signal stable.",
            "Neural channel active.",
            "Music is data.",
            "Type !vibe in chat."
        ]
        self.agent = Agent(
            role="Cyberpunk Radio Host",
            goal="Short radio phrases",
            backstory="AI DJ Nexus",
            verbose=False
        ) if OPENAI_API_KEY else None

    def script(self, mood):
        if not self.agent:
            return random.choice(self.fallback)
        task = Task(
            description=f"Radio phrase. Mood: {mood}. Max 1 sentence.",
            agent=self.agent,
            expected_output="Text"
        )
        try:
            return str(Crew([self.agent], [task]).kickoff())
        except:
            return random.choice(self.fallback)

dj = CrewAIDJ()

# =========================
# 6. MUSIC GEN
# =========================
def gen_music(prompt, path, dur=80):
    audio_model.to(DEVICE)
    with torch.no_grad():
        out = generate_diffusion_cond(
            audio_model,
            steps=60,
            cfg_scale=5.5,
            conditioning=[{"prompt": prompt, "seconds_total": dur}],
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=200,
            device=DEVICE
        )
    audio_model.to("cpu")
    out = rearrange(out, "b d n -> d (b n)")
    out = out / torch
