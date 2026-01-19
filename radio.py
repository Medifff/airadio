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
import torchaudio
from einops import rearrange
from diffusers import StableDiffusionPipeline
import edge_tts
from crewai import Agent, Task, Crew
from huggingface_hub import login
import sys
# === OFFICIAL IMPORTS ===
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

# =========================
# 1. CONFIG & ENV
# =========================
os.environ["HF_HOME"] = "/workspace/hf_cache"

STREAM_KEY = os.environ.get("TWITCH_STREAM_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not STREAM_KEY:
    print("⚠️ WARNING: TWITCH_STREAM_KEY not found.")
if not HF_TOKEN:
    print("❌ CRITICAL: HF_TOKEN not found!")
else:
    login(token=HF_TOKEN)

RTMP_URL = f"rtmp://live.twitch.tv/app/{STREAM_KEY}"
WORKDIR = "/workspace/airadio/data"
os.makedirs(WORKDIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Device: {DEVICE}")

video_queue = queue.Queue(maxsize=3)
TRACKS_BEFORE_DJ = 3


# =========================
# 2. LOAD MODELS
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
# 3. CREW AI (DJ Logic)
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


# =========================
# 4. PROMPTS & AUDIO GEN (UPDATED)
# =========================
def get_vibes():
    # 📌 Suggestion 3: Studio Quality Prompts
    quality_suffix = ", high quality studio recording, clear stereo image, professional mix, no fade out, continuous groove"

    genres = [
        (f"punk rock, fast tempo, distorted electric guitars, live drum kit, real bass guitar, energetic performance{quality_suffix}",
         "punk rock poster, anarchy symbol, graffiti, red and black, grunge texture"),
        (f"post-punk, dark wave, chorus guitar, driving bassline, melancholic, 80s goth vibe, atmospheric{quality_suffix}",
         "post-punk album cover, monochrome, brutalist architecture, dark fog"),
        (f"happy hardcore, 170bpm, energetic piano, heavy kick drum, rave, dance, synthesizer, uplifting{quality_suffix}",
         "colorful rave party, lasers, neon rainbows, high energy"),
        (f"electronic rock, industrial metal, distorted synths, powerful drums, cyberpunk action, cinematic{quality_suffix}",
         "cyberpunk rocker, neon guitar, futuristic city, glitch art"),
        (f"drum and bass, liquid dnb, fast breakbeats, deep sub bass, atmospheric pads, soulful, melodic{quality_suffix}",
         "futuristic tunnel, speed lines, neon blue and orange, liquid fluid abstract")
    ]
    return random.choice(genres)


def gen_music_stable_audio(prompt, out_wav, duration_sec=45):
    print(f"🎧 StableAudio: {prompt[:30]}...")
    cleanup()

    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": duration_sec
    }]

    with torch.no_grad():
        output = generate_diffusion_cond(
            audio_model,
            steps=150,          # 📌 Suggestion 4: Quality Steps
            cfg_scale=5.5,      # 📌 Suggestion 4: Musicality Sweet Spot
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=DEVICE
        )

    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
    audio_np = output.cpu().numpy().T
    sf.write(out_wav, audio_np, sample_rate, subtype='PCM_16')
    return sample_rate


# =========================
# 5. WORKER (PRO AUDIO CHAIN)
# =========================
def generate_segment(idx, is_dj_turn):
    print(f"\n🔨 [Worker] Seg {idx} | DJ: {is_dj_turn}")
    t0 = time.time()

    music_prompt, visual_prompt = get_vibes()

    music_part1 = os.path.join(WORKDIR, f"temp_music_{idx}_1.wav")
    music_part2 = os.path.join(WORKDIR, f"temp_music_{idx}_2.wav")
    voice_path = os.path.join(WORKDIR, f"temp_voice_{idx}.wav") if is_dj_turn else None
    cover_path = os.path.join(WORKDIR, f"temp_cover_{idx}.png")
    final_video = os.path.join(WORKDIR, f"segment_{idx}.ts")

    # A. Generate Music (Два куска по 45с, чтобы обойти лимит 47с модели)
    # Это дает нам ~85 секунд уникального контента без лупов
    gen_music_stable_audio(music_prompt, music_part1, duration_sec=45)
    gen_music_stable_audio(music_prompt, music_part2, duration_sec=45)

    # B. Generate Cover
    with torch.no_grad():
        image = sd_pipe(f"{visual_prompt}, masterpiece, 8k", num_inference_steps=20).images[0]
    image.save(cover_path)

    # C. TTS
    if is_dj_turn:
        mood = music_prompt.split(",")[0]
        dj_text = ai_dj.generate_script(mood=mood)
        print(f"🗣️ DJ: {dj_text}")
        asyncio.run(edge_tts.Communicate(dj_text, "en-US-ChristopherNeural").save(voice_path))

    # D. FFmpeg Pro Mastering
    # Мы склеиваем два трека кроссфейдом, чтобы звучало как один длинный (87 сек)
    total_dur = 85

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    cmd += ["-loop", "1", "-i", cover_path]      # [0] Image

    if is_dj_turn:
        cmd += ["-i", voice_path]                # [1] Voice

    cmd += ["-i", music_part1]                   # [2] (or 1) Music A
    cmd += ["-i", music_part2]                   # [3] (or 2) Music B

    filter_complex = []

    # Индексы меняются в зависимости от того, есть ли голос
    idx_m1 = "2" if is_dj_turn else "1"
    idx_m2 = "3" if is_dj_turn else "2"

    # 1. Склейка музыки (Crossfade) - создает "Continuous groove"
    filter_complex.append(f"[{idx_m1}:a][{idx_m2}:a]acrossfade=d=3:c1=tri:c2=tri[music_raw]")

    if is_dj_turn:
        # 📌 Suggestion 2: Professional Voice Processing
        # Highpass 100Hz (убрать гул), Lowpass 7000Hz (убрать свист), Comp
        filter_complex.append(
            "[1:a]highpass=f=100,lowpass=f=7000,volume=1.8,"
            "acompressor=threshold=-16dB:ratio=6:attack=5:release=80[voice_proc]"
        )

        # Sidechain: Музыка пригибается под голос
        filter_complex.append(
            "[music_raw][voice_proc]sidechaincompress=threshold=0.05:ratio=10:attack=5:release=300[music_ducked]"
        )

        # Mix
        filter_complex.append("[music_ducked][voice_proc]amix=inputs=2:duration=first[pre_master]")
    else:
        filter_complex.append("[music_raw]anull[pre_master]")

    # 📌 Suggestion 5: Loudnorm (Mastering)
    # EBU R128 стандарт (-14 LUFS для стриминга)
    filter_complex.append(
        "[pre_master]aresample=44100:async=1:first_pts=0,"  # <--- ВОТ ЭТО
        "loudnorm=I=-14:TP=-1.0:LRA=11[out]"
    )
    
    cmd += ["-filter_complex", ";".join(filter_complex)]
    cmd += [
        "-map", "0:v", "-map", "[out]",
        "-t", str(total_dur),
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-g", "60",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-f", "mpegts", final_video
    ]

    subprocess.run(cmd, check=True)

    files_to_remove = [music_part1, music_part2, cover_path]
    if is_dj_turn:
        files_to_remove.append(voice_path)
    for f in files_to_remove:
        if f and os.path.exists(f):
            os.remove(f)

    cleanup()
    print(f"✅ [Worker] Seg {idx} ready ({round(time.time()-t0)}s)")
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
    while video_queue.qsize() < 1:
        time.sleep(5)
    print("🔴 GOING LIVE!")

    # 1. Мы убрали -c copy (это корень зла)
    # 2. Добавили перекодировку видео (libx264, очень быстро) и аудио
    # 3. Фильтр aresample выравнивает любые сбои аудио
    stream_cmd = [
        "ffmpeg",
        "-re",                          # Читать вход с нормальной скоростью
        "-fflags", "+genpts+discardcorrupt", # Игнорировать битые метки на входе
        "-i", "pipe:0",                 # Читаем из Python
        
        # --- ВИДЕО ---
        "-c:v", "libx264",              # Кодируем заново (создает новые PTS)
        "-preset", "ultrafast",         # Минимальная нагрузка на CPU
        "-tune", "zerolatency",         # Для стриминга
        "-r", "30",                     # Жестко задаем 30 FPS
        "-g", "60",                     # Keyframe каждые 2 сек (требование Twitch)
        "-b:v", "3000k",                # Битрейт 3000kbps
        "-pix_fmt", "yuv420p",
        
        # --- АУДИО ---
        "-c:a", "aac",
        "-b:a", "160k",
        "-ar", "44100",
        "-af", "aresample=async=1000",  # МАГИЯ: Лечит рассинхрон и щелчки
        
        "-f", "flv", RTMP_URL
    ]

    # Важно: stderr=sys.stderr чтобы видеть ошибки, если они будут
    process = subprocess.Popen(stream_cmd, stdin=subprocess.PIPE, stderr=sys.stderr)

    while True:
        seg_path = video_queue.get()
        print(f"▶️ Playing: {seg_path}")

        try:
            with open(seg_path, "rb") as f:
                while True:
                    chunk = f.read(4096 * 10) # Читаем большими кусками
                    if not chunk:
                        break
                    process.stdin.write(chunk)
            process.stdin.flush()
        except BrokenPipeError:
            print("❌ Stream broken. Restarting...")
            # Тут можно добавить логику перезапуска, но с новым кодом падать не должно
            process = subprocess.Popen(stream_cmd, stdin=subprocess.PIPE, stderr=sys.stderr)
        except Exception as e:
            print(f"❌ Streamer Error: {e}")

        if os.path.exists(seg_path):
            os.remove(seg_path)



if __name__ == "__main__":
    t_worker = threading.Thread(target=worker_thread, daemon=True)
    t_worker.start()
    streamer_thread()
