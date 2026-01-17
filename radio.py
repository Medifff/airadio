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
from transformers import MusicgenForConditionalGeneration, MusicgenProcessor
from diffusers import StableDiffusionPipeline
import edge_tts
from crewai import Agent, Task, Crew

# =========================
# 1. CONFIG & ENV
# =========================
os.environ["HF_HOME"] = "/workspace/hf_cache"

# Ключи
STREAM_KEY = os.environ.get("TWITCH_STREAM_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not STREAM_KEY:
    print("⚠️ WARNING: TWITCH_STREAM_KEY not found.")
if not OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY not found. DJ will be dumb (fallback mode).")

RTMP_URL = f"rtmp://live.twitch.tv/app/{STREAM_KEY}"
WORKDIR = "/workspace/airadio/data"
os.makedirs(WORKDIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Device: {DEVICE}")

# Очередь сегментов
video_queue = queue.Queue(maxsize=4)
TRACKS_BEFORE_DJ = 3  # DJ говорит раз в 3 трека

# =========================
# 2. LOAD MODELS
# =========================
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

print("⏳ Loading MusicGen Medium...")
processor = MusicgenProcessor.from_pretrained("facebook/musicgen-medium")
music_model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-medium",
    torch_dtype=torch.float32, 
    use_safetensors=True
).to(DEVICE)
music_model.eval()

print("⏳ Loading Stable Diffusion...")
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True
).to(DEVICE)
sd_pipe.safety_checker = None

# =========================
# 3. NEWS & CREW AI
# =========================
TECH_KEYWORDS = ["AI", "ML", "OpenAI", "LLM", "NVIDIA", "Robotics", "SpaceX", "Python", "Cyberpunk", "Neural"]

def fetch_tech_news():
    """Парсит Hacker News по теме AI"""
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
            backstory=(
                "You are 'Nexus', an AI host on a futuristic radio station playing Punk and Electronic Rock. "
                "Your voice is energetic but professional. "
                "You love technology, code, and the future. You never talk about politics."
            ),
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
            You are live on air.
            Current Mood: {mood} (Punk/Electronic/Hardcore)
            
            Latest Tech Headlines:
            - {news_str}
            
            Instructions:
            1. Choose ONE headline to mention briefly OR just talk about the digital revolution.
            2. Keep it under 3 sentences.
            3. Be cool, concise, and energetic. Match the high energy of the music.
            4. Smoothly introduce the next track.
            """,
            agent=self.agent,
            expected_output="A short text script for the DJ to read."
        )

        crew = Crew(agents=[self.agent], tasks=[task])
        try:
            return str(crew.kickoff())
        except Exception as e:
            print(f"⚠️ CrewAI Error: {e}")
            return "Data stream synchronized. Listen to this."

ai_dj = CrewAIDJ()

# === ГЛАВНОЕ ОБНОВЛЕНИЕ: НОВЫЕ ЖАНРЫ И ПРОМПТЫ ===
def get_vibes():
    # Чтобы не звучало как "Атари", добавляем: "high quality, studio recording, real instruments"
    genres = [
        (
            "fast tempo punk rock, distorted electric guitars, energetic live drums, rebellion, high fidelity, studio recording", 
            "punk rock poster, anarchy symbol, graffiti, red and black, grunge texture, chaotic"
        ),
        (
            "post-punk, dark wave, chorus guitar, driving bassline, melancholic, 80s goth vibe, high quality", 
            "post-punk album cover, monochrome, grainy, brutalist architecture, dark fog, mysterious"
        ),
        (
            "happy hardcore, uk hardcore, 170bpm, uplifting piano melody, heavy kick drum, energetic rave, dance", 
            "colorful rave party, lasers, anime aesthetic, neon rainbows, high energy, smiley face"
        ),
        (
            "electronic rock, industrial metal, heavy guitar riffs mixed with distorted synths, powerful drums, cyberpunk action", 
            "cyberpunk rocker, neon guitar, futuristic city, glitch art, aggressive style, blue and purple"
        ),
        (
            "alternative rock, grunge, dirty guitar tone, heavy drums, energetic, 90s style, band recording", 
            "grunge aesthetic, flannel shirt, distorted tv static, garage band, moody lighting"
        )
    ]
    return random.choice(genres)

# =========================
# 4. AUDIO PROCESSING
# =========================
def save_audio_normalized(audio_tensor, filename, sr):
    audio_np = audio_tensor[0, 0].cpu().float().numpy()
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        # Чуть уменьшил громкость (0.9 -> 0.85), чтобы дисторшн в панк-роке не клипповал
        audio_np = audio_np / max_val * 0.85
    sf.write(filename, audio_np, sr, subtype='PCM_16')

# =========================
# 5. WORKER (GENERATOR)
# =========================
def generate_segment(idx, is_dj_turn):
    print(f"\n🔨 [Worker] Processing segment {idx} (DJ Turn: {is_dj_turn})...")
    t0 = time.time()
    
    music_prompt, visual_prompt = get_vibes()
    print(f"🎵 Genre: {music_prompt.split(',')[0]}...")
    
    # Файлы
    music_path = os.path.join(WORKDIR, f"temp_music_{idx}.wav")
    voice_path = os.path.join(WORKDIR, f"temp_voice_{idx}.wav") if is_dj_turn else None
    cover_path = os.path.join(WORKDIR, f"temp_cover_{idx}.png")
    final_video = os.path.join(WORKDIR, f"segment_{idx}.ts")

    # A. MusicGen
    inputs = processor(text=[music_prompt], padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        # guidance_scale=3.5 дает более четкое следование жанру
        audio_values = music_model.generate(**inputs, max_new_tokens=1000, guidance_scale=3.5)
    save_audio_normalized(audio_values, music_path, music_model.config.audio_encoder.sampling_rate)

    # B. Stable Diffusion
    with torch.no_grad():
        image = sd_pipe(f"{visual_prompt}, masterpiece, 8k, detailed", num_inference_steps=20).images[0]
    image.save(cover_path)

    # C. TTS (Только если очередь DJ)
    if is_dj_turn:
        # Передаем настроение трека DJ-ю, чтобы он подстроился
        mood = music_prompt.split(",")[0]
        dj_text = ai_dj.generate_script(mood=mood)
        print(f"🗣️ DJ Says: {dj_text}")
        asyncio.run(edge_tts.Communicate(dj_text, "en-US-ChristopherNeural").save(voice_path))

    # D. FFmpeg Assembly
    f = sf.SoundFile(music_path)
    music_dur = len(f) / f.samplerate
    total_dur = music_dur * 3  

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    
    cmd += ["-loop", "1", "-i", cover_path]          # 0
    if is_dj_turn:
        cmd += ["-i", voice_path]                    # 1
    
    cmd += ["-stream_loop", "-1", "-i", music_path]  # 1 или 2
    cmd += ["-t", str(total_dur)]

    if is_dj_turn:
        # Громкая музыка (0.8) + Громкий голос (1.5) + Компрессор, чтобы качало
        filter_str = "[1:a]volume=1.5[v];[2:a]volume=0.85[m];[v][m]amix=inputs=2:duration=longest:dropout_transition=2[mix];[mix]acompressor=ratio=4[aout]"
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
    while video_queue.qsize() < 1:
        time.sleep(2)
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

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    t_worker = threading.Thread(target=worker_thread, daemon=True)
    t_worker.start()
    streamer_thread()
