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

# === IMPORTS FOR STABLE AUDIO & TWITCH ===
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
TWITCH_TOKEN = os.environ.get("TWITCH_TOKEN") # <--- Новый токен для чата
CHANNEL_NAME = os.environ.get("TWITCH_CHANNEL") # Имя твоего канала (логин)

# Если имя канала не задано, попробуем достать его, но лучше задать вручную
if not CHANNEL_NAME:
    print("⚠️ WARNING: TWITCH_CHANNEL env not set. Chat interaction might fail.")
    CHANNEL_NAME = "medi_fff" # Замени на свой логин по умолчанию, если хочешь

if not STREAM_KEY: print("⚠️ WARNING: TWITCH_STREAM_KEY not found.")
if not HF_TOKEN: print("❌ CRITICAL: HF_TOKEN not found!")
else: login(token=HF_TOKEN)

RTMP_URL = f"rtmp://live.twitch.tv/app/{STREAM_KEY}"
WORKDIR = "/workspace/airadio/data"
os.makedirs(WORKDIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Device: {DEVICE}")

video_queue = queue.Queue(maxsize=3)
# Очередь для заказов зрителей (хранит текст промпта)
user_prompt_queue = queue.Queue(maxsize=1) 
TRACKS_BEFORE_DJ = 3 

# =========================
# 2. TWITCH BOT (INTERACTIVE)
# =========================
class Bot(commands.Bot):
    def __init__(self):
        # Используем токен и имя канала
        super().__init__(token=TWITCH_TOKEN, prefix='!', initial_channels=[CHANNEL_NAME])

    async def event_ready(self):
        print(f'🎮 Twitch Bot logged in as | {self.nick}')

    @commands.command(name='vibe', aliases=['вайб'])
    async def vibe_command(self, ctx: commands.Context):
        # Получаем текст после команды !vibe
        content = ctx.message.content
        # Убираем саму команду из текста
        prompt = content.replace("!vibe", "").replace("!вайб", "").strip()
        
        if len(prompt) < 3:
            await ctx.send(f"@{ctx.author.name}, напиши какой жанр ты хочешь. Например: !vibe cyberpunk dark techno")
            return

        # Проверка на спам/плохие слова (базовая)
        if len(prompt) > 100: prompt = prompt[:100]
        
        # Кладем в очередь (если там пусто)
        if user_prompt_queue.empty():
            # Формируем структуру заказа
            order = {
                "user": ctx.author.name,
                "prompt": prompt
            }
            user_prompt_queue.put(order)
            print(f"👾 New Request from {ctx.author.name}: {prompt}")
            await ctx.send(f"@{ctx.author.name}, заказ принят! Нейросеть уже работает над '{prompt}' 🎹")
        else:
            await ctx.send(f"@{ctx.author.name}, очередь занята! Подожди следующий трек.")

def run_twitch_bot():
    if not TWITCH_TOKEN:
        print("⚠️ Twitch Token not found. Chat disabled.")
        return
    
    bot = Bot()
    # Запускаем бота в отдельном event loop, т.к. twitchio асинхронный
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bot.run()

# =========================
# 3. LOAD MODELS
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
            goal="Deliver short updates. Acknowledge user requests if any.",
            backstory="You are 'Nexus', AI host. You play Punk, Electronic Rock and fulfill user requests from chat.",
            verbose=False, allow_delegation=False
        )

    def generate_script(self, mood="high energy", user_request=None):
        if not self.agent: return "System nominal."

        news_items = fetch_tech_news()
        news_str = "\n- ".join(news_items)
        
        # Если есть заказ от зрителя, меняем контекст
        if user_request:
            special_instruction = f"IMPORTANT: Shoutout to user '{user_request['user']}' who requested this track: '{user_request['prompt']}'."
        else:
            special_instruction = "Briefly mention one tech headline."

        task = Task(
            description=f"""
            Live on air. Current Vibe: {mood}.
            {special_instruction}
            Tech Headlines (optional): {news_str}
            
            Keep it under 3 sentences. Be cool, robotic but friendly.
            """,
            agent=self.agent,
            expected_output="Short DJ script."
        )

        crew = Crew(agents=[self.agent], tasks=[task])
        try:
            return str(crew.kickoff())
        except Exception as e:
            return "Request acknowledged. Playing track."

ai_dj = CrewAIDJ()

# =========================
# 5. PROMPTS & AUDIO GEN
# =========================
def get_vibes():
    # 1. Проверяем, есть ли заказ от пользователя
    if not user_prompt_queue.empty():
        order = user_prompt_queue.get()
        print(f"🌟 USING USER PROMPT: {order['prompt']}")
        
        # Добавляем "улучшалки" к пользовательскому промпту
        clean_prompt = f"{order['prompt']}, high quality studio recording, professional mix"
        visual_prompt = f"{order['prompt']}, abstract digital art, 8k, wallpaper"
        
        return clean_prompt, visual_prompt, order # Возвращаем объект заказа

    # 2. Если нет, берем из списка
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
    choice = random.choice(genres)
    return choice[0], choice[1], None # None значит "не заказ"

def gen_music_stable_audio(prompt, out_wav, duration_sec=45):
    print(f"🎧 StableAudio: {prompt[:40]}...")
    cleanup()
    conditioning = [{"prompt": prompt, "seconds_start": 0, "seconds_total": duration_sec}]
    with torch.no_grad():
        output = generate_diffusion_cond(
            audio_model, steps=150, cfg_scale=5.5, conditioning=conditioning,
            sample_size=sample_size, sigma_min=0.3, sigma_max=500,
            sampler_type="dpmpp-3m-sde", device=DEVICE
        )
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
    sf.write(out_wav, output.cpu().numpy().T, sample_rate, subtype='PCM_16')

# =========================
# 6. WORKER (VISUALIZER & CHAT)
# =========================
def generate_segment(idx, is_dj_turn):
    print(f"\n🔨 [Worker] Seg {idx} | DJ: {is_dj_turn}")
    t0 = time.time()
    
    # Получаем промпт (возможно, от пользователя)
    music_prompt, visual_prompt, user_order = get_vibes()
    
    # Если это заказ пользователя, мы ОБЯЗАНЫ включить DJ, чтобы он передал привет
    if user_order:
        is_dj_turn = True
        print("📢 Force enabling DJ for user request!")

    music_part1 = os.path.join(WORKDIR, f"temp_music_{idx}_1.wav")
    music_part2 = os.path.join(WORKDIR, f"temp_music_{idx}_2.wav")
    voice_path = os.path.join(WORKDIR, f"temp_voice_{idx}.wav") if is_dj_turn else None
    cover_path = os.path.join(WORKDIR, f"temp_cover_{idx}.png")
    final_video = os.path.join(WORKDIR, f"segment_{idx}.ts")

    # A. Generate
    gen_music_stable_audio(music_prompt, music_part1, 45)
    gen_music_stable_audio(music_prompt, music_part2, 45)

    with torch.no_grad():
        image = sd_pipe(f"{visual_prompt}, masterpiece, 8k", num_inference_steps=20).images[0]
    image.save(cover_path)

    # B. DJ Script
    if is_dj_turn:
        # Передаем инфо о пользователе в CrewAI
        mood = music_prompt.split(",")[0]
        dj_text = ai_dj.generate_script(mood=mood, user_request=user_order)
        print(f"🗣️ DJ: {dj_text}")
        asyncio.run(edge_tts.Communicate(dj_text, "en-US-ChristopherNeural").save(voice_path))

    # C. FFmpeg with VISUALIZER
    total_dur = 85 
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    
    # Inputs
    cmd += ["-loop", "1", "-i", cover_path]      # [0] Image
    if is_dj_turn: cmd += ["-i", voice_path]     # [1] Voice
    cmd += ["-i", music_part1]                   # [2] Music A
    cmd += ["-i", music_part2]                   # [3] Music B

    filter_complex = []
    
    # Audio Logic (Crossfade + Sidechain + Loudnorm)
    idx_m1 = "2" if is_dj_turn else "1"
    idx_m2 = "3" if is_dj_turn else "2"
    
    filter_complex.append(f"[{idx_m1}:a][{idx_m2}:a]acrossfade=d=3:c1=tri:c2=tri[music_raw]")
    
    if is_dj_turn:
        filter_complex.append(f"[1:a]highpass=f=100,lowpass=f=7000,volume=1.8,acompressor=threshold=-16dB:ratio=6:attack=5:release=80[voice_proc]")
        filter_complex.append(f"[music_raw][voice_proc]sidechaincompress=threshold=0.05:ratio=10:attack=5:release=300[music_ducked]")
        filter_complex.append(f"[music_ducked][voice_proc]amix=inputs=2:duration=first[pre_master]")
    else:
        filter_complex.append(f"[music_raw]anull[pre_master]")

    # Master Output
    filter_complex.append(f"[pre_master]loudnorm=I=-14:TP=-1.0:LRA=11[out_a]")
    
    # === VISUALIZER LOGIC ===
    # Нам нужно дублировать аудио сигнал: один идет на выход, другой на визуализацию
    filter_complex.append(f"[out_a]asplit[a_final][a_vis]")
    
    # Рисуем волну (showwaves)
    # s=1280x240: размер (ширина видео, высота волны)
    # mode=line: стиль линий
    # colors=cyan: цвет
    filter_complex.append(f"[a_vis]showwaves=s=1280x180:mode=line:colors=0x00FFFF@0.6[waves]")
    
    # Накладываем волну на картинку
    # overlay=0:H-h: волна прижимается к низу видео
    filter_complex.append(f"[0:v][waves]overlay=x=0:y=H-h[out_v]")

    cmd += ["-filter_complex", ";".join(filter_complex)]
    cmd += [
        "-map", "[out_v]", "-map", "[a_final]", # Берем обработанное видео и аудио
        "-t", str(total_dur),
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-g", "60",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-f", "mpegts", final_video
    ]
    
    subprocess.run(cmd, check=True)
    
    files_to_remove = [music_part1, music_part2, cover_path]
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
            # Если есть заказ пользователя, tracks_since_dj игнорируется внутри generate_segment
            is_dj_turn = (tracks_since_dj >= TRACKS_BEFORE_DJ)
            
            seg_path = generate_segment(idx, is_dj_turn)
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
            process = subprocess.Popen(stream_cmd, stdin=subprocess.PIPE)
        if os.path.exists(seg_path): os.remove(seg_path)

if __name__ == "__main__":
    # Запускаем бота в отдельном потоке
    t_bot = threading.Thread(target=run_twitch_bot, daemon=True)
    t_bot.start()
    
    t_worker = threading.Thread(target=worker_thread, daemon=True)
    t_worker.start()
    
    streamer_thread()
