# =========================
# 7. WORKER (FIXED RESAMPLING)
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
        
        # БЕЗОПАСНОСТЬ: Проверяем, записался ли голос
        if not os.path.exists(voice_path) or os.path.getsize(voice_path) < 1000:
            print("⚠️ Voice file corrupted or empty! Skipping DJ turn.")
            is_dj_turn = False

    # D. FFmpeg Assembly (With Resampling Fix)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", cover_path]
    
    if is_dj_turn: cmd += ["-i", voice_path]     # Input [1]
    
    cmd += ["-i", music_path, "-i", music_path]  # Input [2] and [3] (or [1] and [2])

    filter_complex = []
    
    # Индексы меняются в зависимости от наличия голоса
    idx_m1 = "2" if is_dj_turn else "1"
    idx_m2 = "3" if is_dj_turn else "2"
    
    # 1. Музыка: Кроссфейд (loop)
    filter_complex.append(f"[{idx_m1}:a][{idx_m2}:a]acrossfade=d=3:c1=tri:c2=tri[music_raw]")
    
    if is_dj_turn:
        # === FIX: RESAMPLING VOICE ===
        # Сначала [1:a] -> aresample=44100 -> [voice_resampled]
        # Это предотвращает краш FFmpeg при сведении разных частот
        filter_complex.append(f"[1:a]aresample=44100,highpass=f=100,lowpass=f=7000,volume=1.8,acompressor=threshold=-16dB:ratio=6:attack=5:release=80[voice_proc_raw]")
        
        # Дублируем для сайдчейна
        filter_complex.append(f"[voice_proc_raw]asplit[voice_sc][voice_mix]")
        
        # Сайдчейн
        filter_complex.append(f"[music_raw][voice_sc]sidechaincompress=threshold=0.05:ratio=10:attack=5:release=300[music_ducked]")
        
        # Микс
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
        # Если упало на сложном фильтре, пробуем аварийный вариант (без голоса)
        if is_dj_turn:
            print("🔄 Retrying without DJ logic...")
            return generate_segment(segment_id, False, forced_genre_idx)
        return None
    
    # Чистка файлов
    files_to_remove = [music_path, cover_path]
    if is_dj_turn and voice_path: files_to_remove.append(voice_path)
    for f in files_to_remove:
        if f and os.path.exists(f): os.remove(f)
    cleanup()
    
    # Обновление пула
    if genre_idx != -1:
        with POOL_LOCK:
            old_file = GENRE_POOL.get(genre_idx)
            GENRE_POOL[genre_idx] = final_video
            print(f"🏊 Pool updated: Genre {genre_idx} refreshed.")
            if old_file and old_file != final_video and os.path.exists(old_file):
                try: os.remove(old_file)
                except: pass

    return final_video
