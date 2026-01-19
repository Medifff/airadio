cmd = [
    "ffmpeg",
    "-ignore_eof", "1",              # <<< КРИТИЧЕСКИ ВАЖНО
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
