# backend/yt_utils.py
from yt_dlp import YoutubeDL

def download_audio_from_youtube(url: str, output_path="audio.mp3") -> str:
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

