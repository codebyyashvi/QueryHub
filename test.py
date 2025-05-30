import yt_dlp
import os
from moviepy.editor import AudioFileClip
def download_audio(url, output_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == '__main__':
    youtube_url = 'https://www.youtube.com/watch?v=9ofL45Mrzj0' # Replace with the actual URL
    output_dir = 'audio' # Replace with your desired directory
    download_audio(youtube_url, output_dir)
    mp4_audio_path = os.path.join(output_dir, "VALUE OF TIME | A Life Changing Motivational Story | Time Story | English Stories | Moral Stories.webm")
    audio_clip = AudioFileClip(mp4_audio_path)
    print("Audio duration (seconds):", audio_clip.duration)
    audio_clip.close()