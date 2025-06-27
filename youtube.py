import logging
import os
import yt_dlp
from pydub import AudioSegment
import speech_recognition as sr
from moviepy.editor import AudioFileClip

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_audio_yt_dlp(url, output_dir, filename):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/{filename}',  # output path with filename
        'quiet': True,
        'no_warnings': True,
        'cookiefile': 'cookies.txt',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def process_youtube_video(youtube_url):
    """Download YouTube audio, convert to wav, split, and transcribe."""

    if not youtube_url or not youtube_url.startswith("https://www.youtube.com/watch?v="):
        return "Invalid YouTube URL. Please provide a valid URL."

    audio_dir = "audio"
    os.makedirs(audio_dir, exist_ok=True)

    mp4_audio_path = os.path.join(audio_dir, "output_audio.webm")  # yt_dlp usually downloads .webm
    wav_audio_path = os.path.join(audio_dir, "output_audio.wav")

    try:
        # Download audio
        download_audio_yt_dlp(youtube_url, audio_dir, "output_audio.webm")

        # Convert to wav
        audio_clip = AudioFileClip(mp4_audio_path)
        audio_clip.write_audiofile(wav_audio_path)
        audio_clip.close()

        # Load audio with pydub for chunking
        audio = AudioSegment.from_wav(wav_audio_path)
        chunk_length_ms = 30 * 1000  # 30 seconds
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        recognizer = sr.Recognizer()
        full_transcription = ""

        for idx, chunk in enumerate(chunks):
            chunk_filename = os.path.join(audio_dir, f"chunk_{idx}.wav")
            chunk.export(chunk_filename, format="wav")

            with sr.AudioFile(chunk_filename) as source:
                audio_data = recognizer.record(source)

            try:
                text = recognizer.recognize_google(audio_data)
                full_transcription += f"[Chunk {idx + 1}]: {text}\n"
            except sr.UnknownValueError:
                full_transcription += f"[Chunk {idx + 1}]: Could not understand the audio.\n"
            except sr.RequestError as e:
                full_transcription += f"[Chunk {idx + 1}]: Google Speech Recognition error: {e}\n"

            os.remove(chunk_filename)  # clean up chunk file

        return full_transcription

    except Exception as e:
        return f"Error processing YouTube video: {e}"

    finally:
        if os.path.exists(wav_audio_path):
            os.remove(wav_audio_path)
        if os.path.exists(mp4_audio_path):
            os.remove(mp4_audio_path)