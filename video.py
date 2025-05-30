import logging  
import os
import tempfile
from moviepy.editor import VideoFileClip
import speech_recognition as sr

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_video(video_file):
    """Process uploaded video file: extract audio and return transcription."""

    # Create directory for saving audio
    audio_dir = "audio"
    os.makedirs(audio_dir, exist_ok=True)

    try:
        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_file.read())
            temp_video_path = temp_video.name

        # Define path for extracted audio
        audio_path = os.path.join(audio_dir, "output_audio.wav")

        # Extract audio from video
        video = VideoFileClip(temp_video_path)
        video.audio.write_audiofile(audio_path)
        video.close()

        # Transcribe audio
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service: {e}"

    except Exception as e:
        return f"Error processing video: {e}"

    finally:
        # Clean up temp video file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)