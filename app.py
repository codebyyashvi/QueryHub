# # Here I am creating a streamlit app that allows the user to upload the pdfs and chat with the pdfs and also one more feature want to add that user can also upload videos and also can chat with that video too I mean with asking question the answer must be from both video and pdf.
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# from moviepy.editor import VideoFileClip
# import tempfile
# from pydub import AudioSegment
# import speech_recognition as sr

# load_dotenv()

# # Set up Google Generative AI API key
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def extract_audio_from_video(video_file):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
#         temp_video.write(video_file.read())
#         temp_video_path = temp_video.name

#     video_clip = VideoFileClip(temp_video_path)
#     temp_audio_path = temp_video_path.replace('.mp4', '.mp3')
#     video_clip.audio.write_audiofile(temp_audio_path)
#     video_clip.close()
#     return temp_audio_path

# def convert_to_wav(audio_path):
#     wav_path = audio_path.replace('.mp3', '.wav')
#     sound = AudioSegment.from_file(audio_path)
#     sound.export(wav_path, format="wav")
#     return wav_path

# def transcribe_audio(audio_path):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_path) as source:
#         audio_data = recognizer.record(source)
#     try:
#         # Using Google's free web speech API
#         text = recognizer.recognize_google(audio_data)
#         return text
#     except sr.UnknownValueError:
#         return "Sorry, could not understand the audio."
#     except sr.RequestError as e:
#         return f"Could not request results; {e}"

# def process_video(video_file):
#     audio_mp3 = extract_audio_from_video(video_file)
#     audio_wav = convert_to_wav(audio_mp3)
#     transcription = transcribe_audio(audio_wav)
#     print("Transcription:", transcription)
#     # Then chunk and index as needed
#     text_chunks = get_text_chunks(transcription)
#     get_vector_store(text_chunks)

    
# def get_pdf_text(pdf_docs):
#     """Extract text from a PDF file."""
#     text = ""
#     for pdf in pdf_docs:
#         reader = PdfReader(pdf)
#         for page in reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     """Split text into manageable chunks."""
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=10000,
#         chunk_overlap=1000,
#         length_function=len # Using len as the length function to determine chunk size
#     )
#     return text_splitter.split_text(text)

# def get_vector_store(text_chunks):
#     """Create a vector store from text chunks."""
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embeddings)
#     vector_store.save_local("faiss_index")

# def get_conversation_chain():
#     """Create a conversation chain for question answering."""
#     llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.2)
#     # Define a prompt template for the question answering chain
#     prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template="Answer the question based on the context: {context}\nQuestion: {question}\nAnswer:"
#     )
#     chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = vector_store.similarity_search(user_question, k=3)
#     chain = get_conversation_chain()
#     response = chain.run(input_documents=docs, question=user_question)
#     print(response)
#     st.write("Answer:", response)

# def main():
#     st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":robot:")
#     st.title("Chat with Multiple PDFs")
    
#     user_question = st.text_input("Ask a question about the PDF documents:")
#     if user_question:
#         user_input(user_question)
    
#     # Sidebar for uploading videos and pdfs
#     with st.sidebar:
#         st.title("Menu")
#         pdf_docs = st.file_uploader("Upload PDF files and click on Submit & Process", accept_multiple_files=True)
#         video_docs = st.file_uploader("Upload Video files", accept_multiple_files=True, type=["mp4", "avi", "mov"])
#         # Submit button to process pdfs and videos
#         if st.button("Submit & Process"):
#             # Process both pdfs and videos 
#             with st.spinner("Processing PDFs and Videos..."):
#                 if pdf_docs:
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks = get_text_chunks(raw_text) 
#                     get_vector_store(text_chunks)
#                     st.success("PDFs processed successfully!")
#                 elif video_docs:
#                     process_video(video_docs[0])
#                     st.success("Video processed successfully!")
#                 else:
#                     st.error("Please upload at least one PDF or Video file.")
                
# if __name__ == "__main__":
#     main()

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
import tempfile
from pydub import AudioSegment
import speech_recognition as sr
import logging
import io
import subprocess
import yt_dlp
from moviepy.editor import AudioFileClip

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set GOOGLE_API_KEY in the .env file.")
    st.stop()

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# def process_youtube_video(youtube_url):
#     """Download YouTube audio to audio/output_audio.mp4, convert to wav, and transcribe."""

#     audio_dir = "audio"
#     os.makedirs(audio_dir, exist_ok=True)

#     # Fixed paths
#     mp4_audio_path = os.path.join(audio_dir, "output_audio.mp4")
#     wav_audio_path = os.path.join(audio_dir, "output_audio.wav")

#     try:
#         # Download audio stream directly to mp4_audio_path
#         yt = YouTube(youtube_url)
#         if not yt:
#             return "Invalid YouTube URL."
#         audio_stream = yt.streams.filter(only_audio=True).first()
#         audio_stream.download(output_path=audio_dir, filename="output_audio.mp4")

#         # Convert mp4 audio to wav
#         audio_clip = AudioFileClip(mp4_audio_path)
#         audio_clip.write_audiofile(wav_audio_path)
#         audio_clip.close()

#         # Transcribe wav audio
#         recognizer = sr.Recognizer()
#         with sr.AudioFile(wav_audio_path) as source:
#             audio_data = recognizer.record(source)

#         try:
#             text = recognizer.recognize_google(audio_data)
#             return text
#         except sr.UnknownValueError:
#             return "Could not understand the audio."
#         except sr.RequestError as e:
#             return f"Google Speech Recognition error: {e}"

#     except Exception as e:
#         return f"Error processing YouTube video: {e}"

#     finally:
#         # Clean up downloaded audio files
#         # if os.path.exists(mp4_audio_path):
#         #     os.remove(mp4_audio_path)
#         if os.path.exists(wav_audio_path):
#             os.remove(wav_audio_path)

def download_audio_yt_dlp(url, output_dir, filename):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/{filename}',  # output path with filename
        'quiet': True,
        'no_warnings': True,
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

def get_pdf_text(pdf_docs):
    """Extract text from a list of PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,  # Reduced chunk size for better granularity
            chunk_overlap=500,
            length_function=len
        )
        return text_splitter.split_text(text)
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        return []

def get_vector_store(text_chunks):
    """Create and save a FAISS vector store from text chunks."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return None

def get_conversation_chain():
    """Create a conversation chain for question answering."""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an assistant that answers questions based on provided documents and video transcriptions. "
                "Use the following context to answer the question accurately and concisely. "
                "If the answer is not clear from the context, say so.\n\n"
                "Context: {context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
        )
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        logger.error(f"Error creating conversation chain: {e}")
        return None

def user_input(user_question):
    """Process user question and return answer from vector store."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if not os.path.exists("faiss_index"):
            st.error("No content has been processed yet. Please upload and process files first.")
            return
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question, k=3)
        chain = get_conversation_chain()
        if not chain:
            st.error("Failed to initialize conversation chain.")
            return
        response = chain.run(input_documents=docs, question=user_question)
        st.write("**Answer:**", response)
    except Exception as e:
        logger.error(f"Error processing user input: {e}")
        st.error(f"An error occurred: {e}")

def main():

    st.set_page_config(page_title="Chat with PDFs and Videos", page_icon=":robot:")
    st.markdown(
        """
         <style>
            /* Main background and text color */
            .stApp {
                background-color: #1a1a1a;
                color: lightgrey;
                font-size: 18px;
            }

            /* Sidebar background and text color */
            section[data-testid="stSidebar"] {
                background-color: #8c8989; /* dark grey */
                color: lightgrey;
            }

            /* Text input and uploader styling */
            .stTextInput > div > div > input,
            .stFileUploader {
                background-color: #8a8787;
                color: lightgrey;
                border: 1px #555;
                border-radius: 10px;
                font-size: 20px;
                font-weight: bold;
            }

            /* Upload PDFs and Videos text on the sidebar in uploader styling */
            .stFileUploader > div > div > label {
                color: lightgrey;
                font-size: 30px;
                font-weight: bold;
            }

            /* General label and markdown text */
            label, .stTextInput label, .css-10trblm, .stMarkdown, .st-bb {
                color: lightgrey !important;
            }

            /* Top toolbar background */
            header[data-testid="stHeader"] {
                background-color: #333232;
            }


            /* Button styling */
            .stButton > button {
                background-color: white; 
                color: #696666;
                border: 1px solid #555;
                border-radius: 10px;
                font-size: 20px;
                font-weight: bold;
            }

            /* Optional: hide Streamlit's menu (≡) and 'Made with Streamlit' footer if needed */
            #MainMenu, footer {
                visibility: hidden;
            }
            </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Chat with PDFs and Videos")

    # Input for user question
    user_question = st.text_input("Ask a question about the uploaded PDFs or videos:")
    if user_question:
        user_input(user_question)

    # Sidebar for uploading files
    with st.sidebar:
        st.title("Upload Files")
        uploaded_files = st.file_uploader(
            "Upload PDFs and Videos", accept_multiple_files=True, type=["pdf", "mp4", "avi", "mov"]
        )
        youtube_url = st.text_input("Or enter a YouTube video URL:")
        if youtube_url:
            if st.button("Process YouTube Video"):
                with st.spinner("Please Wait, this will take some time..."):
                    transcription = process_youtube_video(youtube_url)
                    if transcription:
                        text_chunks = get_text_chunks(transcription)
                        vector_store = get_vector_store(text_chunks)
                        if vector_store:
                            st.success("YouTube video processed successfully!")
                            logger.info(f"Raw text extracted: {transcription[:500]}...")
                        else:
                            st.error("Failed to create vector store from YouTube video.")
                    else:
                        st.error("Failed to transcribe YouTube video.")
                    

        if st.button("Submit & Process"):
            if not uploaded_files:
                st.error("Please upload at least one PDF or video file.")
            else:
                raw_text = ""

                with st.spinner("Please Wait, this will take some time..."):
                    for file in uploaded_files:
                        file_ext = os.path.splitext(file.name)[1].lower()

                        # Process PDF files
                        if file_ext == ".pdf":
                            try:
                                text = get_pdf_text([file])
                                raw_text += text + "\n"
                                st.success(f"Processed PDF: {file.name}")
                            except Exception as e:
                                st.warning(f"Error processing PDF {file.name}: {e}")

                        # Process Video files
                        if file_ext in [".mp4", ".avi", ".mov"]:
                            try:
                                text = process_video(file)
                                if text:
                                    raw_text += text + "\n"
                                    st.success(f"Processed video: {file.name}")
                                else:
                                    st.warning(f"No text extracted from video: {file.name}")
                            except Exception as e:
                                st.warning(f"Error processing video {file.name}: {e}")

                # Common processing after extracting from all files
                if raw_text.strip():
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    if vector_store:
                        st.success("All files processed and vector store created!")
                    else:
                        st.error("Vector store creation failed.")
                else:
                    st.error("No valid text extracted from uploaded files.")
                # Print raw_text for debugging
                logger.info(f"Raw text extracted: {raw_text[:500]}...")

if __name__ == "__main__":
    main()