# QueryHub - A Streamlit app for querying PDFs and videos using Google Generative AI

import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
from pdf import get_pdf_text
from video import process_video
from youtube import process_youtube_video
from vector import get_vector_store, get_text_chunks

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

    st.set_page_config(page_title="Ask from Youtube, Videos and Docs", page_icon=":robot:")
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
    st.title("Ask from Youtube, Videos and Docs")

    # Input for user question
    user_question = st.text_input("Ask a question about the uploaded youtube links or Pdfs and videos:")
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