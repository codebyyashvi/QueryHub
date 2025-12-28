import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_vector_store(path="faiss_index"):
    if os.path.exists(path):
        shutil.rmtree(path)
        logger.info("Old FAISS index cleared.")


def get_vector_store(text_chunks):
    """Create and save a FAISS vector store from text chunks."""
    try:
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        vector_store = FAISS.from_texts(text_chunks, embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return None

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