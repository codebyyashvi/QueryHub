import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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