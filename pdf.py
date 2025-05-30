import logging
from PyPDF2 import PdfReader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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