#from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from config.config import GEMINI_API_KEY, EMBEDDING_MODEL

#def get_embedding_model():
 #   return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.config import GEMINI_API_KEY, EMBEDDING_MODEL

def get_embedding_model():
    """Create a synchronous Gemini embeddings model (compatible with Streamlit Cloud)."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY
    )
