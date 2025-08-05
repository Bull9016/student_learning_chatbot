from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import GEMINI_API_KEY

def get_gemini_model():
    """Return a Gemini chat model instance."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",   # You can change to "gemini-1.5-flash" for faster responses
        google_api_key=GEMINI_API_KEY,
        temperature=0.7
    )
