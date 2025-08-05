import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys (values will come from .env or Streamlit Secrets)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Embedding model
EMBEDDING_MODEL = "models/embedding-001"
