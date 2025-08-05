import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from models.embeddings import get_embedding_model
import streamlit as st

DATA_DIR = "data"

# Ensure data directory exists locally
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def load_and_index_documents():
    """Load all documents from DATA_DIR into a FAISS vectorstore."""
    docs = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif file.endswith(".txt"):
                loader = TextLoader(path)
            else:
                continue
            docs.extend(loader.load())
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = get_embedding_model()
    return FAISS.from_documents(chunks, embeddings)

def add_uploaded_documents(uploaded_file):
    """Handle uploaded files for both local and Streamlit Cloud."""
    docs = []
    try:
        # Save file temporarily so loaders can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load the file depending on extension
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(tmp_path)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            return

        docs.extend(loader.load())

        if docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            embeddings = get_embedding_model()

            if st.session_state.get("vectorstore"):
                st.session_state.vectorstore.add_documents(chunks)
            else:
                st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

            st.success(f"✅ {uploaded_file.name} added to knowledge base!")
        else:
            st.warning(f"No extractable text found in {uploaded_file.name}")

    except Exception as e:
        st.error(f"Error loading file {uploaded_file.name}: {e}")

def query_vectorstore(vectorstore, query, k=3, score_threshold=None):
    """Search the vectorstore for relevant chunks."""
    if not vectorstore:
        return []
    results = vectorstore.similarity_search_with_score(query, k=k)
    if score_threshold is not None:
        results = [doc for doc, score in results if score >= score_threshold]
    else:
        results = [doc for doc, _ in results]
    return results

