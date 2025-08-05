import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models.embeddings import get_embedding_model
from langchain.vectorstores import FAISS
import streamlit as st

DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def load_and_index_documents():
    """Load default docs into FAISS."""
    docs = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
            elif file.endswith(".txt"):
                loader = TextLoader(path)
                docs.extend(loader.load())
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")

    if docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        embeddings = get_embedding_model()
        return FAISS.from_documents(chunks, embeddings)
    return None

def add_uploaded_documents(uploaded_file):
    docs = []
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load based on file type
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

            if st.session_state.vectorstore:
                st.session_state.vectorstore.add_documents(chunks)
            else:
                st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

    except Exception as e:
        st.error(f"Error loading file {uploaded_file.name}: {e}")

def query_vectorstore(vectorstore, query, k=3, score_threshold=0.75):
    """Search with score filtering."""
    if not vectorstore:
        return []
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [doc for doc, score in results if score >= score_threshold]
