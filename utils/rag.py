import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from models.embeddings import get_embedding_model

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
    """Save uploaded file to disk, load it, embed it, and return a FAISS vectorstore."""
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import FAISS
    from models.embeddings import get_embedding_model
    import tempfile

    docs = []
    try:
        suffix = ".pdf" if uploaded_file.name.lower().endswith(".pdf") else ".txt"

        # Save uploaded file to a temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        # Load the file into LangChain docs
        if suffix == ".pdf":
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path)

        loaded_docs = loader.load()
        if not loaded_docs:
            print(f"No text found in {uploaded_file.name}")
            return None

        docs.extend(loaded_docs)

    except Exception as e:
        print(f"Error loading {uploaded_file.name}: {e}")
        return None

    if docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        embeddings = get_embedding_model()
        return FAISS.from_documents(chunks, embeddings)

    return None


def query_vectorstore(vectorstore, query, k=3, score_threshold=0.75):
    """Search with score filtering."""
    if not vectorstore:
        return []
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [doc for doc, score in results if score >= score_threshold]
