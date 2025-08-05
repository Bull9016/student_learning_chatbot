import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.llm import get_gemini_model
from utils.rag import load_and_index_documents, query_vectorstore, add_uploaded_documents
from utils.web_search import google_search

# Load main vectorstore at startup
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_and_index_documents()

# Separate vectorstore for uploaded docs
if "uploaded_vectorstore" not in st.session_state:
    st.session_state.uploaded_vectorstore = None

def get_chat_response(chat_model, messages, system_prompt):
    """Get response from the chat model."""
    try:
        formatted_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))
        response = chat_model.invoke(formatted_messages)
        return response.content
    except Exception as e:
        return f"Error getting response: {str(e)}"

def instructions_page():
    st.title("The Chatbot Blueprint")
    st.markdown("""
    ## 🚀 How to Use This Learning Chatbot for Best Results

    1. **Upload Your Learning Material**  
       - Go to the **Chat** page and upload PDFs or TXT files.
       - The bot will read these and answer questions based on them.

    2. **Choose or Create Your Interest**  
       - Pick a topic like *Cricket*, *F1*, *Cooking* or choose **Other** and type your own.
       - The bot will tailor examples to your interest.

    3. **Select Your Response Mode**  
       - **Concise Mode** → Short summaries.
       - **Detailed Mode** → Longer, in-depth explanations.

    4. **Ask Smart Questions**  
       - Be specific for better answers.

    5. **Use Live Web Search** if your docs don’t have the answer.

    6. **Clear Chat History** anytime from the sidebar.
    """)

def chat_page():
    st.title("🤖 Student Learning ChatBot")

    # Response mode
    mode = st.radio("Response Mode:", ["Concise", "Detailed"], horizontal=True)

    # Interest selection
    interest_choice = st.selectbox(
        "Choose your interest for examples:",
        ["General", "Cricket", "F1", "Cooking", "Other"]
    )
    if interest_choice == "Other":
        interest = st.text_input("Enter your custom interest:")
    else:
        interest = interest_choice

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents for learning (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for file in uploaded_files:
            new_store = add_uploaded_documents(file)
            if new_store:
                if st.session_state.uploaded_vectorstore:
                    st.session_state.uploaded_vectorstore.merge_from(new_store)
                else:
                    st.session_state.uploaded_vectorstore = new_store
        st.success("✅ Documents added to knowledge base!")

    # System prompt
    system_prompt = f"You are a friendly learning assistant. Provide {mode.lower()} answers with examples related to {interest}."

    chat_model = get_gemini_model()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Search uploaded docs first
                local_results = []
                if st.session_state.uploaded_vectorstore:
                    local_results = query_vectorstore(st.session_state.uploaded_vectorstore, prompt, score_threshold=0.75)

                # If none, search default docs
                if not local_results:
                    local_results = query_vectorstore(st.session_state.vectorstore, prompt, score_threshold=0.75)

                # If still none, use web search
                if local_results:
                    context = "\n".join([doc.page_content for doc in local_results])
                    full_prompt = f"{prompt}\n\nContext:\n{context}"
                else:
                    context = google_search(prompt)
                    full_prompt = f"{prompt}\n\nWeb Search Results:\n{context}"

                response = get_chat_response(chat_model, st.session_state.messages, system_prompt + "\n" + full_prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    st.set_page_config(page_title="LangChain Multi-Provider ChatBot", page_icon="🤖", layout="wide")
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"], index=0)
        if page == "Chat":
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()
