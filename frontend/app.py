import os
from typing import List

import requests
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Configure the app
st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Get backend URL from environment variable
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


def upload_documents(files: List[UploadedFile]) -> dict:
    """Upload documents to the backend API"""
    upload_endpoint = f"{BACKEND_URL}/documents"

    files_data = [
        ("files", (file.name, file.getvalue(), "application/pdf"))
        for file in files
    ]

    response = requests.post(upload_endpoint, files=files_data)
    return response.json()


def get_answer(question: str) -> dict:
    """Get answer from the backend API"""
    question_endpoint = f"{BACKEND_URL}/question"

    response = requests.post(
        question_endpoint, json={"question": question}
    )
    return response.json()


# App title
st.title("🤖 RAG System")

with st.expander("Document Upload", expanded=True):
    # File upload section
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF documents", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                result = upload_documents(uploaded_files)
                st.success(
                    f"Processed {result['documents_indexed']} documents with {result['total_chunks']} chunks"
                )

with st.expander("Chat", expanded=True):
    # Chat interface
    st.header("Chat")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "references" in message:
                st.info("References: " + message["references"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_answer(prompt)
                st.write(response["answer"])
                if response["references"]:
                    st.info("References: " + ", ".join(response["references"]))

                # Add assistant message to chat history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response["answer"],
                        "references": ", ".join(response["references"]),
                    }
                )
