import streamlit as st
import rag
import os

st.set_page_config(page_title="Local PDF RAG", layout="centered")
st.title("Local PDF RAG")

TARGET_PDF = rag.CONFIG["PDF_PATH"]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("LLM Model",["qwen3:0.6b", "llama3.2:1b", "gemma3:1b"])
    top_k_val = st.slider("Retrieval Septh (TOP_K)", 1, 10, 3)
    
    st.divider()
    st.subheader("Document Managment")
    uploaded_file = st.file_uploader("Upload PDF", type = "pdf")

    if uploaded_file is not None:
        with open(TARGET_PDF, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("PDF saved! Re-indexing automatically if changed.")

@st.cache_resource
def load_rag(pdf_timestamp):
    return rag.load_knowledge_base()

if not os.path.exists(TARGET_PDF):
    st.warning("Please upload a 'rag.pdf' file or drop a document into the slidebar to start.")
    st.stop()

pdf_timestamp = os.path.getmtime(TARGET_PDF)
index, chunks_metadata = load_rag(pdf_timestamp)

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)
        
question = st.chat_input("Ask a question")

if question:
    with st.chat_message("User"):
        st.write(question) 

    with st.spinner("Thinking..."):
        try:
            rag.CONFIG["TOP_K"] = top_k_val
            context, source = rag.retrieve_context(question, index, chunks_metadata)
            rag.CONFIG["MODEL_NAME"] = model_choice
            answer = rag.ask_llm(question, context, st.session_state.chat_history)
            st.session_state.chat_history.append(("user", question))
            st.session_state.chat_history.append(("assistant", f"{answer}\n\n*[Sources: {source}]*"))
            st.rerun()
        
        except Exception as e:
            st.error(f"Inference pipeline encounter: {e}")
               


