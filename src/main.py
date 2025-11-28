import os
import sys

# Allow local imports
sys.path.append(os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from chatbot_utility import get_subject_list
from get_yt_video import get_yt_video_link


# -----------------------------------------------------------
# Load environment
# -----------------------------------------------------------
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./vector_db")


# -----------------------------------------------------------
# UI CLEANUP (very minimal)
# -----------------------------------------------------------
st.set_page_config(page_title="StudyPal", page_icon="📘", layout="wide")

st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("📘 StudyPal — RAG Powered Study Assistant")


# -----------------------------------------------------------
# Cached Embeddings + LLM
# -----------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)

EMB = get_embeddings()   # preload silently
LLM = get_llm()          # preload silently


# -----------------------------------------------------------
# Setup RAG Chain
# -----------------------------------------------------------
def setup_chain(subject):
    subject_db_path = os.path.join(VECTOR_DB_DIR, subject)

    if not os.path.exists(subject_db_path):
        st.error(f"❌ Vector DB not found for '{subject}'. Run vectorizer first.")
        return None

    vectordb = Chroma(
        persist_directory=subject_db_path,
        embedding_function=EMB
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=LLM,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
        verbose=False
    )
    return chain


# -----------------------------------------------------------
# Session state
# -----------------------------------------------------------
if "selected_subject" not in st.session_state:
    st.session_state.selected_subject = None
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -----------------------------------------------------------
# Subject selector
# -----------------------------------------------------------
subjects = get_subject_list(DATA_DIR)

selected_subject = st.selectbox(
    "Select a subject PDF",
    subjects,
    index=None,
    placeholder="Choose a PDF"
)

if selected_subject and st.session_state.selected_subject != selected_subject:
    st.session_state.chat_chain = setup_chain(selected_subject)
    st.session_state.chat_history = []
    st.session_state.selected_subject = selected_subject

    if st.session_state.chat_chain:
        st.success(f"✔ Loaded: {selected_subject}")


# -----------------------------------------------------------
# Chat Input
# -----------------------------------------------------------
user_input = st.chat_input("Ask anything...")

if user_input:
    if not st.session_state.chat_chain:
        st.error("Please select a subject first.")
    else:
        # User message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant response
        with st.chat_message("assistant"):
            try:
                response = st.session_state.chat_chain.invoke({"question": user_input})
                answer = response["answer"]
            except Exception as e:
                answer = f"❌ Error: {str(e)}"

            st.markdown(answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

        # YouTube recommendation
        yt_link = get_yt_video_link(f"{selected_subject} {user_input} explained")
        if yt_link:
            st.subheader("🎥 Recommended Video")
            st.info(f"[Watch Video]({yt_link})")


# -----------------------------------------------------------
# Display conversation history
# -----------------------------------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
