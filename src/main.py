import logging
import os
import sys
import uuid # For generating unique chat IDs
import io # For capturing and silencing output
import contextlib # For redirecting output

# AGGRESSIVE LOGGER SUPPRESSION 
logging.getLogger().setLevel(logging.ERROR)

# Allow local imports
sys.path.append(os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv

# Vector DB + Embed + LLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# NEW LangChain API (core)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# Local utils (Assuming these are available in your environment)
from chatbot_utility import get_subject_list
from get_yt_video import get_yt_video_link


# -----------------------------------------------------------
# Load env & Global Config
# -----------------------------------------------------------
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./vector_db")

# -----------------------------------------------------------
# UI Configuration (MUST be the first Streamlit command)
# -----------------------------------------------------------
st.set_page_config(page_title="StudyPal", page_icon="📘", layout="wide")

# ==========================================================
# 🥇 INITIALIZATION PHASE: BLANK SCREEN COLD START
# ==========================================================

# 1. Create a large placeholder for the ENTIRE APP content
main_app_placeholder = st.empty()

# Execute loading inside the placeholder container
with main_app_placeholder.container():
    # 2. Execute the resource loading (logs are trapped here)
    # The container itself is blank, so the screen remains blank.
    
    # Define cached functions locally to control their scope
    @st.cache_resource
    def get_embeddings():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    @st.cache_resource
    def get_llm():
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # Use output redirection for the highest chance of silence
    with contextlib.redirect_stdout(io.StringIO()):
        EMB = get_embeddings()
        LLM = get_llm()

# 3. CRUCIAL STEP: Clear the placeholder entirely *after* loading.
# This removes the hidden artifacts and prepares the app for rendering.
main_app_placeholder.empty()

# 4. Define the final, clean app container where all subsequent UI will be rendered.
final_app_container = st.container()

# ==========================================================
# 🥈 RAG COMPONENTS AND UTILITIES 
# ==========================================================
@st.cache_resource
def get_retriever(subject: str):
    """
    Loads and caches the Chroma DB and its retriever for a given subject.
    (Drastically reduces subject load time)
    """
    db_path = os.path.join(VECTOR_DB_DIR, subject)
    
    if not os.path.exists(db_path):
        return None

    db = Chroma(
        persist_directory=db_path,
        embedding_function=EMB
    )
    return db.as_retriever(search_kwargs={"k": 4})


def setup_chain(retriever, history):
    """Builds the RAG chain using cached components."""
    prompt = ChatPromptTemplate.from_template("""
You are StudyPal, a helpful AI tutor. You MUST NOT include any video links in your response, as the app handles video recommendations separately.

Context from textbook:
{context}

Chat History:
{chat_history}

User Question: {question}
""")

    # Chain step: format chat history
    def format_history(_):
        clean_messages = []
        URL_SEPARATOR = "||YT_LINK||"
        for m in history.messages:
            clean_content = m.content.split(URL_SEPARATOR)[0] 
            clean_messages.append(f"{m.type}: {clean_content}")
            
        return "\n".join(clean_messages)

    # Chain pipeline
    chain = (
        RunnableMap({
            "context": retriever,
            "chat_history": format_history,
            "question": RunnablePassthrough()
        })
        | prompt
        | LLM
    )
    return chain

# -----------------------------------------------------------
# Session State Management 
# -----------------------------------------------------------
def init_session_state():
    """Initializes all necessary session state variables."""
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
        
    if "selected_subject" not in st.session_state:
        st.session_state.selected_subject = None

    if "retriever" not in st.session_state:
        st.session_state.retriever = None
        
    if "chain" not in st.session_state:
        st.session_state.chain = None

    if "subjects" not in st.session_state:
         st.session_state.subjects = get_subject_list(DATA_DIR)

init_session_state()

# Helper function to switch/create a new chat
def switch_chat(chat_id=None, subject_change=False):
    """Switches the current chat session or resets to start a new one."""
    if chat_id in st.session_state.chat_sessions:
        st.session_state.current_chat_id = chat_id
        session = st.session_state.chat_sessions[chat_id]
        
        subject = session["subject"]
        st.session_state.selected_subject = subject
        
        st.session_state.retriever = get_retriever(subject)
        
        if st.session_state.retriever:
            st.session_state.chain = setup_chain(st.session_state.retriever, session["history"])
            st.toast(f"✅ Loaded Subject: {subject} successfully!")
        else:
            st.session_state.chain = None
            st.error(f"Could not load retriever for subject: {subject}")
            
    else:
        st.session_state.current_chat_id = None
        st.session_state.selected_subject = None
        st.session_state.retriever = None
        st.session_state.chain = None
        if not subject_change:
            st.toast("Starting a new chat.")
    
    st.rerun() 
    
def create_new_chat(subject_name):
    """Creates a new chat session for a given subject and switches to it."""
    new_chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_chat_id] = {
        "subject": subject_name,
        "history": ChatMessageHistory(),
        "name": f"Chat - {subject_name}"
    }
    switch_chat(new_chat_id, subject_change=True) 

def update_chat_name(chat_id, new_name):
    """Updates the name of a specific chat session."""
    if chat_id in st.session_state.chat_sessions and new_name:
        st.session_state.chat_sessions[chat_id]["name"] = new_name
        st.toast(f"Chat renamed to: {new_name}")

# ==========================================================
# 🥉 MAIN APPLICATION CODE (Rendered inside the final_app_container)
# ==========================================================
with final_app_container:
    # -----------------------------------------------------------
    # UI - Sidebar 
    # -----------------------------------------------------------
    with st.sidebar:
        st.header("Subjects & Chats")
        
        # 1. Subject Selection
        current_subject = st.session_state.selected_subject
        
        subject_index = st.session_state.subjects.index(current_subject) if current_subject in st.session_state.subjects else None
        
        selected_subject_name = st.selectbox(
            "Select a Subject", 
            st.session_state.subjects, 
            index=subject_index,
            key="subject_select" 
        )

        # Subject Change Logic (Triggers a new chat)
        if selected_subject_name and (st.session_state.selected_subject != selected_subject_name):
            create_new_chat(selected_subject_name)
        
        st.markdown("---")
        
        # 2. Chat History Management
        if st.button("➕ New Chat", use_container_width=True):
            if st.session_state.selected_subject:
                create_new_chat(st.session_state.selected_subject)
            else:
                switch_chat(None) 
        
        st.subheader("Chat History")
        
        if st.session_state.chat_sessions:
            # Display chat name input for the currently selected chat 
            if st.session_state.current_chat_id:
                current_session = st.session_state.chat_sessions[st.session_state.current_chat_id]
                
                input_key = f"rename_input_{st.session_state.current_chat_id}"
                
                new_name = st.text_input(
                    "Rename Chat",
                    value=current_session.get("name", f"Chat - {current_session['subject']}"),
                    key=input_key,
                    label_visibility="collapsed"
                )
                
                # Check if the input value is different from the stored name (and not empty)
                if new_name != current_session.get("name") and new_name.strip():
                    update_chat_name(st.session_state.current_chat_id, new_name)
                    st.rerun() 

                st.markdown("---") 

            # List all chats
            sorted_chats = sorted(
                st.session_state.chat_sessions.items(), 
                key=lambda item: item[1]['subject']
            )
            
            for chat_id, session in sorted_chats:
                chat_name = session.get("name", f"Chat ({session['subject']})")
                is_selected = chat_id == st.session_state.current_chat_id
                
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    if st.button(chat_name, key=f"select_{chat_id}", use_container_width=True, type="primary" if is_selected else "secondary"):
                        switch_chat(chat_id)
                
                with col2:
                    if st.button("🗑️", key=f"delete_{chat_id}", use_container_width=True):
                        del st.session_state.chat_sessions[chat_id]
                        if is_selected:
                            switch_chat(None)
                        else:
                            st.rerun()
                            
    # -----------------------------------------------------------
    # UI - Main Content (All Features Integrated)
    # -----------------------------------------------------------
    st.title("📘 StudyPal — RAG Powered Study Assistant")

    # Show current subject header
    if st.session_state.selected_subject:
        st.subheader(f"Subject: {st.session_state.selected_subject}")
    else:
        st.info("Please select a subject from the sidebar to start a new chat.")


    # Define a unique marker to separate the answer from the stored URL
    URL_SEPARATOR = "||YT_LINK||"

    # 1. Display Chat History FIRST (Now correctly shows persistent YouTube links)
    if st.session_state.current_chat_id:
        history = st.session_state.chat_sessions[st.session_state.current_chat_id]["history"]
        
        for msg in history.messages:
            display_type = "user" if msg.type == "human" else "assistant"
            
            # Split content to check for stored URL
            parts = msg.content.split(URL_SEPARATOR)
            response_text = parts[0]
            stored_link = parts[1] if len(parts) > 1 else None

            with st.chat_message(display_type):
                st.write(response_text)
                
                # If it's an AI message and contains a stored link, display it
                if display_type == "assistant" and stored_link:
                    st.info(f"🎥 Recommended Video: [Watch here]({stored_link})")


    # 2. Chat Input and Generation (Updated to store the link)
    user_input = st.chat_input("Ask something from the subject...")

    if user_input:
        # --- Input Validation ---
        if not st.session_state.chain:
            st.error("Select a subject and/or a chat session first.")
            st.stop() 

        # --- Generation Logic ---
        current_history = st.session_state.chat_sessions[st.session_state.current_chat_id]["history"]

        # 1. Show user message immediately
        with st.chat_message("user"):
            st.write(user_input)

        current_history.add_user_message(user_input)

        # 2. Run RAG and display response immediately
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chain.invoke(user_input)
                answer = response.content if hasattr(response, "content") else str(response)
                
                # Fetch the YouTube link
                yt_link = get_yt_video_link(f"{st.session_state.selected_subject} {user_input} tutorial")
                
                # Store the answer AND the link in the history (Persistence Fix)
                stored_content = answer
                if yt_link:
                    stored_content += f"{URL_SEPARATOR}{yt_link}"
                    
                current_history.add_ai_message(stored_content)

                # Display AI response *only once*
                with st.chat_message("assistant"):
                    st.write(answer) # Display clean text
                    
                    # Display YouTube recommendation immediately
                    if yt_link:
                        st.info(f"🎥 Recommended Video: [Watch here]({yt_link})")

            except Exception as e:
                st.error(f"An error occurred during generation. Please try again. Error: {e}")
                if len(current_history.messages) > 0 and current_history.messages[-1].type == "human":
                    current_history.messages.pop() 
                
        st.rerun()