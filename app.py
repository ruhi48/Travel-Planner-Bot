__import__('pysqlite3')
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import streamlit as st
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
import os

# === Set page config with background and branding ===
st.set_page_config(page_title="🌍 Travel Planner Bot", layout="wide")

# Add custom background and styling
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: rgba(0, 0, 0, 0.7);
    color: white;
}
.stButton > button {
    background-color: #ff5a5f;
    color: white;
    border-radius: 10px;
    padding: 10px;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #ff7f7f;
}
div[data-testid="stTextInput"] > div > div {
    color: white;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# === Page header ===
st.markdown("<h1 style='text-align: center; color: white;'>🌍 Your Personalized Travel Planner Bot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Let me create the perfect itinerary for your next adventure!</p>", unsafe_allow_html=True)

# === Initialize memory and embeddings ===
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    st.error(f"⚠️ Failed to load embedding models: {str(e)}")
    embedding_model = None
    semantic_model = None

# === Initialize ChatGroq ===
try:
    chat = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY", "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")
    )
except Exception as e:
    st.error(f"⚠️ Chat model initialization failed: {str(e)}")
    chat = None

# === Initialize ChromaDB ===
try:
    chroma_client = chromadb.PersistentClient(path="./travel_chroma_db")
    collection = chroma_client.get_or_create_collection(name="travel_knowledge_base")
except Exception as e:
    st.error(f"⚠️ Failed to initialize ChromaDB: {str(e)}")
    collection = None

# === Retrieve relevant context ===
def retrieve_context(query, top_k=1):
    if not embedding_model or not collection:
        return ["Embedding model or ChromaDB not initialized."]
    try:
        query_embedding = embedding_model.embed_query(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results["documents"][0] if results and results["documents"] else ["No relevant context found."]
    except Exception as e:
        return [f"⚠️ Retrieval Error: {str(e)}"]

# === Generate itinerary ===
def query_travel_assistant(user_query):
    """Query Llama3 for travel suggestions."""
    if not chat:
        return user_query, "⚠️ Chat model initialization error."
    
    system_prompt = (
        "You are a travel assistant specialized in creating personalized itineraries. "
        "Generate detailed travel plans including attractions, activities, and accommodation suggestions."
    )
    retrieved_context = retrieve_context(user_query)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"DB Context: {retrieved_context}\nQuestion: {user_query}")
    ]

    try:
        response = chat.invoke(messages)
        return user_query, response.content
    except Exception as e:
        return user_query, f"⚠️ API Error: {str(e)}"

# === Chat History ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Display Chat ===
for user_msg, bot_response in st.session_state.chat_history:
    st.markdown(
        f'<div style="background-color:rgba(0, 0, 0, 0.6);padding:10px;border-radius:10px;margin:5px;color:white;">👤 {user_msg}</div>', 
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="background-color:rgba(255, 90, 95, 0.8);padding:10px;border-radius:10px;margin:5px;color:white;">🤖 {bot_response}</div>', 
        unsafe_allow_html=True
    )

# === Input ===
user_query = st.chat_input("Plan your next trip...")
if user_query:
    user_msg, bot_response = query_travel_assistant(user_query)
    st.session_state.chat_history.append((user_msg, bot_response))
    st.rerun()
