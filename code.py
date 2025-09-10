import sys
import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WikipediaLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
import chromadb
from datetime import datetime
import hashlib

# Page config
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stChat {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .assistant-message {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 8px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "document_count" not in st.session_state:
    st.session_state.document_count = 0
if "current_document" not in st.session_state:
    st.session_state.current_document = None

# App Title and Description
st.title("📚 Advanced Document Q&A Assistant")
st.markdown("Upload documents or search Wikipedia to ask questions and get intelligent answers with source citations.")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key configuration
    api_key = st.text_input("OpenAI API Key", type="password", value=st.secrets.get("openai_api_key", ""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.error("Please enter your OpenAI API key to proceed.")
    
    st.divider()
    
    # Model settings
    st.subheader("Model Configuration")
    model_name = st.selectbox(
        "Select Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
        index=0
    )
    
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower values make the output more focused and deterministic"
    )
    
    # Retrieval settings
    st.subheader("Retrieval Settings")
    k_documents = st.slider(
        "Number of relevant chunks to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="More chunks provide more context but may increase response time"
    )
    
    chunk_size = st.slider(
        "Chunk Size",
        min_value=100,
        max_value=1000,
        value=500,
        step=50,
        help="Size of text chunks for processing"
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=200,
        value=50,
        step=10,
        help="Overlap between chunks to maintain context"
    )
    
    st.divider()
    
    # Session management
    st.subheader("Session Management")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.conversation_history = []
        if "memory" in st.session_state:
            st.session_state.memory.clear()
        st.success("Chat history cleared!")
    
    if st.button("🔄 Reset Vector Store"):
        st.session_state.vector_store = None
        st.session_state.document_count = 0
        st.session_state.current_document = None
        st.success("Vector store reset!")
    
    # Display stats
    st.divider()
    st.subheader("📊 Statistics")
    st.metric("Documents Loaded", st.session_state.document_count)
    st.metric("Chat Messages", len(st.session_state.conversation_history))

# Helper Functions
def get_file_hash(file_content):
    """Generate a hash for file content to check for duplicates."""
    return hashlib.md5(file_content).hexdigest()

def load_document(file):
    """Load document based on its file type with error handling."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        name, extension = os.path.splitext(file.name)
        
        if extension.lower() == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif extension.lower() == ".docx":
            loader = Docx2txtLoader(tmp_file_path)
        elif extension.lower() == ".txt":
            loader = TextLoader(tmp_file_path, encoding='utf-8')
        else:
            st.error(f"Unsupported file format: {extension}")
            return None

        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata.update({
                "source": file.name,
                "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return documents
    
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None

def chunk_data(data, chunk_size=500, chunk_overlap=50):
    """Split data into chunks with better strategy."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return text_splitter.split_documents(data)

def create_embeddings_chroma(chunks, collection_name="default", persist_directory='./chroma_db'):
    """Create and persist embeddings in ChromaDB with collection management."""
    try:
        embeddings = OpenAIEmbeddings()
        
        # Create or get collection
        vector_store = Chroma.from_documents(
            chunks,
            embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        return vector_store
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

def format_conversation_history():
    """Format conversation history for display."""
    formatted_history = []
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            formatted_history.append(f'<div class="user-message">👤 **You:** {msg["content"]}</div>')
        else:
            formatted_history.append(f'<div class="assistant-message">🤖 **Assistant:** {msg["content"]}</div>')
            if "sources" in msg:
                sources_html = '<div class="source-box">📌 **Sources:**<br>'
                for source in msg["sources"]:
                    sources_html += f"• {source}<br>"
                sources_html += "</div>"
                formatted_history.append(sources_html)
    return formatted_history

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a file (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        help="Upload a document to analyze and query"
    )
    
    if uploaded_file:
        file_details = {
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.json(file_details)
        
        if st.button("📥 Process Document", type="primary"):
            with st.spinner("Processing document..."):
                progress_bar = st.progress(0)
                
                # Load document
                progress_bar.progress(20)
                data = load_document(uploaded_file)
                
                if data:
                    # Chunk data
                    progress_bar.progress(40)
                    chunks = chunk_data(data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.info(f"Document split into {len(chunks)} chunks")
                    
                    # Create embeddings
                    progress_bar.progress(60)
                    collection_name = f"doc_{get_file_hash(uploaded_file.getvalue())[:8]}"
                    vector_store = create_embeddings_chroma(chunks, collection_name=collection_name)
                    
                    if vector_store:
                        progress_bar.progress(100)
                        st.session_state.vector_store = vector_store
                        st.session_state.document_count += 1
                        st.session_state.current_document = uploaded_file.name
                        st.success(f"✅ Successfully processed '{uploaded_file.name}'!")
                    
                progress_bar.empty()

with col2:
    st.header("🌐 Wikipedia Search")
    
    wikipedia_query = st.text_input(
        "Enter a Wikipedia topic",
        placeholder="e.g., Artificial Intelligence, Climate Change"
    )
    
    max_docs = st.number_input(
        "Number of Wikipedia pages to fetch",
        min_value=1,
        max_value=5,
        value=2
    )
    
    if wikipedia_query and st.button("🔍 Search Wikipedia", type="primary"):
        with st.spinner(f"Searching Wikipedia for '{wikipedia_query}'..."):
            try:
                loader = WikipediaLoader(query=wikipedia_query, load_max_docs=max_docs)
                wiki_data = loader.load()
                
                if wiki_data:
                    # Preview first page
                    with st.expander("📄 Preview Wikipedia Content"):
                        st.write(wiki_data[0].page_content[:500] + "...")
                    
                    wiki_chunks = chunk_data(wiki_data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.info(f"Fetched {len(wiki_data)} pages, split into {len(wiki_chunks)} chunks")
                    
                    # Create embeddings
                    collection_name = f"wiki_{hashlib.md5(wikipedia_query.encode()).hexdigest()[:8]}"
                    vector_store = create_embeddings_chroma(wiki_chunks, collection_name=collection_name)
                    
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.document_count += len(wiki_data)
                        st.session_state.current_document = f"Wikipedia: {wikipedia_query}"
                        st.success(f"✅ Successfully processed Wikipedia data for '{wikipedia_query}'!")
                        
            except Exception as e:
                st.error(f"Error fetching Wikipedia data: {str(e)}")

# Q&A Section
st.divider()
st.header("💬 Ask Questions")

if st.session_state.vector_store and api_key:
    # Display current document
    if st.session_state.current_document:
        st.info(f"📄 Currently loaded: **{st.session_state.current_document}**")
    
    # Initialize retriever with current settings
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k_documents}
    )
    
    # Initialize memory if not exists
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    # Create QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=True
        ),
        retriever=retriever,
        memory=st.session_state.memory,
        chain_type="stuff",
        return_source_documents=True,
        verbose=False
    )
    
    # Question input
    question = st.text_area(
        "Enter your question:",
        placeholder="What would you like to know about the document?",
        height=100
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        ask_button = st.button("🤔 Ask", type="primary")
    with col2:
        if st.button("🔄 Regenerate"):
            if st.session_state.conversation_history:
                question = st.session_state.conversation_history[-2]["content"]
                ask_button = True
    
    if ask_button and question:
        with st.spinner("Thinking..."):
            try:
                # Get answer
                result = qa_chain({"question": question})
                
                # Store in conversation history
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": question
                })
                
                # Extract sources
                sources = []
                if "source_documents" in result:
                    sources = list(set([
                        doc.metadata.get("source", "Unknown")
                        for doc in result["source_documents"]
                    ]))
                
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": sources
                })
                
                # Clear the question input by rerunning
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("📝 Conversation History")
        
        # Display in reverse order (newest first)
        formatted_history = format_conversation_history()
        for msg in reversed(formatted_history):
            st.markdown(msg, unsafe_allow_html=True)
    
else:
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to start.")
    else:
        st.info("📤 Please upload a document or fetch Wikipedia data to start asking questions.")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Made with ❤️ using LangChain and Streamlit | 
        <a href='https://github.com/langchain-ai/langchain' target='_blank'>LangChain Docs</a>
    </div>
    """,
    unsafe_allow_html=True
)
