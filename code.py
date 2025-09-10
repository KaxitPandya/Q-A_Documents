import sys
import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
import time
import shutil
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WikipediaLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback
import chromadb

# Page configuration for a professional look
st.set_page_config(
    page_title="Q&A on Documents",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a more professional UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #1E88E5;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        padding-top: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F44336;
        margin: 1rem 0;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #0D47A1;
    }
    .stProgress .st-bo {
        background-color: #1E88E5;
    }
    div[data-testid="stSidebar"] {
        background-color: #F5F5F5;
    }
    .stTab {
        font-weight: bold;
    }
    .stTextInput input {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "token_count" not in st.session_state:
    st.session_state.token_count = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
if "last_action" not in st.session_state:
    st.session_state.last_action = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0  # 0: Document Upload, 1: Wikipedia, 2: Chat

# Sidebar configuration
with st.sidebar:
    st.subheader("Configuration")
    
    # API Key handling - enhanced with visual confirmation
    st.markdown("### API Key Configuration")
    api_key_option = st.radio("OpenAI API Key", ["Use from Secrets", "Enter Manually"])
    
    api_key_status = None
    
    if api_key_option == "Enter Manually":
        api_key = st.text_input("OpenAI API Key", type="password", 
                               help="Enter your OpenAI API key to use embeddings and Q&A features")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state["api_key_set"] = True
            api_key_status = True
        else:
            st.session_state["api_key_set"] = False
            api_key_status = False
    else:
        if "openai_api_key" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
            st.session_state["api_key_set"] = True
            api_key_status = True
        else:
            st.error("Please add your OpenAI API key to Streamlit secrets or enter it manually to proceed.")
            st.session_state["api_key_set"] = False
            api_key_status = False
    
    # Visual confirmation of API key
    if api_key_status is not None:
        if api_key_status:
            st.success("API key configured successfully")
            # Test the API connection button
            if st.button("Test API Connection"):
                try:
                    # Try a simple completion to test connection
                    from openai import OpenAI
                    client = OpenAI()
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Hello!"}],
                        max_tokens=5
                    )
                    st.success("API connection successful!")
                except Exception as e:
                    st.error(f"API connection failed: {str(e)}")
        else:
            st.warning("No API key configured")
    
    # Model selection
    model_name = st.selectbox(
        "Select Model",
        options=["gpt-3.5-turbo", "gpt-4"],
        index=0,
    )
    
    # Temperature setting
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Higher values make output more random, lower values more deterministic"
    )
    
    # Chunk settings
    chunk_size = st.number_input(
        "Chunk Size", 
        min_value=100, 
        max_value=2000, 
        value=256,
        help="Size of text chunks for embedding"
    )
    
    chunk_overlap = st.number_input(
        "Chunk Overlap", 
        min_value=0, 
        max_value=500, 
        value=0,
        help="Overlap between chunks to maintain context"
    )
    
    # k-value for retrieval
    k_value = st.number_input(
        "Number of chunks to retrieve", 
        min_value=1, 
        max_value=20, 
        value=5,
        help="Number of most relevant chunks to retrieve"
    )
    
    # Reset buttons
    if st.button("Reset Conversation"):
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.chat_history = []
        st.success("Conversation history cleared!")
        
    if st.button("Reset Vector Database"):
        if os.path.exists("./chroma_db"):
            try:
                shutil.rmtree("./chroma_db")
                st.session_state.vector_store = None
                st.success("Vector database cleared successfully!")
            except Exception as e:
                st.error(f"Error clearing database: {str(e)}")

# Helper Functions
def load_document(file) -> Optional[List]:
    """
    Load document based on its file type.
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        List of document chunks or None if file format is unsupported
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        name, extension = os.path.splitext(file.name)
        extension = extension.lower()
        
        if extension == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif extension == ".docx":
            loader = Docx2txtLoader(tmp_file_path)
        elif extension == ".txt":
            loader = TextLoader(tmp_file_path)
        elif extension == ".csv":
            loader = CSVLoader(tmp_file_path)
        else:
            st.error(f"Unsupported file format: {extension}! Please upload PDF, DOCX, TXT, or CSV files.")
            os.unlink(tmp_file_path)  # Clean up temp file
            return None

        documents = loader.load()
        
        # Clean up temp file after loading
        os.unlink(tmp_file_path)
        
        if len(documents) == 0:
            st.warning("Document loaded but contains no extractable text.")
            return None
            
        return documents
        
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None

def chunk_data(data, chunk_size=256, chunk_overlap=0) -> List:
    """
    Split data into chunks for processing.
    
    Args:
        data: Documents to split
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of split document chunks
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(data)
        
        if len(chunks) == 0:
            st.warning("No chunks were created. The document might be empty.")
            return []
            
        return chunks
    except Exception as e:
        st.error(f"Error chunking data: {str(e)}")
        return []

def create_embeddings_chroma(chunks, persist_directory='./chroma_db') -> Optional[Chroma]:
    """
    Create and persist embeddings in ChromaDB.
    
    Args:
        chunks: Document chunks to embed
        persist_directory: Directory to save the embeddings
        
    Returns:
        ChromaDB vector store or None if embeddings failed
    """
    try:
        if not chunks or len(chunks) == 0:
            st.error("No chunks to embed. Please check your document.")
        return None

        # Create a container for embedding process info
        embedding_container = st.container()
        with embedding_container:
            st.info("Starting embedding process...")
            
            # Show progress bar during embedding
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Calculate approx. time (rough estimate)
            estimated_time = len(chunks) * 0.5  # Assume ~0.5s per chunk
            status_text.text(f"Creating embeddings for {len(chunks)} chunks (est. {estimated_time:.1f}s)...")
            
            # Show initial progress
            progress_bar.progress(0.1)
        
        # Log to help debug
        st.session_state.last_action = "Creating vector store"
        
        # Try using OpenAI embeddings first
        try:
            with embedding_container:
                st.text("Using OpenAI embeddings...")
                progress_bar.progress(0.3)
            
            # Get API key from session state or environment
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                with embedding_container:
                    st.error("No OpenAI API key found!")
                    return None
            
            # Create embeddings
            embeddings = OpenAIEmbeddings()
            
            # Create vector store
            with embedding_container:
                progress_bar.progress(0.6)
                status_text.text("Creating vector database...")
            
            vector_store = Chroma.from_documents(
                documents=chunks, 
                embeddings=embeddings,
                persist_directory=persist_directory
            )
            
            # Update progress and status
            with embedding_container:
                progress_bar.progress(1.0)
                status_text.text("Embeddings created successfully!")
                time.sleep(0.5)
            
            # Explicitly set vector store in session state
            st.session_state["vector_store"] = vector_store
            
            # Log successful creation to session state for debugging
            st.session_state["last_action"] = "Embeddings created successfully"
            
            with embedding_container:
                st.success("Vector store saved to session state successfully!")
                st.json({"status": "success", "chunks_processed": len(chunks)})
            
            # Debug information
            st.write("Debug: Vector store created and saved to session state")
            st.write(f"Session state keys: {list(st.session_state.keys())}")
            
            return vector_store
            
        except Exception as inner_e:
            with embedding_container:
                progress_bar.empty()
                status_text.empty()
                st.error(f"OpenAI API Error: {str(inner_e)}")
                st.session_state.last_error = f"OpenAI API Error: {str(inner_e)}"
                st.warning("Please check your OpenAI API key in the sidebar and try again.")
                
                # Debug info for API issues
                st.write("API Debug Information:")
                st.write(f"API key configured: {'Yes' if api_key else 'No'}")
                st.write(f"API key length: {len(api_key) if api_key else 0}")
                st.write(f"Error type: {type(inner_e).__name__}")
            
            return None
        
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        st.session_state.last_error = str(e)
        st.warning("If you're seeing OpenAI API errors, please verify your API key is correct and has sufficient credits.")
        
        # More detailed error debugging
        st.write("General Error Information:")
        st.write(f"Error type: {type(e).__name__}")
        st.write(f"Error details: {str(e)}")
        
        return None

def load_embeddings_chroma(persist_directory='./chroma_db') -> Optional[Chroma]:
    """
    Load embeddings from ChromaDB.
    
    Args:
        persist_directory: Directory where embeddings are stored
        
    Returns:
        ChromaDB vector store or None if loading failed
    """
    try:
        if not os.path.exists(persist_directory):
            st.warning(f"No embeddings found at {persist_directory}. Please create embeddings first.")
            return None
            
    embeddings = OpenAIEmbeddings()
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vector_store
        
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return None

# Main title with styled header
st.markdown('<h1 class="main-header">Q&A on Documents with AI</h1>', unsafe_allow_html=True)

# Debug section (hidden in collapsed section)
with st.expander("Debug Information", expanded=False):
    st.markdown("### Session State Debug")
    
    # Show all session state keys
    st.write("**Session State Keys:**", list(st.session_state.keys()))
    
    # Show specific debug info
    if st.session_state.get("last_action"):
        st.info(f"Last action: {st.session_state.last_action}")
    if st.session_state.get("last_error"):
        st.error(f"Last error: {st.session_state.last_error}")
    if st.session_state.get("vector_store"):
        st.success("Vector store is available in session state")
    else:
        st.warning("No vector store in session state")
    
    # Show API key status
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        st.success(f"API key configured (length: {len(api_key)})")
    else:
        st.error("No API key found in environment")
    
    # Show chunks info
    if st.session_state.get("chunks_count"):
        st.info(f"Chunks count: {st.session_state.chunks_count}")
    
    # Debug buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Clear Error Log"):
            st.session_state.last_error = None
            st.rerun()
    with col2:
        if st.button("Reset Vector Store"):
            st.session_state.vector_store = None
            st.rerun()
    with col3:
        if st.button("Test API Key"):
            try:
                from openai import OpenAI
                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello!"}],
                    max_tokens=5
                )
                st.success("API key works!")
            except Exception as e:
                st.error(f"API key test failed: {str(e)}")

# Create tabs for different functionalities
tab_options = ["Document Upload", "Wikipedia Search", "Chat"]
tab1, tab2, tab3 = st.tabs(tab_options)

# Function to switch active tab
def set_active_tab(tab_index):
    st.session_state.active_tab = tab_index

# Safe rerun function to handle version differences
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            st.write("Please refresh the page manually.")

# File Upload Tab
with tab1:
st.subheader("Upload Your Document")
    uploaded_file = st.file_uploader(
        "Choose a file (PDF, DOCX, TXT, CSV)", 
        type=["pdf", "docx", "txt", "csv"]
    )
    
    col1, col2 = st.columns(2)
    
if uploaded_file:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "File type": uploaded_file.type
        }
        
        with col1:
            st.write("**File Details:**")
            for k, v in file_details.items():
                st.write(f"- {k}: {v}")
        
        # Process document button
        process_doc = st.button("Process Document", key="process_doc")
        
        if process_doc:
    with st.spinner("Loading document..."):
        data = load_document(uploaded_file)
        if data:
                    st.success(f"Document loaded successfully! Found {len(data)} pages/sections.")
                    
                    # Get chunk parameters from sidebar
                    chunks = chunk_data(
                        data, 
                        chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap
                    )
                    
                    if chunks:
            st.write(f"Document split into {len(chunks)} chunks.")

                        # Display sample chunks
                        with st.expander("View sample chunks"):
                            sample_size = min(3, len(chunks))
                            for i in range(sample_size):
                                st.markdown(f"**Chunk {i+1}**")
                                st.text(chunks[i].page_content[:200] + "...")
                                
        # Create embeddings button with better styling
        create_embeddings_col1, create_embeddings_col2 = st.columns([3, 1])
        
        with create_embeddings_col1:
            create_embeddings_btn = st.button(
                "Create Embeddings", 
                key="create_doc_embeddings",
                help="Process document and create vector embeddings for Q&A"
            )
                        
                        # Create embeddings button with immediate feedback
                        if create_embeddings_btn:
                            # Immediate feedback
                            st.markdown("### Processing Request...")
                            st.write("Button clicked successfully!")
                            
                            # Create a status container
                            status_container = st.container()
                            
                            with status_container:
                                st.info("Storing document chunks...")
                                # Store chunks in session state for reference
                                st.session_state["current_chunks"] = chunks
                                st.session_state["chunks_count"] = len(chunks)
                                st.success(f"Stored {len(chunks)} chunks in session state")
                                
                                st.info("Checking API key...")
                                api_key = os.environ.get("OPENAI_API_KEY", "")
                                if api_key:
                                    st.success("API key found")
                                else:
                                    st.error("No API key found!")
                                    st.stop()
                                
                                st.info("Creating embeddings...")
                                
                                # Try to create embeddings with detailed error handling
                                try:
                                    # Create embeddings object
                                    embeddings = OpenAIEmbeddings()
                                    st.success("Embeddings object created")
                                    
                                    # Create vector store
                                    st.info("Building vector database...")
                                    vector_store = Chroma.from_documents(
                                        documents=chunks, 
                                        embeddings=embeddings,
                                        persist_directory='./chroma_db'
                                    )
                                    st.success("Vector store created successfully!")
                                    
                                    # Save to session state
                                    st.info("Saving to session state...")
                                    st.session_state["vector_store"] = vector_store
                                    st.session_state["last_action"] = "Document embeddings created successfully"
                                    st.success("Saved to session state!")
                                    
                                    # Verify session state
                                    if "vector_store" in st.session_state and st.session_state["vector_store"] is not None:
                                        st.success("Session state verification: Vector store is present!")
                                        
                                        # Success message
                                        st.markdown(
                                            '<div class="success-box">Embeddings created and stored successfully!</div>',
                                            unsafe_allow_html=True
                                        )
                                        st.balloons()
                                        
                                        # Navigation button
                                        if st.button("Go to Chat Tab Now", key="go_to_chat_from_doc"):
                                            set_active_tab(2)
                                            safe_rerun()
                                        else:
                                            st.markdown(
                                                '<div class="info-box">You can now ask questions about your document in the Chat tab!</div>',
                                                unsafe_allow_html=True
                                            )
                                    else:
                                        st.error("Session state verification failed!")
                                        
                                except Exception as e:
                                    st.error(f"Error during embedding creation: {str(e)}")
                                    st.write(f"Error type: {type(e).__name__}")
                                    
                                    # Try alternative approach
                                    st.info("Trying alternative approach...")
                                    try:
                                        # Alternative: Create without persist directory
                                        vector_store = Chroma.from_documents(
                                            documents=chunks, 
                                            embeddings=embeddings
                                        )
                                        st.session_state["vector_store"] = vector_store
                                        st.success("Alternative approach successful!")
                                    except Exception as e2:
                                        st.error(f"Alternative approach also failed: {str(e2)}")
                                        st.write("Please check your OpenAI API key and try again.")

# Wikipedia Search Tab
with tab2:
st.subheader("Search Wikipedia")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        wikipedia_query = st.text_input("Enter a Wikipedia topic to search:", key="wiki_query")
    
    with col2:
        wiki_max_docs = st.number_input("Max articles", min_value=1, max_value=5, value=2)
    
if wikipedia_query:
        search_wiki = st.button("Search Wikipedia", key="search_wiki")
        
        if search_wiki:
    with st.spinner("Fetching data from Wikipedia..."):
                try:
                    loader = WikipediaLoader(
                        query=wikipedia_query, 
                        load_max_docs=wiki_max_docs,
                        lang="en"
                    )
        wiki_data = loader.load()
                    
                    if wiki_data:
                        st.success(f"Found {len(wiki_data)} Wikipedia articles.")
                        
                        # Get chunk parameters from sidebar
                        wiki_chunks = chunk_data(
                            wiki_data, 
                            chunk_size=chunk_size, 
                            chunk_overlap=chunk_overlap
                        )
                        
                        if wiki_chunks:
                            st.write(f"Content split into {len(wiki_chunks)} chunks.")
                            
                            # Display sample chunks
                            with st.expander("View sample chunks"):
                                sample_size = min(3, len(wiki_chunks))
                                for i in range(sample_size):
                                    st.markdown(f"**Chunk {i+1}**")
                                    st.text(wiki_chunks[i].page_content[:200] + "...")
                            
                            # Create embeddings button with better styling
                            wiki_btn_col1, wiki_btn_col2 = st.columns([3, 1])
                            
                            with wiki_btn_col1:
                                create_wiki_embeddings_btn = st.button(
                                    "Create Wikipedia Embeddings", 
                                    key="create_wiki_embeddings",
                                    help="Process Wikipedia articles and create vector embeddings for Q&A"
                                )
                            
                            # Create embeddings button for Wikipedia with immediate feedback
                            if create_wiki_embeddings_btn:
                                # Immediate feedback
                                st.markdown("### Processing Wikipedia Request...")
                                st.write("Wikipedia button clicked successfully!")
                                
                                # Create a status container
                                wiki_status_container = st.container()
                                
                                with wiki_status_container:
                                    st.info("Storing Wikipedia chunks...")
                                    # Store chunks in session state for reference
                                    st.session_state["current_chunks"] = wiki_chunks
                                    st.session_state["chunks_count"] = len(wiki_chunks)
                                    st.success(f"Stored {len(wiki_chunks)} Wikipedia chunks in session state")
                                    
                                    st.info("Checking API key...")
                                    api_key = os.environ.get("OPENAI_API_KEY", "")
                                    if api_key:
                                        st.success("API key found")
                                    else:
                                        st.error("No API key found!")
                                        st.stop()
                                    
                                    st.info("Creating Wikipedia embeddings...")
                                    
                                    # Try to create embeddings with detailed error handling
                                    try:
                                        # Create embeddings object
                                        embeddings = OpenAIEmbeddings()
                                        st.success("Embeddings object created")
                                        
                                        # Create vector store
                                        st.info("Building Wikipedia vector database...")
                                        vector_store = Chroma.from_documents(
                                            documents=wiki_chunks, 
                                            embeddings=embeddings,
                                            persist_directory='./chroma_db'
                                        )
                                        st.success("Wikipedia vector store created successfully!")
                                        
                                        # Save to session state
                                        st.info("Saving to session state...")
                                        st.session_state["vector_store"] = vector_store
                                        st.session_state["last_action"] = "Wikipedia embeddings created successfully"
                                        st.success("Saved to session state!")
                                        
                                        # Verify session state
                                        if "vector_store" in st.session_state and st.session_state["vector_store"] is not None:
                                            st.success("Session state verification: Vector store is present!")
                                            
                                            # Success message
                                            st.markdown(
                                                '<div class="success-box">Wikipedia embeddings created and stored successfully!</div>',
                                                unsafe_allow_html=True
                                            )
                                            st.balloons()
                                            
                                            # Navigation button
                                            if st.button("Go to Chat Tab Now", key="go_to_chat_from_wiki"):
                                                set_active_tab(2)
                                                safe_rerun()
                                            else:
                                                st.markdown(
                                                    '<div class="info-box">You can now ask questions about Wikipedia content in the Chat tab!</div>',
                                                    unsafe_allow_html=True
                                                )
                                        else:
                                            st.error("Session state verification failed!")
                                            
                                    except Exception as e:
                                        st.error(f"Error during Wikipedia embedding creation: {str(e)}")
                                        st.write(f"Error type: {type(e).__name__}")
                                        
                                        # Try alternative approach
                                        st.info("Trying alternative approach...")
                                        try:
                                            # Alternative: Create without persist directory
                                            vector_store = Chroma.from_documents(
                                                documents=wiki_chunks, 
                                                embeddings=embeddings
                                            )
                                            st.session_state["vector_store"] = vector_store
                                            st.success("Alternative approach successful!")
                                        except Exception as e2:
                                            st.error(f"Alternative approach also failed: {str(e2)}")
                                            st.write("Please check your OpenAI API key and try again.")
                    else:
                        st.error(f"No articles found for '{wikipedia_query}'.")
                
                except Exception as e:
                    st.error(f"Error fetching Wikipedia data: {str(e)}")
                    st.info("Try a different search term or check your internet connection.")

# Chat Tab
with tab3:
    st.markdown('<h2 class="sub-header">Chat with your Documents</h2>', unsafe_allow_html=True)
    
    if st.session_state.vector_store is None:
        # Show professional looking message when no embeddings are available
        st.markdown(
            """
            <div class="warning-box">
                <h3>No Document Embeddings Available</h3>
                <p>Please complete one of these steps first:</p>
                <ol>
                    <li>Upload and process a document in the <b>Document Upload</b> tab</li>
                    <li>Search and process Wikipedia content in the <b>Wikipedia Search</b> tab</li>
                </ol>
                <p>After creating embeddings, return here to ask questions!</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Quick navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Document Upload", key="goto_doc_upload"):
                set_active_tab(0)
                safe_rerun()
        with col2:
            if st.button("Go to Wikipedia Search", key="goto_wiki_search"):
                set_active_tab(1)
                safe_rerun()
    else:
        # Setup retriever with k value from sidebar
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": k_value}
        )
        
        # Setup the QA chain
        with st.spinner("Setting up the chat system..."):
    qa_chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(
                    model=model_name,
                    temperature=temperature
                ),
        retriever=retriever,
        memory=st.session_state.memory,
                return_source_documents=True,
                verbose=True
            )
        
        # Professional chat interface
        st.markdown(
            '<div style="margin: 15px 0;">Ask a question about your documents:</div>', 
            unsafe_allow_html=True
        )
        
        # Input field with larger, more prominent styling
        question = st.text_input(
            "",  # No label as we use markdown above
            key="question_input",
            placeholder="Type your question here and press Enter...",
            help="You can ask any question related to your uploaded document or Wikipedia content"
        )
        
        # Special commands info
        with st.expander("Special Commands", expanded=False):
            st.markdown("""
            - Type **clear** or **reset chat** to clear the conversation history
            - Type **debug** to show debugging information
            """)
        
    if question:
            # Handle special commands
            if question.lower().strip() in ["clear", "clear chat", "reset", "reset chat"]:
                st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                st.session_state.chat_history = []
                st.markdown('<div class="success-box">Chat history cleared!</div>', unsafe_allow_html=True)
            elif question.lower().strip() == "debug":
                st.session_state.last_action = "Debug command triggered"
                st.json({
                    "vector_store_exists": st.session_state.vector_store is not None,
                    "chat_history_length": len(st.session_state.chat_history),
                    "token_usage": st.session_state.token_count,
                    "last_action": st.session_state.last_action,
                    "last_error": st.session_state.last_error
                })
            else:
                # Process regular questions
                with st.spinner("Thinking..."):
                    with get_openai_callback() as cb:
                        try:
            result = qa_chain({"question": question})
                            
                            # Update token count
                            st.session_state.token_count["prompt_tokens"] += cb.prompt_tokens
                            st.session_state.token_count["completion_tokens"] += cb.completion_tokens
                            st.session_state.token_count["total_tokens"] += cb.total_tokens
                            
                            # Display answer in a nicer format
                            st.markdown('<div style="background-color: #f0f7ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1E88E5; margin: 10px 0;">', unsafe_allow_html=True)
                            st.markdown(f'<h3 style="color: #1E88E5;">Answer</h3>', unsafe_allow_html=True)
                            st.markdown(result["answer"])
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Add to chat history
                            st.session_state.chat_history.append({"question": question, "answer": result["answer"]})
                            
                            # Show source documents with better styling
                            st.markdown('<h3 style="color: #1976D2; margin-top: 20px;">Sources</h3>', unsafe_allow_html=True)
                            source_docs = result["source_documents"]
                            
                            # Display unique source docs in expandable sections
                            unique_sources = {}
                            for i, doc in enumerate(source_docs):
                                # Create a simple hash for content to identify duplicates
                                content_hash = hash(doc.page_content[:100])
                                if content_hash not in unique_sources:
                                    unique_sources[content_hash] = doc
                            
                            for i, doc in enumerate(unique_sources.values()):
                                with st.expander(f"Source {i+1}"):
                                    st.markdown(f"**Content:**")
                                    st.markdown(doc.page_content)
                                    st.markdown("**Metadata:**")
                                    st.json(doc.metadata)
                            
                        except Exception as e:
                            st.error(f"Error generating answer: {str(e)}")
        
        # Display chat history
        if st.session_state.chat_history:
            with st.expander("Chat History", expanded=False):
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    st.markdown(f"**Question {len(st.session_state.chat_history) - i}:** {chat['question']}")
                    st.markdown(f"**Answer:** {chat['answer']}")
                    st.markdown("---")
            
            # Display token usage
            with st.expander("Token Usage", expanded=False):
                st.markdown(f"**Prompt tokens:** {st.session_state.token_count['prompt_tokens']}")
                st.markdown(f"**Completion tokens:** {st.session_state.token_count['completion_tokens']}")
                st.markdown(f"**Total tokens:** {st.session_state.token_count['total_tokens']}")
                
                # Rough cost estimation (approximate)
                if model_name == "gpt-3.5-turbo":
                    cost_per_1k_prompt = 0.0015
                    cost_per_1k_completion = 0.002
                else:  # gpt-4
                    cost_per_1k_prompt = 0.03
                    cost_per_1k_completion = 0.06
                    
                prompt_cost = (st.session_state.token_count['prompt_tokens'] / 1000) * cost_per_1k_prompt
                completion_cost = (st.session_state.token_count['completion_tokens'] / 1000) * cost_per_1k_completion
                total_cost = prompt_cost + completion_cost
                
                st.markdown(f"**Estimated cost:** ${total_cost:.4f}")
                
                if st.button("Reset Token Counter"):
                    st.session_state.token_count = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    st.success("Token counter reset!")
