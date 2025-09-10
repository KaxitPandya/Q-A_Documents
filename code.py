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
from langchain.callbacks import get_openai_callback
import chromadb

# Streamlit App Title
st.title("Q&A on Documents with Wikipedia Search")

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "token_count" not in st.session_state:
    st.session_state.token_count = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# Sidebar configuration
with st.sidebar:
    st.subheader("Configuration")
    
    # API Key handling
    api_key_option = st.radio("OpenAI API Key", ["Use from Secrets", "Enter Manually"])
    
    if api_key_option == "Enter Manually":
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    else:
        if "openai_api_key" not in st.secrets:
            st.error("Please add your OpenAI API key to Streamlit secrets or enter it manually to proceed.")
            st.stop()
        os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
    
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
        # Show progress bar during embedding
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Calculate approx. time (rough estimate)
        estimated_time = len(chunks) * 0.5  # Assume ~0.5s per chunk
        status_text.text(f"Creating embeddings for {len(chunks)} chunks (est. {estimated_time:.1f}s)...")
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create vector store
        for i in range(1, 11):  # Show progress animation
            progress_bar.progress(i * 0.1)
            if i < 10:  # Don't sleep on the last iteration
                time.sleep(estimated_time / 10)
                
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=persist_directory
        )
        
        progress_bar.progress(1.0)
        status_text.text("Embeddings created successfully!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return vector_store
        
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
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

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📄 Document Upload", "🌐 Wikipedia Search", "💬 Chat"])

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
                                
                        # Create embeddings button
                        create_embeddings_btn = st.button("Create Embeddings", key="create_doc_embeddings")
                        
                        if create_embeddings_btn:
                            vector_store = create_embeddings_chroma(chunks)
                            if vector_store:
                                st.session_state.vector_store = vector_store
                                st.success("✅ Embeddings created and stored in ChromaDB!")
                                st.balloons()
                                
                                # Switch to Chat tab
                                st.info("You can now ask questions about your document in the Chat tab!")

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
                            
                            # Create embeddings button
                            create_wiki_embeddings_btn = st.button("Create Wikipedia Embeddings", key="create_wiki_embeddings")
                            
                            if create_wiki_embeddings_btn:
                                vector_store = create_embeddings_chroma(wiki_chunks)
                                if vector_store:
                                    st.session_state.vector_store = vector_store
                                    st.success("✅ Embeddings for Wikipedia data created and stored!")
                                    st.balloons()
                                    
                                    # Switch to Chat tab
                                    st.info("You can now ask questions about the Wikipedia content in the Chat tab!")
                    else:
                        st.error(f"No articles found for '{wikipedia_query}'.")
                
                except Exception as e:
                    st.error(f"Error fetching Wikipedia data: {str(e)}")
                    st.info("Try a different search term or check your internet connection.")

# Chat Tab
with tab3:
    st.subheader("Chat with your Documents")
    
    if st.session_state.vector_store is None:
        st.info("Please upload a document or search Wikipedia and create embeddings first.")
        st.markdown("""
        1. Go to the **Document Upload** tab to process your own files, or
        2. Use the **Wikipedia Search** tab to fetch information on a topic
        
        After processing data, return here to ask questions!
        """)
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
        
        # Chat interface
        question = st.text_input("Ask a question about your document:", key="question_input")
        
        if question:
            if question.lower().strip() in ["clear", "clear chat", "reset", "reset chat"]:
                st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
            else:
                with st.spinner("Generating answer..."):
                    with get_openai_callback() as cb:
                        try:
                            result = qa_chain({"question": question})
                            
                            # Update token count
                            st.session_state.token_count["prompt_tokens"] += cb.prompt_tokens
                            st.session_state.token_count["completion_tokens"] += cb.completion_tokens
                            st.session_state.token_count["total_tokens"] += cb.total_tokens
                            
                            # Display answer
                            st.markdown("### Answer:")
                            st.markdown(result["answer"])
                            
                            # Add to chat history
                            st.session_state.chat_history.append({"question": question, "answer": result["answer"]})
                            
                            # Show source documents
                            st.markdown("### Sources:")
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
