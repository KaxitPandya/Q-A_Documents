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
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
import chromadb
from datetime import datetime
import hashlib
import re
from typing import List, Any, Dict
import time
from functools import lru_cache
import pickle

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
    .confidence-high {
        color: #2e7d32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .confidence-low {
        color: #d32f2f;
        font-weight: bold;
    }
    .response-time {
        color: #666;
        font-size: 0.85em;
        font-style: italic;
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
if "loaded_sources" not in st.session_state:
    st.session_state.loaded_sources = []
if "retrieval_cache" not in st.session_state:
    st.session_state.retrieval_cache = {}
if "streaming_response" not in st.session_state:
    st.session_state.streaming_response = ""

# Custom Streaming Handler for Streamlit
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# App Title and Description
st.title("📚 Fast Document Q&A Assistant")
st.markdown("Upload documents or search Wikipedia to ask questions and get fast, accurate answers.")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key configuration (from secrets only)
    api_key = st.secrets.get("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✅ API Key loaded")
    else:
        st.error("❌ OpenAI API key not found!")
        st.markdown("""
        **To set up your API key:**
        
        1. Create `.streamlit/secrets.toml`:
        ```toml
        openai_api_key = "sk-..."
        ```
        
        2. Or set environment variable:
        ```bash
        export OPENAI_API_KEY="sk-..."
        ```
        """)
    
    st.divider()
    
    # Performance Mode
    st.subheader("⚡ Performance Settings")
    performance_mode = st.radio(
        "Speed Mode",
        ["Fast", "Balanced", "Accurate"],
        index=0,
        help="Fast: Quick responses with good accuracy\nBalanced: Medium speed and accuracy\nAccurate: Best accuracy but slower"
    )
    
    # Set parameters based on mode
    if performance_mode == "Fast":
        default_k = 3
        default_fetch_k = 10
        default_chunk_size = 600
        default_chunk_overlap = 100
        default_compression = False
        default_expansion = False
        default_model = "gpt-3.5-turbo"
    elif performance_mode == "Balanced":
        default_k = 5
        default_fetch_k = 15
        default_chunk_size = 700
        default_chunk_overlap = 150
        default_compression = False
        default_expansion = False
        default_model = "gpt-3.5-turbo-16k"
    else:  # Accurate
        default_k = 8
        default_fetch_k = 20
        default_chunk_size = 800
        default_chunk_overlap = 200
        default_compression = True
        default_expansion = True
        default_model = "gpt-3.5-turbo-16k"
    
    st.divider()
    
    # Model settings
    st.subheader("Model Configuration")
    model_name = st.selectbox(
        "Select Model",
        ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo-preview"],
        index=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo-preview"].index(default_model)
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1
    )
    
    # Retrieval settings
    st.subheader("Retrieval Settings")
    retrieval_mode = st.selectbox(
        "Retrieval Strategy",
        ["similarity", "mmr"],
        index=0 if performance_mode == "Fast" else 1
    )
    
    k_documents = st.slider(
        "Documents to retrieve",
        min_value=1,
        max_value=10,
        value=default_k
    )
    
    if retrieval_mode == "mmr":
        fetch_k = st.slider(
            "Fetch K (for MMR)",
            min_value=10,
            max_value=30,
            value=default_fetch_k
        )
    else:
        fetch_k = k_documents * 2
    
    chunk_size = st.slider(
        "Chunk Size",
        min_value=200,
        max_value=1500,
        value=default_chunk_size,
        step=100
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=300,
        value=default_chunk_overlap,
        step=50
    )
    
    # Advanced settings
    st.subheader("Advanced Options")
    use_compression = st.checkbox(
        "Contextual Compression",
        value=default_compression,
        help="Slower but more accurate"
    )
    
    use_query_expansion = st.checkbox(
        "Query Expansion",
        value=default_expansion,
        help="Slower but finds more relevant content"
    )
    
    use_cache = st.checkbox(
        "Enable Cache",
        value=True,
        help="Cache retrievals for faster repeated queries"
    )
    
    st.divider()
    
    # Session management
    st.subheader("Session Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.conversation_history = []
            st.session_state.retrieval_cache = {}
            if "memory" in st.session_state:
                st.session_state.memory.clear()
            st.success("Cleared!")
    
    with col2:
        if st.button("🔄 Reset All"):
            st.session_state.vector_store = None
            st.session_state.document_count = 0
            st.session_state.current_document = None
            st.session_state.loaded_sources = []
            st.session_state.retrieval_cache = {}
            st.success("Reset!")
    
    # Display stats
    st.divider()
    st.subheader("📊 Statistics")
    st.metric("Total Sources", len(st.session_state.loaded_sources))
    st.metric("Messages", len(st.session_state.conversation_history))
    st.metric("Cache Size", len(st.session_state.retrieval_cache))
    
    # Show loaded sources
    if st.session_state.loaded_sources:
        st.subheader("📚 Loaded Sources")
        for i, source in enumerate(st.session_state.loaded_sources):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"• {source['name']} ({source['chunks']} chunks)")
            with col2:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.loaded_sources.pop(i)
                    st.rerun()

# Custom Prompts (Simplified for speed)
QA_PROMPT = PromptTemplate(
    template="""Answer the question based on the context below. Be concise but accurate.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

QUERY_EXPANSION_PROMPT = PromptTemplate(
    template="""Rephrase this question in 2 different ways:
Question: {question}

Rephrased (one per line):""",
    input_variables=["question"]
)

# Helper Functions
def preprocess_text(text: str) -> str:
    """Clean and preprocess text for better chunking."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text.strip()

def get_file_hash(file_content):
    """Generate a hash for file content."""
    return hashlib.md5(file_content).hexdigest()

@lru_cache(maxsize=100)
def get_cached_embedding(text: str, model: str = "text-embedding-ada-002"):
    """Cache embeddings for repeated queries."""
    embeddings = OpenAIEmbeddings(model=model)
    return embeddings.embed_query(text)

def load_document(file):
    """Load document based on its file type."""
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
        
        # Light preprocessing
        for doc in documents:
            doc.page_content = preprocess_text(doc.page_content)
            doc.metadata.update({
                "source": file.name,
                "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        os.unlink(tmp_file_path)
        return documents
    
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None

def chunk_data(data: List[Document], chunk_size: int = 600, chunk_overlap: int = 100) -> List[Document]:
    """Split data into chunks efficiently."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(data)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    
    return chunks

def expand_query(query: str, llm) -> List[str]:
    """Expand the query for better retrieval."""
    if not use_query_expansion:
        return [query]
    
    try:
        expansion_chain = LLMChain(llm=llm, prompt=QUERY_EXPANSION_PROMPT)
        expanded = expansion_chain.run(question=query)
        alternatives = [q.strip() for q in expanded.split('\n') if q.strip()]
        return [query] + alternatives[:2]  # Limit to 2 alternatives for speed
    except:
        return [query]

def create_embeddings_chroma(chunks, collection_name="default", persist_directory='./chroma_db', add_to_existing=False):
    """Create embeddings with progress tracking."""
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            chunk_size=1000
        )
        
        if add_to_existing and st.session_state.vector_store is not None:
            # Add to existing vector store
            st.session_state.vector_store.add_documents(chunks)
            return st.session_state.vector_store
        else:
            # Create new vector store
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

def calculate_confidence(sources: List[Document], answer: str) -> float:
    """Quick confidence calculation."""
    if not sources:
        return 0.0
    
    # Simplified for speed
    answer_words = set(answer.lower().split()[:20])  # Check first 20 words only
    
    relevance_scores = []
    for doc in sources[:3]:  # Check only top 3 sources
        content_words = set(doc.page_content.lower().split()[:100])
        overlap = len(answer_words.intersection(content_words))
        relevance = overlap / len(answer_words) if answer_words else 0
        relevance_scores.append(relevance)
    
    return min(sum(relevance_scores) / len(relevance_scores), 1.0)

def format_conversation_history():
    """Format conversation history for display."""
    formatted_history = []
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            formatted_history.append(f'<div class="user-message">👤 **You:** {msg["content"]}</div>')
        else:
            confidence = msg.get("confidence", 0.5)
            confidence_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.4 else "confidence-low"
            confidence_emoji = "✅" if confidence > 0.7 else "⚠️" if confidence > 0.4 else "❌"
            
            response_time = msg.get("response_time", 0)
            time_str = f'<span class="response-time">Response time: {response_time:.1f}s</span>' if response_time > 0 else ""
            
            formatted_history.append(
                f'<div class="assistant-message">🤖 **Assistant:** {msg["content"]}<br>'
                f'<span class="{confidence_class}">Confidence: {confidence_emoji} {confidence:.0%}</span> {time_str}</div>'
            )
            
            if "sources" in msg and msg["sources"]:
                sources_html = '<div class="source-box">📌 **Sources:**<br>'
                for source in msg["sources"][:3]:  # Limit to 3 sources for display
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
        type=["pdf", "docx", "txt"]
    )
    
if uploaded_file:
        file_details = {
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.json(file_details)
        
        # Add option to append or replace
        if st.session_state.loaded_sources:
            add_mode = st.radio(
                "Add to existing sources or replace?",
                ["Add to existing", "Replace all"],
                index=0,
                key="doc_add_mode"
            )
            add_to_existing = (add_mode == "Add to existing")
        else:
            add_to_existing = False
        
        if st.button("📥 Process Document", type="primary"):
            with st.spinner("Processing..."):
                start_time = time.time()
                
                # Load document
        data = load_document(uploaded_file)
                
        if data:
                    # Chunk data
                    chunks = chunk_data(data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.info(f"📄 {len(chunks)} chunks created")
                    
                    # Create embeddings
                    collection_name = "combined_docs" if add_to_existing else f"doc_{get_file_hash(uploaded_file.getvalue())[:8]}"
                    vector_store = create_embeddings_chroma(chunks, collection_name=collection_name, add_to_existing=add_to_existing)
                    
                    if vector_store:
                st.session_state.vector_store = vector_store
                        
                        # Update sources tracking
                        if not add_to_existing:
                            st.session_state.loaded_sources = []
                        
                        st.session_state.loaded_sources.append({
                            "name": uploaded_file.name,
                            "type": "document",
                            "chunks": len(chunks),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        st.session_state.document_count = len(st.session_state.loaded_sources)
                        st.session_state.current_document = uploaded_file.name
                        st.session_state.retrieval_cache = {}  # Clear cache
                        
                        process_time = time.time() - start_time
                        st.success(f"✅ Processed in {process_time:.1f}s!")

with col2:
    st.header("🌐 Wikipedia Search")
    
    wikipedia_query = st.text_input("Enter a Wikipedia topic")
    
if wikipedia_query:
        # Add option to append or replace
        if st.session_state.loaded_sources:
            wiki_add_mode = st.radio(
                "Add to existing sources or replace?",
                ["Add to existing", "Replace all"],
                index=0,
                key="wiki_add_mode"
            )
            wiki_add_to_existing = (wiki_add_mode == "Add to existing")
        else:
            wiki_add_to_existing = False
    
    if wikipedia_query and st.button("🔍 Search", type="primary"):
        with st.spinner("Searching..."):
            try:
                start_time = time.time()
                loader = WikipediaLoader(query=wikipedia_query, load_max_docs=1)  # Reduced for speed
        wiki_data = loader.load()
                
                if wiki_data:
                    for doc in wiki_data:
                        doc.page_content = preprocess_text(doc.page_content)
                    
                    wiki_chunks = chunk_data(wiki_data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.info(f"📄 {len(wiki_chunks)} chunks created")
                    
                    collection_name = "combined_docs" if wiki_add_to_existing else f"wiki_{hashlib.md5(wikipedia_query.encode()).hexdigest()[:8]}"
                    vector_store = create_embeddings_chroma(wiki_chunks, collection_name=collection_name, add_to_existing=wiki_add_to_existing)
                    
                    if vector_store:
                st.session_state.vector_store = vector_store
                        
                        # Update sources tracking
                        if not wiki_add_to_existing:
                            st.session_state.loaded_sources = []
                        
                        st.session_state.loaded_sources.append({
                            "name": f"Wikipedia: {wikipedia_query}",
                            "type": "wikipedia",
                            "chunks": len(wiki_chunks),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        st.session_state.document_count = len(st.session_state.loaded_sources)
                        st.session_state.current_document = f"Wikipedia: {wikipedia_query}"
                        st.session_state.retrieval_cache = {}  # Clear cache
                        
                        process_time = time.time() - start_time
                        st.success(f"✅ Processed in {process_time:.1f}s!")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Q&A Section
st.divider()
st.header("💬 Ask Questions")

if st.session_state.vector_store and api_key:
    # Display all loaded sources
    if st.session_state.loaded_sources:
        sources_summary = ", ".join([s["name"] for s in st.session_state.loaded_sources])
        total_chunks = sum(s["chunks"] for s in st.session_state.loaded_sources)
        st.info(f"📚 **Loaded sources ({len(st.session_state.loaded_sources)})**: {sources_summary}")
        st.caption(f"Total chunks: {total_chunks}")
    
    # Initialize LLM with streaming
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        streaming=True
    )
    
    # Configure retriever
    search_kwargs = {"k": k_documents}
    if retrieval_mode == "mmr":
        search_kwargs["fetch_k"] = fetch_k
        search_kwargs["lambda_mult"] = 0.5
    
    base_retriever = st.session_state.vector_store.as_retriever(
        search_type=retrieval_mode,
        search_kwargs=search_kwargs
    )
    
    # Add compression only if enabled
    if use_compression:
        compressor = LLMChainExtractor.from_llm(llm)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    else:
        retriever = base_retriever
    
    # Initialize memory
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    # Question input
    question = st.text_area(
        "Your question:",
        placeholder="What would you like to know?",
        height=80
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        ask_button = st.button("🤔 Ask", type="primary")
    with col2:
        if st.button("🔄 Retry"):
            if st.session_state.conversation_history:
                for msg in reversed(st.session_state.conversation_history):
                    if msg["role"] == "user":
                        question = msg["content"]
                        ask_button = True
                        break
    
    if ask_button and question:
        start_time = time.time()
        
        # Create a placeholder for streaming
        response_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            try:
                # Check cache
                cache_key = hashlib.md5(f"{question}_{retrieval_mode}_{k_documents}".encode()).hexdigest()
                
                if use_cache and cache_key in st.session_state.retrieval_cache:
                    # Use cached results
                    relevant_docs = st.session_state.retrieval_cache[cache_key]
                    retrieval_time = 0
                else:
                    # Retrieve documents
                    retrieval_start = time.time()
                    
                    if use_query_expansion:
                        expanded_queries = expand_query(question, llm)
                        all_docs = []
                        for q in expanded_queries:
                            docs = retriever.get_relevant_documents(q)
                            all_docs.extend(docs)
                        
                        # Remove duplicates
                        seen = set()
                        relevant_docs = []
                        for doc in all_docs:
                            doc_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                            if doc_hash not in seen:
                                seen.add(doc_hash)
                                relevant_docs.append(doc)
                    else:
                        relevant_docs = retriever.get_relevant_documents(question)
                    
                    retrieval_time = time.time() - retrieval_start
                    
                    # Cache results
                    if use_cache:
                        st.session_state.retrieval_cache[cache_key] = relevant_docs
                
                # Create QA chain with streaming
                streaming_handler = StreamlitCallbackHandler(response_placeholder)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=ChatOpenAI(
                        model=model_name,
                        temperature=temperature,
                        streaming=True,
                        callbacks=[streaming_handler]
                    ),
        retriever=retriever,
        memory=st.session_state.memory,
                    chain_type="stuff",
                    return_source_documents=True,
                    combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )

                # Get answer
            result = qa_chain({"question": question})
                
                # Calculate total time
                total_time = time.time() - start_time
                
                # Calculate confidence
                confidence = calculate_confidence(
                    result.get("source_documents", []),
                    result["answer"]
                )
                
                # Store in history
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": question
                })
                
                # Extract sources
                sources = []
                if "source_documents" in result:
                    for doc in result["source_documents"][:5]:  # Limit sources
                        source_info = doc.metadata.get("source", "Unknown")
                        if "chunk_index" in doc.metadata:
                            source_info += f" (chunk {doc.metadata['chunk_index']})"
                        sources.append(source_info)
                
                sources = list(dict.fromkeys(sources))  # Remove duplicates
                
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": sources,
                    "confidence": confidence,
                    "response_time": total_time
                })
                
                # Show timing info
                if retrieval_time > 0:
                    st.info(f"⏱️ Retrieval: {retrieval_time:.1f}s | Total: {total_time:.1f}s")
                
                # Rerun to update display
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Try adjusting settings or rephrasing your question.")
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("📝 Conversation")
        
        # Export button
        if st.button("📥 Export"):
            export_text = ""
            for msg in st.session_state.conversation_history:
                if msg["role"] == "user":
                    export_text += f"Q: {msg['content']}\n"
                else:
                    export_text += f"A: {msg['content']}\n"
                    if "response_time" in msg:
                        export_text += f"Time: {msg['response_time']:.1f}s\n"
                    export_text += "\n"
            
            st.download_button(
                label="Download",
                data=export_text,
                file_name=f"qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        # Display history
        formatted_history = format_conversation_history()
        for msg in reversed(formatted_history[-10:]):  # Show last 10 messages
            st.markdown(msg, unsafe_allow_html=True)
    
else:
    if not api_key:
        st.warning("⚠️ Please set your OpenAI API key in Streamlit secrets or environment variables.")
    else:
        st.info("📤 Please upload a document or search Wikipedia to start.")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Optimized for Speed | Powered by LangChain & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
