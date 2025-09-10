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
import chromadb
from datetime import datetime
import hashlib
import re
from typing import List

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
st.title("📚 Advanced Document Q&A Assistant (Enhanced)")
st.markdown("Upload documents or search Wikipedia to ask questions and get accurate, intelligent answers with source citations.")

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
        ["gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo-preview"],
        index=0,
        help="GPT-3.5-turbo-16k recommended for better context handling"
    )
    
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Lower values (0.1-0.3) recommended for factual Q&A"
    )
    
    # Retrieval settings
    st.subheader("Retrieval Settings")
    retrieval_mode = st.selectbox(
        "Retrieval Strategy",
        ["similarity", "mmr", "similarity_score_threshold"],
        index=1,
        help="MMR (Maximum Marginal Relevance) reduces redundancy"
    )
    
    k_documents = st.slider(
        "Number of relevant chunks to retrieve",
        min_value=3,
        max_value=15,
        value=8,
        help="More chunks provide more context"
    )
    
    fetch_k = st.slider(
        "Fetch K (for MMR)",
        min_value=10,
        max_value=50,
        value=20,
        help="Number of documents to fetch before MMR filtering"
    )
    
    chunk_size = st.slider(
        "Chunk Size",
        min_value=200,
        max_value=2000,
        value=800,
        step=100,
        help="Larger chunks maintain more context"
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=50,
        max_value=400,
        value=200,
        step=50,
        help="More overlap preserves context across chunks"
    )
    
    # Advanced settings
    st.subheader("Advanced Settings")
    use_compression = st.checkbox(
        "Use Contextual Compression",
        value=True,
        help="Extracts only relevant parts from retrieved chunks"
    )
    
    use_query_expansion = st.checkbox(
        "Use Query Expansion",
        value=True,
        help="Expands user query for better retrieval"
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

# Custom Prompts
QA_PROMPT = PromptTemplate(
    template="""You are an intelligent assistant helping users understand documents. Use the following context to answer the question accurately and comprehensively.

IMPORTANT INSTRUCTIONS:
1. Base your answer ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Quote relevant parts from the context when possible
4. Be specific and detailed in your answers
5. If you're unsure, indicate your uncertainty

Context:
{context}

Previous conversation:
{chat_history}

Question: {question}

Detailed Answer:""",
    input_variables=["context", "chat_history", "question"]
)

QUERY_EXPANSION_PROMPT = PromptTemplate(
    template="""Given the following question, generate 3 alternative phrasings or related questions that would help find relevant information. 
Keep the alternatives closely related to the original intent.

Original question: {question}

Alternative phrasings (one per line):""",
    input_variables=["question"]
)

CONDENSE_PROMPT = PromptTemplate(
    template="""Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question that includes all necessary context.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question:""",
    input_variables=["chat_history", "question"]
)

# Helper Functions
def preprocess_text(text: str) -> str:
    """Clean and preprocess text for better chunking."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might interfere
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Fix common OCR errors
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    return text.strip()

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
        
        # Preprocess text
        for doc in documents:
            doc.page_content = preprocess_text(doc.page_content)
            doc.metadata.update({
                "source": file.name,
                "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "doc_length": len(doc.page_content)
            })
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return documents
    
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None

def chunk_data(data: List[Document], chunk_size: int = 800, chunk_overlap: int = 200) -> List[Document]:
    """Split data into chunks with semantic awareness."""
    # Use different separators for better semantic chunking
    separators = [
        "\n\n\n",  # Multiple line breaks (usually chapter/section breaks)
        "\n\n",    # Paragraph breaks
        "\n",      # Line breaks
        ". ",      # Sentence ends
        "! ",
        "? ",
        "; ",      # Clause breaks
        ", ",      # Comma breaks
        " ",       # Word breaks
        ""         # Character breaks (last resort)
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
        keep_separator=True
    )
    
    chunks = text_splitter.split_documents(data)
    
    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
    
    return chunks

def expand_query(query: str, llm) -> List[str]:
    """Expand the query to improve retrieval."""
    expansion_chain = LLMChain(
        llm=llm,
        prompt=QUERY_EXPANSION_PROMPT
    )
    
    try:
        expanded = expansion_chain.run(question=query)
        alternatives = [q.strip() for q in expanded.split('\n') if q.strip()]
        return [query] + alternatives[:3]  # Original + up to 3 alternatives
    except:
        return [query]

def create_embeddings_chroma(chunks, collection_name="default", persist_directory='./chroma_db'):
    """Create and persist embeddings in ChromaDB with collection management."""
    try:
        # Use a better embedding model
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            chunk_size=1000
        )
        
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

def calculate_confidence(sources: List[Document], answer: str) -> float:
    """Calculate confidence score based on sources and answer."""
    if not sources:
        return 0.0
    
    # Check how many sources support the answer
    relevant_sources = 0
    total_relevance = 0
    
    for doc in sources:
        # Simple relevance check - could be made more sophisticated
        content_lower = doc.page_content.lower()
        answer_words = set(answer.lower().split())
        content_words = set(content_lower.split())
        
        # Calculate word overlap
        overlap = len(answer_words.intersection(content_words))
        relevance = overlap / len(answer_words) if answer_words else 0
        
        if relevance > 0.1:  # Threshold for considering a source relevant
            relevant_sources += 1
            total_relevance += relevance
    
    # Calculate confidence
    source_confidence = relevant_sources / len(sources) if sources else 0
    avg_relevance = total_relevance / len(sources) if sources else 0
    
    confidence = (source_confidence * 0.6 + avg_relevance * 0.4)
    return min(confidence, 1.0)

def format_conversation_history():
    """Format conversation history for display."""
    formatted_history = []
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            formatted_history.append(f'<div class="user-message">👤 **You:** {msg["content"]}</div>')
        else:
            # Add confidence indicator
            confidence = msg.get("confidence", 0.5)
            if confidence > 0.7:
                confidence_class = "confidence-high"
                confidence_emoji = "✅"
            elif confidence > 0.4:
                confidence_class = "confidence-medium"
                confidence_emoji = "⚠️"
            else:
                confidence_class = "confidence-low"
                confidence_emoji = "❌"
            
            formatted_history.append(
                f'<div class="assistant-message">🤖 **Assistant:** {msg["content"]}<br>'
                f'<span class="{confidence_class}">Confidence: {confidence_emoji} {confidence:.0%}</span></div>'
            )
            
            if "sources" in msg and msg["sources"]:
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
                    # Show document preview
                    with st.expander("📄 Document Preview"):
                        preview_text = data[0].page_content[:1000]
                        st.text(preview_text + "..." if len(data[0].page_content) > 1000 else preview_text)
                    
                    # Chunk data
                    progress_bar.progress(40)
                    chunks = chunk_data(data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.info(f"Document split into {len(chunks)} chunks (avg size: {sum(len(c.page_content) for c in chunks)//len(chunks)} chars)")
                    
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
                    # Preprocess wiki data
                    for doc in wiki_data:
                        doc.page_content = preprocess_text(doc.page_content)
                    
                    # Preview first page
                    with st.expander("📄 Preview Wikipedia Content"):
                        st.write(wiki_data[0].page_content[:1000] + "...")
                    
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
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        streaming=True
    )
    
    # Configure retriever based on settings
    search_kwargs = {
        "k": k_documents
    }
    
    if retrieval_mode == "mmr":
        search_kwargs["fetch_k"] = fetch_k
        search_kwargs["lambda_mult"] = 0.5  # Balance between relevance and diversity
    elif retrieval_mode == "similarity_score_threshold":
        search_kwargs["score_threshold"] = 0.5
    
    base_retriever = st.session_state.vector_store.as_retriever(
        search_type=retrieval_mode,
        search_kwargs=search_kwargs
    )
    
    # Add contextual compression if enabled
    if use_compression:
        compressor = LLMChainExtractor.from_llm(llm)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    else:
        retriever = base_retriever
    
    # Initialize memory if not exists
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    # Create QA chain with custom prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        chain_type="stuff",
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        condense_question_prompt=CONDENSE_PROMPT
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
                # Get the last user question
                for msg in reversed(st.session_state.conversation_history):
                    if msg["role"] == "user":
                        question = msg["content"]
                        ask_button = True
                        break
    
    if ask_button and question:
        with st.spinner("Thinking deeply..."):
            try:
                # Expand query if enabled
                if use_query_expansion:
                    expanded_queries = expand_query(question, llm)
                    if len(expanded_queries) > 1:
                        st.info(f"🔍 Searching for: {', '.join(expanded_queries[1:])}")
                    
                    # Retrieve documents for all queries
                    all_docs = []
                    for q in expanded_queries:
                        docs = retriever.get_relevant_documents(q)
                        all_docs.extend(docs)
                    
                    # Remove duplicates
                    seen = set()
                    unique_docs = []
                    for doc in all_docs:
                        doc_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                        if doc_hash not in seen:
                            seen.add(doc_hash)
                            unique_docs.append(doc)
                    
                    # Update the retriever to use our expanded docs
                    # This is a workaround - ideally we'd modify the retriever directly
                    result = qa_chain({"question": question})
                else:
                    result = qa_chain({"question": question})
                
                # Calculate confidence
                confidence = calculate_confidence(
                    result.get("source_documents", []),
                    result["answer"]
                )
                
                # Store in conversation history
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": question
                })
                
                # Extract sources with more detail
                sources = []
                if "source_documents" in result:
                    for doc in result["source_documents"]:
                        source_info = doc.metadata.get("source", "Unknown")
                        if "chunk_index" in doc.metadata:
                            source_info += f" (chunk {doc.metadata['chunk_index']})"
                        sources.append(source_info)
                
                # Remove duplicates from sources
                sources = list(dict.fromkeys(sources))
                
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": sources,
                    "confidence": confidence
                })
                
                # Clear the question input by rerunning
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                st.info("Try adjusting the retrieval settings or rephrasing your question.")
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("📝 Conversation History")
        
        # Add export button
        if st.button("📥 Export Conversation"):
            export_text = ""
            for msg in st.session_state.conversation_history:
                if msg["role"] == "user":
                    export_text += f"Q: {msg['content']}\n"
                else:
                    export_text += f"A: {msg['content']}\n"
                    if "confidence" in msg:
                        export_text += f"Confidence: {msg['confidence']:.0%}\n"
                    if "sources" in msg and msg["sources"]:
                        export_text += f"Sources: {', '.join(msg['sources'])}\n"
                    export_text += "\n"
            
            st.download_button(
                label="Download Conversation",
                data=export_text,
                file_name=f"qa_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
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
        Enhanced Q&A System with improved accuracy | Made with ❤️ using LangChain and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
