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
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import chromadb

# Streamlit App Title
st.title("Q&A on Documents with Wikipedia Search")

# Set OpenAI API Key
if "openai_api_key" not in st.secrets:
    st.error("Please add your OpenAI API key to Streamlit secrets to proceed.")
    st.stop()
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

# Helper Functions
def load_document(file):
    """Load document based on its file type."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    name, extension = os.path.splitext(file.name)
    if extension == ".pdf":
        loader = PyPDFLoader(tmp_file_path)
    elif extension == ".docx":
        loader = Docx2txtLoader(tmp_file_path)
    elif extension == ".txt":
        loader = TextLoader(tmp_file_path)
    else:
        st.error("Unsupported file format!")
        return None

    return loader.load()

def chunk_data(data, chunk_size=256):
    """Split data into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    return text_splitter.split_documents(data)

def create_embeddings_chroma(chunks, persist_directory='./chroma_db'):
    """Create and persist embeddings in ChromaDB."""
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    return vector_store

def load_embeddings_chroma(persist_directory='./chroma_db'):
    """Load embeddings from ChromaDB."""
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# File Upload Section
st.subheader("Upload Your Document")
uploaded_file = st.file_uploader("Choose a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
if uploaded_file:
    with st.spinner("Loading document..."):
        data = load_document(uploaded_file)
        if data:
            st.success("Document loaded successfully!")
            chunks = chunk_data(data)
            st.write(f"Document split into {len(chunks)} chunks.")

        if st.button("Create Embeddings"):
            with st.spinner("Creating embeddings..."):
                vector_store = create_embeddings_chroma(chunks)
                st.session_state.vector_store = vector_store
                st.success("Embeddings created and stored in ChromaDB!")


# Wikipedia Search Section
st.subheader("Search Wikipedia")
wikipedia_query = st.text_input("Enter a Wikipedia topic to search:")
if wikipedia_query:
    with st.spinner("Fetching data from Wikipedia..."):
        loader = WikipediaLoader(query=wikipedia_query, load_max_docs=2)
        wiki_data = loader.load()
        wiki_chunks = chunk_data(wiki_data)
        st.write(f"Fetched and split into {len(wiki_chunks)} chunks.")
        if st.button("Create Wikipedia Embeddings"):
            with st.spinner("Creating embeddings for Wikipedia data..."):
                vector_store = create_embeddings_chroma(wiki_chunks)
                st.session_state.vector_store = vector_store
                st.success("Embeddings for Wikipedia data created and stored!")

# Conversational Q&A Section
st.subheader("Ask Questions")
if "vector_store" in st.session_state:
    retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        retriever=retriever,
        memory=st.session_state.memory,
        chain_type="stuff"
    )

    question = st.text_input("Enter your question:")
    if question:
        with st.spinner("Generating answer..."):
            result = qa_chain({"question": question})
            st.write("**Answer:**", result["answer"])
            st.write("**Conversation History:**")
            for message in result["chat_history"]:
                st.write(message)
else:
    st.info("Please upload a document or fetch Wikipedia data to create embeddings first.")

