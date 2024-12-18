import sys
import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import chromadb

# Streamlit App Title
st.title("Q&A on Documents and Wikipedia with LangChain & ChromaDB")

# Set OpenAI API Key
if "openai_api_key" not in st.secrets:
    st.error("Please add your OpenAI API key to Streamlit secrets to proceed.")
    st.stop()
os.environ["OPENAI_API_KEY"] = ""

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

@st.cache_resource
def create_embeddings_chroma(_chunks, persist_directory='./chroma_db'):
    """Create and persist embeddings in ChromaDB."""
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection("my_collection")
    
    documents = [chunk.page_content for chunk in _chunks]
    metadatas = [chunk.metadata for chunk in _chunks]
    ids = [str(i) for i in range(len(_chunks))]
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    return collection

def load_embeddings_chroma(persist_directory='./chroma_db'):
    """Load embeddings from ChromaDB."""
    client = chromadb.PersistentClient(path=persist_directory)
    return client.get_collection("my_collection")

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

            # Embedding Creation
            if st.button("Create Embeddings"):
                with st.spinner("Creating embeddings..."):
                    collection = create_embeddings_chroma(chunks)
                    st.session_state.collection = collection
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
                collection = create_embeddings_chroma(wiki_chunks)
                st.session_state.collection = collection
                st.success("Embeddings for Wikipedia data created and stored!")

# Conversational Q&A Section
st.subheader("Ask Questions")
if "collection" in st.session_state:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("my_collection")
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        client=client,
        collection_name=collection.name,
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
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


# st.write("---")
