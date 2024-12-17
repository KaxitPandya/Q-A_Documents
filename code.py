import streamlit as st
import pysqlite3
import sys

# Replace sqlite3 with pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os
from chromadb.config import Settings
import chromadb

client = chromadb.Client()


# Streamlit App
st.title("Q&A on Documents and Wikipedia with Chroma")

# API Key Setup
if "openai_api_key" not in st.secrets:
    st.error("Please add your OpenAI API key to Streamlit secrets to proceed.")
    st.stop()
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

# Helper Functions
def load_document(file):
    import tempfile

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    # Determine the loader based on file extension
    name, extension = os.path.splitext(file.name)
    if extension == '.pdf':
        loader = PyPDFLoader(tmp_file_path)
    elif extension == '.docx':
        loader = Docx2txtLoader(tmp_file_path)
    elif extension == '.txt':
        loader = TextLoader(tmp_file_path)
    else:
        st.error("Unsupported file format!")
        return None

    data = loader.load()
    return data

def chunk_data(data, chunk_size=256):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    return text_splitter.split_documents(data)

def create_embeddings_chroma(chunks, persist_directory='./chroma_db'):
    client = chromadb.Client()
    collection = client.create_collection("my_collection")
    
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [str(i) for i in range(len(chunks))]
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    return collection


def load_embeddings_chroma(persist_directory='./chroma_db'):
    chroma_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory
    )
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        client_settings=chroma_settings
    )

def ask_and_get_answer(collection, q, k=3):
    results = collection.query(
        query_texts=[q],
        n_results=k
    )
    
    # Process the results as needed
    # You may need to adjust this part based on your specific requirements
    return results


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
                    vector_store = create_embeddings_chroma(chunks)
                    st.success("Embeddings created and stored in Chroma DB!")

# Wikipedia Search Section
st.subheader("Search Wikipedia")
wikipedia_query = st.text_input("Enter a Wikipedia topic to search:")
if wikipedia_query:
    with st.spinner("Fetching data from Wikipedia..."):
        loader = WikipediaLoader(query=wikipedia_query, load_max_docs=2)
        wiki_data = loader.load()
        wiki_chunks = chunk_data(wiki_data)
        st.write(f"Fetched and split into {len(wiki_chunks)} chunks.")

# Conversational Q&A Section
st.subheader("Ask Questions")
if 'vector_store' in locals() or 'vector_store' in globals():
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    crc = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model='gpt-3.5-turbo', temperature=0),
        retriever=retriever,
        memory=memory,
        chain_type='stuff'
    )
    question = st.text_input("Enter your question:")
    if question:
        with st.spinner("Generating answer..."):
            result = crc.invoke({'question': question})
            st.write("**Answer:**", result['answer'])
            st.write("**Conversation History:**")
            for item in result['chat_history']:
                st.write(item)
else:
    st.info("Please upload a document and create embeddings first.")

st.write("---")
st.info("This app supports PDF, DOCX, TXT files, and Wikipedia queries for Q&A.")
