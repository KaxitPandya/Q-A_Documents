import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os

# Streamlit App
st.title("Q&A on Documents and Wikipedia with Chroma")

# API Key Setup
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
if not openai_api_key:
    st.warning("Please provide your OpenAI API key to proceed.")
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_api_key

# Helper Functions
def load_document(file):
    name, extension = os.path.splitext(file.name)
    if extension == '.pdf':
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        loader = TextLoader(file)
    else:
        st.error("Unsupported file format!")
        return None
    return loader.load()

def chunk_data(data, chunk_size=256):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    return text_splitter.split_documents(data)

def create_embeddings_chroma(chunks, persist_directory='./chroma_db'):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    return vector_store

def load_embeddings_chroma(persist_directory='./chroma_db'):
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def ask_and_get_answer(vector_store, q, k=3):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return chain.run(q)

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
