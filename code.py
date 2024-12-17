import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile

# Streamlit App
st.title("Q&A on Documents and Wikipedia with Chroma")

# API Key Setup
if "openai_api_key" not in st.secrets:
    st.error("Please add your OpenAI API key to Streamlit secrets to proceed.")
    st.stop()
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

# Helper Functions
def load_document(file):
    name, extension = os.path.splitext(file.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

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
    os.unlink(tmp_file_path)
    return data

def load_from_wikipedia(query, lang='en', load_max_docs=2):
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    return loader.load()

def chunk_data(data, chunk_size=256):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    return text_splitter.split_documents(data)

@st.cache_resource
def create_embeddings_chroma(chunks, persist_directory='./chroma_db'):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    return Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)

def ask_and_get_answer(vector_store, q, k=3):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return chain.invoke(q)

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
                    st.session_state.vector_store = vector_store
                    st.success("Embeddings created and stored in Chroma DB!")

# Wikipedia Search Section
st.subheader("Search Wikipedia")
wikipedia_query = st.text_input("Enter a Wikipedia topic to search:")
if wikipedia_query:
    with st.spinner("Fetching data from Wikipedia..."):
        wiki_data = load_from_wikipedia(wikipedia_query)
        wiki_chunks = chunk_data(wiki_data)
        st.write(f"Fetched and split into {len(wiki_chunks)} chunks.")
        if st.button("Create Wikipedia Embeddings"):
            with st.spinner("Creating Wikipedia embeddings..."):
                wiki_vector_store = create_embeddings_chroma(wiki_chunks, persist_directory='./wiki_chroma_db')
                st.session_state.wiki_vector_store = wiki_vector_store
                st.success("Wikipedia embeddings created and stored!")

# Conversational Q&A Section
st.subheader("Ask Questions")
if 'vector_store' in st.session_state or 'wiki_vector_store' in st.session_state:
    vector_store = st.session_state.get('vector_store') or st.session_state.get('wiki_vector_store')
    
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    crc = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model='gpt-3.5-turbo', temperature=0),
        retriever=vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5}),
        memory=st.session_state.memory,
        chain_type='stuff'
    )
    
    question = st.text_input("Enter your question:")
    if question:
        with st.spinner("Generating answer..."):
            result = crc.invoke({'question': question})
            st.write("**Answer:**", result['answer'])
            
            st.write("**Conversation History:**")
            for i, message in enumerate(st.session_state.memory.chat_memory.messages):
                if i % 2 == 0:
                    st.write("Human: ", message.content)
                else:
                    st.write("AI: ", message.content)
else:
    st.info("Please upload a document or search Wikipedia and create embeddings first.")

st.write("---")
st.info("This app supports PDF, DOCX, TXT files, and Wikipedia queries for Q&A.")
