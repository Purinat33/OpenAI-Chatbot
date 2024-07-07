import streamlit as st
import os
import openai

# LlamaIndex Stuff
from llama_index.core import SimpleDirectoryReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from llama_index.core import Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, ServiceContext
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI

load_dotenv()

cache_folder = './cache/'
persist_dir = './persist/'
document_folder = './docs/'

# Embedding Model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

store = LocalFileStore(cache_folder)
embedder = CacheBackedEmbeddings.from_bytes_store(
    hf, store, namespace=model_name
)


# OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
st.header("OpenAI Chatbot")

openai_model = 'gpt-3.5-turbo'
# Set global setting for llamaindex
Settings.embed_model = LangchainEmbedding(embedder)
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=16)
Settings.llm = OpenAI(model=openai_model)

# Session state to keep track of history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me something about cell biology or PyGame!"}
    ]


@st.cache_resource(show_spinner=False)
def load_content():
    with st.spinner(text="Loading and indexing data. Hang Tight!"):
        # Storage
        if os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0:
            storage_context = StorageContext.from_defaults(
                persist_dir=persist_dir)
            index = load_index_from_storage(storage_context=storage_context)
        else:
            documents = SimpleDirectoryReader(document_folder).load_data()
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=persist_dir)
        return index


index = load_content()

# Create chat engine
chat_engine = index.as_chat_engine(chat_mode='condense_question', verbose=True)
if prompt := st.chat_input("Your Question"):
    st.session_state.messages.append({'role': "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])
        
if st.session_state.messages[-1]['role'] != 'assistant':
    with st.chat_message("assistant"):
        with st.spinner('Thinking...'):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {'role': "assistant", 'content': response.response}
            st.session_state.messages.append(message)
