import os
from langchain.llms import OpenAI
import streamlit as st 
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma


# import API KEY
os.environ['OPENAI_API_KEY'] = 'apikey'

llm = OpenAI(temperature=0.9)

# Create text input box for user
prompt = st.text_input('Input your prompt here')

# Create and load pdf loader
loader = PyPDFLoader('FullTime_Resume.pdf')
# Split pages from pdf
pages = loader.load_and_split()
# Load documents into vector database (ChromaDB)
store = Chroma.from_documents(pages, collection_name='FullTime_Resume')


# If the user hits enter 
if prompt:
    response = llm(prompt)
    st.write(response)
