import os
from langchain.llms import OpenAI
import streamlit as st 
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from apikey import apikey
from langchain.embeddings.openai import OpenAIEmbeddings

# Import vector store stuff
from langchain.agents.agent_toolkits import(
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)


# import API KEY
os.environ['OPENAI_API_KEY'] = apikey

llm = OpenAI(temperature=0.9)

# Create text input box for user
prompt = st.text_input('Input your prompt here')

# Create and load pdf loader
loader = PyPDFLoader('tesla_annual_report.pdf')
# Split pages from pdf
pages = loader.load_and_split(embedding_function=OpenAIEmbeddings())
# Load documents into vector database (ChromaDB)
store = Chroma.from_documents(pages, collection_name='tesla_annual_report')
# Create vectorstore info object (works as metadata repo)
vectorstore_info = VectorStoreInfo(
    name = "tesla_annual_report",
    description = "financial report on fiscal year for tesla 2022",
    vectorstore = store
)

# Convert document store into vector toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC (gives our model access to pdf)
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True 
)

# If the user hits enter 
if prompt:
    # response = llm(prompt)
    respone = agent_executor.run(prompt)
    st.write(response)

    # With a streamlit expander
    with st.expander('Document Similarity Search'):
        # searches for relevant passages based on input from user
        search = store.similarity_search_with_score(prompt)
        st.write(search[0][0].page_content)


