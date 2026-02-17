import time
import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.

<context>
{context}
</context>

Question: {input}
"""
)


def create_vector_embedding():
    if "vector_store" not in st.session_state:
        embedding = OllamaEmbeddings(model="mxbai-embed-large")
        loader = PyPDFDirectoryLoader("research_paper")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        split_docs = text_splitter.split_documents(docs[:5])

        vector_store = FAISS.from_documents(split_docs, embedding)

        st.session_state.vector_store = vector_store
        st.session_state.retriever = vector_store.as_retriever()


st.title("RAG Document Q&A with Groq")

question = st.text_input("Enter your question here:")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.success("Document embedding created successfully!")

if question and "vector_store" in st.session_state:

    retriever = st.session_state.retriever

    rag_chain = (
        {
            "context": retriever,
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    start = time.process_time()
    answer = rag_chain.invoke({"input": question})
    end = time.process_time()

    st.write("### Answer")
    st.write(answer)
    st.write("Processing time:", end - start, "seconds")

    # Show retrieved documents inside expander
    retrieved_docs = retriever.invoke(question)

    with st.expander("Retrieved Documents"):
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"**Document {i+1}:**")
            st.write(doc.page_content)
            st.markdown("---")
