import streamlit as st
import os
from dotenv import load_dotenv
from operator import itemgetter

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableWithMessageHistory,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

st.title("Conversational RAG With PDF + Ollama + Groq")
st.write("Upload PDFs and chat with their content")

api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:

    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.1-8b-instant"
    )

    embedding = OllamaEmbeddings(model="mxbai-embed-large")

    session_id = st.text_input("Session ID", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files:

        documents = []

        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp_{uploaded_file.name}"
            with open(temp_pdf, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs[:5])

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
        )

        splits = text_splitter.split_documents(documents)

        # FAISS vector store with Ollama embeddings
        vectorstore = FAISS.from_documents(splits, embedding)
        retriever = vectorstore.as_retriever()

        # ---------------------------
        # 1️⃣ HISTORY AWARE RETRIEVER
        # ---------------------------

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "Given chat history and the latest user question, "
                 "rewrite the question so it can be understood standalone. "
                 "Do NOT answer it."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_rewriter = (
            contextualize_q_prompt
            | llm
            | StrOutputParser()
        )

        history_aware_retriever = (
            {
                "input": RunnablePassthrough(),
                "chat_history": itemgetter("chat_history"),
            }
            | question_rewriter
            | retriever
        )

        # ---------------------------
        # 2️⃣ STUFF DOCUMENTS CHAIN
        # ---------------------------

        join_docs = RunnableLambda(
            lambda docs: "\n\n".join(doc.page_content for doc in docs)
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are an assistant for question-answering tasks. "
                 "Use the following context to answer. "
                 "If unknown, say you don't know.\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = (
            qa_prompt
            | llm
            | StrOutputParser()
        )

        # ---------------------------
        # 3️⃣ FULL RAG CHAIN
        # ---------------------------

        rag_chain = (
            {
                "context": history_aware_retriever | join_docs,
                "input": RunnablePassthrough(),
                "chat_history": itemgetter("chat_history"),
            }
            | question_answer_chain
        )

        # ---------------------------
        # 4️⃣ MEMORY WRAPPER
        # ---------------------------

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        user_input = st.text_input("Your question:")

        if user_input:

            session_history = get_session_history(session_id)

            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            st.write("Assistant:", response)
            st.write("Chat History:", session_history.messages)

else:
    st.warning("Please enter the Groq API Key")
