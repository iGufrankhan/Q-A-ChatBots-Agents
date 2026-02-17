import streamlit as st
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries clearly."),
        ("user", "Question: {question}")
    ]
)

# Function to generate response
def generate_response(question, api_key, model, temperature, max_tokens):

    # Set Google API key
    os.environ["GOOGLE_API_KEY"] = api_key

    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    answer = chain.invoke({"question": question})

    return answer


# Streamlit App
st.title("🤖 Q&A Chatbot With GenAi")

# Welcome banner at the top
st.markdown(
    """
    <h1 style='text-align: center; font-size: 3rem; font-weight: bold; margin-bottom: 1rem; color: #00BFFF; letter-spacing: 2px;'>
        Welcome to Tech_AI
    </h1>
    """,
    unsafe_allow_html=True
)

# Sidebar Settings
st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "Enter your Google API Key:",
    type="password"
)

# Model Selection
model = st.sidebar.selectbox(
    "Select Gemini Model",
    [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-001"
    ]
)


temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7
)

max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=50,
    max_value=2048,
    value=512
)

# Add custom CSS for chat bubbles
st.markdown("""
    <style>
    .chat-bubble {
        padding: 1rem;
        border-radius: 1.2rem;
        margin-bottom: 0.5rem;
        max-width: 80%;
        word-break: break-word;
    }
    .user-bubble {
        background-color: #DCF8C6;
        align-self: flex-end;
        margin-left: 20%;
    }
    .bot-bubble {
        background-color: #F1F0F0;
        align-self: flex-start;
        margin-right: 20%;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.write("how can we help you please provide context 👇")

# Chat input box at the bottom
user_input = st.chat_input("Type your question here...")

# Display chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div style="padding:1rem;border-radius:1.2rem;margin-bottom:0.5rem;max-width:80%;word-break:break-word;background-color:#DCF8C6;color:#222;align-self:flex-end;margin-left:20%;">{msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="padding:1rem;border-radius:1.2rem;margin-bottom:0.5rem;max-width:80%;word-break:break-word;background-color:#F1F0F0;color:#222;align-self:flex-start;margin-right:20%;">{msg["content"]}</div>',
            unsafe_allow_html=True
        )
st.markdown('</div>', unsafe_allow_html=True)

# Handle user input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    if api_key:
        response = generate_response(
            user_input,
            api_key,
            model,
            temperature,
            max_tokens
        )
        st.session_state.messages.append({"role": "bot", "content": response})
        st.rerun()
    else:
        st.warning("Please enter your Google API Key in the sidebar.")

elif user_input:
    st.warning("Please enter your Google API Key in the sidebar.")
