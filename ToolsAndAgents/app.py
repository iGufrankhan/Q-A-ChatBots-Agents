import streamlit as st  
import os 
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

load_dotenv()

# Tools setup
api_wiki_tools = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wiki_tools)

api_arvi_tools = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
ariv = ArxivQueryRun(api_wrapper=api_arvi_tools)

search = DuckDuckGoSearchRun()

st.title("Tools and Agents with Streamlit and Langchain")
st.sidebar.title("Settings")

groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I'm a chatbot who can search the web. How can I help you today?"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if groq_api_key:
        llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant", streaming=True)
        tools = [search, ariv, wiki]
        
        search_agent = initialize_agent(
            tools, 
            llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(prompt, callbacks=[st_cb])
            st.session_state.messages.append({'role': 'assistant', "content": response})
            st.write(response)
    else:
        st.error("Please enter your Groq API Key in the sidebar")







