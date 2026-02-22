import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.llm_math.base import LLMMathChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Math & Data Assistant",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 AI Math & Reasoning Assistant")
st.divider()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("🔑 Settings")
    groq_api_key = st.text_input("Groq API Key", type="password")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key,
    temperature=0
)

# ---------------- TOOLS ----------------
wikipedia_wrapper = WikipediaAPIWrapper()

wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Search for general knowledge information."
)

math_chain = LLMMathChain.from_llm(llm=llm)

calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Solve mathematical expressions only."
)

prompt = """
You are an agent tasked with solving users' mathematical questions.
Logically arrive at the solution and provide a clear explanation.

Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="Solve logic-based reasoning problems."
)

assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# ---------------- CHAT MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hi! I'm your AI Math Assistant. Ask me any math question!"
        }
    ]

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- CHAT INPUT ----------------
user_input = st.chat_input("Type your question here...")

if user_input:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            st_cb = StreamlitCallbackHandler(
                st.container(),
                expand_new_thoughts=False
            )

           
            response = assistant_agent.run(
                user_input,
                callbacks=[st_cb]
            )

        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )