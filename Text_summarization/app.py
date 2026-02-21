import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


## Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')


## Sidebar
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")


prompt_template = """
Provide a concise summary (max 200 words) of the following content:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


if st.button("Summarize the Content from YT or Website"):

    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")

    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")

    else:
        try:
            with st.spinner("Processing..."):

                
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    groq_api_key=groq_api_key,
                    temperature=0,
                    max_tokens=300
                )

                # Load content
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=False   
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )

                docs = loader.load()

                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,   
                    chunk_overlap=200
                )

                split_docs = text_splitter.split_documents(docs)

                # 🔥 Use only first 2 chunks to reduce latency
                split_docs = split_docs[:2]

                # Summarization chain
                chain = load_summarize_chain(
                    llm,
                    chain_type="stuff",
                    prompt=prompt
                )

                output_summary = chain.run(split_docs)

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception:{e}")