from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import re


class YouTubeRAGTool:
    def __init__(self):
        self.embedding = OllamaEmbeddings(model="nomic-embed-text")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.persist_directory = "./chroma_db"

    def extract_video_id(self, url):
        pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        raise ValueError("Invalid YouTube URL")

    def load_video(self, youtube_url):
        video_id = self.extract_video_id(youtube_url)

        ytt_api = YouTubeTranscriptApi()
        transcript_data = ytt_api.fetch(video_id)

        text = " ".join([entry.text for entry in transcript_data])

        documents = [Document(page_content=text)]
        splits = self.text_splitter.split_documents(documents)

        self.vectorstore = Chroma(
            collection_name="youtube_collection",
            embedding_function=self.embedding,
            persist_directory=self.persist_directory
        )

        self.vectorstore.add_documents(splits)
        self.vectorstore.persist()

    def query(self, question):
        retriever = self.vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(question)
        return "\n\n".join([doc.page_content for doc in docs])