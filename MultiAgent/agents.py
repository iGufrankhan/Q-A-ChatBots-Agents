from crewai import Agent
import os
from dotenv import load_dotenv

load_dotenv()


os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

blog_researcher = Agent(
    role="YouTube Research Specialist",
    goal="Extract deep insights from transcript data about {topic}",
    backstory="Expert AI researcher specializing in AI, ML, Data Science and GenAI.",
    verbose=True,
    memory=True,
    llm="groq/llama-3.1-8b-instant"   
)

blog_writer = Agent(
    role="Technical Blog Writer",
    goal="Write structured and engaging blog content about {topic}",
    backstory="Professional technical writer skilled at simplifying AI concepts.",
    verbose=True,
    memory=True,
    llm="groq/llama-3.1-8b-instant" 
)