# crew.py

from crewai import Crew, Process
from tasks import research_task, write_task
from agents import blog_researcher, blog_writer
from toolsp import YouTubeRAGTool



youtube_url = "https://www.youtube.com/watch?v=I2wURDqiXdM"
topic = "AI vs ML vs DL vs Data Science"

yt_tool = YouTubeRAGTool()
yt_tool.load_video(youtube_url)

context = yt_tool.query(topic)

# Inject transcript context into tasks
research_task.description += f"\n\nTranscript Context:\n{context}"
write_task.description += f"\n\nResearch Context:\n{context}"


# 🔥 STEP 2 — Create Crew
crew = Crew(
    agents=[blog_researcher, blog_writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    memory=True,
    cache=True
)


# 🔥 STEP 3 — Run
result = crew.kickoff(inputs={"topic": topic})

print("\n\nFINAL OUTPUT:\n")
print(result)