# tasks.py

from crewai import Task
from agents import blog_researcher, blog_writer


research_task = Task(
    description=(
        "Analyze the provided transcript context about {topic} "
        "and produce a detailed 3-paragraph research summary."
    ),
    expected_output="Three detailed research paragraphs.",
    agent=blog_researcher
)

write_task = Task(
    description=(
        "Using the research summary, create a well-structured "
        "blog post about {topic} in markdown format."
    ),
    expected_output="Complete blog post in markdown format.",
    agent=blog_writer,
    output_file="new-blog-post.md"
)