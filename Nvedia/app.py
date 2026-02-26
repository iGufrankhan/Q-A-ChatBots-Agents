from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st


import os
load_dotenv()


apikey = os.getenv("NVEDIA_API_KEY")

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = apikey
)



completion = client.chat.completions.create(
  model="meta/llama-3.1-70b-instruct",
  messages=[{"content":"provide a summary of machine learning","role":"user"}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices and chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")




