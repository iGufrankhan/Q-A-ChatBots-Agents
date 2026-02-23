import requests
from bs4 import BeautifulSoup
import gradio as gr
import json


url='http://localhost:11434/api/generate'
  

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

history = []

def generate_response(prompt):
    history.append(prompt)
    final_prompt = "\n".join(history)
    data = {
         "model": "example",
        "prompt": final_prompt,
        "stream": False
        
    }
    
    response= requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        result = response.text
        data= json.loads(result)
        actual_response = data.get("response", "")
        return actual_response
    else:
        return "Error: Unable to get response from the server."
    

interface = gr.Interface(fn=generate_response, inputs=gr.Textbox(lines=5, placeholder="Input Prompt"), outputs="text", title="Q&A System")
        

interface.launch()