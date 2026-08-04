import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import base64

def local_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

image_path = "/home/happytth/video_rag_project/saved_frames/frame_51.jpg"

if os.path.exists(image_path):
    b64_str = local_image_to_base64(image_path)
    llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0.2, api_key=os.environ.get("GROQ_API_KEY"))
    msg = [
        {"type": "text", "text": "Is a man standing in a crop field in this image? Answer clearly in 2 sentences."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}}
    ]
    res = llm.invoke([HumanMessage(content=msg)])
    ans = res.content
    if "</think>" in ans:
        ans = ans.split("</think>")[-1].strip()
    print("Direct Groq Vision Inspection of frame_51.jpg:\n", ans)
