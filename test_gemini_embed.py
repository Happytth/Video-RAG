import os
from google import genai
from dotenv import load_dotenv
load_dotenv()

key = os.environ.get("GEMINI_API_KEY")
if key:
    client = genai.Client(api_key=key)
    for model_name in ["models/gemini-embedding-2", "models/gemini-embedding-2-preview", "models/gemini-embedding-001"]:
        try:
            res = client.models.embed_content(model=model_name, contents="test query")
            vals = res.embeddings[0].values
            print(f"Model '{model_name}' SUCCESS! Vector Dim: {len(vals)}")
        except Exception as e:
            print(f"Model '{model_name}' failed: {e}")
