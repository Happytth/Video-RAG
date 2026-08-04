import os
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
for m in client.models.list().data:
    print(m.id)
