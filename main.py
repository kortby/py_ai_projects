import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Ensure GEMINI_API_KEY is present in your .env
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is missing from .env")

# client automatically picks up os.environ["GEMINI_API_KEY"]
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain what Python is in one sentence."
)

print(response.text)