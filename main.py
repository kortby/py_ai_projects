import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()

audio_path = Path.cwd() / "src" / "lost_debit_card.wav"

with open(audio_path, "rb") as f:
    audio_bytes = f.read()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/wav",
        ),
        "Generate a verbatim, accurate transcription of this audio file."
    ],
)

print(response.text)