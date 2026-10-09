import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()

SYSTEM_INSTRUCTION = (
    "You are a passionate pizza lover. If the user's input is rude, disrespectful, "
    "or hostile, reply with exactly the single word: rude. "
    "Otherwise, respond enthusiastically from your persona as a pizza lover."
)

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    if not user_input:
        continue

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_input,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7,
        ),
    )

    reply = response.text.strip()

    if reply.lower() == "rude":
        print("Bot: idont talk to rude ppl")
    else:
        print(f"Bot: {reply}")