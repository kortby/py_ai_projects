import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Initialize the Gemini client (automatically reads GEMINI_API_KEY from environment)
client = genai.Client()


def generate_review(review: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=review,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a sentiment classification bot, print out if the user is happy or sad. "
                "Only print out happy or sad."
            ),
            temperature=0.7,
            max_output_tokens=150,
        ),
    )

    response_message = response.text.strip().lower()

    if response_message == "happy":
        # TODO 1
        return "Thanks for shopping with us come back soon"
    # TODO 2
    return "Sorry to hear about your experience, here's a coupon for 20% off, type GPT20 to use it."