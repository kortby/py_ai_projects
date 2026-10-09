import os
from google import genai
from google.genai import types
from google.genai.errors import APIError


class PizzaBotService:
    SYSTEM_INSTRUCTION = (
        "You are an enthusiastic pizza-loving bot. You adore discussing pizza styles, "
        "toppings, dough, and recipes. "
        "Rule 1: If the user says anything rude, hostile, insulting, or disrespectful, output ONLY the single word: rude. "
        "Rule 2: Otherwise, respond conversationally with your passionate pizza-loving persona."
    )

    def __init__(self, model: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment or .env file.")

        self.model = model
        # Initialize client once per service instance
        self.client = genai.Client()

    def respond(self, message: str) -> str:
        clean_message = message.strip()
        if not clean_message:
            return ""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=clean_message,
                config=types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_INSTRUCTION,
                    temperature=0.7,
                    max_output_tokens=300,
                ),
            )

            # Safeguard against empty or missing response payload
            reply = (response.text or "").strip()

            if reply.lower() == "rude":
                return "idont talk to rude ppl"

            return reply

        except APIError as e:
            raise RuntimeError(f"Gemini API error: {e.message}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {str(e)}") from e