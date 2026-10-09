import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Define the Pydantic schema
class Furniture(BaseModel):
    type: str = Field(description="the type of furniture")
    style: str = Field(description="the style of furniture")
    colour: str = Field(description="colour")


# 2. Initialize the Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
)

# 3. Enforce structured schema output natively using Gemini
structured_llm = llm.with_structured_output(Furniture)

# 4. Define prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the structured furniture details from the user request."),
    ("human", "{request}"),
])

# 5. Build LCEL chain
chain = prompt | structured_llm

# 6. Execute extraction
furniture_request = "I'd like a blue mid century chair"
result: Furniture = chain.invoke({"request": furniture_request})

print("Extracted Object:", result)
print("Type:", result.type)
print("Style:", result.style)
print("Colour:", result.colour)