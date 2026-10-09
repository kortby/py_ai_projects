import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 1. Initialize the Gemini chat model
# It automatically reads GEMINI_API_KEY from the environment
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)

# 2. Build the prompt template with system instructions and user inputs
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an enthusiastic pizza chef. Give brief, punchy advice."),
    ("human", "{question}"),
])

# 3. Create the chain using LCEL
chain = prompt | llm | StrOutputParser()

# 4. Invoke the chain
response = chain.invoke({"question": "What is the secret to a crispy pizza crust?"})

print(response)