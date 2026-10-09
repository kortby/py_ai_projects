import os
from dotenv import load_dotenv

# Document Loading & Chunking
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter

# Vector Store & Embeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# LLM & RAG Chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Load the webpage
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Tea")
documents = loader.load()

# 2. Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 3. Embed text chunks using Gemini embeddings and store in FAISS
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
docsearch = FAISS.from_documents(texts, embeddings)

# 4. Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
)

# 5. Build modern RAG Chain (replaces legacy RetrievalQA)
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(docsearch.as_retriever(), question_answer_chain)

# 6. Execute query
response = rag_chain.invoke({"input": "When did tea originate?"})

print("Answer:", response["answer"])