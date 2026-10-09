
import csv
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http import models as rest


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")


# 1. Load CSV rows as LangChain documents
class CSVLoader(BaseLoader):
    def __init__(
        self,
        file_path: str,
        source_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        csv_args: Optional[Dict] = None,
        encoding: Optional[str] = "utf-8",
    ):
        self.file_path = file_path
        self.source_column = source_column
        self.metadata_columns = metadata_columns or []
        self.csv_args = csv_args or {}
        self.encoding = encoding

    def load(self) -> List[Document]:
        docs = []

        with open(
            self.file_path,
            newline="",
            encoding=self.encoding,
        ) as csvfile:
            reader = csv.DictReader(csvfile, **self.csv_args)

            for i, row in enumerate(reader):
                content = "\n".join(
                    f"{key.strip()}: {(value or '').strip()}"
                    for key, value in row.items()
                    if key is not None
                )

                if self.source_column:
                    if self.source_column not in row:
                        raise ValueError(
                            f"Source column '{self.source_column}' "
                            "not found in CSV."
                        )
                    source = row[self.source_column] or self.file_path
                else:
                    source = self.file_path

                metadata = {
                    "source": source,
                    "row": i,
                }

                for key in self.metadata_columns:
                    if key in row:
                        metadata[key] = row[key] or ""

                docs.append(
                    Document(
                        page_content=content,
                        metadata=metadata,
                    )
                )

        return docs


# 2. Define the filters Gemini should extract
class BookSearch(BaseModel):
    year: str = Field(
        description="Publication year requested by the user"
    )
    genre: str = Field(
        description="Book genre requested by the user"
    )


parser = PydanticOutputParser(pydantic_object=BookSearch)

filter_prompt = PromptTemplate(
    template=(
        "Extract the requested book publication year and genre.\n"
        "If either is not specified, use an empty string for that field.\n"
        "Do not invent missing values.\n"
        "{format_instructions}\n"
        "User request: {query}"
    ),
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

# 3. Configure Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
)

filter_chain = filter_prompt | llm | StrOutputParser()


def get_parsed_result(book_request: str) -> BookSearch:
    output = filter_chain.invoke({"query": book_request})
    return parser.parse(output)


# 4. Build a Qdrant metadata filter
def create_filter(parsed: BookSearch):
    conditions = []

    if parsed.genre:
        conditions.append(
            rest.FieldCondition(
                key="metadata.categories",
                match=rest.MatchValue(value=parsed.genre),
            )
        )

    if parsed.year:
        conditions.append(
            rest.FieldCondition(
                key="metadata.published_year",
                match=rest.MatchValue(value=parsed.year),
            )
        )

    return rest.Filter(must=conditions)


# 5. Load the catalog and create its vector index
loader = CSVLoader(
    file_path="./src/dataset_small.csv",
    source_column="title",
    metadata_columns=["categories", "published_year"],
)

data = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key,
)

vector_store = QdrantVectorStore.from_documents(
    documents=data,
    embedding=embeddings,
    location=":memory:",
    collection_name="book",
)


# 6. Prepare the answer prompt
answer_prompt = PromptTemplate.from_template(
    """
You are a helpful AI librarian.

Answer the user's question using only the book information
provided in the context.

Do not provide ISBN numbers.
If no relevant books are provided, explain that no matching
books were found in the catalog.

Book information:
{context}

User question:
{question}

Answer:
"""
)

answer_chain = answer_prompt | llm | StrOutputParser()


# 7. Run the librarian
while True:
    user_input = input(
        "\nHi, I'm an AI librarian. What can I help you with?\n"
    ).strip()

    if user_input.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not user_input:
        continue

    try:
        parsed_result = get_parsed_result(user_input)

        # Retrieve matching books from Qdrant.
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": create_filter(parsed_result),
            }
        )

        documents = retriever.invoke(user_input)

        print(f"\nMatching books found: {len(documents)}")

        if not documents:
            print("No matching books found in the catalog.")
            continue

        context = "\n\n".join(
            document.page_content for document in documents
        )

        answer = answer_chain.invoke(
            {
                "context": context,
                "question": user_input,
            }
        )

        print(f"\n{answer}")

    except Exception as error:
        print(f"An error occurred: {error}")