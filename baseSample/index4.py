from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_ollama import ChatOllama

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


class Person(BaseModel):
    """Information about a person."""

    name: str = Field(..., description="The name of the person")
    height_in_meters: float = Field(
        ..., description="The height of the person expressed in meters."
    )


class People(BaseModel):
    """Identifying information about all people in a text."""

    people: List[Person]


# Set up a parser
parser = PydanticOutputParser(pydantic_object=People)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user query. Wrap the output in `json` tags\n{format_instructions}",
        ),
        ("human", "{query}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

query = "Anna is 23 years old and she is 6 feet tall. tom is 33 years old and she is 8 feet tall"

print(prompt.invoke(query).to_string())


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# llm = OllamaFunctions(model="llama3.1:8b", temperature=0)
# llm = ChatOllama(
#     model="llama3.1:8b",
#     temperature=0,
#     # other params...
# )

chain = prompt | llm | parser

res = chain.invoke({"query": query})
print(res)