from typing import List
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


# Schema for structured response
class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="The person's name", required=True)
    height: float = Field(description="The person's height", required=True)
    hair_color: str = Field(description="The person's hair color")

class People(BaseModel):
    """Identifying information about all people in a text."""
    people: List[Person]

# Prompt template
prompt = PromptTemplate.from_template(
    """
{context}    

Human: {question}
AI: """
)

# Chain
llm = OllamaFunctions(model="llama3.1:8b", format="json", temperature=0)
structured_llm = llm.with_structured_output(People)
chain = prompt | structured_llm

context = """Alex is 5 feet tall. 
Claudia is 1 feet taller than Alex and jumps higher than him.
Claudia is a brunette and Alex is blonde."""

alex = chain.invoke({"context": context, "question":"Describe information about all people in a text."})
print(alex)