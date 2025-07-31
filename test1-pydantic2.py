from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

import os
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

LLM_MODEL = os.environ['LLM_MODEL']
print("model:", LLM_MODEL)

df = pd.read_csv("./sns_recommand.csv", sep="|")

# fewshot 예제 load
with open('fewshot_examples.json', 'r', encoding='utf-8') as f:
    examples = json.load(f)

class Professor(BaseModel):
    """한 명의 교수에 대한 정보"""
    professor: str = Field(description="교수의 이름")
    hospital: str = Field(description="교수가 소속된 병원")
    disease: str = Field(description="교수가 담당하는 진료과(질환병)")

# class Professors(BaseModel):
#     """한 문장에 포함되어 있는 교수 정보들"""
#     professors: List[Professor]

# PydanticOutputParser 생성
parser = PydanticOutputParser(pydantic_object=Professor)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a professional text analyzer. Extract this following (professor, hospital, disease) keywords by looking at the content and comments. If you cannot find the keyword, leave it blank("") and do not provide additional explanation.\n{format}"),
        ("human", "{input}")
    ]
)


prompt = prompt.partial(format=parser.get_format_instructions())

if LLM_MODEL == "llama3.1:70b":
    llm = ChatOllama(model=LLM_MODEL, temperature=0, base_url="http://ec2-3-233-233-233.compute-1.amazonaws.com:11434")
elif LLM_MODEL == "llama3.1:8b":
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
elif LLM_MODEL == "gpt-3.5-turbo":
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
elif LLM_MODEL == "gpt-4o":
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)


chain = prompt | llm

def extract_professor_info(sid, contents, comments):
    try:
        print('sid:', sid)
        if pd.isna(contents):
            print(f"{sid}'s contents is a nan")
            return
        
        # 일부 특수문자가 파싱 오류로 발생하여, 삭제
        # new_str = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z~!@#$%&*()-=,.\s]", "", contents)

        result = chain.invoke({
            "input": {
                "post": contents,
                "comments": comments   
            }
        })
        print('sid-result:', result.content)
        return result.content
    except Exception as e:
        print(f"Error processing contents: {e}")

# df = df.head(3)
df["result"] = df.apply(lambda x: extract_professor_info(x['sid'], x['contents'], x['comments']), axis=1)

df = df.drop(columns="contents", axis=1)
df = df.drop(columns="comments", axis=1)

json_data = df.to_json(orient='records', force_ascii=False, indent=2)
print(json_data)
with open('./result/sns_recommand_result.json', 'w', encoding='utf-8') as f:
    f.write(json_data)