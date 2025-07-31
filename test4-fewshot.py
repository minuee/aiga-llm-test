from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain_openai import AzureOpenAI, AzureChatOpenAI

from langchain_community.callbacks import get_openai_callback

# from langchain.llms.bedrock import Bedrock
# from langchain_community.chat_models import BedrockChat
from langchain_aws import ChatBedrock

from langchain_core.output_parsers import StrOutputParser

import os
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

LLM_MODEL = os.environ['LLM_MODEL']
# print("model:", LLM_MODEL)

df = pd.read_csv("./request/sns_recommand.csv", sep="|")


# fewshot 예제 load
with open('./fewshot/fewshot.json', 'r', encoding='utf-8') as f:
    examples = json.load(f)

# print(examples)

# 예시 prompt
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}")
    ]
)

# few shot prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

# 최종 prompt 
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("""
You are an AI that extracts doctor recommendation information from raw data collected from various SNS. The better you are at extracting physician referral information, the better it will help keep people healthy.
Look at the title, body, and comments of the user-registered post, and if there is content recommending a professor (doctor) in the comment, extract keywords (professor's name, affiliated hospital name, disease name). 
If you cannot find it, leave it blank (""). Please determine the Korean initial consonants by name. Do not provide any additional description other than keywords."""),
        few_shot_prompt,
        ("human", "{input}")
    ]
)

# LLM 모델 선택, 편히상 주석 처리
# if LLM_MODEL == "llama3.1:70b":
#     llm = ChatOllama(model=LLM_MODEL, temperature=0, base_url="http://ec2-3-233-233-233.compute-1.amazonaws.com:11434")
# elif LLM_MODEL == "llama3.1:8b":
#     llm = ChatOllama(model=LLM_MODEL, temperature=0)
# elif LLM_MODEL == "gpt-3.5-turbo":
#     llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
# elif LLM_MODEL == "gpt-4o":
#     llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# 아래 모델 선택으로 처리
# llm = ChatOllama(model="llama3.1:70b", temperature=0, base_url="http://ec2-3-233-233-233.compute-1.amazonaws.com:11434")
# llm = ChatOllama(model="llama3.1:8b", temperature=0)
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# llm = ChatOpenAI(model="gpt-4o", temperature=0)

llm = ChatBedrock( #Bedrock llm 클라이언트 생성
    # credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #AWS 자격 증명에 사용할 프로필 이름 설정 (기본값이 아닌 경우)
    region_name="us-east-1", #리전 설정 (기본값이 아닌 경우)
    # endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #엔드포인트 URL 설정 (필요한 경우)
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0" #파운데이션 모델 설정
)

# llm = AzureOpenAI(
#     model_name="gpt-4o",
#     deployment_name="gpt-4o",
#     openai_api_version=os.getenv("OPENAI_API_VERSION"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     openai_api_key=os.getenv("AZURE_OPENAI_API_KEY")
# )

# llm = AzureChatOpenAI(
#     # deployment_name="gpt-4o",
#     model_name="gpt-4o",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version="2023-03-15-preview"
# )


# Output Parser
parser = StrOutputParser()

# Chaining
chain = final_prompt | llm | parser


# 토큰 예측 값 측정.
# input = {
#     "title": df["title"][4],
#     "post": df["contents"][4],
#     "comments": df["comments"][4]
# }
# with get_openai_callback() as cb:
#     result = chain.invoke({"input": input})
#     print(cb)


def extract_professor_info(sid, title, content, comments):
    try:
        print('sid:', sid)
        if pd.isna(content):
            print(f"{sid}'s article is a nan")
            return
        
        # 일부 특수문자가 파싱 오류로 발생하여, 삭제
        # new_str = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z~!@#$%&*()-=,.\s]", "", contents)

        result = chain.invoke({
            "input": {
              "title": title,
              "content": content,
              "comments": comments
            }
        })
        print('sid-result:', result)
        return result
    except Exception as e:
        print(f"Error processing article: {e}")

# df = df.head(3)
df["result"] = df.apply(lambda x: extract_professor_info(x['sid'], x['title'], x['contents'], x['comments']), axis=1)

df = df.drop(columns="title", axis=1)
df = df.drop(columns="contents", axis=1)
df = df.drop(columns="comments", axis=1)

json_data = df.to_json(orient='records', force_ascii=False, indent=2)
print(json_data)
with open('./result/sns_recommand_result.json', 'w', encoding='utf-8') as f:
    f.write(json_data)