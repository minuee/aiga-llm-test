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

from ast import literal_eval
import re
import sys

from logger import log

import os
import json
import pandas as pd
from dotenv import load_dotenv
from database import saveEvaluation

load_dotenv()

SOURCE_ID = os.getenv('SOURCE_ID')

def getFewShot():
    fewshot = f'./fewshot/{SOURCE_ID}/fewshot.json'
    with open(fewshot, 'r', encoding='utf-8') as f:
        return json.load(f)

def getSystemMessage():
    systemMessage = f'./SystemMessage/{SOURCE_ID}/message.txt'
    with open(systemMessage, 'r', encoding='utf-8') as f:
        return f.read()

fewshot = getFewShot()
systemMessage = getSystemMessage()

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}")
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=fewshot
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("""{systemMessage}"""),
        few_shot_prompt,
        ("human", "{input}")
    ]

)
# Output Parser
parser = StrOutputParser()

# LLM 모델 선택
llm = ChatBedrock( #Bedrock llm 클라이언트 생성
    # credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #AWS 자격 증명에 사용할 프로필 이름 설정 (기본값이 아닌 경우)
    region_name="us-east-1", #리전 설정 (기본값이 아닌 경우)
    model_kwargs=dict(temperature=0),
    # endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #엔드포인트 URL 설정 (필요한 경우)
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0" #파운데이션 모델 설정
)
# llm = AzureChatOpenAI(
#     # deployment_name="gpt-4o",
#     model_name="gpt-4o",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version="2023-03-15-preview"
# )

def checkLLMResult(result):
    stop_reason = result.additional_kwargs['stop_reason']
    if stop_reason != 'end_turn':
        # print('stop_reason:', stop_reason)
        raise Exception("LLM 결과에서 에러가 감지되었습니다.", 'stop_reason:', stop_reason)
    return result

# Chaining
chain = final_prompt | llm | checkLLMResult | parser

def extract_professor_info(review):
    try:
        # if pd.isna(contents):
        #     print(f"{review_id}'s review is a nan")
        #     return
        
        # 일부 특수문자가 파싱 오류로 발생하여, 삭제
        # new_str = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z~!@#$%&*()-=,.\s]", "", contents)
        print('\n final_prompt:', final_prompt)
        print('\n llm:', llm)
        print('\n Chaining:', checkLLMResult)
        print('\n parser:', parser)
        print('\n review_id:', review['review_id'], end="")

        input = {
            "articleNo": review['review_id'],
            "title": review["title"],
            "post": review["contents"],
            "comments": review["comments"]
        }
        # print('chain.invoke input:', input)

        result = chain.invoke({
            "systemMessage": systemMessage,
            "input": input
        })                              
        print(', extract_professor_info result: ', result, end='\n')
        return result
    except Exception as e:
        msg = f"Error, review_id({review['review_id']}), extract_professor_info: {e}"
        log.LogTextOut(msg)


def evaluateDoctor(reviews):

    results = []
    doctor_evals = []  
    for review in reviews:
        result = extract_professor_info(review)
        print("evaluateDoctor result:", result)
        # LLM 수집의 오류가 발생하면, 바로 다음 진행
        if result is None:
            doctor_evals.append({'review_id': review['review_id']})
            continue

        try:
            # pattern = r"'([^']*)'"
            # result = re.sub(pattern, r'"\1"', result)
            # print(result)
            # parsed_result = json.loads(result)
            # None 값 때문에 eval() 사용.
            parsed_result = literal_eval(result)
            print(parsed_result)

            results.append(parsed_result)
        except Exception as e:
            msg = f"Error, review_id({review['review_id']}), in converting dictionary from result: {e}"
            log.LogTextOut(msg)
            continue

        # results.append(parsed_result)

        if len(parsed_result["doctor_list"]) > 0:
            for doctor_eval in parsed_result["doctor_list"]:
                saveEvaluation(review, {
                    "review_id": review['review_id'],
                    "eval": doctor_eval
                })
        else:
            # doctor_evals.append({'review_id': review['review_id']})
            saveEvaluation(review, {"review_id": review['review_id']})

    print('\n', "results:\n", results)

    # 리스트를 JSON 파일로 저장
    with open("./result/result.json", "w", encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=2)

    return doctor_evals
        












