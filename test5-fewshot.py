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

import ast
import re
import sys

import os
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

LLM_MODEL = os.environ['LLM_MODEL']
# print("model:", LLM_MODEL)

with open('./SystemMessage/SystemMessage copy 2.txt', 'r', encoding='utf-8') as f:
    systemMessage = f.read()

df = pd.read_csv("./userPrompt/userPrompt.csv", sep="|")

# fewshot 예제 load
with open('./fewshot/test5-fewshot.json', 'r', encoding='utf-8') as f:
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
{systemMessage}
"""),
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


def checkLLMResult(result):
    stop_reason = result.additional_kwargs['stop_reason']
    if stop_reason != 'end_turn':
        # print('stop_reason:', stop_reason)
        raise Exception("LLM 결과에서 에러가 감지되었습니다.", 'stop_reason:', stop_reason)
    return result

# Chaining
chain = final_prompt | llm | checkLLMResult | parser
# chain = final_prompt | llm


# 토큰 예측 값 측정.
# input = {
#     "title": df["title"][4],
#     "post": df["contents"][4],
#     "comments": df["comments"][4]
# }
# with get_openai_callback() as cb:
#     result = chain.invoke({"input": input})
#     print(cb)

def extract_professor_info(sid, title, contents, comments):
    try:
        
        # if pd.isna(contents):
        #     print(f"{sid}'s article is a nan")
        #     return
        
        # 일부 특수문자가 파싱 오류로 발생하여, 삭제
        # new_str = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z~!@#$%&*()-=,.\s]", "", contents)
        
        print('sid:', sid, end="")
        # print('contents:', contents)
        # print('comments:', comments)

        # pattern = r'\n|\s+\n'
        # title = re.sub(pattern, ' ', title)

        # print('title:', title)

        input = {
            "articleNo": sid,
            "title": title,
            "post": contents,
            "comments": comments
        }
        # print('input:', input)

        result = chain.invoke({
            "systemMessage": systemMessage,
            "input": input
        })                              
        print(', result:', result, end='\n')
        return result
    except Exception as e:
        print(f"Error processing article: {e}")

# res = extract_professor_info(
#     19955,
#     "당뇨망막병증 말기 안과 질문입니다.",
#     {"writer":"살자구마","content":"안녕하세요,.,36살에 당뇨를 방치 하다가 이번에 앞이 안보여서 병원을 갔더니 양쪽다 당뇨망박병증 말기라고 합니다..내과도 다녀왔는데 단백뇨도 있고 합병증이 많이 왔다고 합니다...​우측은 적출하지 않는 것이 목표라는 말까지 들었네요. 유리체절제술이 가능 할줄 알았는데 불가능 하다고 했습니다.,,​좌측또한 말기 면서 시력검사하니 0,6정도 나오네요...광시증이 심해서 낮에는 썬그라스 없이는 활동이 힘듭니다.​저번주 아바스틴 주사 양쪽에 맞았고 다음주 좌측 레이저 하러 오라고 했습니다...​병원을 더 옮겨 다녀볼 필요성이 있는것인지 너무 머리가 복잡해 질문드려요..현재 진료 받는 곳은 대구누네안과병원입니다..현재 다니는 내과 원장님이 대구장우혁안과의원과 부산대학병원 안과를 추천해서 장우혁 8월 6일 오후 , 부산대 8월 13일 오후 예약을 잡아 놓은 상태 입니다...서울에 있는 김안과를 가야할지...​아니면 계속 누네안과를 다니는것이 맞는건지 판단이 너무 안섭니다...일주일째 잠이 안오고 죽고싶네요...​"},
#     [{"writer":"당뇨에서벗어나자","comment":"아우 왜 제 얘기를 하는거 같죠...."},{"writer":"서울T반디얌","comment":"불안하시면 몇군데 다녀보시고 그 중에 제일 나은 곳 선택하는 것도 방법일거 같아요. 시기 놓치지 않도록 주의하시고 잘 치료 받으시길 바랍니다."},{"writer":"정신쫌차려","comment":"너무 심란하시겠어요.ㅠㅜ"},{"writer":"포돌이","comment":"대학병원가세요"},{"writer":"빅빅터","comment":"몸이 1000냥이면 눈은 900냥입니다. 일단 가장 중요한 눈의 문제이니. 우리나라에서 제일 잘하는병원으로 일단 가시라고 말씀 드리고 싶습니다.일단 서울로 가셔야 하고요.. 지방이라고 무시하는것은 아니고요..대부분의 명의는 서울쪽에 있습니다.개인적으로는 강남성모병원 / 아산병원 / 서울 영등포 김안과 / 서울 누네안과 / 로 가셔서 진찰으시고...  설사 결과와 수술방법은 비슷하다고  결과가 나와도. 그래도 후회없이가장 잘하는 분들에게 진료 받으세요..   그래야 나중에 후회도 없고최선을 다한거라 후회없습니다.  그리고 지방에서 안된다고 하는데아무래도 수술의 경험치면에서 차이가 많습니다.  지방에서 안된다고하지만 실제 서울 최고병원에서는 된다고 하는 경우가 많습니다.(안과정밀기계도 최신식기계와 가격차도 많이 납니다.)강남성모병원이나 아산병원에 예약하시면 기본 3~4달을 기다려야하고대기 시간이 너무 길긴합니다.  빠르게 진료 받을수 있다면 그쪽도 좋고요..아니면 전문 안과 병원으로는 김안과 와 누네안과는 훨씬 빨리 진료를 받을수 있습니다.이병원들은 위에 성모나 아산병원에서 과장이상인분들을 스카우트를 해서 진료하시는 분들이많습니다..  구체적으로 어느의사분들을 언급하는것은 좀 아닌것 같아서 검색해보시면  명의들의 이름이 나옵니다.그분들한테 에약하시면 됩니다...좀더 빠르게 에약 받으려면  눈에 출혈이 심해서 좀 빠르게 진찰받고 싶다고 하면 조정도 어느정도 가능할것으로알고 있습니다."},{"writer":"빅빅터","comment":"제가 잘아는 지방대학병원  내과과장도  자기 아들 눈이 문제 생기자... 서울에서 제일 잘하는 의사한테 맡깁니다.자기병원에도 안과전문의는 있지만요..  눈문제는 더 그렇겠지요..  비지니스와 본인 가족문제는 별개이니.."},{"writer":"ㅈin","comment":"토닥토닥ᆢ지금 어떤 말도 위로가 안될꺼라는거 알아요.저도 실명 이야기 듣고 많이 울었거든요ᆢ양안 치료 받다 결국 수술했고ᆢ결국 시력장애판정 받았어요.신장도 망가져서 외출도 힘들어요.숨차요ᆢ다리도 불편해요.어깨도 불편해서 옷 갈아 입을때 마니 힘들어요.10번중 8번은 가족 도움 받아요. 어린아이도 아닌데ᆢ만세 합니다ㅎㅎ뭐가 문제냐면ᆢ저는 여자고ᆢ남편은 남자고ᆢ아이는 고등 머스마입니다.그래도 질긴게 목숨이라고 살아집니다.살다 보면 웃을 일도 생기더군요.저 같은 사람도 살아요ᆢ저 보다 더 불편하신 분들도 많겠지요~아직 결정된건 없자나요~~일단 치료 받아보세요.좋다는데 찾아가서 매달려도 보세요.여기 저기 다 가보세요.검사한거 CD나 서류 복사해서 다 가보세요.나에게 맞는 병원이 있을껍니다.내 눈이이니까 끝까지 포기하지 마세요.우는건 나중에 울어도 됩니다.지금은 울지 마세요.우는것도 눈에 안좋아요.지금은 병원을 찾아보세요.지방 병원을 무시하는거 아니구요.저도 서울의 모 대학병원 다니고 있지만ᆢ대학병원이라고 다 좋은거 아니라는거 배웠어요.자기 전공 아니면 안과인데도 몰라요.모르는데 자존심(?)인건지ᆢ다른 교수에게 넘기지도 않아서ᆢ개고생만적도 있어요.근데ᆢ서울이 그래도 수술 경험이 더 맜고진료 환자도 더 많이 봐서ᆢ그 경험은 무시 못한다고 생각해요.내 눈이니까ᆢ우는건 좀 미루시고ᆢ여기 저기 다 예약해 보세요.지금은 울지마세요."},{"writer":"쮸시","comment":"서울 순천향 병원(2차병원)  이성진 교수 -- 망막전문아산 삼성 성모병원중 1곳 해서 2군데  진료 예약해보세요계신곳 3차 병원도 가보시고   힘내고 할수있는데까지  힘써보세요     지푸라기라도 잡아셔야지....냉동블루베리  적상추  당근  적양배추  매일같이 폭풍흡입하시고요   뭐라도해야지요  의사한테만 의지하지말고요이성진 교수는 작년엔 신규환자 안받는다했으나 예약실에 사정좀하고...   시도는 해봐야지요어디서 진료보던  의무기록사본 복사해서 기록 남기세요"},{"writer":"쮸시","comment":"당뇨있으니  3차병원 가정의학과에서  채혈한번하고  선생님한테 사정말하고 안과로 점프시켜 달라고 하는것도 방법이겠네요"},{"writer":"Hu77","comment":"사실 양방쪽은 증상이 나와야 치료해주는거라  그때그때  한달뒤보자는 식이더군요. 저희누님같은경우도 혈전이 막혀 신생혈관이 터지고 출혈이발생해 실명단계까지 갔어요성모병원 김안과 갔었는데  딱히 해줄게 없다고만 하더라구요 다행이 근처에 중의사님 도움으로 막힌 혈전을 뚫고  출혈도 지혈되고 지금은 눈이 호전되서 정상적으로 카페 운욍하고 계세요. 요는 긍정적인 마음으로 병원만 의지하지말고 기능의학한방  운동  식습관개선등 모든걸 다 해보시기바랍니다."},{"writer":"러너가이","comment":"초면에 죄송한 질문인데 저도 신생혈관출혈로 3번 수술했어요. 혹시 누님은 어디서 치료를 받으셨는지 지푸라기잡는 심정으로 여쭤봅니다. 지금도 혈관이 또 터진 상태라서..."},{"writer":"Hu77","comment":"메일주시면  정보드릴게요~ 공개적으로 올리면 안될거 같아서요~^^"},{"writer":"러너가이","comment":"Hu77님 친절 감사합니다. 쪽지로도 받을수 있을까요?"},{"writer":"Hu77","comment":"카페등급이 제가 입문이라 채팅이 제한이되네요 혹시 쪽지주심 답드릴게요^^"},{"writer":"러너가이","comment":"감사합니다.저도 입문단계인데 아무것도 몰랐네요."},{"writer":"러너가이","comment":"지금 사적번호 지우시면 제가 메일 새로 만들어 올릴게요."},{"writer":"Hu77","comment":"넵 지웠습니다~^^"},{"writer":"살자구마","comment":"죄송하지만 혹시 어디인지 알수 있을까요 ?"},{"writer":"Hu77","comment":"여기다가 올리는건 광고가될거같아서요~\n메일주시면 정보드릴게요~^^"},{"writer":"살자구마","comment":"쪽지 드렸습니다 ㅜ"},{"writer":"Hu77","comment":"제가등급이안되서 쪽지확인이 안되는거같아요 댓글로 메일주심  제가 캡쳐하겠습니다. 그리고 삭제해주심 메일드릴게요"},{"writer":"Hu77","comment":"지우셔두됩니다~"},{"writer":"Hu77","comment":"제가 일정이 바뻐 이제야 메일보내려했는데 메일주신 댓글을 지우셨네요다시보내주심 정보드리겠습니다."},{"writer":"러너가이","comment":"아이구 죄송요 바쁘신데 제가 늦게 염치없게 부탁드린 줄 알고 민망해서 지웠어요."},{"writer":"Hu77","comment":"메일보내드렸습니다. 메일댓글은 지워주셔두 되요~^^"},{"writer":"Hu77","comment":"한번터진혈관은 약해져서 계속 터진다고 해요 근원적으로 혈관을  강해가하려면\n비타민c  메가도스가 도움된다고 합니다"},{"writer":"러너가이","comment":"네 감사합니다~"},{"writer":"러너가이","comment":"그렇군요. 눈을 3번이나 유리체절제술을 했는데 의사샘이 왜자꾸 터지지 라고만 하셔 몹시 불안해하고 있었는데 조언해주셔 감사합니다."},{"writer":"Hu77","comment":"양방쪽에선 예방을위한약은 없자나요\n항상 병이나와야 치료해주는거라.\n계속해소 혈관이 터지는건 기존혈관이 막혀 우회혈관(신생혈관)이 뚤리는걸로 알고있어요 혹은 약해져서 찢어지던가요\n혹시 드시는약중 아스피린같은 혈전용해제 있으시면 드시면 큰일나요."},{"writer":"러너가이","comment":"사실 오늘부터 아스피린을 먹었어요.신경외과 샘으로부터 똑같은 얘기를 듣고요. Hu77님의 글에 혈전에 의해 막혔다는 글을 보고 관심갖고 부탁드린거예요. 메일의 그곳 참조하여 생각해보겠습니다. 진심어린 조언 거듭 감사합니다."},{"writer":"Hu77","comment":"양방쪽의 혈전용해제를 망막에 쓰지않는 이유는 얇은 모세혈관으로 이루어져있어 오히려 출혈을 일으키고 지혈을 방해하는걸로 알고 있어요 가뜩이나 망막혈관이 터졌는데 아스피린을 드시는것 불난집에 부채질입니다. 당장 중단 하셔야 해요 큰일납니다. 정말!!"},{"writer":"러너가이","comment":"네 이곳에서 질문할 용기를 낸게 정말 다행이네요. 친절한 조언 감사합니다."},{"writer":"Hu77","comment":"진심입니다. 중의사님 약드세요.\n비용도 많이 안들어요. 그리고 언능주무세요\n충분한 휴식이 눈건강에는 최고니까요!!!"},{"writer":"러너가이","comment":"네 ㅎㅎ"}]
# )

# df = df.head(3)
df["result"] = df.apply(lambda x: extract_professor_info(x['sid'], x['title'], x['contents'], x['comments']), axis=1)
df = df.drop(columns="title", axis=1)
df = df.drop(columns="contents", axis=1)
df = df.drop(columns="comments", axis=1)


# # print(df["result"].iloc[0])
listResult = df["result"].tolist()
print("result:\n", listResult, end='\n')

# # 각 행의 JSON 문자열을 파싱하여 리스트에 저장
# parsed_json = [json.loads(row) for row in listResult]
# print("result3:\n", parsed_json, end='\n')

# # ast.literal_eval: 문자열을 파이썬 딕셔너리로 안전하게 변환
# parsed_json = [ast.literal_eval(row) for row in listResult]
# print('results:', parsed_json, end='\n')

# 리스트 항목을 돌면서 작업 수행
parsed_json = []
for result in listResult:
    try:
        # ast.literal_eval: 문자열을 파이썬 딕셔너리로 안전하게 변환
        new_result = ast.literal_eval(result)
        parsed_json.append(new_result)
    except Exception as e:
        print(e)  # 실패한 항목에 대한 에러 메시지 출력

# 리스트를 JSON 파일로 저장
with open("./result/result.json", "w", encoding='utf-8') as json_file:
    json.dump(parsed_json, json_file, ensure_ascii=False, indent=2)