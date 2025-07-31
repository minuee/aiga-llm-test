import os
from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import pandas as pd
pd.set_option('display.max_columns', None)

load_dotenv()

df = pd.read_csv("./sns_recommand.csv", sep="|")

LLAMA_MODEL = os.environ['LLAMA_MODEL']

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

prompt = PromptTemplate.from_template(
    """
You are a helpful assistant. 
Answer the user's question in Korean(한글), according to the given format, and do not provide additional explanations.

QUESTION:
{question}

ARTICLE:
{article}

FORMAT:
{format}
"""
)

prompt = prompt.partial(format=parser.get_format_instructions())

# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatOllama(model=LLAMA_MODEL, temperature=0)

chain = prompt | llm

def extract_person_hospital_info(article):
    try:
        # print(article)
        result = chain.invoke({
          "article": article,
          "question": "글 내용(Article)에서 교수 정보를 최대 3명까지 찾아 주세요",
        })
        return result.content
    except Exception as e:
        print(f"Error processing article: {e}")

samples = [
  {"sid": 0, "contents": "", "comments": "", "article": "저는 만71세 남자로 국내 최 장기 이식 생존자 처럼 ROTC 제대와 동시에 대기업에 입사하며 받은 신체검사에서,단백뇨가 검출되어 정밀 검사와 조직검사를 받고 앞으로 16년후에는 이식이나 투석을 받아야 됩니다 하는,경희의료원 김명재 교수의 진단 그대로 26세에에 진단후 41세에 바로 신장이식을 받아 현재 32년차로 Cr 1.46~1.5 , BUN 28 ,사구체여과율 48 정도로 유지 하고 살고 있습니다,​,서울성모병원 양철우 교수님 진료를 받았으나 이제 곧 정년 하시므로 다른 교수님으로 주치의가 비뀌겠지요,그동안 아무 신장 관련 카페에 가입 하지 않고 지내다 신촌 세브란스병원에서 Cr이 1.69~1.77이 나온 일이 있었,기에 깜짝 놀라 관련 카페를 찾다가 여기 동호회가 신뢰 할 만하고 도움을 받을수 있는곳 임을 알고 가입하여 여러,환우님들과 더불어 건강 생활 하고 활동도 열심히 하고 싶습니다,환우님들 모두 지금보다 훨씬 나은 신약이나 의료 신기술이 개발되어 건강한 삶을 사시기 바랍니다~~~,혹시 누군가 저의 도움이 필요 하신분이 있으시면 글로서 도와 드리고 저도 좋은글을 읽고 삶의 희망을 가지고,살고 싶습니다,국내 최장기 이식신의 기록이 42년 이라 하여 저도 희망이 있답니다,고생하는 한우 여러분 모두 힘 내세요!,저는 만71세 남자로 국내 최 장기 이식 생존자 처럼 ROTC 제대와 동시에 대기업에 입사하며 받은 신체검사에서단백뇨가 검출되어 정밀 검사와 조직검사를 받고 앞으로 16년후에는 이식이나 투석을 받아야 됩니다 하는경희의료원 김명재 교수의 진단 그대로 26세에에 진단후 41세에 바로 신장이식을 받아 현재 32년차로 Cr 1.46~1.5 , BUN 28 ,사구체여과율 48 정도로 유지 하고 살고 있습니다​서울성모병원 양철우 교수님 진료를 받았으나 이제 곧 정년 하시므로 다른 교수님으로 주치의가 비뀌겠지요그동안 아무 신장 관련 카페에 가입 하지 않고 지내다 신촌 세브란스병원에서 Cr이 1.69~1.77이 나온 일이 있었기에 깜짝 놀라 관련 카페를 찾다가 여기 동호회가 신뢰 할 만하고 도움을 받을수 있는곳 임을 알고 가입하여 여러 환우님들과 더불어 건강 생활 하고 활동도 열심히 하고 싶습니다환우님들 모두 지금보다 훨씬 나은 신약이나 의료 신기술이 개발되어 건강한 삶을 사시기 바랍니다~~~혹시 누군가 저의 도움이 필요 하신분이 있으시면 글로서 도와 드리고 저도 좋은글을 읽고 삶의 희망을 가지고 살고 싶습니다국내 최장기 이식신의 기록이 42년 이라 하여 저도 희망이 있답니다고생하는 한우 여러분 모두 힘 내세요! 신장이식 32년차에 상기 수치는 괜찮은 것 같습니다. 잘 관리하면 평생 사용 가능 할 듯 합니다. 감사합니다.,남편한테 신장공여자 45세 입니다. 남편도  오래 잘 유지할수 있겠다는  희망이  생기네요^^,와.. 대단하고 멋지시네요..,안녕하세요1980년 생으로 올 6월12일 다낭신으로 인한 말기신분전으로 신장이식 수술을 받았습니다.회원님 글을 보니 신장이식 후 장기간 잘 유지 하신것 같은데 어떻게 유지 하셨는지 평소 생활에 도움이 될만한게 있으면 공유 부탁드립니다.감사합니다,쭉~~~~~~~~\n계속 유지하시길 바랍니다\n좋은기운 받아 갑니다~^^,글에서 늠름함이 느껴집니다// 식이조절이 어렵던데 관리를 참 잘하셨나봐요,대단하세요. 어떻게 관리하셨는지 정말 궁금합니다. 글 부탁드려요.,이런~고맙게도또 생각지도 않았던 큰 응원을 받아 봅니다많이 들어 잘 아시겠지만 그냥 평범 하게 살았어요피곤하면 쉬고  무리 없이~약을 제 시간에  먹느건 필수고요가급적 덜 맵고 덜 짜게  먹구요좋아하던 닭고기  아주 조금 먹고  기름기 있는  고기 피하지요생선과  과일 좋아하고  소고기 살코기로  먹고요신앙생활 하면서 늘 살아있음을 감사 합니다잘 먹고 잘자고  긍정적으로  생각 하고요모두들 하늘이 주시는  천수를 다 누리시길 기도 합니다,39세 어린자녀2명있는 맘입니다..신기능 5%미만이고 이식예정인데요...저도 장기간 유지 잘하고 살수있을것같은 희망이되네요! 좋은 기운받아갑니다 감사드려요!!,좋은 약 훌륭한 의사가 있어 수술 성공하고 잘 사실수 있습니다제 경우에 잔여 생존 기간이 5년 이었지만 32년째 살고 있고앞으로 명대로 살라는 주치의 선생님의 진료중 말씀을 믿고 삽니다걱정 마시고 의사샘 말만 잘 듣고 마음 편히 사시기 바랍니다아이가 있으면 엄마는 강한 힘이 나오잖아요?,대단합니다 전이제 수술5일차입니다"},
  {"sid": 1, "contents": "", "comments": "", "article": "안녕하세요,50대 남편이 최근 쭈그리고 앉아 일하고 볼링도 치고 하더니 고관절 통증이 심해 병원에 갔떠니,고관절무혈성괴사라고 하더라구요.,근데 어릴때부터 그쪽이 좋지않았고, 10여년전에도 나중에 수술해야한다고는 했대요,카페글을 읽어보니 다들 큰병원 가시는거같은데,,저희는 분당 나우에서 진료받았거든요,삼성병원은 9월9일에야 진료 예약이 되더라구요. 박찬우교수님으로요. 임교수님은 ARS 에서 아예 진료가 안된다며 박교수님으로 넘기더라구요.,근데 삼성병원은 9월 진료면 올해안에 수술이 가능할까요?,그리고 큰병원이라 비용부담도 되구요. 1000만원정도 든다고 본거같아요. 전체 재활병원 비용까지겠죠?,​,차라리 언제 할지 알수도 없고 여러모로 편의성이 있는 가까운 분당 나우서 당장 하는게 나을지,그래도 기다려서 삼성병원 가야는지 고민입니다.,겨울에 목발짚고 다니면 더 위험하지안하해서요.,일단 나우에서는 2주 약먹어보고 안되면 수술하자고 했거든요.,​,혹시 분당 나우에서 하신분 계실까요? 평촌 나우는 있는데 분당 나우는 후기가 없는 것 같아서요.,그리고 최근 삼성에서 하신분들 얼마나 기다리셨느지요?,​,조언부탁드립니다 감사합니다,안녕하세요50대 남편이 최근 쭈그리고 앉아 일하고 볼링도 치고 하더니 고관절 통증이 심해 병원에 갔떠니 고관절무혈성괴사라고 하더라구요. 근데 어릴때부터 그쪽이 좋지않았고, 10여년전에도 나중에 수술해야한다고는 했대요카페글을 읽어보니 다들 큰병원 가시는거같은데,저희는 분당 나우에서 진료받았거든요삼성병원은 9월9일에야 진료 예약이 되더라구요. 박찬우교수님으로요. 임교수님은 ARS 에서 아예 진료가 안된다며 박교수님으로 넘기더라구요.근데 삼성병원은 9월 진료면 올해안에 수술이 가능할까요?그리고 큰병원이라 비용부담도 되구요. 1000만원정도 든다고 본거같아요. 전체 재활병원 비용까지겠죠?​차라리 언제 할지 알수도 없고 여러모로 편의성이 있는 가까운 분당 나우서 당장 하는게 나을지그래도 기다려서 삼성병원 가야는지 고민입니다.겨울에 목발짚고 다니면 더 위험하지안하해서요.  일단 나우에서는 2주 약먹어보고 안되면 수술하자고 했거든요.​혹시 분당 나우에서 하신분 계실까요? 평촌 나우는 있는데 분당 나우는 후기가 없는 것 같아서요. 그리고 최근 삼성에서 하신분들 얼마나 기다리셨느지요?​조언부탁드립니다 감사합니다 병원추천은 조심스러운 부분인거같아요\n혹시 결과 안좋으면 원망받기 쉽상이거든요\n\n제 생각은 여기서 언급되는 병원들은 다 잘하는거 같아요\n\n지난글들 마니 검색해보시고 결정하시면 될듯합니다,네 감사합니다 다들 큰병원 많이가시려는것같아요 분당 나우는 후기가 별로 없네요..,삼성서울 임교수님 어제 진료받는데 수술은 내년 1월말이나 가능하다고 하셨어요 취소 나오면 연락주신다고 ㅠㅠ,임교수님은 아예 예약이 안되더라구요.ㅠ.,전화로 예약하셨나요? 전 전화로 임교수님지명하여 예약했었거든요,네네 다시 저나해서 7.22에 예약했습니다 근데 7월초 가셧는데도 내년1월말 수술이 가능하대요? 와 너무 먼데요 큰일이네요,대학병원은 수술비250~350만원 정도 들어요!전문병원은 500~600만원정도 들고요!수술 많이하는곳으로가세요! 그게 조금이라두예후가 좋은거 같아요!,아 대학병원이 더 싼건가요? 근데 대학병원은 입원을 오래 안시키고 재활로 보내느게 맞죠?,꼭 후기가 많은병원 경험이 많으신분께 수술받으세요. 생각보다 큰수술입니다. 저도 이제3달째 되었는데 다 괜찮다 생각되다가도 한번씩 불편할때가 있습니다. 수술후 관리도 정말 중요한것 같습니다,고관절 수술은 정형외과에서 제일 큰 수술이라고 합니다.그러니 매일 수술하고 경험이 많은신분에게 몸을 맡기는게 좋을 듯합니다.위분처럼 병원추천은 조심스럽네요~~,고관절명의는 평촌나우에  있어요~,다른 병원도 있으니 카페글보시는거 추천드려요\n병원 다니시다보면 확~~~끌리시는 병원이 있을꺼예요\n한번 받은수술은 돌아오지않으니 힘드시더라도 여러군데 다녀오시길요,평촌나우가 건물 새로 지어서 너무 좋아요....\n 윤필환샘두 강추 드리구요.... \n저두 분당에서 평촌 다녔는데 충분히 가까워요~ 30~40분이면 가는걸요."}
]

# text = samples[0]["article"]
# output = extract_person_hospital_info(text)
# print(output)

df = df.head(0)
def combine_2rd_columns(col_1, col_2):
    result = col_1
    if not pd.isna(col_2):
        result += " " + str(col_2)
    return result

df["article"] = df.apply(lambda x: combine_2rd_columns(x['contents'], x['comments']), axis=1)

# df.append(
#     samples, 
#     ignore_index=True
# )

df.loc[len(df.index)] = samples[0] 
# print(df)

df.loc[len(df.index)] = samples[1] 
# print(df)


df["result"] = df.apply(lambda x: extract_person_hospital_info(x['article']), axis=1)
result_list = df["result"].to_list()

for n in result_list:
    print(n)


