from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
import json
from ast import literal_eval

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
주어진 목록에서 영어로 된 사람 이름을 한글 이름으로 바꿔죠. 
영어와 한글은 성과 이름 순서가 바뀌어 있다는 것은 참고해줘. 출력은 예시와 같이 꼭 json형식이 되어야 해.
그외의 부연 설명은 필요 없어.

출력 예시:
[

    {{
        "englishName": "Jihee Park",
        "koreanName": "박지희"
    }}
]

""",
        ),
        ("user", "{text}"),
    ]
)    

parser = StrOutputParser()

# llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm = ChatBedrock( #Bedrock llm 클라이언트 생성
    # credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #AWS 자격 증명에 사용할 프로필 이름 설정 (기본값이 아닌 경우)
    region_name="us-east-1", #리전 설정 (기본값이 아닌 경우)
    # endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #엔드포인트 URL 설정 (필요한 경우)
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0" #파운데이션 모델 설정
)

# runnable = prompt | llm.with_structured_output(schema=Data)
# runnable = prompt | llm | parser
runnable = prompt | llm

# text = """
# 아내고통 바라보며 힘빠지는 요즘 힘드네요..
# 아프지만나을거야,content:60대 후반 은퇴한 남성입니다. 이미 교직원으로서의 커리어 마무리하고 은퇴를 했지만, 최근 들어 마음이 더더욱 무겁기만 하네요. 은퇴후 여유자금도 온데간데 없고, 저희 집사람은 50대 초반부터 갑상선암, 갱년기 무사 다치렀지만 몇년전부터는 만성위염에 허리협착증도 앓고 있는데 저희 주머니상황때문에 통증이 심해도 수술안하고 자꾸 참고 지내는것 같아서 허리 구부정하게 걷거나 일어설때 힘들어할 때마다 마음이 너무 아픕니다. ​사실 경제적으로 여유가 없어 집사람 제대로 된 치료를 해줄 수 없는 상황인데, 처가집에 묶인 돈이 많은데, 그게 사실 저의 은퇴자금 다 빌려줬거든요. 그래서인지 집사람은 눈치만 더 보는것 같아요. 최근에 첫째아들도 늦었지만 장가보낸지 얼마안돼서 아픈 것도 참고 있는돈없는돈 다 긁어모아서 줘버렸거든요...그러다 보니 저 역시 속으로만 끙끙 앓고 있습니다.​그동안 아파트뒷산에서 산책이라도 즐겼지만 이제는 집사람이 5분만 지나도 한걸음더 걷는 것도 종종 힘들어하네요. 설상가상으로 잘 못 움직이니 소화도 더 안 되고, 위염도 더심해져서 아예 속도 매번 부대끼고 변도 잘 보지 못해 매일 복통에도 시달리구요. 그럴 때마다 제가 해줄 수 있는 것이 너무나도 적어 슬프고 무기력해집니다. 혹시라도 비슷한 상황을 겪고 계신 분들이 계시다면 지혜 조금만 나눠주세요... 매일 아침밥 먹던 제가 할수있는것은. 아침 거스르는것이랑 누워있는 안사람 손잡고 혈자리지압해주면서 소화잘되라 등두드려주는것 밖에 없네요. 작은 팁이라도 좋으니, 소중한 경험과 지혜 나눠주시면 감사하겠습니다:

# 허리통증노모,comment:열심히 사신 것 같은데 마음이 아프네요. 힘내세요. 좋은날 올겁니다.저도  스트레칭을 해보려고 노력 중이에요. 통증 너무 심할때는 그것도 힘들지만, 신전자세는 힘드시겠죠?스트레칭이 가능하시다면 그것만으로도 신진대사 좀 올라갈거에요,아프지만나을거야,comment:용기주셔서. 감사합니다. 저도 아내랑 스트레칭 해보려 노력중이에요. 힘들어할땐 힘들어하죠...,미라미야,comment:저희엄마도 협착증으로\n오래걷기가 불가해서 대학병원에서수술하셨어요\n저렴한곳에서 mri찍어서 꼭 가까운 대학병원가보셔요\n척전보다 대학병원이 수술비 저렴해요,아프지만나을거야,comment:정말 죄송하지만 저희는 실비밖에 없어서 그런데 다 적용되던가요? 하도 못배우고 살아서 아프면 다 참고 살았는데, 제가 잘못키워서인지 저희아들도 병원은 고사하고 참고만 살았거든요.집전체가 저때문에 다 이렇게 된것같네요,메로로로로,comment:실비는 입원시 5천만원 한도내에서  90프로 이상 커버됩니다 \n입원 안하고 통원시 사진에 나오는 자부담을 제외한 한도 내에서 보험금 줍니다\n병원비 걱정 마시고 언능 잘하는데서 치료 받으셔요,메로로로로,comment:,미라미야,comment:실비로 거의커버될거에요 협착은 대학병원가서 수술할정도인지 꼭상담해보세요 걸음이안되시면 근육도줄어들테고..쇠약해지실까봐 걱정되네요  많이힘드실거에요 저렴한곳mri찍어서 씨디로 만들어달라해서 대학병원 신경외과진료시 제출하면됩니다,미라미야,comment:협착은 기다린다고 낫는게 아니라 적극적으로 꼭 치료받으셔야해요,허리우스,comment:저희 부모님도 다른 상황이긴 하지만, 잘 못 움직이시는 상황이셔서 온열마사지를 해주는 기기 사드려봤어요. 처음엔 큰 기대 없었는데, 사용하고 나서 복부통증도 많이 완화되고 방구도 잘 나온다 하시더라고요. 한 번 고려해보셔도 좋을 것 같아요.,아프지만나을거야,comment:가격이 부담되지 않을까 걱정이지만, 조금이라도 도움되면 정말 좋겠습니다. 저희도 그런것 사본적 있었는데 아프기만하고 별효과 못본적 있어서 그런데 혹시 사용해보신 모델이나 제품이 있으면 알려주실 수 있을까요?,허리우스,comment:어버이날 근처에 추천받아 사드린거라 제가 이름은 기억안나구요. 결제내역은 나중에 보고 쪽지로 알려드릴게요!,허리우스,comment:https://post.naver.com/viewer/postView.naver?volumeNo=37720584&memberNo=48164597건강카페인지 유튜브 돌다가 이글보고 샀던것 같아요! 저희부모님이 뭣보다 배에도 허리에도 사용둘다 잘하고 계시다하셔서 아내분 통증완화랑 소화에도 도움되셨으면 좋겠네요!,도락산,comment:보험회사에 정확히 문의해보세요..\n저도 실비보험으로 1년전수술하고 잘지내고있어요.,아프지만나을거야,comment:수술 실비로 하셨어요? 내일 바로 확인해봐야겠습니다. 잘돼셔서 다행이빈다. 혹시. 후유증은 없으신지요.. 나이가 있어서요,도락산,comment:저도60후반.男입니다..수술후14개월지났는데.정상의90%까지는 좋아진듯하구요.수술후 좋아졌다 나빠졌다 과정이 1~2번있었지만 걷기위주로 운동해주고 무리안하니 저개인적으론 수술잘했다싶습니다.수술전증세는 전방위증&협착 동시에 있었는데 방사통은없었고.신경차단술 2회받고 효과없어 바로수술받았지요.저는 잘몰라서 그냥척추전문병원에서 천만원넘게 들었는데 대부분들 대학병원에서들 하는것 같아요.잘판단해보세요...,천년연가,comment:머리 어깨 등~ 종아리 발바닥 ~손에 악력이 있으시다면 종종 주물러 주시는 것도 작은 도움이 되지 않을까 싶네요~두분 힘내시길 바랍니다.,구윰이,comment:실비로 꼭 치료받으시길바랍니다~~,billy11,comment:실비가 있으시면 대학병원에서 수술하시면 거의 다 나온다라고 생각하시면 됩니다 만약 진료후 바로 수술을 안 받을경우는 mri비용이 부담이 되실수 있으니 동네병원 진료후 소견서 받으신 다음 건강관리협회에서 mri 촬영 후 실비 청구하시면 비용 거의 다 나옵니다 너무 낙심하지마시고 더 늦기전에 적극적으로 진료 받아보세요~,메로로로로,comment:실비 있으시면 부담 많이 줄거 같은데요~제가 알기론  90프로 이상 커버될건데\n이번에 청구할건데요 니오면 공유드릴께요~,메로로로로,comment:그리고 그냥 잘하는데서 수술에든 하시는게 맘편합니다 나이 먹고 보존치료한다고 아픈몸 이끌고 병원 왔가갔다 차도는 있는지 없는지 나을지 기약도 없늘바엔 적극적 치료로 하루라도 빨리 수술로 고치는게 낫다고 봅니다,노갱315,comment:저도 90%가량 실비로 수술했어요. 엄마아빠 모습 보는것 같아 마음이 아픕니다ㅠ 보존치료나 기기등에 돈 많이 쓰지마시고 꼭 사모님 모시고 병원가서 치료하세요.,hellokee,comment:안녕하세요. 30대 디스크 수술환자 입니다.은퇴하셨는데도 불구하고 계속해서 자식 챙겨주시려고 하는 마음이 저희 부모님이 생각나 몇자 적습니다. 저의 경우는 실비보험이 있어 입원, MRI, 수술비까지 대략 500만원 가량되는 부분을 지원 받았습니다. 여기서 보조기구는 간이영수증으로 지원이 안되서 30만원가량 받지 못했구요. 제 경우에는 단 1초도 서있지 못했고 누워서 소변보는것 조차 5번 중 1번 성공할 정도, 그리고 다리에 감각이 무뎌지는 것까지 느끼고 수술을 결정했습니다. 되도록이면 누워서 2~3개월간 척추 전만자세 유지하시고 결정하시라 하고 싶지만 고통이 극심하고 마비가 느껴진다면 어쩔수 없는것 같아요. 평생 관리하면서 살아야 합니다. 또 집안에 있는 모든 것들을 허리 높이 위로 올리시고, 청소용품 같은 것도 허리 높이위로 바꾸셔서 조심하시는것이 좋아요.꼭 무통하시길 바라고, 건강하고 행복한 삶, 긍정적인 삶을 이어나가셨으면 합니다.,난행운아2,comment:아내분에 대한 사랑이 느껴져서 참 마음이 그렇습니다.  협착증 수술 많이들 하십니다. 좋은 의사선생님 만나셔서 잘 치료되시고 백년해로 행복하시길 기도합니다,하리디스크환자1922,comment:오랜만에 다쳐서 다시 왔는데 사랑하는 마음에 감동받아 글 씁니다 저랑 비슷하시네요 저는 현재 누워서 생활중인데 소화도 잘 안되서 고통입니다이게 밥먹고 바로 누우면 안되느데 통증이 심해서 누울수 밖에 없고 눕자니 소화 안되고 위산 올라옵니다역류성식도염베개라는게 있습니다 이게 또 허리에 안 좋을 수도 있는데 막상 밥먹고 바로 눕는게 되게 안 좋거든요 그래서 전 식후에 3시간 정도에만 씁니다 아마 누워계시다보면서 위산 역류되고 안좋으실것 같아요 선택이긴 한데 장시간 사용보다 식후에 누워야 되시는 상황이니 1~2시간 정도만 사용해보세요 조금 조심스럽습니다 너무 길게 쓰지 마시고 식후에 눕는 상황에서는 도움 많이 되실거에요 힘내세요,햇살가득한거리,comment:실비 있다고 하셨는데 그럼 갑상선암 등등 질병에 그동안 진료비 청구 안하신건가요? 3년 이내의 진료비 모두 청구 가능하니 알아보시고요 대학병원 진료 보세요,아트록,comment:안타까운 마음에 조금이라도 도움이 될수 있을지 모르겠지만 빨리 병원가셔서 MRI 찍어서 의사하고 정확한 진료상담을 해보시는게 좋을것 같아요. 제경험상 여러병원 다니시는게 좋고 가급적 대학병원이나 그분야에서 경험많은 의사(특히수술인경우에)를 추천드려요. 그리고 병원비는 제경험상(저는 척추유합술) 천차만별인데 척추전문병원은 1,000만원하는데도 있는데, ㅇㅇㄷㅅㅁㅂㅇ이 2차병원이라 저렴했던것 같아요. 저같은 경우에 수술후 1주일 입원에 500만원 정도 나왔구요. 전액 실손 보험금으로 커버 됐어요.가입하고 계신 실손 보험있으면 걱정하지마시고 하루빨리 병원가보세요.,개허리,comment:지난번 ebs 프로그램에 나온 운동법인데 제가 2주넘게 하고있는데 통증이 많이 없어졌어요.꾸준히 따라해 보세요 어렵지 않습니다.https://youtu.be/Ro3E9A3sowQ?si=2qdI6qBlkpwgXrMT,비빔칼국수,comment:힘내십시오 선생님. 좋은 결과 있으시길 바랍니다 정말루요
# """
text = """
Jiyun Oh
Almir G V Bitencourt
E-Ryung Choi, Ok Hee Woo
Jeongju Kim
Mi-Ri Kwon
Jihee Park
Haejung Kim
Domiziana Santucci
Marion Fontaine
Ka Eun Kim
Harim Kim
"""

# response = runnable.invoke({"text": text})
results = []
def read_file_in_chunks(file_path, chunk_size=20):
    with open(file_path, 'r') as file:
        loop = 0
        while True:
            try:
                # 5000줄씩 읽어옴
                lines = [file.readline() for _ in range(chunk_size)]
                
                # 더 이상 읽을 줄이 없으면 종료
                if not lines:
                    break
                    
                # 빈 줄이 있을 경우 그만큼 줄이 남아있지 않은 것이므로 필터링
                lines = [line for line in lines if line.strip()]
                if len(lines) == 0:
                    break

                # 각 chunk에 대해 처리할 작업을 여기에 넣을 수 있음
                # 예: print(lines) 또는 데이터를 다른 곳에 저장
                print(f"Read {(loop + 1) * len(lines)} lines")
                # 예시 처리: lines를 출력하거나 다른 작업을 할 수 있음
                # for line in lines:
                #     print(line.strip())  # 줄바꿈 제거 후 출력

                nameList = ''.join(lines)
                response = runnable.invoke({"text": nameList})
                print(response.content)
                parsed_result = literal_eval(response.content)
                results.extend(parsed_result)

                # 리스트를 JSON 파일로 저장
                with open("./nameList-Ex-Reverse.json", "w", encoding='utf-8') as json_file:
                    json.dump(results, json_file, ensure_ascii=False, indent=2)

                loop += 1
            except Exception as e:
                msg = f"Error, read_file_in_chunks: {e}"
                print(msg)
# 예시 사용법
read_file_in_chunks("pubmed_distinct.csv")
print('출력', results)



