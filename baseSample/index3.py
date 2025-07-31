from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

class Car(BaseModel):
    """Information about a car."""
    make: Optional[str] = Field(default=None, description="The make of the car")
    model_name: Optional[str] = Field(default=None, description="The model name of the car")
    model_year: Optional[int] = Field(
        default=None, description="The year the car model was manufactured"
    )
    color: Optional[str] = Field(default=None, description="The color of the car")
    price: Optional[float] = Field(default=None, description="The price of the car")
    mileage: Optional[float] = Field(default=None, description="The mileage of the car")

from typing import List
class Data(BaseModel):
    """Extracted data about cars."""
    cars: List[Car] = Field(
        default=None, description="Extracted information about cars"
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)    


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

runnable = prompt | llm.with_structured_output(schema=Data)

text = """
현재 환경부와 산업통상자원부가 심사 중인 BYD의 소형 해치백 차량인 ‘돌핀’과 중형 세단 차량인 ‘씰’의 중국 내 최저 판매 가격은 각각 1900만 원, 3900만 원이다. 특히 돌핀은 국내에서 가장 값싼 경형 전기차인 ‘기아 레이EV(세제 혜택 전 2775만 원)’와 비교해도 압도적으로 저렴하다.
씰은 BYD의 셀투보디(CTB) 기술이 세계 최초로 적용된 차량으로 가격 대비 높은 성능을 자랑한다. CTB란 차량 본체와 배터리·배터리관리시스템(BMS) 등을 하나로 통합해 강성과 효율성을 모두 높이는 기술을 뜻한다. 두 차량 모두 유럽의 신차 안정성 프로그램(euro NCAP)에서 최고 등급을 받기도 했다.
한국 시장 진입을 위해 BYD가 현지 판매가와 유사한 수준으로 가격을 책정할 가능성도 있다. 통상 국내 시장 진입 시 가격을 더 높여잡는 게 일반적이지만 중국산 제품에 대한 한국의 부정적 인식을 고려해 가격 경쟁력을 최우선적으로 확보할 수 있다는 것이다. 스위스 투자은행(IB) UBS에 따르면 BYD는 배터리, 차량용 반도체, 소프트웨어 등 전체 부품 75%에 대한 수직 계열화를 이루면서 경쟁사 대비 30% 수준의 가격 우위를 확보하고 있다. 아울러 리튬·인산·철(LFP) 배터리에 대한 환경부의 불리한 규정에도 일정 수준의 보조금 확보도 가능하다. 현재 돌핀과 씰의 판매 가격은 국내 전기차 보조금 전액 지원 기준인 5500만 원을 충족한다. 유럽 인증 기준을 만족시키는 최대 427㎞(돌핀), 570㎞(씰)에 이르는 1회 충전 주행거리도 유리한 요소다.
BYD의 대항마로는 최근 기아가 출시한 소형 스포츠유틸리티차량(SUV) ‘EV3’가 꼽힌다. EV3는 니켈·코발트·망간(NCM) 배터리를 탑재해 롱레인지 모델 기준 1회 충전에 501㎞ 주행거리를 확보했다. 가격은 보조금 적용 시 3000만 원 중반대로 전기차 대중화라는 목표를 이루기 위한 기아의 주력 모델이다. KG모빌리티의 코란도EV(3000만 원대)도 BYD의 경쟁 상대다.
"""
response = runnable.invoke({"text": text})
print(response)