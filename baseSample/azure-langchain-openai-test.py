import os
from dotenv import load_dotenv
load_dotenv()

# Import Azure OpenAI
# from langchain_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI

# llm = AzureOpenAI(
#     deployment_name="gpt-35-turbo-instruct-0914",
#     # model_name="gpt-4o",
#     # deployment_name="gpt-4o",
#     openai_api_version=os.getenv("OPENAI_API_VERSION"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     openai_api_key=os.getenv("AZURE_OPENAI_API_KEY")
# )

llm = AzureChatOpenAI(
    # deployment_name="gpt-4o",
    model_name="gpt-4o",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-03-15-preview"
)

# Run the LLM
res = llm.invoke("Tell me a joke")
print(res)