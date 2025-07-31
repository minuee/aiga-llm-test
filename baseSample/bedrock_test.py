import boto3
from botocore.exceptions import ClientError

client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
user_message = "Bedrock에서 모델 액세스 권한을 받았어. 내가 구독한 모델의 내역을 조회하는 AWS CLI 명령은?"
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    # Send the message to the model, using a basic inference configuration.
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
    )

    # Extract and print the response text.
    response_text = response["output"]["message"]["content"][0]["text"]
    print(response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)