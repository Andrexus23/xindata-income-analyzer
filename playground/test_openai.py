import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
load_dotenv()
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        SystemMessage("На одни и те же вопросы давай всегда одни и те же ответы"),
        UserMessage("сколько лет России"),
    ],
    temperature=0.0,
    top_p=1.0,
    model=model
)

print(response.choices[0].message.content)


