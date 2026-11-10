from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = init_chat_model(model="gpt-4o-mini", model_provider="openai")

response = model.invoke("Hello, how are you?")
# print(response)

system_template = "Translate the following from English into {target_language}:"

prompt_template = ChatPromptTemplate.from_messages([("system", system_template), ("user", "{text}")])

prompt = prompt_template.invoke({"target_language": "French", "text": "Hello, how are you?"})

response = model.invoke(prompt)
print(response)
