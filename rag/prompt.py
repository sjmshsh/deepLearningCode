import os
from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="",
)

completion = client.chat.completions.create(
    model="",
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '你是谁？'}],
)

print(completion.choices[0].message.content)
