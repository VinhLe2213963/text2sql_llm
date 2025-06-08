import base64
import os
from google import genai
from google.genai import types
from openai import OpenAI
from mistralai import Mistral, UserMessage, SystemMessage
from utils import extract_sql_from_llm

class Gemini:
    def __init__(self, model_name, api):
        self.model = model_name
        self.client = genai.Client(
            api_key=api,
        )
        
    def generate(self, prompt, temperature):
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            temperature=temperature
        )

        response = ""
        
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        ):
            response += chunk.text

        return extract_sql_from_llm(response)
    
class GPT:
    def __init__(self, model, endpoint, api):
        self.model = model
        self.client = OpenAI(
            base_url=endpoint,
            api_key=api,
        )

    def generate(self, prompt, temperature):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=temperature,
            top_p=1.0,
            max_tokens=1000,
            model=self.model
        )

        return extract_sql_from_llm(response.choices[0].message.content)

class Mistral:
    def __init__(self, model, endpoint, api):
        self.model = model
        self.client = Mistral(api_key=api, server_url=endpoint)

    def generate(self, prompt, temperature):
        response = self.client.chat.complete(
            model=self.model,
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content=prompt),
            ],
            temperature=temperature,
            max_tokens=1000,
            top_p=1.0
        )
        return extract_sql_from_llm(response.choices[0].message.content)
