import base64
import os
from google import genai
from google.genai import types
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