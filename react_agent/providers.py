import openai
import os
from dotenv import load_dotenv

load_dotenv()


class Provider():
    def __init__(self, temperature=0.7):
        self.client: openai.OpenAI = None
        self.model: str = None
        self.temperature: float = temperature

    def generate(self, messages: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

class DeepseekProvider(Provider):
    def __init__(self, temperature=0.7):
        super().__init__(temperature)
        self.client = openai.OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"))
        self.model = os.getenv("DEEPSEEK_MODEL")
        
class OllamaProvider(Provider):
    def __init__(self, temperature=0.7):
        super().__init__(temperature)
        self.client = openai.OpenAI(api_key=os.getenv("OLLAMA_API_KEY"))
        self.model = os.getenv("OLLAMA_MODEL")