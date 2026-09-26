import os

from dotenv import load_dotenv
from openai import OpenAI
from app.config.config import DEEPSEEK_API_KEY
from langchain_ollama import ChatOllama

from llm.llm import get_llm

load_dotenv()


class LLMClient:

    def __init__(self, api_key=None):
        self.llm = get_llm()

    def generate_answer(self, prompt):
        response = self.llm.invoke(
            input=prompt
        )
        return response.content