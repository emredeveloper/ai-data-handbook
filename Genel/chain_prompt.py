from langchain_core.language_models import LLM
from typing import List, ClassVar
import google.generativeai as genai

# Gemini yapılandırması
genai.configure(api_key="your_api_key")

class GeminiLLM(LLM):
    model: ClassVar = genai.GenerativeModel("gemini-2.5-flash")

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        response = self.model.generate_content(prompt)
        return response.text.strip()

    @property
    def _llm_type(self) -> str:
        return "gemini"

# Artık LangChain içinde kullanılabilir
llm = GeminiLLM()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("What is the capital of {country}?")
    | llm
    | StrOutputParser()
    | ChatPromptTemplate.from_template("Briefly tell me about {text}.")
    | llm
    | StrOutputParser()
)

print(chain.invoke({"country": "Spain"}))
