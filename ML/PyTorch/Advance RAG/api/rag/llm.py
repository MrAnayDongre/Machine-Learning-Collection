from typing import Annotated
from fastapi import Depends
from langchain_community.llms.ollama import Ollama
from langchain_core.language_models import BaseLLM
from api.settings import settings


def _llm_factory() -> BaseLLM:
    return Ollama(base_url=settings.OLLAMA_SERVER, model=settings.LLM_MODEL)


LLMFactory = Annotated[BaseLLM, Depends(_llm_factory)]
