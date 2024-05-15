from typing import Annotated
from fastapi import Depends
from api.settings import settings
from api.rag.llm import LLMFactory
from api.rag.retreiver import RetrieverFactory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable


def _chain_factory(
    llm: LLMFactory,
    retriever: RetrieverFactory,
) -> RunnableSerializable[str, str]:
    rag_prompt = settings.RAG_PROMPT

    prompt = ChatPromptTemplate.from_template(rag_prompt)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


ChainFactory = Annotated[RunnableSerializable[str, str], Depends(_chain_factory)]
