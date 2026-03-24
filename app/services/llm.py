from typing import AsyncGenerator, Iterator

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from loguru import logger

from app.core.config import settings

"""LLM service — wraps Groq via LangChain for both regular and streaming responses."""

_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Use the following context to answer the question. "
        "If you cannot find the answer in the context, say "
        '"I don\'t have enough information to answer this question."\n\n'
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)


class LLMService:
    """Calls a Groq-hosted LLM with full and streaming response modes."""

    def __init__(self) -> None:
        logger.info(f"Initialising LLM Service — model: {settings.LLM_MODEL}")
        self.llm = ChatGroq(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            groq_api_key=settings.GROQ_API_KEY,
        )
        self.prompt_template = _RAG_PROMPT

    #Helpers 

    def _build_prompt(self, query: str, context: str) -> str:
        return self.prompt_template.format(context=context, question=query)

    #Public API

    def call_llm(self, query: str, context: str) -> str:
        """Return a complete (non-streaming) LLM response."""
        logger.info(f"Calling LLM — query: '{query}'")
        try:
            response = self.llm.invoke(self._build_prompt(query, context))
            answer: str = response.content if hasattr(response, "content") else str(response)
            logger.info("LLM response generated successfully")
            return answer
        except Exception as exc:
            logger.error(f"Error calling LLM: {exc}")
            raise

    def call_llm_streaming(self, query: str, context: str) -> Iterator[str]:
        """Yield LLM response tokens synchronously."""
        logger.info(f"Calling LLM (streaming) — query: '{query}'")
        try:
            for chunk in self.llm.stream(self._build_prompt(query, context)):
                if chunk.content:
                    yield chunk.content
            logger.info("LLM streaming response completed")
        except Exception as exc:
            logger.error(f"Error calling LLM (streaming): {exc}")
            raise

    async def call_llm_streaming_async(
        self, query: str, context: str
    ) -> AsyncGenerator[str, None]:
        """Yield LLM response tokens in an async context."""
        logger.info(f"Calling LLM (async streaming) — query: '{query}'")
        try:
            for chunk in self.llm.stream(self._build_prompt(query, context)):
                if chunk.content:
                    yield chunk.content
            logger.info("LLM async streaming response completed")
        except Exception as exc:
            logger.error(f"Error calling LLM (async streaming): {exc}")
            raise
