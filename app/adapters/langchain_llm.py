"""LangChain OpenAI-compatible LLM adapter with retry and backoff."""

import time

from app.core.errors import ExternalServiceError
from app.services.rag_service import LLMClient
from langchain_openai import ChatOpenAI


class LangChainLlm(LLMClient):
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float,
                 max_retries: int, backoff_seconds: float, temperature: float = 0.0):
        # ponytail: retries delegated to langchain max_retries + backoff; explicit loop only if retry/error-class control is needed
        self.client = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            retry_wait=backoff_seconds,
        )

    def complete(self, prompt: str) -> str:
        try:
            response = self.client.invoke(prompt)
            return response.content
        except Exception as e:
            raise ExternalServiceError(f"LLM invocation failed: {e}") from e
