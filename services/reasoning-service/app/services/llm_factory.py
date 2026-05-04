"""
LLM abstraction layer — supports OpenAI and Groq via a unified factory.
LLM instances are created once and reused across agents.
"""
import os
import logging
from functools import lru_cache
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


def get_llm(provider: str | None = None, model: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    """
    Return a chat model instance based on provider env-var or explicit argument.

    Provider resolution order:
    1. `provider` argument
    2. LLM_PROVIDER env var  (openai | groq)
    3. Falls back to openai

    Model resolution order:
    1. `model` argument
    2. LLM_MODEL env var
    3. Provider default
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    logger.info("Initializing LLM | provider=%s", provider)

    if provider == "groq":
        from langchain_groq import ChatGroq  # type: ignore
        resolved_model = model or os.getenv("LLM_MODEL", "llama3-70b-8192")
        logger.info("Using Groq model: %s", resolved_model)
        return ChatGroq(
            model=resolved_model,
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

    # Default: OpenAI
    from langchain_openai import ChatOpenAI
    resolved_model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
    logger.info("Using OpenAI model: %s", resolved_model)
    return ChatOpenAI(
        model=resolved_model,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


@lru_cache(maxsize=4)
def get_cached_llm(provider: str = "openai", model: str = "gpt-4o-mini", temperature: float = 0.0) -> BaseChatModel:
    """Cached LLM factory — avoids re-initialising on every agent call."""
    return get_llm(provider=provider, model=model, temperature=temperature)
