"""
LLM abstraction layer — supports OpenAI and Groq via a unified factory.
LLM instances are created once and reused across agents.
Reads config from app.config.settings (validated at startup).
"""
import logging
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from app.config.settings import settings
from app.core.exceptions import LLMConfigError

logger = logging.getLogger(__name__)


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """
    Return a chat model instance.

    Resolution order for each param:
      explicit argument → settings value → provider default
    """
    resolved_provider = (provider or settings.llm_provider).lower()
    resolved_temp = temperature if temperature is not None else settings.llm_temperature

    logger.info("Initializing LLM | provider=%s", resolved_provider)

    try:
        if resolved_provider == "groq":
            from langchain_groq import ChatGroq  # type: ignore
            resolved_model = model or settings.llm_model or "llama3-70b-8192"
            logger.info("Using Groq model: %s", resolved_model)
            return ChatGroq(
                model=resolved_model,
                temperature=resolved_temp,
                groq_api_key=settings.groq_api_key,
            )

        if resolved_provider == "openai":
            from langchain_openai import ChatOpenAI
            resolved_model = model or settings.llm_model or "gpt-4o-mini"
            logger.info("Using OpenAI model: %s", resolved_model)
            return ChatOpenAI(
                model=resolved_model,
                temperature=resolved_temp,
                api_key=settings.openai_api_key,
            )

        raise LLMConfigError(
            message=f"Unknown LLM provider: '{resolved_provider}'. Supported: openai, groq.",
            details={"provider": resolved_provider},
        )

    except LLMConfigError:
        raise
    except Exception as exc:
        raise LLMConfigError(
            message=f"Failed to initialise LLM provider '{resolved_provider}': {exc}",
            details={"provider": resolved_provider, "cause": str(exc)},
        ) from exc


@lru_cache(maxsize=4)
def get_cached_llm(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> BaseChatModel:
    """Cached LLM factory — one instance per (provider, model, temperature)."""
    return get_llm(provider=provider, model=model, temperature=temperature)
