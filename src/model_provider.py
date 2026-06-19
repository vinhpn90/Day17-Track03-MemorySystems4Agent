from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Student TODO: map aliases like `anthorpic` -> `anthropic`."""
    val = value.lower().strip()
    if val in ("openai", "open_ai"):
        return "openai"
    if val in ("gemini", "google"):
        return "gemini"
    if val in ("anthropic", "claude"):
        return "anthropic"
    if val in ("ollama",):
        return "ollama"
    if val in ("openrouter", "open_router"):
        return "openrouter"
    if val in ("custom", "vllm"):
        return "custom"
    return val


def build_chat_model(config: ProviderConfig):
    """Student TODO: instantiate the real chat model for the selected provider.

    Pseudocode:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter` (or ChatOpenAI with OpenRouter endpoints)
    """
    provider = normalize_provider(config.provider)
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key
        )
    elif provider == "custom":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or "custom",
            base_url=config.base_url,
            max_retries=10,
            timeout=120.0
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url
        )
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url or "https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def invoke_with_retry(model, messages, max_retries=10, initial_delay=1.0):
    import time
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return model.invoke(messages)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"      [Connection Warning] Attempt {attempt+1}/{max_retries} failed ({e}). Retrying in {delay:.1f}s...", flush=True)
            time.sleep(delay)
            delay = min(delay * 1.5, 10.0)
