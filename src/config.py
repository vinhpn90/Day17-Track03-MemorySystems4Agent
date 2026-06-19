from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Student TODO: load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """
    import os
    from dotenv import load_dotenv

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    
    # Load .env if it exists
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)

    # Create state directory
    state_dir = root / "state"
    state_dir.mkdir(exist_ok=True, parents=True)

    # Sensible defaults for compact memory
    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "300"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    # Choose provider and model configuration
    provider = os.getenv("LLM_PROVIDER", "gemini")
    model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # Resolve API Key / Base URL
    api_key = None
    base_url = None
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://14.241.208.1:443")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    elif provider in ("custom", "vllm"):
        api_key = os.getenv("CUSTOM_API_KEY") or os.getenv("VLLM_API_KEY")
        base_url = os.getenv("CUSTOM_BASE_URL") or os.getenv("VLLM_BASE_URL", "http://14.241.208.1:443/v1")

    model_config = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    # Setup judge model configuration (defaulting to the main LLM config if not specified)
    judge_provider = os.getenv("JUDGE_PROVIDER", provider)
    judge_model_name = os.getenv("JUDGE_MODEL", model_name)
    judge_api_key = os.getenv("JUDGE_API_KEY", api_key)
    judge_base_url = os.getenv("JUDGE_BASE_URL", base_url)

    judge_config = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=judge_api_key,
        base_url=judge_base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model_config,
        judge_model=judge_config,
    )
