"""LLM Adapter — unified interface for multiple LLM providers.

Provides a thin wrapper around the OpenAI-compatible API to connect to
OpenAI, Ollama, Google Gemini, NVIDIA NIM, OpenCode, and ZAI.
Each provider is configured with a base URL and an API key (sourced from
environment variables at import time).

Usage:
    from my_llm import Provider, Model, get_client

    client = get_client(Provider.OPENAI)
    response = client.chat.completions.create(
        model=Model.GPT_4O_MINI.value,
        messages=[{"role": "user", "content": "Hello!"}],
    )
"""

import os
import subprocess
from enum import Enum
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Provider & Model enums
# ---------------------------------------------------------------------------


class Provider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    GEMINI = "gemini"
    NVIDIA_NIM = "nvidia_nim"
    OPENCODE = "opencode"
    ZAI = "zai"


class Model(Enum):
    """Available models grouped by provider.

    Enum values are the model identifiers accepted by each provider's API.
    """

    # OpenAI
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_4_MINI = "gpt-5.4-mini"

    # Google Gemini
    GEMINI_FLASH = "gemini-flash-latest"
    GEMINI_FLASH_LITE = "gemini-flash-lite-latest"

    # NVIDIA NIM
    NVIDIA_NIM_GLM_5 = "z-ai/glm5"
    NVIDIA_NIM_GLM_4_7 = "z-ai/glm4.7"
    NVIDIA_NIM_KIMI_K2_5 = "moonshotai/kimi-k2.5"
    NVIDIA_NIM_MINIMAX_M2_5 = "minimaxai/minimax-m2.5"
    NVIDIA_NIM_QWEN3_5_397B_A17B = "qwen/qwen3.5-397b-a17b"
    NVIDIA_NIM_DEEPSEEK_3_2 = "deepseek-ai/deepseek-v3.2"
    NVIDIA_NIM_GPT_OSS_120B = "openai/gpt-oss-120b"

    # OpenCode
    OPENCODE_GLM_5 = "glm-5"
    OPENCODE_MINIMAX_M2_5 = "minimax-m2.5"
    OPENCODE_MINIMAX_M2_7 = "minimax-m2.7"
    OPENCODE_KIMI_K2_5 = "kimi-k2.5"

    # ZAI
    ZAI_GLM_5_1 = "glm-5.1"
    ZAI_GLM_5_TURBO = "glm-5-turbo"
    ZAI_GLM_5 = "glm-5"
    ZAI_GLM_4_7 = "glm-4.7"

    # Ollama (local)
    OLLAMA_QWEN3 = "qwen3-4b-instruct-2507-i1-q4_k_m"
    OLLAMA_QWEN3_HERETIC = "qwen3-4b-instruct-2507-heretic-av2-i1-q4_k_m"


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

# Base URLs for each provider's OpenAI-compatible API endpoint.
_BASE_URLS = {
    Provider.OPENAI: "https://api.openai.com/v1",
    Provider.OLLAMA: "http://localhost:11434/v1",
    Provider.GEMINI: "https://generativelanguage.googleapis.com/v1beta/openai/",
    Provider.NVIDIA_NIM: "https://integrate.api.nvidia.com/v1/",
    Provider.OPENCODE: "https://opencode.ai/zen/go/v1",
    Provider.ZAI: "https://api.z.ai/api/coding/paas/v4",
}

# API keys read from environment variables at module load time.
# Ollama runs locally and uses a placeholder key.
_API_KEYS = {
    Provider.OPENAI: os.getenv("OPENAI_API_KEY"),
    Provider.OLLAMA: "ollama",
    Provider.GEMINI: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    Provider.NVIDIA_NIM: os.getenv("NVIDIA_NIM_API_KEY"),
    Provider.OPENCODE: os.getenv("OPENCODE_API_KEY"),
    Provider.ZAI: os.getenv("ZAI_API_KEY"),
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_client(provider: Provider) -> OpenAI:
    """Return an OpenAI-compatible client configured for the given provider.

    Args:
        provider: The LLM provider to connect to.

    Returns:
        An ``OpenAI`` client instance pointed at the provider's base URL
        and authenticated with the corresponding API key.

    Raises:
        EnvironmentError: If the API key for the provider is missing.
    """
    validate_keys([provider])
    return OpenAI(
        base_url=_BASE_URLS[provider],
        api_key=_API_KEYS[provider],
    )


def validate_keys(providers: Optional[List[Provider]] = None) -> bool:
    """Check that API keys are present for the specified providers.

    Args:
        providers: Providers to validate.  Defaults to all providers.

    Returns:
        ``True`` if every requested provider has a non-empty key.

    Raises:
        EnvironmentError: If a required key is missing or empty.
    """
    if providers is None:
        providers = list(Provider)

    for provider in providers:
        key = _API_KEYS[provider]
        if not key:
            env_var = {
                Provider.OPENAI: "OPENAI_API_KEY",
                Provider.OLLAMA: None,
                Provider.GEMINI: "GEMINI_API_KEY or GOOGLE_API_KEY",
                Provider.NVIDIA_NIM: "NVIDIA_NIM_API_KEY",
                Provider.OPENCODE: "OPENCODE_API_KEY",
                Provider.ZAI: "ZAI_API_KEY",
            }.get(provider)
            if env_var:
                raise EnvironmentError(f"Missing {env_var} for {provider.value}")
    return True


def list_ollama_models() -> List[str]:
    """List locally available Ollama models.

    Runs ``ollama list`` and parses the output to extract model names.

    Returns:
        A list of model identifier strings.

    Raises:
        FileNotFoundError: If the ``ollama`` CLI is not installed.
    """
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
    )
    model_list = []
    for line in result.stdout.splitlines():
        # Skip the header row that contains "NAME"
        if "NAME" in line:
            continue
        parts = line.split()
        if parts:
            model_list.append(parts[0])
    return model_list
