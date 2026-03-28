# LLM Adapter

A thin Python wrapper around the OpenAI-compatible API that provides a unified interface for multiple LLM providers.

## Supported Providers

| Provider      | Env Variable                         |
| ------------- | ------------------------------------ |
| OpenAI        | `OPENAI_API_KEY`                     |
| Google Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| NVIDIA NIM    | `NVIDIA_NIM_API_KEY`                 |
| OpenCode      | `OPENCODE_API_KEY`                   |
| ZAI           | `ZAI_API_KEY`                        |
| Ollama        | _(local, no key needed)_             |

## Setup

Requires Python 3.14+.

```bash
pip install .
```

Create a `.env` file in your project root with the API keys for the providers you want to use:

```bash
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
NVIDIA_NIM_API_KEY=...
OPENCODE_API_KEY=...
ZAI_API_KEY=...
```

Load the `.env` file **before** importing this library so the keys are available in the environment.

## Usage

```python
from dotenv import load_dotenv
load_dotenv(override=True)  # Load .env file

from my_llm import Provider, Model, get_client

client = get_client(Provider.OPENAI)
response = client.chat.completions.create(
    model=Model.GPT_4O_MINI.value,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

`get_client` validates the API key before returning the client and will raise `EnvironmentError` if it's missing.

List locally available Ollama models:

```python
from my_llm import list_ollama_models

print(list_ollama_models())
```

## Adding Your Own Providers, Models, and Keys

The library is designed to be extended directly in `my_llm.py`.

### 1. Add a new provider

Add an entry to the `Provider` enum:

```python
class Provider(Enum):
    # ...existing providers...
    MY_PROVIDER = "my_provider"
```

### 2. Add models for the provider

Add entries to the `Model` enum:

```python
class Model(Enum):
    # ...existing models...
    MY_PROVIDER_MODEL_A = "model-a-id"
    MY_PROVIDER_MODEL_B = "model-b-id"
```

### 3. Configure the base URL and API key

Add entries to `_BASE_URLS` and `_API_KEYS`:

```python
_BASE_URLS = {
    # ...existing entries...
    Provider.MY_PROVIDER: "https://api.example.com/v1",
}

_API_KEYS = {
    # ...existing entries...
    Provider.MY_PROVIDER: os.getenv("MY_PROVIDER_API_KEY"),
}
```

### 4. Add the key to your `.env` file

```bash
MY_PROVIDER_API_KEY=...
```

That's it — `get_client(Provider.MY_PROVIDER)` will now work like any built-in provider, including key validation.
