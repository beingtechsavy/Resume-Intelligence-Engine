"""Shared OpenRouter configuration for the local resume tools."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek/deepseek-v4.1-flash-1017"
_PROVIDER_SORTS = {"price", "throughput", "latency"}


@dataclass(frozen=True)
class AIConfig:
    """Provider settings loaded from the local environment."""

    api_key: str
    model: str
    base_url: str
    default_headers: dict[str, str]
    provider_options: dict[str, Any]
    reasoning_options: dict[str, Any]

    def client(self) -> OpenAI:
        """Create an OpenAI-compatible client pointed at OpenRouter."""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=self.default_headers or None,
        )

    def request_kwargs(self) -> dict[str, Any]:
        """Return OpenRouter-specific request fields for the selected model."""
        extra_body: dict[str, Any] = {}
        if self.provider_options:
            extra_body["provider"] = self.provider_options
        if self.reasoning_options:
            extra_body["reasoning"] = self.reasoning_options
        return {"extra_body": extra_body} if extra_body else {}


def _parse_bool(value: str, default: bool | None = None) -> bool | None:
    normalized = value.strip().casefold()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(
        "\n[ERROR] Invalid boolean OpenRouter setting: "
        f"'{value}'. Use true/false.\n"
    )


def _parse_price(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(f"\n[ERROR] {name} must be a number in USD per million tokens.\n") from None
    if value < 0:
        raise SystemExit(f"\n[ERROR] {name} must not be negative.\n")
    return value


def _provider_options(model: str) -> dict[str, Any]:
    """Build a conservative provider policy without hard-pinning one host."""
    options: dict[str, Any] = {}

    # The DeepSeek model is used for two concurrent document calls. Prefer
    # throughput while keeping the route below the current balanced price band;
    # OpenRouter can fall back if the preferred endpoint is unavailable.
    if model == DEFAULT_DEEPSEEK_MODEL:
        options.update(
            {
                "sort": "throughput",
                "allow_fallbacks": True,
                "max_price": {"prompt": 0.10, "completion": 0.60},
            }
        )

    sort = os.getenv("OPENROUTER_PROVIDER_SORT", "").strip().casefold()
    if sort:
        if sort not in _PROVIDER_SORTS:
            raise SystemExit(
                "\n[ERROR] OPENROUTER_PROVIDER_SORT must be one of: "
                f"{', '.join(sorted(_PROVIDER_SORTS))}.\n"
            )
        options["sort"] = sort

    allow_fallbacks = _parse_bool(os.getenv("OPENROUTER_PROVIDER_ALLOW_FALLBACKS", ""))
    if allow_fallbacks is not None:
        options["allow_fallbacks"] = allow_fallbacks

    order = [
        provider.strip()
        for provider in os.getenv("OPENROUTER_PROVIDER_ORDER", "").split(",")
        if provider.strip()
    ]
    if order:
        options["order"] = order

    prompt_limit = _parse_price("OPENROUTER_PROVIDER_MAX_PROMPT_PRICE")
    completion_limit = _parse_price("OPENROUTER_PROVIDER_MAX_COMPLETION_PRICE")
    if prompt_limit is not None or completion_limit is not None:
        max_price = dict(options.get("max_price", {}))
        if prompt_limit is not None:
            max_price["prompt"] = prompt_limit
        if completion_limit is not None:
            max_price["completion"] = completion_limit
        options["max_price"] = max_price

    return options


def _reasoning_options(model: str) -> dict[str, Any]:
    effort = os.getenv("OPENROUTER_REASONING_EFFORT", "").strip().casefold()
    if not effort and model == DEFAULT_DEEPSEEK_MODEL:
        effort = "none"
    return {"effort": effort} if effort else {}


def load_ai_config(
    env_file: Path,
    fallback_env_file: Path | None = None,
) -> AIConfig:
    """Load and validate OpenRouter settings from a local .env file."""
    used_fallback = False
    if fallback_env_file and fallback_env_file.exists() and fallback_env_file != env_file:
        load_dotenv(fallback_env_file)
        used_fallback = True
    if env_file.exists():
        load_dotenv(env_file, override=True)
    if used_fallback and not env_file.exists():
        print(f"[INFO] Using provider settings from {fallback_env_file}")

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "").strip()
    base_url = os.getenv(
        "OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL
    ).strip()

    missing = []
    if not api_key:
        missing.append("OPENROUTER_API_KEY")
    if not model:
        missing.append("OPENROUTER_MODEL")
    if missing:
        raise SystemExit(
            "\n[ERROR] Missing OpenRouter configuration: "
            + ", ".join(missing)
            + f"\n       Set them in '{env_file}'.\n"
        )

    default_headers: dict[str, str] = {}
    http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    app_title = os.getenv("OPENROUTER_APP_TITLE", "").strip()
    if http_referer:
        default_headers["HTTP-Referer"] = http_referer
    if app_title:
        default_headers["X-OpenRouter-Title"] = app_title

    return AIConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        default_headers=default_headers,
        provider_options=_provider_options(model),
        reasoning_options=_reasoning_options(model),
    )
