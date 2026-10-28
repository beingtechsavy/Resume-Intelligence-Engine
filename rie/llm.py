"""Provider-agnostic LLM configuration for the Resume Intelligence Engine.

Settings come from the local environment (.env) using neutral ``RIE_*`` keys.
The historical ``OPENROUTER_*`` keys remain fully supported as the fallback in
each pair, so an existing .env keeps working unchanged.

``RIE_PROVIDER`` (``openrouter`` | ``openai_compatible``) gates every
OpenRouter-specific request extra: ``extra_body.provider`` routing,
``extra_body.reasoning`` and the ``HTTP-Referer`` / ``X-OpenRouter-Title``
headers. Vanilla OpenAI-compatible endpoints receiving those fields can 4xx,
so non-OpenRouter configurations get a plain request. When unset, the provider
is auto-detected from the resolved base_url host.

The cover-letter slot can run a different model than the resume slot:
``RIE_CL_MODEL`` (fallback ``OPENROUTER_CL_MODEL``) overrides ``RIE_MODEL``
for ``load_ai_config(..., slot="cover_letter")``. All other provider settings
are shared, so a split CV/CL configuration differs by at most one slug.
"""

from dataclasses import dataclass
import dataclasses
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek/deepseek-v4.1-flash-1017"
_PROVIDER_SORTS = {"price", "throughput", "latency"}

# Neutral key configures first; legacy key is the fallback.
_ENV_PAIRS = {
    "model": ("RIE_MODEL", "OPENROUTER_MODEL"),
    "cl_model": ("RIE_CL_MODEL", "OPENROUTER_CL_MODEL"),
    "api_key": ("RIE_API_KEY", "OPENROUTER_API_KEY"),
    "base_url": ("RIE_BASE_URL", "OPENROUTER_BASE_URL"),
    "provider_sort": ("RIE_PROVIDER_SORT", "OPENROUTER_PROVIDER_SORT"),
    "provider_allow_fallbacks": (
        "RIE_PROVIDER_ALLOW_FALLBACKS",
        "OPENROUTER_PROVIDER_ALLOW_FALLBACKS",
    ),
    "provider_order": ("RIE_PROVIDER_ORDER", "OPENROUTER_PROVIDER_ORDER"),
    "provider_max_prompt_price": (
        "RIE_PROVIDER_MAX_PROMPT_PRICE",
        "OPENROUTER_PROVIDER_MAX_PROMPT_PRICE",
    ),
    "provider_max_completion_price": (
        "RIE_PROVIDER_MAX_COMPLETION_PRICE",
        "OPENROUTER_PROVIDER_MAX_COMPLETION_PRICE",
    ),
    "reasoning_effort": ("RIE_REASONING_EFFORT", "OPENROUTER_REASONING_EFFORT"),
    "http_referer": ("RIE_HTTP_REFERER", "OPENROUTER_HTTP_REFERER"),
    "app_title": ("RIE_APP_TITLE", "OPENROUTER_APP_TITLE"),
}

_OPENROUTER_HOSTS = ("openrouter.ai",)
PROVIDER_OPENROUTER = "openrouter"
PROVIDER_OPENAI_COMPATIBLE = "openai_compatible"


@dataclass(frozen=True)
class AIConfig:
    """Provider settings loaded from the local environment."""

    api_key: str
    model: str
    base_url: str
    provider: str
    default_headers: dict[str, str]
    provider_options: dict[str, Any]
    reasoning_options: dict[str, Any]

    def client(self) -> OpenAI:
        """Create an OpenAI-compatible client at the configured endpoint."""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=self.default_headers or None,
        )

    def request_kwargs(self) -> dict[str, Any]:
        """Return provider-specific request fields for the selected model.

        OpenRouter routes and reasoning extras are only sent when the provider
        is OpenRouter; other OpenAI-compatible endpoints receive a plain body.
        """
        if self.provider != PROVIDER_OPENROUTER:
            return {}
        extra_body: dict[str, Any] = {}
        if self.provider_options:
            extra_body["provider"] = self.provider_options
        if self.reasoning_options:
            extra_body["reasoning"] = self.reasoning_options
        return {"extra_body": extra_body} if extra_body else {}

    def with_model(self, model: str) -> "AIConfig":
        """Copy of this config pointed at another model on the same endpoint.

        Model-dependent option defaults (the DeepSeek routing/reasoning
        special case) are recomputed for the new slug; explicit env-level
        provider/reasoning settings are intentionally not reapplied because
        they are resolved before a config instance exists.
        """
        if model == self.model:
            return self
        is_openrouter = self.provider == PROVIDER_OPENROUTER
        return dataclasses.replace(
            self,
            model=model,
            provider_options=_provider_options(model, is_openrouter),
            reasoning_options=_reasoning_options(model, is_openrouter),
        )


def _env_setting(name: str) -> str:
    """Resolve a setting from its RIE_*/legacy pair, first-wins."""
    primary, legacy = _ENV_PAIRS[name]
    for key in (primary, legacy):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return ""


def _parse_bool(value: str, default: bool | None = None) -> bool | None:
    normalized = value.strip().casefold()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(
        "\n[ERROR] Invalid boolean provider setting: "
        f"'{value}'. Use true/false.\n"
    )


def _parse_price(raw: str, label: str) -> float | None:
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(
            f"\n[ERROR] {label} must be a number in USD per million tokens.\n"
        ) from None
    if value < 0:
        raise SystemExit(f"\n[ERROR] {label} must not be negative.\n")
    return value


def _resolve_provider(base_url: str) -> str:
    """Detect the provider flavour from RIE_PROVIDER or the base_url host."""
    explicit = os.getenv("RIE_PROVIDER", "").strip().casefold()
    if explicit:
        if explicit not in {PROVIDER_OPENROUTER, PROVIDER_OPENAI_COMPATIBLE}:
            raise SystemExit(
                "\n[ERROR] RIE_PROVIDER must be one of: "
                f"{PROVIDER_OPENROUTER}, {PROVIDER_OPENAI_COMPATIBLE}.\n"
            )
        return explicit

    host = base_url.split("//", 1)[-1].split("/", 1)[0].casefold()
    if host in _OPENROUTER_HOSTS:
        return PROVIDER_OPENROUTER
    return PROVIDER_OPENAI_COMPATIBLE


def _provider_options(model: str, is_openrouter: bool) -> dict[str, Any]:
    """Build a conservative provider policy without hard-pinning one host."""
    options: dict[str, Any] = {}

    if is_openrouter:
        # The DeepSeek model is used for two concurrent document calls. Prefer
        # throughput while keeping the route below the current balanced price
        # band; OpenRouter can fall back if the preferred endpoint is down.
        if model == DEFAULT_DEEPSEEK_MODEL:
            options.update(
                {
                    "sort": "throughput",
                    "allow_fallbacks": True,
                    "max_price": {"prompt": 0.10, "completion": 0.60},
                }
            )

    sort = _env_setting("provider_sort").casefold()
    if sort:
        if sort not in _PROVIDER_SORTS:
            raise SystemExit(
                "\n[ERROR] Provider sort setting must be one of: "
                f"{', '.join(sorted(_PROVIDER_SORTS))}.\n"
            )
        options["sort"] = sort

    allow_fallbacks = _parse_bool(_env_setting("provider_allow_fallbacks"))
    if allow_fallbacks is not None:
        options["allow_fallbacks"] = allow_fallbacks

    order = [
        provider.strip()
        for provider in _env_setting("provider_order").split(",")
        if provider.strip()
    ]
    if order:
        options["order"] = order

    labels = _ENV_PAIRS["provider_max_prompt_price"]
    prompt_limit = _parse_price(
        _env_setting("provider_max_prompt_price"), labels[0]
    )
    completion_limit = _parse_price(
        _env_setting("provider_max_completion_price"),
        _ENV_PAIRS["provider_max_completion_price"][0],
    )
    if prompt_limit is not None or completion_limit is not None:
        max_price = dict(options.get("max_price", {}))
        if prompt_limit is not None:
            max_price["prompt"] = prompt_limit
        if completion_limit is not None:
            max_price["completion"] = completion_limit
        options["max_price"] = max_price

    return options


def _reasoning_options(model: str, is_openrouter: bool) -> dict[str, Any]:
    effort = _env_setting("reasoning_effort").casefold()
    if not effort and is_openrouter and model == DEFAULT_DEEPSEEK_MODEL:
        effort = "none"
    return {"effort": effort} if effort else {}


def cl_model_override(env_file: Path | None = None) -> str:
    """Return the cover-letter model override, or "" when unset.

    Assumes dotenv state already resolved by ``load_ai_config``; loading the
    env file here as well is harmless and keeps the helper self-sufficient.
    """
    if env_file is not None and env_file.exists():
        load_dotenv(env_file, override=True)
    return _env_setting("cl_model")

def load_ai_config(
    env_file: Path,
    fallback_env_file: Path | None = None,
    slot: str = "cv",
) -> AIConfig:
    """Load and validate provider settings from a local .env file.

    ``slot="cover_letter"`` picks up the ``RIE_CL_MODEL`` override so the
    cover-letter call can use a different model than the resume call.
    """
    used_fallback = False
    if fallback_env_file and fallback_env_file.exists() and fallback_env_file != env_file:
        load_dotenv(fallback_env_file)
        used_fallback = True
    if env_file.exists():
        load_dotenv(env_file, override=True)
    if used_fallback and not env_file.exists():
        print(f"[INFO] Using provider settings from {fallback_env_file}")

    api_key = _env_setting("api_key")
    model = _env_setting("model")
    if slot == "cover_letter":
        model = _env_setting("cl_model") or model
    base_url = _env_setting("base_url") or DEFAULT_OPENROUTER_BASE_URL

    missing = []
    if not api_key:
        missing.append("RIE_API_KEY (or OPENROUTER_API_KEY)")
    if not model:
        missing.append("RIE_MODEL (or OPENROUTER_MODEL)")
    if missing:
        raise SystemExit(
            "\n[ERROR] Missing provider configuration: "
            + ", ".join(missing)
            + f"\n       Set them in '{env_file}'.\n"
        )

    provider = _resolve_provider(base_url)
    is_openrouter = provider == PROVIDER_OPENROUTER

    default_headers: dict[str, str] = {}
    if is_openrouter:
        http_referer = _env_setting("http_referer")
        app_title = _env_setting("app_title")
        if http_referer:
            default_headers["HTTP-Referer"] = http_referer
        if app_title:
            default_headers["X-OpenRouter-Title"] = app_title

    return AIConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
        default_headers=default_headers,
        provider_options=_provider_options(model, is_openrouter),
        reasoning_options=_reasoning_options(model, is_openrouter),
    )
