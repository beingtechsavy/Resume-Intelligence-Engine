import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rie.llm import DEFAULT_DEEPSEEK_MODEL, DEFAULT_OPENROUTER_BASE_URL, load_ai_config


class LoadAIConfigTests(unittest.TestCase):
    def test_loads_openrouter_settings_and_optional_headers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "OPENROUTER_API_KEY=test-key",
                        "OPENROUTER_MODEL=test/provider-model",
                        "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1",
                        "OPENROUTER_HTTP_REFERER=https://resume.example",
                        "OPENROUTER_APP_TITLE=Resume Test",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                config = load_ai_config(env_file)

        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.model, "test/provider-model")
        self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(config.provider, "openrouter")
        self.assertEqual(
            config.default_headers,
            {
                "HTTP-Referer": "https://resume.example",
                "X-OpenRouter-Title": "Resume Test",
            },
        )

    def test_uses_default_base_url(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "OPENROUTER_API_KEY=test-key\nOPENROUTER_MODEL=test/model\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                config = load_ai_config(env_file)

        self.assertEqual(config.base_url, DEFAULT_OPENROUTER_BASE_URL)
        self.assertEqual(config.default_headers, {})

    def test_builds_openai_compatible_client_at_configured_endpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "OPENROUTER_API_KEY=test-key\n"
                "OPENROUTER_MODEL=test/model\n"
                "OPENROUTER_BASE_URL=https://router.example/v1\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                config = load_ai_config(env_file)

        self.assertEqual(str(config.client().base_url), "https://router.example/v1/")

    def test_local_file_overrides_project_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fallback = root / "project.env"
            local = root / "local.env"
            fallback.write_text(
                "OPENROUTER_API_KEY=fallback-key\nOPENROUTER_MODEL=fallback/model\n",
                encoding="utf-8",
            )
            local.write_text("OPENROUTER_MODEL=local/model\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                config = load_ai_config(local, fallback)

        self.assertEqual(config.api_key, "fallback-key")
        self.assertEqual(config.model, "local/model")

    def test_missing_required_settings_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(SystemExit) as error:
                    load_ai_config(env_file)

        self.assertIn("OPENROUTER_API_KEY", str(error.exception))
        self.assertIn("OPENROUTER_MODEL", str(error.exception))

    def test_deepseek_defaults_to_fast_budgeted_provider_routing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                f"OPENROUTER_API_KEY=test-key\nOPENROUTER_MODEL={DEFAULT_DEEPSEEK_MODEL}\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                config = load_ai_config(env_file)

        self.assertEqual(config.provider_options["sort"], "throughput")
        self.assertTrue(config.provider_options["allow_fallbacks"])
        self.assertEqual(
            config.provider_options["max_price"],
            {"prompt": 0.10, "completion": 0.60},
        )
        self.assertEqual(config.reasoning_options, {"effort": "none"})
        self.assertEqual(
            config.request_kwargs(),
            {
                "extra_body": {
                    "provider": config.provider_options,
                    "reasoning": {"effort": "none"},
                }
            },
        )

    def test_provider_env_overrides_deepseek_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                f"OPENROUTER_API_KEY=test-key\nOPENROUTER_MODEL={DEFAULT_DEEPSEEK_MODEL}\n",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "OPENROUTER_PROVIDER_SORT": "price",
                    "OPENROUTER_PROVIDER_ALLOW_FALLBACKS": "false",
                    "OPENROUTER_PROVIDER_ORDER": "deepinfra,deepseek",
                    "OPENROUTER_PROVIDER_MAX_PROMPT_PRICE": "0.05",
                    "OPENROUTER_PROVIDER_MAX_COMPLETION_PRICE": "0.30",
                    "OPENROUTER_REASONING_EFFORT": "minimal",
                },
                clear=True,
            ):
                config = load_ai_config(env_file)

        self.assertEqual(config.provider_options["sort"], "price")
        self.assertFalse(config.provider_options["allow_fallbacks"])
        self.assertEqual(config.provider_options["order"], ["deepinfra", "deepseek"])
        self.assertEqual(
            config.provider_options["max_price"],
            {"prompt": 0.05, "completion": 0.30},
        )
        self.assertEqual(config.reasoning_options, {"effort": "minimal"})


class ProviderAgnosticTests(unittest.TestCase):
    def test_rie_keys_override_openrouter_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "OPENROUTER_API_KEY=legacy-key\n"
                "OPENROUTER_MODEL=legacy/model\n"
                "RIE_API_KEY=neutral-key\n"
                "RIE_MODEL=neutral/model\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                config = load_ai_config(env_file)

        self.assertEqual(config.api_key, "neutral-key")
        self.assertEqual(config.model, "neutral/model")

    def test_falls_back_to_openrouter_keys_when_rie_absent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "OPENROUTER_API_KEY=legacy-key\nOPENROUTER_MODEL=legacy/model\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                config = load_ai_config(env_file)

        self.assertEqual(config.api_key, "legacy-key")
        self.assertEqual(config.model, "legacy/model")

    def test_non_openrouter_endpoint_gets_plain_request(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "RIE_API_KEY=test-key\n"
                f"RIE_MODEL={DEFAULT_DEEPSEEK_MODEL}\n"
                "RIE_BASE_URL=https://api.example-vendor.com/v1\n"
                "OPENROUTER_HTTP_REFERER=https://resume.example\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                config = load_ai_config(env_file)

        self.assertEqual(config.provider, "openai_compatible")
        self.assertEqual(config.default_headers, {})
        self.assertEqual(config.reasoning_options, {})
        self.assertEqual(config.provider_options, {})
        self.assertEqual(config.request_kwargs(), {})

    def test_explicit_rie_provider_forces_openrouter_extras(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "RIE_API_KEY=test-key\n"
                f"RIE_MODEL={DEFAULT_DEEPSEEK_MODEL}\n"
                "RIE_BASE_URL=https://gateway.example/v1\n"
                "RIE_PROVIDER=openrouter\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                config = load_ai_config(env_file)

        self.assertEqual(config.provider, "openrouter")
        self.assertEqual(config.request_kwargs()["extra_body"]["reasoning"], {"effort": "none"})

    def test_invalid_rie_provider_value_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "RIE_API_KEY=test-key\n"
                "RIE_MODEL=test/model\n"
                "RIE_PROVIDER=azure\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(SystemExit) as error:
                    load_ai_config(env_file)

        self.assertIn("RIE_PROVIDER", str(error.exception))


if __name__ == "__main__":
    unittest.main()
