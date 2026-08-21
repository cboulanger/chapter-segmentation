"""Unit tests for evaluation/inference_endpoints.py."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from evaluation.inference_endpoints import (
    ModelEndpoint, OpenAICompatibleLLMClient, load_mpcdf_sessions, resolve_endpoint_from_env,
    resolve_endpoints_from_config_file,
)

_NO_SESSION_FILE = Path("/nonexistent/mpcdf-sessions-test-isolation.txt")


class TestResolveEndpointFromEnv(unittest.TestCase):
    def test_builds_endpoint_from_all_three_vars(self):
        env = {
            "MPCDF_A_BASE_URL": "https://example.invalid/v1",
            "MPCDF_A_API_KEY": "secret-key",
            "MPCDF_A_MODEL": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        }
        with patch.dict(os.environ, env, clear=False):
            endpoint = resolve_endpoint_from_env("MPCDF_A")

        self.assertIsInstance(endpoint, ModelEndpoint)
        self.assertEqual(endpoint.label, "MPCDF_A")
        self.assertEqual(endpoint.model_id, "Qwen/Qwen3-VL-30B-A3B-Instruct")
        self.assertIn("https://example.invalid/v1", str(endpoint.client.base_url))
        self.assertEqual(endpoint.client.api_key, "secret-key")

    def test_missing_base_url_raises_naming_that_var(self):
        env = {"MPCDF_B_API_KEY": "k", "MPCDF_B_MODEL": "m"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                resolve_endpoint_from_env("MPCDF_B")
        self.assertIn("MPCDF_B_BASE_URL", str(ctx.exception))

    def test_missing_api_key_raises_naming_that_var(self):
        env = {"MPCDF_C_BASE_URL": "https://example.invalid/v1", "MPCDF_C_MODEL": "m"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                resolve_endpoint_from_env("MPCDF_C")
        self.assertIn("MPCDF_C_API_KEY", str(ctx.exception))

    def test_missing_model_raises_naming_that_var(self):
        env = {"MPCDF_D_BASE_URL": "https://example.invalid/v1", "MPCDF_D_API_KEY": "k"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                resolve_endpoint_from_env("MPCDF_D")
        self.assertIn("MPCDF_D_MODEL", str(ctx.exception))

    def test_all_missing_names_all_three(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                resolve_endpoint_from_env("MPCDF_E")
        message = str(ctx.exception)
        self.assertIn("MPCDF_E_BASE_URL", message)
        self.assertIn("MPCDF_E_API_KEY", message)
        self.assertIn("MPCDF_E_MODEL", message)


class TestLoadMpcdfSessions(unittest.TestCase):
    _TABLE_A = (
        "framework\tvLLM\n"
        "framework_args\t--model=mistralai/Pixtral-12B-2409 --tensor-parallel-size=2 --trust-remote-code\n"
        "framework_envs\t\n"
        "host\t10.179.7.234:24100\n"
        "key\tsekret-a\n"
        "reservation\t\n"
        "url\thttps://llm.mpcdf.mpg.de/session-a\n"
    )
    _TABLE_B = (
        "framework\tvLLM\n"
        "framework_args\t--model=Qwen/Qwen3-Omni-30B-A3B-Instruct --tensor-parallel-size=2\n"
        "key\tsekret-b\n"
        "url\thttps://llm.mpcdf.mpg.de/session-b/v1\n"
    )

    def test_parses_two_pasted_tables_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.txt"
            path.write_text(self._TABLE_A + "\n" + self._TABLE_B)

            sessions = load_mpcdf_sessions(path)

        self.assertEqual(len(sessions), 2)
        self.assertEqual(sessions[0], {
            "base_url": "https://llm.mpcdf.mpg.de/session-a/v1",
            "api_key": "sekret-a",
            "model": "mistralai/Pixtral-12B-2409",
        })
        # Already has a /v1 suffix -- must not be doubled.
        self.assertEqual(sessions[1], {
            "base_url": "https://llm.mpcdf.mpg.de/session-b/v1",
            "api_key": "sekret-b",
            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        })

    def test_missing_file_returns_empty_list(self):
        self.assertEqual(load_mpcdf_sessions(_NO_SESSION_FILE), [])

    def test_block_missing_model_arg_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.txt"
            path.write_text("key\tk\nurl\thttps://example.invalid\n")

            sessions = load_mpcdf_sessions(path)

        self.assertEqual(sessions, [])


class TestResolveEndpointsFromConfigFile(unittest.TestCase):
    _TABLE_A = TestLoadMpcdfSessions._TABLE_A
    _TABLE_B = TestLoadMpcdfSessions._TABLE_B

    def test_builds_one_endpoint_per_table_labeled_by_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.txt"
            path.write_text(self._TABLE_A + "\n" + self._TABLE_B)

            endpoints = resolve_endpoints_from_config_file(path)

        self.assertEqual(len(endpoints), 2)
        self.assertEqual(endpoints[0].label, "config:A")
        self.assertEqual(endpoints[0].model_id, "mistralai/Pixtral-12B-2409")
        self.assertIn("session-a", str(endpoints[0].client.base_url))
        self.assertEqual(endpoints[1].label, "config:B")
        self.assertEqual(endpoints[1].model_id, "Qwen/Qwen3-Omni-30B-A3B-Instruct")
        self.assertIn("session-b", str(endpoints[1].client.base_url))

    def test_missing_file_raises_naming_the_path(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_endpoints_from_config_file(_NO_SESSION_FILE)
        self.assertIn(str(_NO_SESSION_FILE), str(ctx.exception))

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.txt"
            path.write_text("")
            with self.assertRaises(ValueError):
                resolve_endpoints_from_config_file(path)


class TestOpenAICompatibleLLMClient(unittest.IsolatedAsyncioTestCase):
    async def test_uses_the_given_client_not_a_new_one(self):
        fake_client = MagicMock()
        message = MagicMock()
        message.content = "hello"
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        fake_client.chat.completions.create = AsyncMock(return_value=response)

        llm_client = OpenAICompatibleLLMClient(model="model-x", client=fake_client)
        result = await llm_client.generate("prompt", max_tokens=10, temperature=0.0)

        self.assertEqual(result, "hello")
        fake_client.chat.completions.create.assert_awaited_once_with(
            model="model-x", messages=[{"role": "user", "content": "prompt"}], max_tokens=10, temperature=0.0,
        )


if __name__ == "__main__":
    unittest.main()
