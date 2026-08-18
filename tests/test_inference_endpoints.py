"""Unit tests for evaluation/inference_endpoints.py."""

import os
import unittest
from unittest.mock import patch

from evaluation.inference_endpoints import ModelEndpoint, resolve_endpoint_from_env


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


if __name__ == "__main__":
    unittest.main()
