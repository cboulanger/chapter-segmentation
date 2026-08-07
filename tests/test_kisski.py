"""Unit tests for evaluation/kisski.py -- KISSKI model listing and
selection. See design spec
docs/superpowers/specs/2026-08-07-per-strategy-evaluation-design.md
"LLM cache refresh"."""

import unittest
from unittest.mock import MagicMock, patch

from evaluation.kisski import KisskiModel, fetch_kisski_models, select_full_regen, select_gap_fill, select_top5


class TestKisskiModelAvailability(unittest.TestCase):
    def test_zero_demand_is_available(self):
        self.assertEqual(KisskiModel(id="a", name="A", demand=0).availability, "available")

    def test_low_demand_is_busy(self):
        self.assertEqual(KisskiModel(id="a", name="A", demand=5).availability, "busy")

    def test_high_demand_is_very_busy(self):
        self.assertEqual(KisskiModel(id="a", name="A", demand=6).availability, "very busy")


class TestFetchKisskiModels(unittest.TestCase):
    def test_posts_to_models_endpoint_with_bearer_auth(self):
        response = MagicMock()
        response.json.return_value = {"data": [{"id": "model-a", "name": "Model A", "demand": 0}]}
        response.raise_for_status.return_value = None
        with patch("evaluation.kisski.httpx.post", return_value=response) as mock_post:
            models = fetch_kisski_models("https://example.test/v1", "secret-key")
        self.assertEqual(models, [KisskiModel(id="model-a", name="Model A", demand=0)])
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://example.test/v1/models")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret-key")

    def test_strips_trailing_slash_on_base_url(self):
        response = MagicMock()
        response.json.return_value = {"data": []}
        response.raise_for_status.return_value = None
        with patch("evaluation.kisski.httpx.post", return_value=response) as mock_post:
            fetch_kisski_models("https://example.test/v1/", "secret-key")
        self.assertEqual(mock_post.call_args[0][0], "https://example.test/v1/models")

    def test_missing_name_falls_back_to_id(self):
        response = MagicMock()
        response.json.return_value = {"data": [{"id": "model-a", "demand": 0}]}
        response.raise_for_status.return_value = None
        with patch("evaluation.kisski.httpx.post", return_value=response):
            models = fetch_kisski_models("https://example.test/v1", "secret-key")
        self.assertEqual(models[0].name, "model-a")


class TestSelectTop5(unittest.TestCase):
    def test_prefers_available_over_busy(self):
        models = [
            KisskiModel(id="busy-1", name="Busy 1", demand=3),
            KisskiModel(id="avail-1", name="Avail 1", demand=0),
        ]
        selected = select_top5(models)
        self.assertEqual([m.id for m in selected], ["avail-1", "busy-1"])

    def test_excludes_very_busy(self):
        models = [KisskiModel(id="very-busy", name="Very Busy", demand=10)]
        self.assertEqual(select_top5(models), [])

    def test_caps_at_five(self):
        models = [KisskiModel(id=f"m{i}", name=f"M{i}", demand=0) for i in range(8)]
        self.assertEqual(len(select_top5(models)), 5)

    def test_ascending_demand_order_within_same_availability(self):
        models = [
            KisskiModel(id="m-demand-2", name="M2", demand=2),
            KisskiModel(id="m-demand-1", name="M1", demand=1),
        ]
        selected = select_top5(models)
        self.assertEqual([m.id for m in selected], ["m-demand-1", "m-demand-2"])


class TestSelectGapFill(unittest.TestCase):
    def test_skips_already_covered_models(self):
        models = [
            KisskiModel(id="covered", name="Covered", demand=0),
            KisskiModel(id="uncovered", name="Uncovered", demand=0),
        ]
        selected = select_gap_fill(models, covered_model_ids={"covered"})
        self.assertEqual([m.id for m in selected], ["uncovered"])

    def test_excludes_very_busy_even_if_uncovered(self):
        models = [KisskiModel(id="very-busy", name="Very Busy", demand=10)]
        self.assertEqual(select_gap_fill(models, covered_model_ids=set()), [])

    def test_respects_limit(self):
        models = [KisskiModel(id=f"m{i}", name=f"M{i}", demand=0) for i in range(8)]
        selected = select_gap_fill(models, covered_model_ids=set(), limit=3)
        self.assertEqual(len(selected), 3)

    def test_all_covered_returns_empty(self):
        models = [KisskiModel(id="a", name="A", demand=0)]
        self.assertEqual(select_gap_fill(models, covered_model_ids={"a"}), [])


class TestSelectFullRegen(unittest.TestCase):
    def test_selects_only_cached_models(self):
        models = [
            KisskiModel(id="cached-1", name="Cached 1", demand=0),
            KisskiModel(id="uncached", name="Uncached", demand=0),
        ]
        selected = select_full_regen(models, cached_model_ids={"cached-1"})
        self.assertEqual([m.id for m in selected], ["cached-1"])

    def test_includes_very_busy_cached_models(self):
        # Unlike select_top5/select_gap_fill, a deliberate full
        # regeneration must not skip a cached model just because it's
        # currently busy -- the whole point is to redo everything cached.
        models = [KisskiModel(id="very-busy", name="Very Busy", demand=10)]
        selected = select_full_regen(models, cached_model_ids={"very-busy"})
        self.assertEqual([m.id for m in selected], ["very-busy"])

    def test_no_cap_on_result_size(self):
        models = [KisskiModel(id=f"m{i}", name=f"M{i}", demand=0) for i in range(8)]
        selected = select_full_regen(models, cached_model_ids={m.id for m in models})
        self.assertEqual(len(selected), 8)

    def test_ascending_demand_order(self):
        models = [
            KisskiModel(id="m-demand-2", name="M2", demand=2),
            KisskiModel(id="m-demand-1", name="M1", demand=1),
        ]
        selected = select_full_regen(models, cached_model_ids={"m-demand-1", "m-demand-2"})
        self.assertEqual([m.id for m in selected], ["m-demand-1", "m-demand-2"])

    def test_cached_model_no_longer_offered_is_silently_excluded(self):
        # fetch_kisski_models only returns what KISSKI currently offers --
        # a retired model can't appear in `models`, so it's naturally
        # dropped here; refresh_llm_cache.py's _main is what prints a
        # warning about it separately.
        models = [KisskiModel(id="still-offered", name="Still", demand=0)]
        selected = select_full_regen(models, cached_model_ids={"still-offered", "retired-model"})
        self.assertEqual([m.id for m in selected], ["still-offered"])


if __name__ == "__main__":
    unittest.main()
