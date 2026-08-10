"""Unit tests for evaluation/nuextract2_common.py -- the NuExtract-2.0-4B
fine-tuning pilot's shared, dependency-light helpers. See design spec
docs/superpowers/specs/2026-08-10-nuextract2-finetuning-pilot-design.md.

None of these tests import torch/transformers/peft -- every heavy
dependency (a real tokenizer, a real chat template) is injected as a
plain callable/value, so this file runs under the default `uv run pytest`
with no `nuextract-finetune` extras installed."""

import json
import unittest

from evaluation.nuextract2_common import (
    build_chat_prompt,
    build_target,
    pad_batch,
    stratified_split,
    tokenize_training_example,
)


class TestBuildChatPrompt(unittest.TestCase):
    def test_passes_single_user_message_with_text(self):
        captured = {}

        def fake_apply_chat_template(messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return "PROMPT"

        result = build_chat_prompt("scan window text", {"chapters": []}, fake_apply_chat_template)

        self.assertEqual(result, "PROMPT")
        self.assertEqual(captured["messages"], [{"role": "user", "content": "scan window text"}])
        self.assertEqual(captured["kwargs"]["tokenize"], False)
        self.assertEqual(captured["kwargs"]["add_generation_prompt"], True)
        self.assertIn('"chapters"', captured["kwargs"]["template"])


class TestBuildTarget(unittest.TestCase):
    def test_uses_start_of_citation_pages_as_page_number(self):
        chapters = [{"title": "Introduction", "authors": ["A"], "citation_pages": "1-31"}]
        self.assertEqual(
            build_target(chapters),
            {"chapters": [{"title": "Introduction", "authors": ["A"], "printed_page_number": "1"}]},
        )

    def test_null_citation_pages_becomes_null_page_number(self):
        chapters = [{"title": "Foreword", "authors": [], "citation_pages": None}]
        self.assertEqual(
            build_target(chapters),
            {"chapters": [{"title": "Foreword", "authors": [], "printed_page_number": None}]},
        )

    def test_handles_roman_numeral_citation_pages(self):
        chapters = [{"title": "Foreword", "authors": [], "citation_pages": "vii-ix"}]
        self.assertEqual(build_target(chapters)["chapters"][0]["printed_page_number"], "7")

    def test_defaults_missing_authors_to_empty_list(self):
        chapters = [{"title": "Index", "citation_pages": None}]
        self.assertEqual(build_target(chapters)["chapters"][0]["authors"], [])


class TestStratifiedSplit(unittest.TestCase):
    def test_splits_each_corpus_independently_by_eval_count(self):
        corpus_stems = {
            "open-access": [f"oa{i}" for i in range(10)],
            "copyrighted-scans": [f"cs{i}" for i in range(5)],
        }
        result = stratified_split(
            corpus_stems, eval_counts={"open-access": 3, "copyrighted-scans": 2}, seed=1,
        )

        eval_by_corpus = {"open-access": 0, "copyrighted-scans": 0}
        for corpus, _stem in result["eval"]:
            eval_by_corpus[corpus] += 1
        self.assertEqual(eval_by_corpus, {"open-access": 3, "copyrighted-scans": 2})
        self.assertEqual(len(result["train"]), 10)

    def test_train_and_eval_are_disjoint_and_cover_every_book(self):
        corpus_stems = {"open-access": [f"oa{i}" for i in range(6)]}
        result = stratified_split(corpus_stems, eval_counts={"open-access": 2}, seed=7)

        train_set = set(result["train"])
        eval_set = set(result["eval"])
        self.assertEqual(train_set & eval_set, set())
        self.assertEqual(train_set | eval_set, {("open-access", f"oa{i}") for i in range(6)})

    def test_deterministic_for_same_seed(self):
        corpus_stems = {"open-access": [f"oa{i}" for i in range(8)]}
        result_a = stratified_split(corpus_stems, eval_counts={"open-access": 3}, seed=99)
        result_b = stratified_split(corpus_stems, eval_counts={"open-access": 3}, seed=99)
        self.assertEqual(result_a, result_b)

    def test_corpus_absent_from_eval_counts_goes_entirely_to_train(self):
        corpus_stems = {"pending": ["p0", "p1"]}
        result = stratified_split(corpus_stems, eval_counts={"open-access": 8}, seed=1)
        self.assertEqual(result["eval"], [])
        self.assertEqual(len(result["train"]), 2)


class TestTokenizeTrainingExample(unittest.TestCase):
    @staticmethod
    def _fake_apply_chat_template(messages, **kwargs):
        return f"PROMPT[{messages[0]['content']}]"

    @staticmethod
    def _fake_encode(text):
        return [ord(c) for c in text]  # one fake token id per character

    def test_masks_prompt_and_keeps_completion_in_labels(self):
        result = tokenize_training_example(
            "abc", {"chapters": []}, self._fake_apply_chat_template, self._fake_encode, eos_token_id=999,
        )
        prompt_len = len(self._fake_encode("PROMPT[abc]"))
        completion_ids = self._fake_encode(json.dumps({"chapters": []}))

        self.assertEqual(result["labels"][:prompt_len], [-100] * prompt_len)
        self.assertEqual(result["labels"][prompt_len:-1], completion_ids)
        self.assertEqual(result["labels"][-1], 999)
        self.assertEqual(len(result["input_ids"]), len(result["labels"]))
        self.assertEqual(result["attention_mask"], [1] * len(result["input_ids"]))


class TestPadBatch(unittest.TestCase):
    def test_pads_to_longest_example(self):
        examples = [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [1], "labels": [1], "attention_mask": [1]},
        ]
        batch = pad_batch(examples, pad_token_id=0)
        self.assertEqual(batch["input_ids"], [[1, 2, 3], [1, 0, 0]])
        self.assertEqual(batch["labels"], [[-100, 2, 3], [1, -100, -100]])
        self.assertEqual(batch["attention_mask"], [[1, 1, 1], [1, 0, 0]])


if __name__ == "__main__":
    unittest.main()
