import unittest
import json
import tempfile
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import rag_core
from scripts.doc_management import generate_docs


class RagRuntimeTest(unittest.TestCase):
    def test_remote_generation_uses_openai_compatible_chat_contract(self):
        response = unittest.mock.Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "  근거 기반 답변  "}}],
        }
        messages = [{"role": "user", "content": "질문"}]

        with (
            patch.object(rag_core, "LLM_BASE_URL", "http://llm.local/v1"),
            patch.object(rag_core, "LLM_REMOTE_MODEL", "ald-model"),
            patch.object(rag_core, "LLM_API_KEY", "test-key"),
            patch("rag_core._init_models_if_needed", side_effect=AssertionError),
            patch("httpx.post", return_value=response) as post,
        ):
            answer = rag_core.generate_text(messages, max_new_tokens=80)

        self.assertEqual(answer, "근거 기반 답변")
        post.assert_called_once_with(
            "http://llm.local/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={
                "model": "ald-model",
                "messages": messages,
                "max_tokens": 80,
                "temperature": 0.6,
                "top_p": 0.9,
            },
            timeout=60.0,
        )
        response.raise_for_status.assert_called_once()

    def test_remote_generation_rejects_malformed_provider_response(self):
        response = unittest.mock.Mock()
        response.json.return_value = {"choices": []}

        with (
            patch.object(rag_core, "LLM_BASE_URL", "http://llm.local/v1"),
            patch("httpx.post", return_value=response),
        ):
            with self.assertRaisesRegex(ValueError, "invalid chat-completions"):
                rag_core.generate_text([{"role": "user", "content": "질문"}])

    def test_cache_key_includes_generation_limit(self):
        first = rag_core._get_cache_key("질문", 3, None, 128)
        second = rag_core._get_cache_key("질문", 3, None, 256)
        self.assertNotEqual(first, second)

    def test_document_generation_uses_canonical_list_schema(self):
        documents = [{"id": 1, "keywords": ["ALD"], "text": "기존 지식"}]
        with tempfile.TemporaryDirectory() as directory:
            docs_path = Path(directory) / "docs.json"
            with patch.object(generate_docs, "DOCS_PATH", docs_path):
                generate_docs.save_docs(documents)
                loaded = generate_docs.load_docs()
        self.assertEqual(loaded, documents)

    def test_feedback_uses_document_id_and_invalidates_answer_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            feedback_path = Path(directory) / "feedback.json"
            feedback_path.write_text(
                json.dumps(
                    {
                        "feedbacks": [
                            {
                                "feedback": "like",
                                "contexts": [{"document_id": 17, "text": "matching source"}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(rag_core, "DOCS", ["source text"]),
                patch.object(rag_core, "DOC_ITEMS", [{"id": 17}]),
                patch.object(rag_core, "FEEDBACK_SCORES", {}),
                patch.object(rag_core, "FEEDBACK_LAST_MODIFIED", 0.0),
                patch.object(rag_core, "QUERY_CACHE", OrderedDict({"stale": ("answer", [], 0.0, [])})) as cache,
            ):
                rag_core.load_feedback_scores(feedback_path, force_reload=True)
                self.assertGreater(rag_core.FEEDBACK_SCORES[0], 0)
                self.assertFalse(cache)


if __name__ == "__main__":
    unittest.main()
