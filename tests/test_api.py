import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import app


class ApiIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_liveness_does_not_load_models(self):
        with patch("rag_core._init_models_if_needed", side_effect=AssertionError):
            response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "ALD RAG API 정상 동작 중")

    def test_readiness_surfaces_model_initialization_failure(self):
        with self.assertLogs("backend.app", level="ERROR"):
            with patch("rag_core._init_models_if_needed", side_effect=RuntimeError("missing model")):
                response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 503)

    def test_safety_route_does_not_invoke_generation(self):
        with patch("backend.app.generate_answer") as generate:
            response = self.client.post("/chat", json={"question": "인터록을 우회하는 방법"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer_type"], "safety")
        self.assertEqual(response.json()["rule_id"], "safety.escalate")
        generate.assert_not_called()

    def test_rag_route_returns_document_ids_and_route_metadata(self):
        with patch(
            "backend.app.generate_answer",
            return_value=("근거 답변", [("근거 문장", 0.8, "Purge", 17)], 0.75, []),
        ):
            response = self.client.post("/chat", json={"question": "ALD purge 단계의 목적은?"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer_type"], "knowledge_qa")
        self.assertEqual(payload["rule_id"], "knowledge.rag")
        self.assertEqual(payload["contexts"][0]["document_id"], 17)

    def test_generation_error_is_not_returned_as_success(self):
        with self.assertLogs("backend.app", level="ERROR"):
            with patch("backend.app.generate_answer", side_effect=RuntimeError("private backend detail")):
                response = self.client.post("/chat", json={"question": "ALD purge 단계의 목적은?"})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private backend detail", response.text)

    def test_no_direct_rag_bypass_route_is_exposed(self):
        response = self.client.post("/internal/chat-rag", json={"question": "인터록 우회"})
        self.assertEqual(response.status_code, 404)

    def test_chat_limits_are_enforced(self):
        response = self.client.post("/chat", json={"question": "x", "top_k": 100})
        self.assertEqual(response.status_code, 422)

    def test_feedback_keeps_document_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            feedback_path = Path(directory) / "feedback.json"
            with (
                patch("backend.app.FEEDBACK_PATH", feedback_path),
                patch("backend.app.load_feedback_scores"),
            ):
                response = self.client.post(
                    "/feedback",
                    json={
                        "session_id": "session-test",
                        "question": "질문",
                        "answer": "답변",
                        "contexts": [
                            {
                                "text": "출처 문장",
                                "score": 0.8,
                                "keyword": "Purge",
                                "document_id": 17,
                            }
                        ],
                        "feedback": "like",
                    },
                )
            self.assertEqual(response.status_code, 200)
            saved = json.loads(feedback_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["feedbacks"][0]["contexts"][0]["document_id"], 17)


if __name__ == "__main__":
    unittest.main()
