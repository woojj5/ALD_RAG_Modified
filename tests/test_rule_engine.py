import unittest

from rule_engine import route_question


class RuleEngineTest(unittest.TestCase):
    def test_safety_route_precedes_sop_and_rag(self):
        result = route_question("승인된 SOP에 따라 인터록을 우회할 수 있나요?")
        self.assertEqual(result.route, "safety")
        self.assertEqual(result.rule_id, "safety.escalate")
        self.assertIsNotNone(result.answer)

    def test_time_conversion_is_deterministic(self):
        result = route_question("500 ms를 s로 변환해줘")
        self.assertEqual(result.route, "calculation")
        self.assertEqual(result.answer, "500 ms = 0.5 s")

    def test_pressure_conversion_is_deterministic(self):
        result = route_question("1 kPa to Pa")
        self.assertEqual(result.answer, "1 kPa = 1000 Pa")

    def test_incompatible_units_are_rejected(self):
        result = route_question("1 s to Pa")
        self.assertEqual(result.rule_id, "unit.dimension_mismatch")

    def test_sop_route_does_not_invent_procedure(self):
        result = route_question("SOP-123 절차를 알려줘")
        self.assertEqual(result.route, "approved_sop")
        self.assertIsNotNone(result.answer)

    def test_equipment_question_uses_rag(self):
        result = route_question("VG12와 VG13의 차이를 알려줘")
        self.assertEqual(result.route, "equipment_lookup")
        self.assertIsNone(result.answer)

    def test_general_question_uses_rag(self):
        result = route_question("ALD에서 purge 단계가 중요한 이유는?")
        self.assertEqual(result.route, "knowledge_qa")
        self.assertIsNone(result.answer)


if __name__ == "__main__":
    unittest.main()
