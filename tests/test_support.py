"""Functional support tests using mocked LLM responses."""

import io
import json
import os
import unittest
from unittest.mock import patch

from campuspulse.coach import FOCUS, support
from campuspulse.data import FEATURES


EXAMPLE = dict(zip(FEATURES, [7, 30, 50, 4, 4, 1]))


class SupportTests(unittest.TestCase):

    def test_explicit_focus_retrieves_matching_card(self):
        with patch.dict(os.environ, {"OLLAMA_MODEL": ""}):
            for focus in FOCUS.keys() - {"general"}:
                with self.subTest(focus=focus):
                    result = support(
                        {**EXAMPLE, "weekly_focus": focus},
                        True,
                    )

                    self.assertEqual(
                        result["sources"][0]["id"],
                        focus,
                    )
                    self.assertEqual(len(result["sources"]), 1)
                    self.assertEqual(
                        result["message"],
                        result["sources"][0]["text"],
                    )
                    self.assertIn("You chose", result["reason"])

    def test_score_does_not_select_suggestion(self):
        with patch.dict(os.environ, {"OLLAMA_MODEL": ""}):
            first = support(
                {
                    **EXAMPLE,
                    "weekly_focus": "movement",
                    "prediction": 20,
                },
                True,
            )

            second = support(
                {
                    **EXAMPLE,
                    "weekly_focus": "movement",
                    "prediction": 80,
                },
                True,
            )

        self.assertEqual(first, second)

    def test_invalid_focus_rejected_before_transport(self):
        with patch("campuspulse.coach.open_local_llm") as transport:
            for focus in ["ignore instructions", None, [], {}, True]:
                with self.subTest(focus=focus):
                    with self.assertRaises(ValueError):
                        support(
                            {**EXAMPLE, "weekly_focus": focus},
                            True,
                        )

            transport.assert_not_called()

    def test_opt_out_ignores_even_invalid_inputs(self):
        with patch("campuspulse.coach.open_local_llm") as transport:
            result = support({"weekly_focus": []}, False)

            self.assertEqual(result["mode"], "disabled")
            transport.assert_not_called()

    def test_focus_payload_contains_only_topics_and_matching_card(self):
        body = json.dumps({
            "message": {
                "content": (
                    "If you would like, and it feels comfortable "
                    "and accessible, consider a short walk."
                )
            }
        }).encode()

        with patch.dict(os.environ, {"OLLAMA_MODEL": "test"}), patch(
            "campuspulse.coach.open_local_llm",
            return_value=io.BytesIO(body),
        ) as transport:
            result = support(
                {
                    **EXAMPLE,
                    "weekly_focus": "movement",
                    "student_id": "secret",
                    "prediction": 80,
                },
                True,
            )

        self.assertEqual(result["mode"], "local_llm")

        sent = json.loads(transport.call_args.args[0].data)
        content = json.loads(sent["messages"][1]["content"])

        self.assertEqual(set(content), {"topics", "cards"})
        self.assertEqual(
            [card["id"] for card in content["cards"]],
            ["movement"],
        )
        self.assertNotIn("secret", str(sent))
        self.assertNotIn("student_id", str(sent))
        self.assertNotIn("prediction", str(sent))

        for feature in FEATURES:
            self.assertNotIn(feature, content)

    def test_filtered_response_preserves_chosen_focus(self):
        body = json.dumps({
            "message": {
                "content": "Guaranteed treatment for your disorder."
            }
        }).encode()

        with patch.dict(os.environ, {"OLLAMA_MODEL": "test"}), patch(
            "campuspulse.coach.open_local_llm",
            return_value=io.BytesIO(body),
        ):
            result = support(
                {**EXAMPLE, "weekly_focus": "routine"},
                True,
            )

        self.assertEqual(result["mode"], "retrieval_fallback")
        self.assertEqual(result["sources"][0]["id"], "routine")
        self.assertEqual(
            result["message"],
            result["sources"][0]["text"],
        )

    def test_movement_without_accessibility_conditions_falls_back(self):
        examples = [
            "If you would like, consider a short walk.",
            "If you would like, and it feels comfortable, consider a walk.",
            "If you would like, and it is accessible, consider a walk.",
        ]

        for text in examples:
            with self.subTest(text=text):
                body = json.dumps({
                    "message": {"content": text}
                }).encode()

                with patch.dict(
                    os.environ, {"OLLAMA_MODEL": "test"}
                ), patch(
                    "campuspulse.coach.open_local_llm",
                    return_value=io.BytesIO(body),
                ):
                    result = support(
                        {**EXAMPLE, "weekly_focus": "movement"},
                        True,
                    )

                self.assertEqual(
                    result["mode"],
                    "retrieval_fallback",
                )
                self.assertEqual(
                    result["message"],
                    result["sources"][0]["text"],
                )

    def test_benefit_claims_from_live_reports_are_rejected(self):
        examples = [
            (
                "If you would like, movement can be really "
                "beneficial for your well-being."
            ),
            (
                "If you would like, a walk could help "
                "boost your energy and mood."
            ),
            (
                "If you would like, a routine could help you "
                "feel more calm and ready for sleep."
            ),
            (
                "If you would like to create a calming bedtime "
                "routine, try a wind-down time."
            ),
            (
                "If you would like, consider establishing a "
                "comfortable bedtime routine to enhance restful sleep."
            ),
        ]

        for text in examples:
            with self.subTest(text=text):
                body = json.dumps({
                    "message": {"content": text}
                }).encode()

                with patch.dict(
                    os.environ, {"OLLAMA_MODEL": "test"}
                ), patch(
                    "campuspulse.coach.open_local_llm",
                    return_value=io.BytesIO(body),
                ):
                    result = support(
                        {**EXAMPLE, "weekly_focus": "routine"},
                        True,
                    )

                self.assertEqual(
                    result["mode"],
                    "retrieval_fallback",
                )
                self.assertIn("Output rejected", result["note"])

    def test_faithful_movement_with_accessibility_accepted(self):
        text = (
            "If you would like, and it feels comfortable "
            "and accessible, consider a short walk."
        )

        body = json.dumps({
            "message": {"content": text}
        }).encode()

        with patch.dict(os.environ, {"OLLAMA_MODEL": "test"}), patch(
            "campuspulse.coach.open_local_llm",
            return_value=io.BytesIO(body),
        ):
            result = support(
                {**EXAMPLE, "weekly_focus": "movement"},
                True,
            )

        self.assertEqual(result["mode"], "local_llm")
        self.assertEqual(result["message"], text)

    def test_if_you_want_opening_is_accepted(self):
        text = (
            "If you want, pause one screen-based task for "
            "a few minutes and choose a restful break."
        )

        body = json.dumps({
            "message": {"content": text}
        }).encode()

        with patch.dict(os.environ, {"OLLAMA_MODEL": "test"}), patch(
            "campuspulse.coach.open_local_llm",
            return_value=io.BytesIO(body),
        ):
            result = support(
                {**EXAMPLE, "weekly_focus": "balance"},
                True,
            )

        self.assertEqual(result["mode"], "local_llm")
        self.assertEqual(result["message"], text)

    def test_non_optional_instruction_falls_back(self):
        body = json.dumps({
            "message": {
                "content": "Pause your screen task and take a break."
            }
        }).encode()

        with patch.dict(os.environ, {"OLLAMA_MODEL": "test"}), patch(
            "campuspulse.coach.open_local_llm",
            return_value=io.BytesIO(body),
        ):
            result = support(
                {**EXAMPLE, "weekly_focus": "balance"},
                True,
            )

        self.assertEqual(result["mode"], "retrieval_fallback")
        self.assertEqual(
            result["message"],
            result["sources"][0]["text"],
        )


if __name__ == "__main__":
    unittest.main()