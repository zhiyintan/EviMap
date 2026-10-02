from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import openai

from experiments.catalogue import llm
from experiments import evaluate_topic_quality_full as evaluator


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(
        message="Connection error.",
        request=httpx.Request("POST", "http://127.0.0.1:18020/v1/chat/completions"),
    )


def response(content: str = '{"ok": true}') -> SimpleNamespace:
    message = SimpleNamespace(content=content, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


class FakeCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def create(self, **_: object) -> object:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def fake_client(outcomes: list[object]) -> tuple[SimpleNamespace, FakeCompletions]:
    completions = FakeCompletions(outcomes)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


class BackendRecoveryTests(unittest.TestCase):
    def test_explicit_max_tokens_and_length_finish_raise_truncation(self) -> None:
        truncated = response('{"partial":')
        truncated.choices[0].finish_reason = "length"
        client, _ = fake_client([truncated])
        with (
            patch.object(llm, "_client_for", return_value=client),
            patch.object(llm, "_acquire_url", return_value="http://backend/v1"),
            patch.object(llm, "_release_url"),
        ):
            with self.assertRaises(llm.CompletionTruncatedError) as raised:
                llm.chat_completion(
                    [{"role": "user", "content": "test"}],
                    max_retries=0,
                    max_tokens=1024,
                )
        self.assertEqual(raised.exception.max_tokens, 1024)

    def test_connection_failure_heartbeats_until_healthy_then_retries(self) -> None:
        client, completions = fake_client([connection_error(), response("recovered")])
        with (
            patch.object(llm, "_client_for", return_value=client),
            patch.object(llm, "_acquire_url", return_value="http://backend/v1"),
            patch.object(llm, "_release_url"),
            patch.object(llm, "BACKEND_RECOVERY_WAIT_S", 600.0),
            patch.object(llm, "BACKEND_RECOVERY_HEARTBEAT_S", 60.0),
            patch.object(
                llm,
                "_probe_backend_health",
                side_effect=[False, False, True],
            ) as heartbeat,
            patch.object(llm.time, "sleep") as sleep,
        ):
            result = llm.chat_completion([{"role": "user", "content": "test"}], max_retries=0)

        self.assertEqual(result, "recovered")
        self.assertEqual(completions.calls, 2)
        self.assertEqual(heartbeat.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [60.0] * 3)

    def test_unhealthy_backend_gets_ten_heartbeats_before_final_retry(self) -> None:
        client, completions = fake_client([connection_error(), response("recovered")])
        with (
            patch.object(llm, "_client_for", return_value=client),
            patch.object(llm, "_acquire_url", return_value="http://backend/v1"),
            patch.object(llm, "_release_url"),
            patch.object(llm, "BACKEND_RECOVERY_WAIT_S", 600.0),
            patch.object(llm, "BACKEND_RECOVERY_HEARTBEAT_S", 60.0),
            patch.object(llm, "_probe_backend_health", return_value=False) as heartbeat,
            patch.object(llm.time, "sleep") as sleep,
        ):
            result = llm.chat_completion([{"role": "user", "content": "test"}], max_retries=0)

        self.assertEqual(result, "recovered")
        self.assertEqual(completions.calls, 2)
        self.assertEqual(heartbeat.call_count, 10)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [60.0] * 10)

    def test_second_connection_failure_is_final_for_that_request(self) -> None:
        client, completions = fake_client([connection_error(), connection_error()])
        with (
            patch.object(llm, "_client_for", return_value=client),
            patch.object(llm, "_acquire_url", return_value="http://backend/v1"),
            patch.object(llm, "_release_url"),
            patch.object(llm, "BACKEND_RECOVERY_WAIT_S", 600.0),
            patch.object(llm, "BACKEND_RECOVERY_HEARTBEAT_S", 60.0),
            patch.object(llm, "_probe_backend_health", return_value=True),
            patch.object(llm.time, "sleep") as sleep,
        ):
            with self.assertRaises(llm.BackendRecoveryExhausted):
                llm.chat_completion([{"role": "user", "content": "test"}], max_retries=3)

        self.assertEqual(completions.calls, 2)
        sleep.assert_called_once_with(60.0)

    def test_json_retry_loop_does_not_repeat_backend_recovery(self) -> None:
        failure = llm.BackendRecoveryExhausted("already retried")
        with patch.object(llm, "chat_completion", side_effect=failure) as completion:
            with self.assertRaises(llm.BackendRecoveryExhausted):
                llm.chat_json("system", "user", max_retries=3)
        self.assertEqual(completion.call_count, 1)

    def test_evaluator_retry_loop_does_not_repeat_backend_recovery(self) -> None:
        job = {
            "job_id": "j",
            "task": "alignment",
            "system": "EviMap",
            "payload": {
                "document_text": "test document",
                "assignments": [],
                "assignments_truncated": False,
            },
        }
        failure = llm.BackendRecoveryExhausted("already retried")
        with patch.object(evaluator, "chat_json", side_effect=failure) as judge:
            with self.assertRaises(llm.BackendRecoveryExhausted):
                evaluator.evaluate_job(job, "model", 0, 0.0, True, 2)
        self.assertEqual(judge.call_count, 1)

    def test_evaluator_escalates_output_budget_until_json_is_complete(self) -> None:
        job = {
            "job_id": "alignment::EviMap::d1",
            "task": "alignment",
            "system": "EviMap",
            "payload": {
                "document_text": "test document",
                "assignments": [],
                "assignments_truncated": False,
            },
        }
        complete = {
            "assignment_checks": [],
            "missing_themes": [],
            "rationale": "complete",
        }
        with (
            patch.object(
                evaluator,
                "chat_json",
                side_effect=[
                    llm.CompletionTruncatedError(512, "{"),
                    llm.CompletionTruncatedError(1024, "{"),
                    complete,
                ],
            ) as judge,
            patch.object(evaluator.time, "sleep"),
        ):
            row = evaluator.evaluate_job(job, "model", 0, 0.0, True, 2)

        self.assertEqual(
            [call.kwargs["max_tokens"] for call in judge.call_args_list],
            [512, 1024, 2048],
        )
        self.assertEqual(row["output_max_tokens"], 2048)
        self.assertEqual(row["adaptive_attempt"], 3)


if __name__ == "__main__":
    unittest.main()
