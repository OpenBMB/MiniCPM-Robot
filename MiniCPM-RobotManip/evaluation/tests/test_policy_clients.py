from __future__ import annotations

import unittest
from unittest import mock

import numpy as np


def action_chunk() -> np.ndarray:
    steps = np.arange(30, dtype=np.float32)[:, None] * 100
    dims = np.arange(80, dtype=np.float32)[None, :]
    return (steps + dims)[None, ...]


class FakeWireClient:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.calls: list[dict] = []
        self.closed = False
        self.actions = action_chunk()

    def get_server_metadata(self) -> dict:
        return {
            "action_chunk_size": 30,
            "action_normalization": "none",
            "actions_ready_for_execution": True,
        }

    def predict_action(self, payload: dict) -> dict:
        self.calls.append(payload)
        return {
            "ok": True,
            "status": "ok",
            "type": "inference_result",
            "data": {"actions": self.actions},
        }

    def close(self) -> None:
        self.closed = True


class LiberoModelClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from evaluation.libero import model2libero_interface
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"LIBERO client dependency unavailable: {exc}") from exc
        cls.module = model2libero_interface

    def test_two_views_and_chunk_cache_without_normalization_fields(self) -> None:
        fake = FakeWireClient()
        with mock.patch.object(
            self.module,
            "WebsocketClientPolicy",
            return_value=fake,
        ):
            client = self.module.ModelClient(
                action_ensemble=False,
                image_size=(8, 8),
            )

        first_view = np.full((8, 8, 3), 11, dtype=np.uint8)
        wrist_view = np.full((8, 8, 3), 22, dtype=np.uint8)
        example = {
            "image": [first_view, wrist_view],
            "lang": "pick up the block",
        }

        first = client.step(example, step=0)["raw_action"]
        second = client.step(example, step=1)["raw_action"]
        client.step(example, step=30)

        self.assertEqual(len(fake.calls), 2)
        payload = fake.calls[0]
        self.assertEqual(set(payload), {"examples"})
        sent_example = payload["examples"][0]
        self.assertEqual(set(sent_example), {"image", "lang"})
        np.testing.assert_array_equal(sent_example["image"][0], first_view)
        np.testing.assert_array_equal(sent_example["image"][1], wrist_view)
        np.testing.assert_array_equal(first["world_vector"], [0, 1, 2])
        np.testing.assert_array_equal(first["rotation_delta"], [3, 4, 5])
        np.testing.assert_array_equal(first["open_gripper"], [6])
        np.testing.assert_array_equal(second["world_vector"], [100, 101, 102])

    def test_rejects_unsuccessful_or_nonfinite_response(self) -> None:
        fake = FakeWireClient()
        with mock.patch.object(
            self.module,
            "WebsocketClientPolicy",
            return_value=fake,
        ):
            client = self.module.ModelClient(
                action_ensemble=False,
                image_size=(8, 8),
            )

        example = {
            "image": [np.zeros((8, 8, 3), dtype=np.uint8)],
            "lang": "test",
        }
        fake.predict_action = lambda payload: {
            "ok": False,
            "error": {"message": "failed"},
        }
        with self.assertRaisesRegex(RuntimeError, "ok=true"):
            client.step(example, step=0)

        fake.predict_action = lambda payload: {
            "ok": True,
            "data": {
                "actions": np.full((1, 30, 80), np.nan, dtype=np.float32)
            },
        }
        with self.assertRaisesRegex(ValueError, "finite"):
            client.step(example, step=0)


class RoboTwinModelClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from evaluation.robotwin import model2robotwin_interface
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(
                f"RoboTwin client dependency unavailable: {exc}"
            ) from exc
        cls.module = model2robotwin_interface

    def test_three_view_order_chunk_cache_and_action_reorder(self) -> None:
        fake = FakeWireClient()
        with mock.patch.object(
            self.module,
            "WebsocketClientPolicy",
            return_value=fake,
        ):
            client = self.module.ModelClient(image_size=(8, 8))

        class FakeTaskEnv:
            take_action_cnt = 0

            def __init__(self) -> None:
                self.action = None

            def get_instruction(self):
                return "stack the blocks"

            def take_action(self, action):
                self.action = action

        task_env = FakeTaskEnv()
        observation = {
            "observation": {
                "head_camera": {
                    "rgb": np.full((8, 8, 3), 1, dtype=np.uint8)
                },
                "left_camera": {
                    "rgb": np.full((8, 8, 3), 2, dtype=np.uint8)
                },
                "right_camera": {
                    "rgb": np.full((8, 8, 3), 3, dtype=np.uint8)
                },
            },
            "joint_action": {"vector": np.arange(14, dtype=np.float32)},
        }

        self.module.eval(task_env, client, observation)
        client.step(
            {
                "image": [
                    observation["observation"]["head_camera"]["rgb"],
                    observation["observation"]["left_camera"]["rgb"],
                    observation["observation"]["right_camera"]["rgb"],
                ],
                "lang": "stack the blocks",
            },
            step=1,
        )
        client.step(
            {
                "image": [
                    observation["observation"]["head_camera"]["rgb"],
                    observation["observation"]["left_camera"]["rgb"],
                    observation["observation"]["right_camera"]["rgb"],
                ],
                "lang": "stack the blocks",
            },
            step=30,
        )

        self.assertEqual(len(fake.calls), 2)
        sent = fake.calls[0]["examples"][0]
        self.assertNotIn("state", sent)
        self.assertEqual([int(view[0, 0, 0]) for view in sent["image"]], [1, 2, 3])
        np.testing.assert_array_equal(
            task_env.action,
            [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13],
        )

    def test_non_absolute_mode_and_bad_image_fail_fast(self) -> None:
        fake = FakeWireClient()
        with mock.patch.object(
            self.module,
            "WebsocketClientPolicy",
            return_value=fake,
        ):
            with self.assertRaisesRegex(ValueError, "action_mode='abs'"):
                self.module.ModelClient(action_mode="delta")

            client = self.module.ModelClient(image_size=(8, 8))

        with self.assertRaisesRegex(ValueError, "dtype uint8"):
            client.step(
                {
                    "image": [np.zeros((8, 8, 3), dtype=np.float32)],
                    "lang": "test",
                },
                step=0,
            )


class LegacyWebsocketsClientCompatibilityTest(unittest.TestCase):
    def test_connect_without_proxy_parameter(self) -> None:
        from deployment.model_server.tools import msgpack_numpy
        from deployment.model_server.tools import websocket_policy_client

        class FakeConnection:
            def __init__(self) -> None:
                self.closed = False

            def recv(self, timeout=None):
                del timeout
                return msgpack_numpy.packb({"action_chunk_size": 30})

            def close(self):
                self.closed = True

        connection = FakeConnection()

        # Mirrors the websockets 13 connect signature relevant to this client:
        # there is no `proxy` keyword.
        def legacy_connect(
            uri,
            *,
            compression,
            max_size,
            open_timeout,
            ping_interval,
        ):
            del uri, compression, max_size, open_timeout, ping_interval
            return connection

        with mock.patch.object(
            websocket_policy_client.websockets.sync.client,
            "connect",
            legacy_connect,
        ):
            client = websocket_policy_client.WebsocketClientPolicy()

        self.assertEqual(client.get_server_metadata()["action_chunk_size"], 30)
        client.close()
        self.assertTrue(connection.closed)


class ReadinessProbeTest(unittest.TestCase):
    def test_transport_exception_is_retryable_but_metadata_mismatch_is_not(self) -> None:
        from evaluation.common import probe_server

        class FakeClient:
            def __init__(self, metadata, *, fail_ping=False) -> None:
                self.metadata = metadata
                self.fail_ping = fail_ping

            def get_server_metadata(self):
                return self.metadata

            def ping(self, **kwargs):
                del kwargs
                if self.fail_ping:
                    raise TimeoutError("transient timeout")
                return {"ok": True, "type": "ping"}

            def close(self):
                pass

        valid_metadata = {
            "server": "minicpm_robot_manip",
            "ckpt_path": "fake/model",
            "default_embodiment_id": 0,
            "action_normalization": "none",
            "actions_ready_for_execution": True,
            "action_dim": 80,
            "action_chunk_size": 30,
        }
        arguments = [
            "--host",
            "127.0.0.1",
            "--port",
            "10093",
            "--checkpoint",
            "fake/model",
            "--embodiment-id",
            "0",
            "--min-action-dim",
            "14",
        ]

        with mock.patch.object(
            probe_server,
            "WebsocketClientPolicy",
            return_value=FakeClient(valid_metadata, fail_ping=True),
        ):
            self.assertEqual(probe_server.main(arguments), 2)

        mismatched = dict(valid_metadata, ckpt_path="wrong/model")
        with mock.patch.object(
            probe_server,
            "WebsocketClientPolicy",
            return_value=FakeClient(mismatched),
        ):
            self.assertEqual(probe_server.main(arguments), 3)


if __name__ == "__main__":
    unittest.main()
