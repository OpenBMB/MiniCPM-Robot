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
        fake.actions = np.zeros((1, 30, 80), dtype=np.float32)
        fake.actions[0, :, 7:10] = (
            np.arange(30, dtype=np.float32)[:, None]
            + np.array([1, 2, 3], dtype=np.float32)
        )
        fake.actions[0, :, 10:16] = [1, 0, 0, 0, 1, 0]
        fake.actions[0, :, 16] = 0.75
        with mock.patch.object(
            self.module,
            "WebsocketClientPolicy",
            return_value=fake,
        ):
            client = self.module.ModelClient(
                action_ensemble=False,
                image_size=(8, 8),
                unified_ee6d=True,
            )

        first_view = np.full((8, 8, 3), 11, dtype=np.uint8)
        wrist_view = np.full((8, 8, 3), 22, dtype=np.uint8)
        initial_state = np.array(
            [0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0],
            dtype=np.float32,
        )
        example = {
            "image": [first_view, wrist_view],
            "lang": "pick up the block",
            "state": initial_state,
        }

        first = client.step(example, step=0)["raw_action"]
        second = client.step(example, step=1)["raw_action"]
        client.step(example, step=30)

        self.assertEqual(len(fake.calls), 2)
        payload = fake.calls[0]
        self.assertEqual(set(payload), {"examples"})
        sent_example = payload["examples"][0]
        self.assertEqual(set(sent_example), {"image", "lang", "state"})
        np.testing.assert_array_equal(sent_example["image"][0], first_view)
        np.testing.assert_array_equal(sent_example["image"][1], wrist_view)
        np.testing.assert_array_equal(sent_example["state"][:7], np.zeros(7))
        np.testing.assert_array_equal(sent_example["state"][7:17], initial_state)
        np.testing.assert_array_equal(sent_example["state"][17:], np.zeros(63))
        np.testing.assert_array_equal(first["world_vector"], [1, 2, 3])
        np.testing.assert_allclose(first["rotation_delta"], [0, 0, 0])
        np.testing.assert_array_equal(first["open_gripper"], [1])
        np.testing.assert_array_equal(second["world_vector"], [2, 3, 4])
        chained_state = fake.calls[1]["examples"][0]["state"]
        np.testing.assert_array_equal(
            chained_state[7:17],
            [2, 3, 4, 1, 0, 0, 0, 1, 0, 1],
        )

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
            "state": np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0]),
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

    @staticmethod
    def _observation(
        left_xyz=(1, 2, 3),
        right_xyz=(4, 5, 6),
        left_gripper=1.0,
        right_gripper=0.0,
    ) -> dict:
        return {
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
            "joint_action": {"vector": np.arange(1, 15, dtype=np.float32)},
            "endpose": {
                "left_endpose": [*left_xyz, 1, 0, 0, 0],
                "left_gripper": left_gripper,
                "right_endpose": [*right_xyz, 1, 0, 0, 0],
                "right_gripper": right_gripper,
            },
        }

    def _client(self, fake: FakeWireClient, image_size=None):
        with mock.patch.object(
            self.module,
            "WebsocketClientPolicy",
            return_value=fake,
        ):
            return self.module.ModelClient(image_size=image_size)

    def test_state_action_and_measured_state_at_chunk_boundary(self) -> None:
        fake = FakeWireClient()
        fake.actions = np.zeros((1, 30, 80), dtype=np.float32)
        steps = np.arange(30, dtype=np.float32)
        fake.actions[0, :, 7:10] = np.stack(
            (steps + 1, np.full(30, 2), np.full(30, 3)), axis=1
        )
        fake.actions[0, :, 10:16] = [1, 0, 0, 1, 0, 0]
        fake.actions[0, :, 16] = 0.75
        fake.actions[0, :, 24:27] = np.stack(
            (steps + 4, np.full(30, 5), np.full(30, 6)), axis=1
        )
        fake.actions[0, :, 27:33] = [1, 0, 0, 1, 0, 0]
        fake.actions[0, :, 33] = 0.25
        client = self._client(fake)

        class FakeTaskEnv:
            def __init__(self) -> None:
                self.take_action_cnt = 0
                self.action = None
                self.action_type = None

            def get_instruction(self):
                return "stack the blocks"

            def take_action(self, action, action_type=None):
                self.action = action
                self.action_type = action_type
                self.take_action_cnt += 1

        task_env = FakeTaskEnv()
        observation = self._observation()
        self.module.eval(task_env, client, observation)

        first_payload = fake.calls[0]
        self.assertEqual(first_payload["embodiment_id"], 4)
        sent = first_payload["examples"][0]
        self.assertEqual(
            sent["lang"],
            "The robot is RoboTwin2 ALOHA-AgileX, a simulated dual-arm "
            "ALOHA-style manipulator. Its action control method is absolute "
            "dual-arm end-effector pose in the unified 80D layout with gripper "
            "closed commands, and its action FPS is 15 Hz. Task: stack the blocks",
        )
        self.assertEqual(sent["state"].shape, (1, 80))
        expected_state = np.zeros(80, dtype=np.float32)
        expected_state[7:17] = [1, 2, 3, 1, 0, 0, 1, 0, 0, 0]
        expected_state[24:34] = [4, 5, 6, 1, 0, 0, 1, 0, 0, 1]
        np.testing.assert_array_equal(sent["state"][0], expected_state)
        # image_size=None is the ablation: frames are forwarded untouched.
        self.assertEqual([int(view[0, 0, 0]) for view in sent["image"]], [1, 2, 3])
        self.assertEqual([view.shape for view in sent["image"]], [(8, 8, 3)] * 3)
        np.testing.assert_allclose(
            task_env.action,
            [1, 2, 3, 1, 0, 0, 0, 0, 4, 5, 6, 1, 0, 0, 0, 1],
        )
        self.assertEqual(task_env.action_type, "ee")

        self.module.eval(task_env, client, observation)
        np.testing.assert_allclose(
            task_env.action,
            [2, 2, 3, 1, 0, 0, 0, 0, 5, 5, 6, 1, 0, 0, 0, 1],
        )

        # Drive to the next chunk boundary; the new state must report the
        # measured endpose of the current observation, not the last command.
        moved = self._observation(
            left_xyz=(10, 20, 30),
            right_xyz=(40, 50, 60),
            left_gripper=0.0,
            right_gripper=1.0,
        )
        while client.control_step < 30:
            self.module.eval(task_env, client, moved)
        # control_step is now 30, so this call crosses the chunk boundary.
        self.module.eval(task_env, client, moved)

        self.assertEqual(len(fake.calls), 2)
        boundary_state = fake.calls[1]["examples"][0]["state"][0]
        expected_boundary = np.zeros(80, dtype=np.float32)
        # Raw gripper 0.0 is not > 0.4, so the closed flag is 1; 1.0 gives 0.
        expected_boundary[7:17] = [10, 20, 30, 1, 0, 0, 1, 0, 0, 1]
        expected_boundary[24:34] = [40, 50, 60, 1, 0, 0, 1, 0, 0, 0]
        np.testing.assert_array_equal(boundary_state, expected_boundary)
        self.assertIsNone(client._chained_ee6d)

    def test_client_resize_matches_opencv_inter_area(self) -> None:
        client = self._client(FakeWireClient(), image_size=(2, 2))
        image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        expected = np.asarray(
            [
                [[8, 9, 10], [14, 15, 16]],
                [[32, 33, 34], [38, 39, 40]],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(client._prepare_image(image), expected)

    def test_gripper_threshold_treats_half_as_closed(self) -> None:
        from evaluation.robotwin import unified_ee6d

        action = np.zeros(80, dtype=np.float32)
        action[10:16] = [1, 0, 0, 1, 0, 0]
        action[27:33] = [1, 0, 0, 1, 0, 0]
        # The reference client decides open as `1.0 - value > 0.5`, so the threshold
        # itself commands a closed gripper.
        for prediction, expected_open in ((0.49, 1.0), (0.5, 0.0), (0.51, 0.0)):
            with self.subTest(prediction=prediction):
                action[16] = prediction
                action[33] = prediction
                decoded = unified_ee6d.robotwin_ee6d_to_env_action(
                    unified_ee6d.unpack_robotwin_eef(action)
                )
                self.assertEqual(decoded[7], expected_open)
                self.assertEqual(decoded[15], expected_open)

    def test_rotation6d_round_trip(self) -> None:
        from evaluation.robotwin import unified_ee6d

        quaternion = np.asarray(
            [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32
        )
        rotation6d = unified_ee6d.robotwin_wxyz_to_rotate6d(quaternion)
        restored = unified_ee6d.robotwin_rotate6d_to_wxyz(rotation6d)
        self.assertAlmostEqual(abs(float(np.dot(quaternion, restored))), 1.0, places=6)

    def test_non_absolute_mode_bad_image_and_bad_response_fail_fast(self) -> None:
        fake = FakeWireClient()
        with mock.patch.object(
            self.module,
            "WebsocketClientPolicy",
            return_value=fake,
        ):
            with self.assertRaisesRegex(ValueError, "action_mode='abs'"):
                self.module.ModelClient(action_mode="delta")

            client = self.module.ModelClient()

        example = {
            "image": [
                np.zeros((8, 8, 3), dtype=np.float32),
                np.zeros((8, 8, 3), dtype=np.float32),
                np.zeros((8, 8, 3), dtype=np.float32),
            ],
            "lang": "test",
            "joint_state": np.zeros(14, dtype=np.float32),
            "endpose": np.asarray(
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32
            ),
        }
        with self.assertRaisesRegex(ValueError, "dtype uint8"):
            client.step(example, step=0)

        good_example = {
            **example,
            "image": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)],
        }
        fake.predict_action = lambda payload: {
            "ok": True,
            "data": {"actions": np.full((1, 30, 80), np.nan, dtype=np.float32)},
        }
        with self.assertRaisesRegex(ValueError, "finite"):
            client.step(good_example, step=0)


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
        ):
            del uri, compression, max_size, open_timeout
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
