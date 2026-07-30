from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from evaluation.robotwin import unified_ee6d


IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def identity_endpose(left_gripper: float, right_gripper: float) -> np.ndarray:
    """16D RoboTwin endpose: xyz + quat_wxyz + raw gripper, per arm."""
    return np.array(
        [
            0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, left_gripper,
            0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, right_gripper,
        ],
        dtype=np.float32,
    )


def joint_state() -> np.ndarray:
    """14D: left_joint6 + left_gripper + right_joint6 + right_gripper."""
    return np.arange(1, 15, dtype=np.float32)


class UnifiedEE6DTest(unittest.TestCase):
    def test_pack_places_joints_eef_and_leaves_the_rest_zero(self) -> None:
        joints = joint_state()
        state = unified_ee6d.pack_robotwin_state(joints, identity_endpose(1.0, 0.0))

        self.assertEqual(state.shape, (80,))
        np.testing.assert_allclose(state[0:6], joints[0:6])
        np.testing.assert_allclose(state[17:23], joints[7:13])
        np.testing.assert_allclose(state[7:10], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(state[10:16], IDENTITY_ROT6D)
        np.testing.assert_allclose(state[24:27], [0.4, 0.5, 0.6])
        np.testing.assert_allclose(state[27:33], IDENTITY_ROT6D)
        # The seventh joint slot of each arm and the reserved tail stay zero.
        self.assertEqual(state[6], 0.0)
        self.assertEqual(state[23], 0.0)
        np.testing.assert_allclose(state[34:], np.zeros(46))

    def test_raw_gripper_open_threshold_maps_to_closed_flag(self) -> None:
        # RoboTwin's raw value is "open"; the model channel is "closed".
        # The boundary is compared in float32: exactly 0.4 counts as closed.
        state = unified_ee6d.pack_robotwin_state(
            joint_state(), identity_endpose(0.4, 0.41)
        )
        self.assertEqual(state[16], 1.0)
        self.assertEqual(state[33], 0.0)

    def test_ee6d_override_replaces_eef_but_keeps_joints_measured(self) -> None:
        joints = joint_state()
        override = np.arange(100, 120, dtype=np.float32)
        state = unified_ee6d.pack_robotwin_state(
            joints, identity_endpose(1.0, 1.0), ee6d_override=override
        )

        np.testing.assert_allclose(state[0:6], joints[0:6])
        np.testing.assert_allclose(state[17:23], joints[7:13])
        np.testing.assert_allclose(state[7:17], override[0:10])
        np.testing.assert_allclose(state[24:34], override[10:20])

    def test_unpack_eef_round_trips_through_pack(self) -> None:
        override = np.arange(20, dtype=np.float32)
        state = unified_ee6d.pack_robotwin_state(
            joint_state(), identity_endpose(1.0, 1.0), ee6d_override=override
        )
        np.testing.assert_allclose(unified_ee6d.unpack_robotwin_eef(state), override)

    def test_rotation6d_round_trip(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(64):
            quaternion = rng.normal(size=4)
            quaternion = (quaternion / np.linalg.norm(quaternion)).astype(np.float32)
            rot6d = unified_ee6d.robotwin_wxyz_to_rotate6d(quaternion)
            recovered = unified_ee6d.robotwin_rotate6d_to_wxyz(rot6d)
            # q and -q are the same rotation.
            if np.dot(recovered, quaternion) < 0:
                recovered = -recovered
            np.testing.assert_allclose(recovered, quaternion, atol=1e-5)

    def test_env_action_gripper_threshold_is_inclusive(self) -> None:
        # The reference client decides open as `1.0 - value > 0.5`, so the threshold
        # value itself commands a closed gripper.
        ee6d = np.zeros(20, dtype=np.float32)
        ee6d[3:9] = IDENTITY_ROT6D
        ee6d[13:19] = IDENTITY_ROT6D
        ee6d[9] = 0.5
        ee6d[19] = 0.49

        action = unified_ee6d.robotwin_ee6d_to_env_action(ee6d)
        self.assertEqual(action.shape, (16,))
        self.assertEqual(action[7], 0.0)
        self.assertEqual(action[15], 1.0)

    def test_env_action_honours_custom_threshold_and_close_position(self) -> None:
        ee6d = np.zeros(20, dtype=np.float32)
        ee6d[3:9] = IDENTITY_ROT6D
        ee6d[13:19] = IDENTITY_ROT6D
        ee6d[9] = 0.3
        ee6d[19] = 0.1

        action = unified_ee6d.robotwin_ee6d_to_env_action(
            ee6d, threshold=0.2, gripper_close_position=0.05
        )
        self.assertAlmostEqual(float(action[7]), 0.05)
        self.assertEqual(action[15], 1.0)

    def test_binarize_only_touches_gripper_channels(self) -> None:
        ee6d = np.arange(20, dtype=np.float32) / 20.0
        binarized = unified_ee6d.binarize_ee6d_grippers(ee6d)

        self.assertEqual(binarized[9], 0.0)  # 0.45 < 0.5
        self.assertEqual(binarized[19], 1.0)  # 0.95 >= 0.5
        untouched = [i for i in range(20) if i not in (9, 19)]
        np.testing.assert_allclose(binarized[untouched], ee6d[untouched])
        self.assertAlmostEqual(float(ee6d[9]), 0.45)  # input is not mutated


def action_chunk(gripper_closed: float) -> np.ndarray:
    chunk = np.zeros((1, 30, 80), dtype=np.float32)
    chunk[0, :, 7:10] = np.arange(30, dtype=np.float32)[:, None] + [1.0, 2.0, 3.0]
    chunk[0, :, 10:16] = IDENTITY_ROT6D
    chunk[0, :, 16] = gripper_closed
    chunk[0, :, 24:27] = np.arange(30, dtype=np.float32)[:, None] + [4.0, 5.0, 6.0]
    chunk[0, :, 27:33] = IDENTITY_ROT6D
    chunk[0, :, 33] = gripper_closed
    return chunk


class FakeWireClient:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.calls: list[dict] = []
        self.closed = False
        self.actions = action_chunk(0.9)

    def get_server_metadata(self) -> dict:
        return {"action_chunk_size": 30}

    def predict_action(self, payload: dict) -> dict:
        self.calls.append(payload)
        return {"ok": True, "data": {"actions": self.actions}}

    def close(self) -> None:
        self.closed = True


class RobotwinModelClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from evaluation.robotwin import model2robotwin_interface

        cls.module = model2robotwin_interface

    def make_client(self, fake: FakeWireClient, **kwargs):
        with mock.patch.object(
            self.module, "WebsocketClientPolicy", return_value=fake
        ):
            return self.module.ModelClient(**kwargs)

    @staticmethod
    def example(step: int) -> dict:
        return {
            "lang": "pick up the bottle",
            "image": [np.zeros((240, 320, 3), dtype=np.uint8)] * 3,
            "joint_state": joint_state() + step,
            "endpose": identity_endpose(1.0, 1.0),
        }

    def test_request_carries_prompt_and_client_side_resize(self) -> None:
        fake = FakeWireClient()
        client = self.make_client(fake)
        client.step(self.example(0), step=0)

        self.assertEqual(len(fake.calls), 1)
        payload = fake.calls[0]
        self.assertEqual(payload["embodiment_id"], 4)
        model_example = payload["examples"][0]

        self.assertTrue(model_example["lang"].startswith("The robot is RoboTwin2"))
        self.assertIn("gripper closed commands", model_example["lang"])
        self.assertTrue(model_example["lang"].endswith("Task: pick up the bottle"))
        # The client owns the resize, matching the reference client's cv2 INTER_AREA.
        for image in model_example["image"]:
            self.assertEqual(image.shape, (448, 448, 3))

        # Joint channels stay zero by default.
        state = model_example["state"][0]
        np.testing.assert_allclose(state[0:6], np.zeros(6))
        np.testing.assert_allclose(state[17:23], np.zeros(6))

    def test_native_resolution_frames_for_the_ablation(self) -> None:
        fake = FakeWireClient()
        client = self.make_client(fake, image_size=None)
        client.step(self.example(0), step=0)

        for image in fake.calls[0]["examples"][0]["image"]:
            self.assertEqual(image.shape, (240, 320, 3))

    def test_second_chunk_state_chains_the_last_commanded_eef(self) -> None:
        fake = FakeWireClient()
        client = self.make_client(fake, chain_eef=True)
        for step in range(31):
            client.step(self.example(step), step=step)

        self.assertEqual(len(fake.calls), 2)
        state = fake.calls[1]["examples"][0]["state"][0]

        # Chunk row 29 was the last command before the boundary.
        last_row = fake.actions[0, 29]
        np.testing.assert_allclose(state[7:16], last_row[7:16])
        np.testing.assert_allclose(state[24:33], last_row[24:33])
        # Gripper channels are binarized, not the raw 0.9 regression.
        self.assertEqual(state[16], 1.0)
        self.assertEqual(state[33], 1.0)
        # Joint channels stay zero regardless of chaining.
        np.testing.assert_allclose(state[0:6], np.zeros(6))
        np.testing.assert_allclose(state[17:23], np.zeros(6))

    def test_state_keeps_the_measured_endpose_by_default(self) -> None:
        fake = FakeWireClient()
        client = self.make_client(fake)
        for step in range(31):
            client.step(self.example(step), step=step)

        state = fake.calls[1]["examples"][0]["state"][0]
        np.testing.assert_allclose(state[7:10], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(state[10:16], IDENTITY_ROT6D)
        # Nothing is chained when the default is in effect.
        self.assertIsNone(client._chained_ee6d)

    def test_reset_clears_the_chained_target(self) -> None:
        fake = FakeWireClient()
        client = self.make_client(fake, chain_eef=True)
        client.step(self.example(0), step=0)
        self.assertIsNotNone(client._chained_ee6d)

        self.module.reset_model(client)
        self.assertIsNone(client._chained_ee6d)
        self.assertIsNone(client.raw_actions)
        self.assertEqual(client.control_step, 0)

        client.step(self.example(0), step=0)
        state = fake.calls[1]["examples"][0]["state"][0]
        np.testing.assert_allclose(state[7:10], [0.1, 0.2, 0.3])

    def test_joint_channels_can_be_filled_for_the_ablation(self) -> None:
        fake = FakeWireClient()
        client = self.make_client(fake, use_joint_state=True)
        client.step(self.example(0), step=0)

        state = fake.calls[0]["examples"][0]["state"][0]
        np.testing.assert_allclose(state[0:6], joint_state()[0:6])
        np.testing.assert_allclose(state[17:23], joint_state()[7:13])
        # The seventh joint slot of each arm still stays zero.
        self.assertEqual(state[6], 0.0)
        self.assertEqual(state[23], 0.0)

    def test_returned_action_is_the_16d_absolute_ee_command(self) -> None:
        fake = FakeWireClient()
        client = self.make_client(fake)
        action = client.step(self.example(0), step=0)

        self.assertEqual(action.shape, (16,))
        np.testing.assert_allclose(action[0:3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(action[3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
        self.assertEqual(action[7], 0.0)  # closed=0.9 > 0.5 -> closed command
        np.testing.assert_allclose(action[8:11], [4.0, 5.0, 6.0])
        self.assertEqual(action[15], 0.0)

    def test_get_model_reads_the_alignment_switches(self) -> None:
        fake = FakeWireClient()
        env = {
            "ROBOTWIN_STATE_JOINTS": "1",
            "ROBOTWIN_CHAIN_EEF": "1",
            "ROBOTWIN_CLIENT_RESIZE": "0",
            "ROBOTWIN_PROMPT_STYLE": "legacy",
            "ROBOTWIN_GRIPPER_CLOSED_THRESHOLD": "0.6",
            "ROBOTWIN_GRIPPER_CLOSE_POSITION": "0.02",
        }
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
            self.module, "WebsocketClientPolicy", return_value=fake
        ):
            client = self.module.get_model({"host": "127.0.0.1", "port": 10093})

        self.assertTrue(client.use_joint_state)
        self.assertTrue(client.chain_eef)
        self.assertIsNone(client.image_size)
        self.assertTrue(client.prompt_template.startswith("The robot is RoboTwin ALOHA"))
        self.assertAlmostEqual(client.gripper_threshold, 0.6)
        self.assertAlmostEqual(client.gripper_close_position, 0.02)

    def test_get_model_defaults_match_reference_client(self) -> None:
        fake = FakeWireClient()
        removed = (
            "ROBOTWIN_STATE_JOINTS",
            "ROBOTWIN_CHAIN_EEF",
            "ROBOTWIN_CLIENT_RESIZE",
            "ROBOTWIN_PROMPT_STYLE",
        )
        with mock.patch.dict(os.environ, clear=False) as env:
            for name in removed:
                env.pop(name, None)
            with mock.patch.object(
                self.module, "WebsocketClientPolicy", return_value=fake
            ):
                client = self.module.get_model({})

        self.assertFalse(client.chain_eef)
        self.assertEqual(client.image_size, (448, 448))
        self.assertFalse(client.use_joint_state)

    def test_unknown_prompt_style_is_rejected(self) -> None:
        with mock.patch.dict(
            os.environ, {"ROBOTWIN_PROMPT_STYLE": "nope"}, clear=False
        ):
            with self.assertRaises(ValueError):
                self.module.get_model({})


class FakeTaskEnv:
    """Minimal RoboTwin TASK_ENV stand-in for the eval() hook."""

    def __init__(self, cnt_stride: int = 2) -> None:
        self.actions: list[np.ndarray] = []
        self.cnt_stride = cnt_stride
        # Deliberately advances faster than the policy steps so a client that
        # still read take_action_cnt would re-infer on the wrong steps.
        self.take_action_cnt = 0

    def get_instruction(self) -> str:
        return "pick up the bottle"

    def take_action(self, action: np.ndarray, action_type: str = "ee") -> None:
        assert action_type == "ee"
        self.actions.append(action)
        self.take_action_cnt += self.cnt_stride


def observation() -> dict:
    endpose = identity_endpose(1.0, 1.0)
    return {
        "observation": {
            "head_camera": {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)},
            "left_camera": {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)},
            "right_camera": {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)},
        },
        "joint_action": {"vector": joint_state()},
        "endpose": {
            "left_endpose": endpose[0:7],
            "left_gripper": float(endpose[7]),
            "right_endpose": endpose[8:15],
            "right_gripper": float(endpose[15]),
        },
    }


class RobotwinEvalHookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from evaluation.robotwin import model2robotwin_interface

        cls.module = model2robotwin_interface

    def make_client(self, fake: FakeWireClient, **kwargs):
        with mock.patch.object(
            self.module, "WebsocketClientPolicy", return_value=fake
        ):
            return self.module.ModelClient(**kwargs)

    def test_eval_drives_the_clients_own_control_step(self) -> None:
        fake = FakeWireClient()
        client = self.make_client(fake)
        env = FakeTaskEnv()

        for _ in range(31):
            self.module.eval(env, client, observation())

        # One model target per eval() call, chunk size 30 -> exactly two chunks.
        self.assertEqual(client.control_step, 31)
        self.assertEqual(len(env.actions), 31)
        self.assertEqual(len(fake.calls), 2)
        # take_action_cnt ran ahead and did not drive the chunk boundary.
        self.assertEqual(env.take_action_cnt, 62)

    def test_reset_model_rewinds_the_control_step(self) -> None:
        fake = FakeWireClient()
        client = self.make_client(fake)
        env = FakeTaskEnv()

        for _ in range(5):
            self.module.eval(env, client, observation())
        self.assertEqual(client.control_step, 5)

        self.module.reset_model(client)
        self.assertEqual(client.control_step, 0)

        self.module.eval(env, client, observation())
        # A fresh episode re-infers on its first step.
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(client.control_step, 1)


if __name__ == "__main__":
    unittest.main()
