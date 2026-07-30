from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

from evaluation.calvin import model2calvin_interface


class FakeWireClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.closed = False
        self.metadata = {
            "action_chunk_size": 30,
            "action_dim": 80,
            "state_dim": 80,
            "action_normalization": "none",
            "actions_ready_for_execution": True,
        }
        self.actions = np.zeros((1, 30, 80), dtype=np.float32)
        self.actions[0, :, 10:16] = [1, 0, 0, 1, 0, 0]

    def get_server_metadata(self) -> dict:
        return self.metadata

    def predict_action(self, payload: dict) -> dict:
        self.calls.append(payload)
        return {"ok": True, "data": {"actions": self.actions}}

    def close(self) -> None:
        self.closed = True


class CalvinModelClientTest(unittest.TestCase):
    def make_client(self, fake: FakeWireClient):
        with mock.patch.object(
            model2calvin_interface,
            "WebsocketClientPolicy",
            return_value=fake,
        ):
            return model2calvin_interface.ModelClient(image_size=(8, 8))

    @staticmethod
    def example(state: np.ndarray) -> dict:
        return {
            "image": [
                np.full((8, 8, 3), 11, dtype=np.uint8),
                np.full((8, 8, 3), 22, dtype=np.uint8),
            ],
            "lang": "CALVIN prompt",
            "state": state,
        }

    def test_payload_chunk_cache_and_fresh_measured_state(self) -> None:
        fake = FakeWireClient()
        steps = np.arange(30, dtype=np.float32)
        fake.actions[0, :, 7:10] = np.stack(
            (steps + 1, steps + 2, steps + 3), axis=1
        )
        fake.actions[0, :, 16] = 0.75
        client = self.make_client(fake)

        state0 = np.array(
            [0.1, 0.2, 0.3, 1, 0, 0, 1, 0, 0, 0], dtype=np.float32
        )
        state30 = state0.copy()
        state30[:3] = [3.1, 3.2, 3.3]
        action0 = client.step(self.example(state0), step=0)
        action1 = client.step(self.example(state30), step=1)
        action30 = client.step(self.example(state30), step=30)

        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0]["embodiment_id"], 1)
        first_example = fake.calls[0]["examples"][0]
        second_example = fake.calls[1]["examples"][0]
        self.assertEqual(first_example["state"].shape, (1, 80))
        np.testing.assert_array_equal(first_example["state"][0, :7], np.zeros(7))
        np.testing.assert_array_equal(first_example["state"][0, 7:17], state0)
        np.testing.assert_array_equal(first_example["state"][0, 17:], np.zeros(63))
        np.testing.assert_array_equal(second_example["state"][0, 7:17], state30)
        self.assertEqual(
            [int(image[0, 0, 0]) for image in first_example["image"]], [11, 22]
        )

        self.assertEqual(action0["type"], "cartesian_abs")
        np.testing.assert_allclose(
            action0["action"], [1, 2, 3, 0, 0, 0, -1], atol=1e-6
        )
        np.testing.assert_allclose(
            action1["action"], [2, 3, 4, 0, 0, 0, -1], atol=1e-6
        )
        np.testing.assert_allclose(
            action30["action"], [1, 2, 3, 0, 0, 0, -1], atol=1e-6
        )

    def test_interleaved_rotation_round_trip_and_gripper_boundary(self) -> None:
        euler = np.array([0.3, -0.4, 0.2])
        matrix = Rotation.from_euler("xyz", euler).as_matrix()
        ee6d = np.concatenate(
            ([1.0, 2.0, 3.0], matrix[:, :2].reshape(6), [0.5])
        )
        result = model2calvin_interface.calvin_ee6d_to_action(ee6d)
        restored = Rotation.from_euler("xyz", result["action"][3:6]).as_matrix()

        np.testing.assert_allclose(restored, matrix, atol=1e-6)
        self.assertEqual(float(result["action"][-1]), 1.0)

        ee6d[-1] = np.nextafter(np.float64(0.5), np.float64(1.0))
        result = model2calvin_interface.calvin_ee6d_to_action(ee6d)
        self.assertEqual(float(result["action"][-1]), -1.0)

    def test_rejects_bad_metadata_and_response(self) -> None:
        bad_metadata = FakeWireClient()
        bad_metadata.metadata["state_dim"] = 10
        with self.assertRaisesRegex(ValueError, "state_dim"):
            self.make_client(bad_metadata)
        self.assertTrue(bad_metadata.closed)

        fake = FakeWireClient()
        client = self.make_client(fake)
        fake.predict_action = lambda payload: {"ok": False, "error": "failed"}
        with self.assertRaisesRegex(RuntimeError, "ok=true"):
            client.step(self.example(np.zeros(10, dtype=np.float32)), step=0)

        fake.predict_action = lambda payload: {
            "ok": True,
            "data": {
                "actions": np.full((1, 30, 80), np.nan, dtype=np.float32)
            },
        }
        with self.assertRaisesRegex(ValueError, "finite"):
            client.step(self.example(np.zeros(10, dtype=np.float32)), step=0)


if __name__ == "__main__":
    unittest.main()
