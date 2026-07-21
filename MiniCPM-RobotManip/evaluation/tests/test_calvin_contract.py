from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

import numpy as np


def _module(name: str, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _import_calvin_evaluator():
    fake_omega = types.SimpleNamespace(
        load=lambda path: {},
        create=lambda value: value,
    )
    fake_modules = {
        "hydra": _module(
            "hydra",
            utils=types.SimpleNamespace(instantiate=lambda config, **kwargs: config),
        ),
        "calvin_agent": _module("calvin_agent"),
        "calvin_agent.evaluation": _module("calvin_agent.evaluation"),
        "calvin_agent.evaluation.utils": _module(
            "calvin_agent.evaluation.utils",
            collect_plan=lambda *args, **kwargs: None,
            count_success=lambda results: [0.0] * 5,
            get_env_state_for_initial_condition=lambda state: (
                np.zeros(1),
                np.zeros(1),
            ),
            get_log_dir=lambda path: path,
            print_and_save=lambda *args, **kwargs: None,
        ),
        "moviepy": _module("moviepy"),
        "moviepy.editor": _module(
            "moviepy.editor",
            ImageSequenceClip=object,
        ),
        "omegaconf": _module("omegaconf", OmegaConf=fake_omega),
        "termcolor": _module(
            "termcolor",
            colored=lambda text, color: text,
        ),
        "tqdm": _module("tqdm", tqdm=lambda iterable, **kwargs: iterable),
        "tyro": _module("tyro", cli=lambda function: function),
    }
    with mock.patch.dict(sys.modules, fake_modules):
        sys.modules.pop("evaluation.calvin.eval_calvin", None)
        return importlib.import_module("evaluation.calvin.eval_calvin")


class FakeSharedModelClient:
    def __init__(self, *args, **kwargs) -> None:
        self.init_args = args
        self.init_kwargs = kwargs
        self.calls: list[tuple[dict, int]] = []

    def step(self, example: dict, step: int = 0) -> dict:
        self.calls.append((example, step))
        return {
            "raw_action": {
                "world_vector": np.array([1, 2, 3], dtype=np.float32),
                "rotation_delta": np.array([4, 5, 6], dtype=np.float32),
                "open_gripper": np.array([0.75], dtype=np.float32),
            }
        }


class CalvinContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _import_calvin_evaluator()

    def test_two_view_policy_adapter_and_step_counter(self) -> None:
        with mock.patch.object(
            self.module,
            "ModelClient",
            FakeSharedModelClient,
        ):
            policy = self.module.CalvinPolicyClient(
                host="127.0.0.1",
                port=10093,
                resize_size=448,
            )

        static = np.full((8, 8, 3), 10, dtype=np.uint8)
        gripper = np.full((8, 8, 3), 20, dtype=np.uint8)
        observation = {
            "rgb_obs": {
                "rgb_static": static,
                "rgb_gripper": gripper,
            }
        }

        action0 = policy.step(observation, "open the drawer")
        action1 = policy.step(observation, "open the drawer")

        np.testing.assert_array_equal(action0, [1, 2, 3, 4, 5, 6, 0.75])
        np.testing.assert_array_equal(action1, action0)
        self.assertEqual(
            [step for _, step in policy.client.calls],
            [0, 1],
        )
        sent = policy.client.calls[0][0]
        self.assertEqual([int(view[0, 0, 0]) for view in sent["image"]], [10, 20])
        self.assertNotIn("state", sent)

    def test_rollout_converts_gripper_to_calvin_sign(self) -> None:
        class FakePolicy:
            plan = None

            def reset(self):
                pass

            def step(self, obs, annotation):
                del obs, annotation
                return np.array([0, 0, 0, 0, 0, 0, -0.25], dtype=np.float32)

        class FakeEnv:
            def __init__(self) -> None:
                self.actions = []

            def reset(self, **kwargs):
                del kwargs

            def get_obs(self):
                return {"rgb_obs": {"rgb_static": np.zeros((2, 2, 3))}}

            def get_info(self):
                return {"before": True}

            def step(self, action):
                self.actions.append(action.copy())
                return self.get_obs(), 0, False, {"after": True}

        class FakeOracle:
            def get_task_info_for_set(self, start, current, tasks):
                del start, current
                return set(tasks)

        env = FakeEnv()
        success = self.module.rollout(
            env=env,
            policy=FakePolicy(),
            task_oracle=FakeOracle(),
            subtask="open_drawer",
            val_annotations={"open_drawer": ["Open the drawer."]},
            plans=defaultdict(list),
            debug=False,
        )

        self.assertTrue(success)
        self.assertEqual(float(env.actions[0][-1]), -1.0)

    def test_bundled_sequences_and_num_sequences_slice(self) -> None:
        sequence_path = (
            Path(self.module.__file__).resolve().with_name("eval_sequences.json")
        )
        sequences = json.loads(sequence_path.read_text(encoding="utf-8"))
        self.assertEqual(len(sequences), 1000)
        self.assertTrue(all(len(entry) == 2 for entry in sequences))
        self.assertTrue(all(len(entry[1]) == 5 for entry in sequences))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_config = (
                root
                / "callbacks"
                / "rollout"
                / "tasks"
                / "new_playtable_tasks.yaml"
            )
            annotation_config = (
                root / "annotations" / "new_playtable_validation.yaml"
            )
            task_config.parent.mkdir(parents=True)
            annotation_config.parent.mkdir(parents=True)
            task_config.write_text("{}\n", encoding="utf-8")
            annotation_config.write_text("{}\n", encoding="utf-8")
            small_sequences = root / "sequences.json"
            small_sequences.write_text(
                json.dumps(sequences[:4]),
                encoding="utf-8",
            )

            with (
                mock.patch.object(
                    self.module,
                    "evaluate_sequence",
                    return_value=1,
                ) as evaluate_sequence,
                mock.patch.object(self.module, "print_and_save") as print_and_save,
            ):
                results = self.module.evaluate_policy(
                    policy=object(),
                    env=object(),
                    epoch=0,
                    calvin_conf_path=root,
                    eval_sequences_path=small_sequences,
                    num_sequences=2,
                    eval_log_dir=root / "logs",
                    debug=True,
                )

        self.assertEqual(results, [1, 1])
        self.assertEqual(evaluate_sequence.call_count, 2)
        self.assertEqual(len(print_and_save.call_args.args[1]), 2)


if __name__ == "__main__":
    unittest.main()
