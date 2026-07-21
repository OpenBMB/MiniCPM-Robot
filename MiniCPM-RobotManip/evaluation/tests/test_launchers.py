from __future__ import annotations

import ast
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock


MINICPM_ROOT = Path(__file__).resolve().parents[2]


def run(command, *, env=None, timeout=30):
    merged_env = dict(os.environ)
    if env:
        merged_env.update({key: str(value) for key, value in env.items()})
    return subprocess.run(
        command,
        cwd=MINICPM_ROOT,
        env=merged_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def available_port_range(count: int) -> int:
    for base in range(21000, 61000):
        sockets = []
        try:
            for offset in range(count):
                sock = socket.socket()
                sockets.append(sock)
                sock.bind(("127.0.0.1", base + offset))
        except OSError:
            pass
        else:
            return base
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError(f"Unable to reserve {count} consecutive test ports")


def assert_ports_available(test: unittest.TestCase, base: int, count: int) -> None:
    for offset in range(count):
        with socket.socket() as sock:
            sock.settimeout(0.2)
            test.assertNotEqual(
                sock.connect_ex(("127.0.0.1", base + offset)),
                0,
                f"Port {base + offset} still has a live listener",
            )


def process_is_running(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        fields = stat_path.read_text(encoding="utf-8").split()
    except FileNotFoundError:
        return False
    return len(fields) > 2 and fields[2] != "Z"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def make_fake_minicpm_python(root: Path) -> Path:
    fake_server = root / "fake_server.py"
    fake_server.write_text(
        textwrap.dedent(
            """
            import argparse
            from deployment.model_server import protocol
            from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

            parser = argparse.ArgumentParser()
            parser.add_argument("--checkpoint", required=True)
            parser.add_argument("--device")
            parser.add_argument("--host", default="127.0.0.1")
            parser.add_argument("--port", type=int, required=True)
            parser.add_argument("--default-embodiment-id", type=int, default=0)
            args, _ = parser.parse_known_args()

            class FakePolicy:
                def predict_action(self, **payload):
                    raise RuntimeError("readiness-only fake policy")

            metadata = protocol.build_server_metadata(
                {
                    "action_chunk_size": 30,
                    "action_dim": 80,
                    "state_dim": 80,
                    "default_embodiment_id": args.default_embodiment_id,
                    "action_normalization": "none",
                    "actions_ready_for_execution": True,
                },
                checkpoint=args.checkpoint,
            )
            WebsocketPolicyServer(
                FakePolicy(),
                host=args.host,
                port=args.port,
                metadata=metadata,
            ).serve_forever()
            """
        ),
        encoding="utf-8",
    )
    wrapper = root / "fake_minicpm_python"
    write_executable(
        wrapper,
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os
            import sys

            if sys.argv[1:3] == ["-m", "deployment.model_server.server_policy"]:
                os.execv({sys.executable!r}, [{sys.executable!r}, {str(fake_server)!r}, *sys.argv[3:]])
            os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
            """
        ),
    )
    return wrapper


def make_fake_libero_python(root: Path) -> Path:
    wrapper = root / "fake_libero_python"
    write_executable(
        wrapper,
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json
            import os
            import sys
            import time

            if sys.argv[1:3] == ["-m", "evaluation.libero.eval_libero"]:
                port = sys.argv[sys.argv.index("--args.port") + 1]
                trace = os.environ["FAKE_EVAL_TRACE"]
                with open(trace, "a", encoding="utf-8") as file:
                    file.write(json.dumps({{"event": "start", "port": port, "time": time.time()}}) + "\\n")
                expected = int(os.environ.get("FAKE_EVAL_EXPECTED", "1"))
                deadline = time.time() + 10
                while time.time() < deadline:
                    with open(trace, encoding="utf-8") as file:
                        started = sum('"event": "start"' in line for line in file)
                    if started >= expected:
                        break
                    time.sleep(0.05)
                else:
                    raise SystemExit(8)
                time.sleep(float(os.environ.get("FAKE_EVAL_SLEEP", "0.3")))
                with open(trace, "a", encoding="utf-8") as file:
                    file.write(json.dumps({{"event": "end", "port": port, "time": time.time()}}) + "\\n")
                if port == os.environ.get("FAKE_FAIL_PORT"):
                    raise SystemExit(7)
                raise SystemExit(0)
            os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
            """
        ),
    )
    return wrapper


def make_fake_calvin_python(root: Path) -> Path:
    wrapper = root / "fake_calvin_python"
    write_executable(
        wrapper,
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os
            import subprocess
            import sys
            import time

            if sys.argv[1:3] == ["-m", "evaluation.calvin.eval_calvin"]:
                child = subprocess.Popen([{sys.executable!r}, "-c", "import time; time.sleep(30)"])
                with open(os.environ["FAKE_CALVIN_PID_FILE"], "w", encoding="utf-8") as file:
                    file.write(str(child.pid))
                time.sleep(30)
                raise SystemExit(0)
            os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
            """
        ),
    )
    return wrapper


def read_intervals(trace_path: Path):
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    starts = {record["port"]: record["time"] for record in records if record["event"] == "start"}
    ends = {record["port"]: record["time"] for record in records if record["event"] == "end"}
    return starts, ends


class LauncherStaticTest(unittest.TestCase):
    def test_all_shell_scripts_parse_and_help(self) -> None:
        scripts = [
            "evaluation/libero/eval_libero.sh",
            "evaluation/libero/run_policy_server.sh",
            "evaluation/libero/auto_eval_scripts/eval_libero_parallel.sh",
            "evaluation/libero/auto_eval_scripts/auto_eval_libero.sh",
            "evaluation/calvin/eval_calvin.sh",
            "evaluation/calvin/run_policy_server.sh",
            "evaluation/robotwin/eval.sh",
            "evaluation/robotwin/run_policy_server.sh",
            "evaluation/robotwin/start_eval.sh",
        ]
        for script in scripts:
            with self.subTest(script=script, mode="syntax"):
                result = run(["bash", "-n", script])
                self.assertEqual(result.returncode, 0, result.stderr)
            with self.subTest(script=script, mode="help"):
                result = run(["bash", script, "--help"])
                self.assertEqual(
                    result.returncode,
                    0,
                    f"stdout={result.stdout}\nstderr={result.stderr}",
                )

    def test_multi_gpu_dry_runs(self) -> None:
        libero = run(
            [
                "bash",
                "evaluation/libero/auto_eval_scripts/auto_eval_libero.sh",
                "--dry-run",
                "--checkpoint",
                "fake/model",
            ],
            env={
                "GPU_LIST": "2 5",
                "TASK_SUITES": "libero_goal libero_object",
                "RUN_ID": "dry-run",
                "EMBODIMENT_ID": "0",
            },
        )
        self.assertEqual(libero.returncode, 0, libero.stderr)
        self.assertIn("slot=0 gpu=2 egl=2 port=10093", libero.stdout)
        self.assertIn("slot=1 gpu=5 egl=5 port=10094", libero.stdout)

        robotwin = run(
            [
                "bash",
                "evaluation/robotwin/start_eval.sh",
                "--mode",
                "demo_clean",
                "--run-name",
                "dry-run",
                "--checkpoint",
                "fake/model",
                "--default-embodiment-id",
                "0",
                "--dry-run",
                "adjust_bottle,open_laptop",
            ],
            env={
                "CUDA_VISIBLE_DEVICES": "2,5",
                "MINICPM_PYTHON": sys.executable,
                "ROBOTWIN_PYTHON": sys.executable,
            },
        )
        self.assertEqual(robotwin.returncode, 0, robotwin.stderr)
        self.assertIn("slot=0 gpu=2", robotwin.stdout)
        self.assertIn("slot=1 gpu=5", robotwin.stdout)
        self.assertIn("FIFO[2]=open_laptop", robotwin.stdout)

        missing_libero_id = run(
            [
                "bash",
                "evaluation/libero/auto_eval_scripts/auto_eval_libero.sh",
                "--dry-run",
                "--checkpoint",
                "fake/model",
            ],
            env={"EMBODIMENT_ID": ""},
        )
        self.assertNotEqual(missing_libero_id.returncode, 0)
        self.assertIn("EMBODIMENT_ID is required", missing_libero_id.stderr)

        missing_robotwin_id = run(
            [
                "bash",
                "evaluation/robotwin/start_eval.sh",
                "--mode",
                "demo_clean",
                "--run-name",
                "dry-run",
                "--checkpoint",
                "fake/model",
                "--dry-run",
                "adjust_bottle",
            ],
            env={
                "MINICPM_PYTHON": sys.executable,
                "ROBOTWIN_PYTHON": sys.executable,
                "ROBOTWIN_DEFAULT_EMBODIMENT_ID": "",
            },
        )
        self.assertNotEqual(missing_robotwin_id.returncode, 0)
        self.assertIn(
            "--default-embodiment-id is required",
            missing_robotwin_id.stderr,
        )

    def test_python38_syntax_and_gpu_disable(self) -> None:
        from evaluation.robotwin import launcher

        for path in (
            MINICPM_ROOT / "evaluation" / "robotwin" / "launcher.py",
            MINICPM_ROOT / "evaluation" / "robotwin" / "probe_server.py",
        ):
            ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
                feature_version=(3, 8),
            )

        with mock.patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "-1"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "disables all GPUs"):
                launcher.detect_cuda_devices()

    def test_worker_internal_error_is_recorded(self) -> None:
        from evaluation.robotwin import launcher

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = launcher.Config(
                root=MINICPM_ROOT,
                script_dir=MINICPM_ROOT / "evaluation" / "robotwin",
                robotwin_path=None,
                minicpm_python=sys.executable,
                robotwin_python=sys.executable,
                checkpoint="fake/model",
                embodiment_id=0,
                mode="demo_clean",
                run_name="test",
                seed=0,
                device="cuda",
                host="127.0.0.1",
                server_timeout=1,
                output_root=root,
                dry_run=False,
            )
            coordinator = launcher.Coordinator(
                config,
                [launcher.Slot(0, "0", 10093)],
                ["adjust_bottle"],
                root,
            )
            with mock.patch.object(
                coordinator,
                "run_task",
                side_effect=OSError("manifest unavailable"),
            ):
                coordinator.worker(launcher.Slot(0, "0", 10093))

        self.assertEqual(coordinator.work.qsize(), 0)
        self.assertEqual(len(coordinator.internal_errors), 1)
        self.assertIn("manifest unavailable", coordinator.internal_errors[0])

    def test_cleanup_kills_children_after_group_leader_exits(self) -> None:
        from evaluation.common.launcher_utils import terminate_process_group

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            child_pid_file = root / "child.pid"
            script = root / "leader.py"
            script.write_text(
                textwrap.dedent(
                    """\
                    import os
                    import subprocess
                    import sys

                    child = subprocess.Popen(
                        [sys.executable, "-c", "import time; time.sleep(30)"]
                    )
                    with open(os.environ["CHILD_PID_FILE"], "w", encoding="utf-8") as file:
                        file.write(str(child.pid))
                    raise SystemExit(7)
                    """
                ),
                encoding="utf-8",
            )
            process = subprocess.Popen(
                [sys.executable, str(script)],
                env={**os.environ, "CHILD_PID_FILE": str(child_pid_file)},
                start_new_session=True,
            )
            self.assertEqual(process.wait(timeout=5), 7)
            child_pid = int(child_pid_file.read_text(encoding="utf-8"))
            self.assertTrue(process_is_running(child_pid))
            terminate_process_group(process)
            for _ in range(50):
                if not process_is_running(child_pid):
                    break
                time.sleep(0.1)
            else:
                self.fail(f"orphaned process-group child PID {child_pid}")


class MultiGpuLauncherIntegrationTest(unittest.TestCase):
    def test_libero_jobs_overlap_and_failure_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            minicpm_python = make_fake_minicpm_python(root)
            libero_python = make_fake_libero_python(root)
            libero_home = root / "LIBERO"
            libero_home.mkdir()
            trace = root / "trace.jsonl"
            base_port = available_port_range(2)
            environment = {
                "MINICPM_PYTHON": minicpm_python,
                "LIBERO_PYTHON": libero_python,
                "LIBERO_HOME": libero_home,
                "OUTPUT_ROOT": root / "outputs",
                "GPU_LIST": "0 1",
                "TASK_SUITES": "libero_goal libero_object",
                "BASE_PORT": base_port,
                "SERVER_TIMEOUT": 10,
                "EMBODIMENT_ID": 0,
                "RUN_ID": "parallel-test",
                "FAKE_EVAL_TRACE": trace,
                "FAKE_EVAL_EXPECTED": 2,
                "FAKE_EVAL_SLEEP": 0.2,
            }
            command = [
                "bash",
                "evaluation/libero/auto_eval_scripts/auto_eval_libero.sh",
                "--checkpoint",
                "fake/model",
            ]
            result = run(command, env=environment, timeout=30)
            self.assertEqual(
                result.returncode,
                0,
                f"stdout={result.stdout}\nstderr={result.stderr}",
            )
            starts, ends = read_intervals(trace)
            self.assertEqual(len(starts), 2)
            self.assertLess(max(starts.values()), min(ends.values()))
            assert_ports_available(self, base_port, 2)

            trace.write_text("", encoding="utf-8")
            failure_environment = dict(environment)
            failure_environment.update(
                {
                    "GPU_LIST": "0",
                    "TASK_SUITES": "libero_goal",
                    "BASE_PORT": available_port_range(1),
                    "RUN_ID": "failure-test",
                    "FAKE_FAIL_PORT": str(available_port_range(1)),
                    "FAKE_EVAL_EXPECTED": 1,
                    "FAKE_EVAL_SLEEP": 0.1,
                }
            )
            failure_environment["BASE_PORT"] = failure_environment["FAKE_FAIL_PORT"]
            failed = run(command, env=failure_environment, timeout=30)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("jobs failed", failed.stderr)
            assert_ports_available(
                self,
                int(failure_environment["BASE_PORT"]),
                1,
            )

    def test_robotwin_tasks_overlap_and_write_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            minicpm_python = make_fake_minicpm_python(root)
            robotwin_root = root / "RoboTwin"
            script_dir = robotwin_root / "script"
            script_dir.mkdir(parents=True)
            trace = root / "robotwin-trace.jsonl"
            (script_dir / "eval_policy.py").write_text(
                textwrap.dedent(
                    """\
                    import json
                    import os
                    import sys
                    import time

                    config_path = sys.argv[sys.argv.index("--config") + 1]
                    port = ""
                    with open(config_path, encoding="utf-8") as file:
                        for line in file:
                            if line.startswith("port:"):
                                port = line.split(":", 1)[1].strip()
                    trace = os.environ["FAKE_ROBOTWIN_TRACE"]
                    with open(trace, "a", encoding="utf-8") as file:
                        file.write(json.dumps({"event": "start", "port": port, "time": time.time()}) + "\\n")
                    expected = int(os.environ.get("FAKE_ROBOTWIN_EXPECTED", "1"))
                    deadline = time.time() + 10
                    while time.time() < deadline:
                        with open(trace, encoding="utf-8") as file:
                            started = sum('"event": "start"' in line for line in file)
                        if started >= expected:
                            break
                        time.sleep(0.05)
                    else:
                        raise SystemExit(8)
                    time.sleep(0.2)
                    with open(trace, "a", encoding="utf-8") as file:
                        file.write(json.dumps({"event": "end", "port": port, "time": time.time()}) + "\\n")
                    print("Success rate: 1.0", flush=True)
                    """
                ),
                encoding="utf-8",
            )
            base_port = available_port_range(2)
            output_root = root / "outputs"
            result = run(
                [
                    "bash",
                    "evaluation/robotwin/start_eval.sh",
                    "--mode",
                    "demo_clean",
                    "--run-name",
                    "integration",
                    "--checkpoint",
                    "fake/model",
                    "--base-port",
                    str(base_port),
                    "--server-timeout",
                    "10",
                    "--default-embodiment-id",
                    "0",
                    "adjust_bottle",
                    "open_laptop",
                ],
                env={
                    "CUDA_VISIBLE_DEVICES": "0,1",
                    "MINICPM_PYTHON": minicpm_python,
                    "ROBOTWIN_PYTHON": sys.executable,
                    "ROBOTWIN_PATH": robotwin_root,
                    "OUTPUT_ROOT": output_root,
                    "FAKE_ROBOTWIN_TRACE": trace,
                    "FAKE_ROBOTWIN_EXPECTED": 2,
                },
                timeout=30,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout={result.stdout}\nstderr={result.stderr}",
            )
            starts, ends = read_intervals(trace)
            self.assertEqual(len(starts), 2)
            self.assertLess(max(starts.values()), min(ends.values()))
            run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            status_lines = (
                run_dirs[0] / "status.tsv"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(status_lines), 3)
            self.assertTrue(all(line.endswith("\t0") for line in status_lines[1:]))
            self.assertTrue((run_dirs[0] / "run_manifest.tsv").is_file())
            self.assertTrue((run_dirs[0] / "schedule.tsv").is_file())
            assert_ports_available(self, base_port, 2)

    def test_libero_sigterm_reaps_server_and_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            minicpm_python = make_fake_minicpm_python(root)
            libero_python = make_fake_libero_python(root)
            libero_home = root / "LIBERO"
            libero_home.mkdir()
            trace = root / "signal-trace.jsonl"
            base_port = available_port_range(1)
            environment = dict(os.environ)
            environment.update(
                {
                    "MINICPM_PYTHON": str(minicpm_python),
                    "LIBERO_PYTHON": str(libero_python),
                    "LIBERO_HOME": str(libero_home),
                    "OUTPUT_ROOT": str(root / "outputs"),
                    "GPU_LIST": "0",
                    "TASK_SUITES": "libero_goal",
                    "BASE_PORT": str(base_port),
                    "SERVER_TIMEOUT": "10",
                    "EMBODIMENT_ID": "0",
                    "RUN_ID": "signal-test",
                    "FAKE_EVAL_TRACE": str(trace),
                    "FAKE_EVAL_SLEEP": "30",
                }
            )
            process = subprocess.Popen(
                [
                    "bash",
                    "evaluation/libero/auto_eval_scripts/auto_eval_libero.sh",
                    "--checkpoint",
                    "fake/model",
                ],
                cwd=MINICPM_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                for _ in range(150):
                    if trace.is_file() and '"event": "start"' in trace.read_text(
                        encoding="utf-8"
                    ):
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(0.1)
                self.assertIsNone(
                    process.poll(),
                    "launcher exited before the fake evaluator started",
                )
                process.terminate()
                stdout, stderr = process.communicate(timeout=25)
                self.assertNotEqual(
                    process.returncode,
                    0,
                    f"stdout={stdout}\nstderr={stderr}",
                )
                assert_ports_available(self, base_port, 1)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)

    def test_calvin_sigterm_reaps_evaluator_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            minicpm_python = make_fake_minicpm_python(root)
            calvin_python = make_fake_calvin_python(root)
            calvin_root = root / "CALVIN"
            dataset = root / "dataset"
            calvin_root.mkdir()
            dataset.mkdir()
            child_pid_file = root / "calvin-child.pid"
            base_port = available_port_range(1)
            environment = dict(os.environ)
            environment.update(
                {
                    "MINICPM_PYTHON": str(minicpm_python),
                    "CALVIN_PYTHON": str(calvin_python),
                    "CALVIN_ROOT": str(calvin_root),
                    "CALVIN_DATASET_PATH": str(dataset),
                    "CHECKPOINT": "fake/model",
                    "EMBODIMENT_ID": "0",
                    "PORT": str(base_port),
                    "READY_TIMEOUT": "10",
                    "OUTPUT_ROOT": str(root / "outputs"),
                    "FAKE_CALVIN_PID_FILE": str(child_pid_file),
                }
            )
            process = subprocess.Popen(
                ["bash", "evaluation/calvin/eval_calvin.sh"],
                cwd=MINICPM_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                for _ in range(150):
                    if child_pid_file.is_file():
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(0.1)
                self.assertTrue(
                    child_pid_file.is_file(),
                    "fake CALVIN evaluator didn't start its child process",
                )
                child_pid = int(child_pid_file.read_text(encoding="utf-8"))
                process.terminate()
                stdout, stderr = process.communicate(timeout=25)
                self.assertNotEqual(
                    process.returncode,
                    0,
                    f"stdout={stdout}\nstderr={stderr}",
                )
                for _ in range(50):
                    if not process_is_running(child_pid):
                        break
                    time.sleep(0.1)
                else:
                    self.fail(f"CALVIN evaluator child PID {child_pid} leaked")
                assert_ports_available(self, base_port, 1)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)

    def test_robotwin_sigterm_reaps_server_and_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            minicpm_python = make_fake_minicpm_python(root)
            robotwin_root = root / "RoboTwin"
            script_dir = robotwin_root / "script"
            script_dir.mkdir(parents=True)
            started_file = root / "robotwin-started"
            (script_dir / "eval_policy.py").write_text(
                textwrap.dedent(
                    """\
                    import os
                    import time
                    from pathlib import Path

                    Path(os.environ["FAKE_ROBOTWIN_STARTED"]).write_text(
                        "started", encoding="utf-8"
                    )
                    time.sleep(30)
                    """
                ),
                encoding="utf-8",
            )
            base_port = available_port_range(1)
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "MINICPM_PYTHON": str(minicpm_python),
                    "ROBOTWIN_PYTHON": sys.executable,
                    "ROBOTWIN_PATH": str(robotwin_root),
                    "OUTPUT_ROOT": str(root / "outputs"),
                    "FAKE_ROBOTWIN_STARTED": str(started_file),
                }
            )
            process = subprocess.Popen(
                [
                    "bash",
                    "evaluation/robotwin/start_eval.sh",
                    "--mode",
                    "demo_clean",
                    "--run-name",
                    "signal",
                    "--checkpoint",
                    "fake/model",
                    "--base-port",
                    str(base_port),
                    "--server-timeout",
                    "10",
                    "--default-embodiment-id",
                    "0",
                    "adjust_bottle",
                ],
                cwd=MINICPM_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                for _ in range(150):
                    if started_file.is_file():
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(0.1)
                self.assertTrue(
                    started_file.is_file(),
                    "fake RoboTwin evaluator didn't start",
                )
                process.terminate()
                stdout, stderr = process.communicate(timeout=25)
                self.assertNotEqual(
                    process.returncode,
                    0,
                    f"stdout={stdout}\nstderr={stderr}",
                )
                assert_ports_available(self, base_port, 1)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
