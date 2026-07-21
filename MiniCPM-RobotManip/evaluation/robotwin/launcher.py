# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

"""Multi-GPU RoboTwin evaluation launcher.

This module replaces the large Bash scheduler migrated from starVLA. One
worker thread owns each GPU slot and runs server/evaluator subprocesses
serially; different slots execute concurrently.
"""

from __future__ import annotations

import argparse
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Sequence


ALL_TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle_horizontally",
    "shake_bottle",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)


@dataclass(frozen=True)
class Slot:
    index: int
    gpu: str
    port: int


@dataclass(frozen=True)
class WorkItem:
    number: int
    task: str


@dataclass
class Config:
    root: Path
    script_dir: Path
    robotwin_path: Path | None
    minicpm_python: str
    robotwin_python: str
    checkpoint: str
    embodiment_id: int
    mode: str
    run_name: str
    seed: int
    device: str
    host: str
    server_timeout: int
    output_root: Path
    dry_run: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RoboTwin tasks with bounded MiniCPM GPU slots."
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        choices=("demo_clean", "demo_randomized"),
    )
    parser.add_argument(
        "-n",
        "--run-name",
        "--name",
        dest="run_name",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        "--ckpt",
        dest="checkpoint",
        required=True,
        help="Hugging Face model ID or local checkpoint directory",
    )
    parser.add_argument(
        "--default-embodiment-id",
        type=int,
        default=None,
        help="Required MiniCPM RoboTwin embodiment ID",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=int(os.environ.get("ROBOTWIN_SEED", "0")),
    )
    parser.add_argument(
        "-j",
        "--jobs-per-gpu",
        type=int,
        default=int(os.environ.get("ROBOTWIN_JOBS_PER_GPU", "1")),
    )
    parser.add_argument(
        "-p",
        "--base-port",
        type=int,
        default=int(os.environ.get("ROBOTWIN_BASE_PORT", "10093")),
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=int(os.environ.get("ROBOTWIN_SERVER_TIMEOUT", "600")),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "tasks",
        nargs="+",
        help="Task names, comma-separated tasks, task files, or 'all'",
    )
    return parser


def resolve_executable(value: str, label: str) -> str:
    candidate = Path(value).expanduser()
    if "/" in value:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise ValueError(f"{label} is not executable: {candidate}")
        return str(candidate.resolve())
    resolved = shutil.which(value)
    if resolved is None:
        raise ValueError(f"{label} is not available on PATH: {value}")
    return resolved


def resolve_checkpoint(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError(f"Local checkpoint must be a directory: {candidate}")
        return str(candidate.resolve())
    if value.startswith(("/", "./", "../")):
        raise ValueError(f"Local checkpoint directory does not exist: {value}")
    return value


def _add_task_text(tasks: list[str], text: str) -> None:
    for value in text.split(","):
        value = value.strip()
        if not value:
            continue
        if value == "all":
            tasks.extend(ALL_TASKS)
        else:
            tasks.append(value)


def resolve_tasks(values: Sequence[str]) -> list[str]:
    tasks: list[str] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                _add_task_text(tasks, line.split("#", 1)[0].strip())
        else:
            _add_task_text(tasks, value)
    if not tasks:
        raise ValueError("No RoboTwin tasks were resolved from input")
    return tasks


def detect_cuda_devices() -> list[str]:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible = os.environ["CUDA_VISIBLE_DEVICES"].strip()
        if not visible or visible == "-1":
            raise ValueError("CUDA_VISIBLE_DEVICES explicitly disables all GPUs")
        devices = [value.strip() for value in visible.split(",") if value.strip()]
        if devices:
            return devices
        raise ValueError("CUDA_VISIBLE_DEVICES contains no usable GPU IDs")

    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        devices = []
        if result.returncode == 0:
            devices = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip().isdigit()
            ]
        if devices:
            return devices

    raise ValueError(
        "No GPU was detected. Set CUDA_VISIBLE_DEVICES to the intended GPU IDs."
    )


def port_available(host: str, port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def allocate_slots(
    devices: Sequence[str],
    jobs_per_gpu: int,
    base_port: int,
    host: str,
) -> list[Slot]:
    if jobs_per_gpu < 1:
        raise ValueError("jobs-per-gpu must be positive")
    slots: list[Slot] = []
    candidate = base_port
    for device in devices:
        for _ in range(jobs_per_gpu):
            while candidate <= 65535 and not port_available(host, candidate):
                candidate += 1
            if candidate > 65535:
                raise ValueError("No available policy port remains")
            slots.append(Slot(len(slots), device, candidate))
            candidate += 1
    return slots


def safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    return label or "unnamed"


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def terminate_process(
    process: subprocess.Popen | None,
    *,
    timeout: float = 5,
) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=timeout)


class Coordinator:
    def __init__(
        self,
        config: Config,
        slots: Sequence[Slot],
        tasks: Sequence[str],
        log_dir: Path,
    ) -> None:
        self.config = config
        self.slots = list(slots)
        self.log_dir = log_dir
        self.work: queue.Queue[WorkItem] = queue.Queue()
        for index, task in enumerate(tasks, start=1):
            self.work.put(WorkItem(index, task))
        self.interrupted = threading.Event()
        self.lock = threading.Lock()
        self.active: dict[int, tuple[subprocess.Popen | None, subprocess.Popen | None]] = {}
        self.failures: list[str] = []
        self.internal_errors: list[str] = []
        self.schedule_path = log_dir / "schedule.tsv"
        self.status_path = log_dir / "status.tsv"

    def register(
        self,
        slot: Slot,
        server: subprocess.Popen | None,
        evaluator: subprocess.Popen | None,
    ) -> None:
        with self.lock:
            self.active[slot.index] = (server, evaluator)

    def stop_all(self) -> None:
        self.interrupted.set()
        with self.lock:
            processes = list(self.active.values())
        for server, evaluator in processes:
            terminate_process(evaluator)
            terminate_process(server)

    def append_schedule(
        self,
        item: WorkItem,
        slot: Slot,
        server_log: Path,
        eval_log: Path,
    ) -> None:
        with self.lock, self.schedule_path.open("a", encoding="utf-8") as file:
            file.write(
                f"{item.number}\t{item.task}\t{slot.index}\t{slot.gpu}\t"
                f"{slot.port}\t{server_log}\t{eval_log}\n"
            )

    def append_status(self, item: WorkItem, slot: Slot, status: int) -> None:
        with self.lock, self.status_path.open("a", encoding="utf-8") as file:
            file.write(f"{item.task}\t{slot.index}\t{status}\n")
            if status:
                self.failures.append(item.task)

    def probe(self, slot: Slot) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["PYTHONPATH"] = (
            f"{self.config.root}"
            + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
        )
        return subprocess.run(
            [
                self.config.minicpm_python,
                "-m",
                "evaluation.common.probe_server",
                "--host",
                self.config.host,
                "--port",
                str(slot.port),
                "--checkpoint",
                self.config.checkpoint,
                "--embodiment-id",
                str(self.config.embodiment_id),
                "--min-action-dim",
                "14",
                "--request-id",
                "robotwin-readiness",
                "--timeout",
                "2",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

    def wait_ready(self, slot: Slot, server: subprocess.Popen) -> None:
        deadline = time.monotonic() + self.config.server_timeout
        last_error = ""
        while time.monotonic() < deadline:
            if self.interrupted.is_set():
                raise RuntimeError("Evaluation interrupted")
            status = server.poll()
            if status is not None:
                raise RuntimeError(
                    f"Policy server exited before readiness with status {status}"
                )
            try:
                probe = self.probe(slot)
            except subprocess.TimeoutExpired:
                last_error = "readiness probe timed out"
            else:
                if probe.returncode == 0:
                    return
                last_error = (probe.stderr or probe.stdout).strip()
                if probe.returncode == 3:
                    raise RuntimeError(last_error or "Server metadata mismatch")
            time.sleep(1)
        raise RuntimeError(
            f"Timed out waiting for {self.config.host}:{slot.port}: {last_error}"
        )

    def run_task(self, slot: Slot, item: WorkItem) -> int:
        task_safe = safe_label(item.task)
        slot_label = f"slot{slot.index}_gpu{safe_label(slot.gpu)}_port{slot.port}"
        server_log = (
            self.log_dir
            / f"{item.number:03d}_{task_safe}_{self.config.mode}_{slot_label}_server.log"
        )
        eval_log = (
            self.log_dir
            / f"{item.number:03d}_{task_safe}_{self.config.mode}_{slot_label}_eval.log"
        )
        self.append_schedule(item, slot, server_log, eval_log)
        print(
            f"[INFO] Launching task={item.task} mode={self.config.mode} "
            f"gpu={slot.gpu} port={slot.port}",
            flush=True,
        )

        server: subprocess.Popen | None = None
        evaluator: subprocess.Popen | None = None
        status = 1
        try:
            with server_log.open(
                "w", encoding="utf-8"
            ) as server_stream, eval_log.open(
                "w", encoding="utf-8"
            ) as eval_stream:
                server = subprocess.Popen(
                    [
                        "bash",
                        str(self.config.script_dir / "run_policy_server.sh"),
                        self.config.checkpoint,
                        slot.gpu,
                        str(slot.port),
                        self.config.host,
                        self.config.device,
                        str(self.config.embodiment_id),
                    ],
                    stdout=server_stream,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        "MINICPM_PYTHON": self.config.minicpm_python,
                    },
                    start_new_session=True,
                )
                self.register(slot, server, None)
                self.wait_ready(slot, server)

                if self.interrupted.is_set():
                    raise RuntimeError("Evaluation interrupted")
                evaluator = subprocess.Popen(
                    [
                        "bash",
                        str(self.config.script_dir / "eval.sh"),
                        item.task,
                        self.config.mode,
                        self.config.run_name,
                        str(self.config.seed),
                        slot.gpu,
                        str(slot.port),
                        self.config.host,
                    ],
                    stdout=eval_stream,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        "ROBOTWIN_PATH": str(self.config.robotwin_path),
                        "ROBOTWIN_PYTHON": self.config.robotwin_python,
                        "ROBOTWIN_POLICY_MODULE": (
                            "evaluation.robotwin.model2robotwin_interface"
                        ),
                    },
                    start_new_session=True,
                )
                self.register(slot, server, evaluator)
                if self.interrupted.is_set():
                    raise RuntimeError("Evaluation interrupted")
                status = evaluator.wait()

                if status == 0 and server.poll() is not None:
                    status = 1
                    print(
                        f"[ERROR] Server exited before task={item.task} completed.",
                        file=sys.stderr,
                    )
        except Exception as exc:
            print(f"[ERROR] Task {item.task}: {exc}", file=sys.stderr)
            status = 1
        finally:
            terminate_process(evaluator)
            terminate_process(server)
            self.register(slot, None, None)

        if eval_log.is_file():
            for line in eval_log.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if "Success rate" in line:
                    print(f"[RESULT] {item.task}: {line}")
        self.append_status(item, slot, status)
        return status

    def worker(self, slot: Slot) -> None:
        while not self.interrupted.is_set():
            try:
                item = self.work.get_nowait()
            except queue.Empty:
                return
            try:
                try:
                    self.run_task(slot, item)
                except Exception as exc:
                    message = (
                        f"slot={slot.index} task={item.task}: "
                        f"internal worker error: {exc}"
                    )
                    print(f"[ERROR] {message}", file=sys.stderr)
                    with self.lock:
                        self.internal_errors.append(message)
            finally:
                self.work.task_done()


def write_manifests(
    config: Config,
    slots: Sequence[Slot],
    tasks: Sequence[str],
    log_dir: Path,
) -> None:
    with (log_dir / "run_manifest.tsv").open("w", encoding="utf-8") as file:
        fields = {
            "source": "starVLA@631aae02afe6d95876e923ff518e8ff2ab9a2f88",
            "checkpoint": config.checkpoint,
            "minicpm_revision": git_revision(config.root),
            "robotwin_revision": (
                git_revision(config.robotwin_path)
                if config.robotwin_path is not None
                else "unknown"
            ),
            "run_name": config.run_name,
            "task_config": config.mode,
            "seed": str(config.seed),
            "default_embodiment_id": str(config.embodiment_id),
            "gpus": ",".join(slot.gpu for slot in slots),
            "ports": ",".join(str(slot.port) for slot in slots),
            "total_tasks": str(len(tasks)),
            "minicpm_python": config.minicpm_python,
            "robotwin_python": config.robotwin_python,
        }
        for key, value in fields.items():
            file.write(f"{key}\t{value}\n")
    (log_dir / "schedule.tsv").write_text(
        "task_number\ttask\tslot\tgpu\tport\tserver_log\teval_log\n",
        encoding="utf-8",
    )
    (log_dir / "status.tsv").write_text(
        "task\tslot\texit_status\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    embodiment_id = args.default_embodiment_id
    if embodiment_id is None:
        value = os.environ.get("ROBOTWIN_DEFAULT_EMBODIMENT_ID", "")
        if not value:
            print(
                "--default-embodiment-id is required because the checkpoint "
                "doesn't publish a RoboTwin mapping.",
                file=sys.stderr,
            )
            return 2
        try:
            embodiment_id = int(value)
        except ValueError:
            print(
                "ROBOTWIN_DEFAULT_EMBODIMENT_ID must be an integer.",
                file=sys.stderr,
            )
            return 2
    if embodiment_id < 0:
        print("--default-embodiment-id must be non-negative.", file=sys.stderr)
        return 2
    if args.jobs_per_gpu < 1:
        print("--jobs-per-gpu must be positive.", file=sys.stderr)
        return 2
    if not 1 <= args.base_port <= 65535:
        print("--base-port must be in [1, 65535].", file=sys.stderr)
        return 2
    if args.server_timeout < 1:
        print("--server-timeout must be positive.", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[2]
    script_dir = Path(__file__).resolve().parent
    try:
        checkpoint = resolve_checkpoint(args.checkpoint)
        tasks = resolve_tasks(args.tasks)
        minicpm_python = resolve_executable(
            os.environ.get("MINICPM_PYTHON", "python"),
            "MINICPM_PYTHON",
        )
        robotwin_python = resolve_executable(
            os.environ.get("ROBOTWIN_PYTHON", "python"),
            "ROBOTWIN_PYTHON",
        )
        slots = allocate_slots(
            detect_cuda_devices(),
            args.jobs_per_gpu,
            args.base_port,
            "127.0.0.1",
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    robotwin_value = os.environ.get("ROBOTWIN_PATH", "")
    robotwin_path = Path(robotwin_value).expanduser() if robotwin_value else None
    if not args.dry_run:
        if (
            robotwin_path is None
            or not (robotwin_path / "script" / "eval_policy.py").is_file()
        ):
            print(
                "ROBOTWIN_PATH must contain script/eval_policy.py.",
                file=sys.stderr,
            )
            return 2

    if args.jobs_per_gpu > 1:
        print(
            f"[WARN] jobs-per-gpu={args.jobs_per_gpu} loads multiple models "
            "and simulators on each GPU.",
            file=sys.stderr,
        )
    print(f"[INFO] MiniCPM python: {minicpm_python}")
    print(f"[INFO] RoboTwin python: {robotwin_python}")
    print(
        f"[INFO] mode={args.mode} run_name={args.run_name} seed={args.seed}"
    )
    print(
        f"[INFO] checkpoint={checkpoint} "
        f"default_embodiment_id={embodiment_id}"
    )
    print(
        f"[INFO] GPUs={','.join(slot.gpu for slot in slots)} "
        f"slots={len(slots)}"
    )
    print(f"[INFO] tasks ({len(tasks)}): {', '.join(tasks)}")

    if args.dry_run:
        print("[DRY-RUN] No server, simulator, or output directory will start.")
        for slot in slots:
            print(
                f"[DRY-RUN] slot={slot.index} gpu={slot.gpu} port={slot.port}"
            )
        for index, task in enumerate(tasks, start=1):
            print(f"[DRY-RUN] FIFO[{index}]={task}")
        return 0

    output_root = Path(
        os.environ.get(
            "OUTPUT_ROOT",
            root / "outputs" / "evaluation" / "robotwin",
        )
    ).expanduser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_root / (
        f"{safe_label(args.run_name)}_{args.mode}_"
        f"{safe_label(Path(checkpoint).name)}_{timestamp}_{os.getpid()}"
    )
    log_dir.mkdir(parents=True)

    config = Config(
        root=root,
        script_dir=script_dir,
        robotwin_path=robotwin_path,
        minicpm_python=minicpm_python,
        robotwin_python=robotwin_python,
        checkpoint=checkpoint,
        embodiment_id=embodiment_id,
        mode=args.mode,
        run_name=args.run_name,
        seed=args.seed,
        device=os.environ.get("MINICPM_DEVICE", "cuda"),
        host="127.0.0.1",
        server_timeout=args.server_timeout,
        output_root=output_root,
        dry_run=False,
    )
    write_manifests(config, slots, tasks, log_dir)
    coordinator = Coordinator(config, slots, tasks, log_dir)

    interrupted_signal = 0

    def handle_signal(signum, frame) -> None:
        del frame
        nonlocal interrupted_signal
        interrupted_signal = signum
        print(
            f"[WARN] Received signal {signum}; stopping all RoboTwin jobs.",
            file=sys.stderr,
        )
        coordinator.stop_all()

    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.signal(signum, handle_signal)

    try:
        threads = [
            threading.Thread(
                target=coordinator.worker,
                args=(slot,),
                name=f"robotwin-slot-{slot.index}",
            )
            for slot in slots
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        coordinator.stop_all()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    if interrupted_signal:
        return 128 + interrupted_signal
    if coordinator.internal_errors or not coordinator.work.empty():
        for error in coordinator.internal_errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        if not coordinator.work.empty():
            print(
                f"[ERROR] {coordinator.work.qsize()} RoboTwin tasks were not run.",
                file=sys.stderr,
            )
        return 1
    if coordinator.failures:
        print(
            "[ERROR] RoboTwin evaluation failures: "
            + ", ".join(coordinator.failures),
            file=sys.stderr,
        )
        print(f"[ERROR] Logs are under {log_dir}", file=sys.stderr)
        return 1
    print("[INFO] RoboTwin evaluation finished successfully.")
    print(f"[INFO] Logs are under {log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
