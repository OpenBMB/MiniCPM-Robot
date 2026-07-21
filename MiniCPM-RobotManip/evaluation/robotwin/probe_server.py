"""Backward-compatible RoboTwin readiness entry point."""

import sys

from evaluation.common.probe_server import main


if __name__ == "__main__":
    raise SystemExit(
        main([*sys.argv[1:], "--min-action-dim", "14"])
    )
