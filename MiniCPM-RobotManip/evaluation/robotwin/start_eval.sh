#!/usr/bin/env bash
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MINICPM_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LAUNCHER_PYTHON="${LAUNCHER_PYTHON:-python3}"

export PYTHONPATH="${MINICPM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${LAUNCHER_PYTHON}" -m evaluation.robotwin.launcher "$@"
