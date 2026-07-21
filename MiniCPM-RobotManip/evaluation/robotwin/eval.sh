#!/usr/bin/env bash
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Modifications Copyright 2026 The OpenBMB Team.

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  bash eval.sh <task_name> <task_config> <run_name> <seed> <gpu_id> [policy_port] [policy_host]

Environment:
  ROBOTWIN_PATH       External RoboTwin checkout (required)
  ROBOTWIN_PYTHON     Python from the RoboTwin environment (default: python)
  ROBOTWIN_POLICY_MODULE
                      Dotted RoboTwin policy module
  DEPLOY_POLICY_TEMPLATE_PATH
                      Optional deploy_policy.yml template override
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if (( $# < 5 || $# > 7 )); then
    usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINICPM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

task_name="$1"
task_config="$2"
run_name="$3"
seed="$4"
gpu_id="$5"
policy_port="${6:-${ROBOTWIN_POLICY_PORT:-10093}}"
policy_host="${7:-${ROBOTWIN_POLICY_HOST:-127.0.0.1}}"
policy_module="${ROBOTWIN_POLICY_MODULE:-evaluation.robotwin.model2robotwin_interface}"
robotwin_python="${ROBOTWIN_PYTHON:-python}"
robotwin_path="${ROBOTWIN_PATH:-}"
deploy_policy_template="${DEPLOY_POLICY_TEMPLATE_PATH:-${SCRIPT_DIR}/deploy_policy.yml}"

if [[ -z "${robotwin_path}" || ! -d "${robotwin_path}" ]]; then
    echo "ROBOTWIN_PATH must point to an existing RoboTwin checkout: ${robotwin_path:-<unset>}" >&2
    exit 1
fi
if [[ ! -f "${robotwin_path}/script/eval_policy.py" ]]; then
    echo "RoboTwin eval entry does not exist: ${robotwin_path}/script/eval_policy.py" >&2
    exit 1
fi
if [[ ! -f "${deploy_policy_template}" ]]; then
    echo "Deploy policy template does not exist: ${deploy_policy_template}" >&2
    exit 1
fi
if ! command -v "${robotwin_python}" >/dev/null 2>&1; then
    echo "ROBOTWIN_PYTHON is not executable: ${robotwin_python}" >&2
    exit 1
fi
if [[ ! "${policy_port}" =~ ^[0-9]+$ ]] || (( policy_port < 1 || policy_port > 65535 )); then
    echo "Invalid policy port: ${policy_port}" >&2
    exit 1
fi
if [[ ! "${policy_host}" =~ ^[A-Za-z0-9.-]+$ ]]; then
    echo "Invalid policy host: ${policy_host}" >&2
    exit 1
fi

runtime_deploy_policy="$(
    mktemp "${TMPDIR:-/tmp}/minicpm_robotwin_deploy.${BASHPID}.XXXXXX.yml"
)"
cleanup() {
    rm -f "${runtime_deploy_policy}"
}
trap cleanup EXIT

sed \
    -e "s|^host:.*|host: \"${policy_host}\"|" \
    -e "s|^port:.*|port: ${policy_port}|" \
    "${deploy_policy_template}" > "${runtime_deploy_policy}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${MINICPM_ROOT}:${robotwin_path}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[INFO] RoboTwin task=${task_name} config=${task_config} run=${run_name}"
echo "[INFO] GPU=${gpu_id} policy=${policy_module} endpoint=${policy_host}:${policy_port}"

cd "${robotwin_path}"
PYTHONWARNINGS=ignore::UserWarning \
"${robotwin_python}" script/eval_policy.py \
    --config "${runtime_deploy_policy}" \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${run_name}" \
    --seed "${seed}" \
    --policy_name "${policy_module}"
