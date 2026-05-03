#!/usr/bin/env bash
# Drive AutoDL from a CPU-only laptop: remote git pull + train, then fetch runs/.
#
# One-time setup (local machine):
#   cp scripts/autodl/env.example scripts/autodl/env.local && nano scripts/autodl/env.local
#   chmod +x scripts/autodl/drive_autodl.sh
#
# Day-to-day:
#   1) Push code from Cursor: git commit && git push origin main
#   2) ./scripts/autodl/drive_autodl.sh pull-train -- --max-epochs 20 --num-envs 512 --save-frequency 1
#   3) ./scripts/autodl/drive_autodl.sh fetch-runs
#   4) tensorboard --logdir "${HOME}/Downloads/autodl_simtoolreal_runs/runs" --bind_all 127.0.0.1 --port 6006
#   5) (optional) VLC: open any *.mp4 under .../runs/ after remote-play
#
# Isaac Gym trains/renders only on GPU; your laptop views TensorBoard + videos — no local GPU needed.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/env.local"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}"
  echo "Run: cp scripts/autodl/env.example scripts/autodl/env.local && edit it"
  exit 1
fi

# shellcheck disable=SC1090
set -a
source "${ENV_FILE}"
set +a

: "${AUTO_DL_HOST:?Set in env.local}"
: "${AUTO_DL_PORT:?}"
: "${AUTO_DL_USER:?}"
: "${AUTO_DL_REPO_DIR:?}"

LOCAL_RUNS_DIR="${LOCAL_RUNS_DIR:-${HOME}/Downloads/autodl_simtoolreal_runs}"

SSH_ARGS=(-p "${AUTO_DL_PORT}" -o StrictHostKeyChecking=accept-new)
SCP_ARGS=(-P "${AUTO_DL_PORT}" -o StrictHostKeyChecking=accept-new)

if [[ -n "${SSH_IDENTITY_FILE:-}" ]]; then
  SSH_ARGS+=(-i "${SSH_IDENTITY_FILE}")
  SCP_ARGS+=(-i "${SSH_IDENTITY_FILE}")
fi

REMOTE="${AUTO_DL_USER}@${AUTO_DL_HOST}"

if [[ -n "${AUTO_DL_PASSWORD:-}" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "Install sshpass OR clear AUTO_DL_PASSWORD and use pubkey auth."
    exit 1
  fi
  SSHWRAP=(sshpass -p "${AUTO_DL_PASSWORD}")
else
  SSHWRAP=()
fi

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") pull-train -- [<py args for knife_fruit_quick_train.py>]
      Example:
        $(basename "$0") pull-train -- --max-epochs 15 --num-envs 512 --save-frequency 1

  $(basename "$0") fetch-runs
      scp remote \${AUTO_DL_REPO_DIR}/runs -> ${LOCAL_RUNS_DIR}/runs

  $(basename "$0") remote-play phase1_trajectory | phase2_peel_surrogate
      Headless policy rollout + RecordVideo on AutoDL (then run fetch-runs for MP4s).

Workflow: push locally -> pull-train on GPU -> fetch-runs -> TensorBoard/VLC locally.
USAGE
}

cmd_pull_train() {
  if [[ "${1:-}" == "--" ]]; then shift; fi
  declare -a user_args=("$@")
  local q_py
  q_py=$(printf '%q ' "${user_args[@]}")

  local q_repo
  q_repo=$(printf '%q' "${AUTO_DL_REPO_DIR}")

  "${SSHWRAP[@]}" ssh "${SSH_ARGS[@]}" "${REMOTE}" bash <<EOF
set -euo pipefail
AUTO_DL_REPO_DIR=${q_repo}
source /etc/network_turbo 2>/dev/null || true
cd "\${AUTO_DL_REPO_DIR}"
if [[ ! -d .venv ]]; then
  echo ".venv missing — complete one-time setup on AutoDL (uv + Isaac Gym) first."
  exit 1
fi
source .venv/bin/activate
git fetch origin main
git checkout main
git pull --ff-only origin main || {
  echo "git pull --ff-only failed. On AutoDL shell: cd \${AUTO_DL_REPO_DIR} && git status"
  exit 1
}
python scripts/knife_fruit_quick_train.py ${q_py}
EOF
}

cmd_fetch_runs() {
  mkdir -p "${LOCAL_RUNS_DIR}"
  echo "scp ${REMOTE}:${AUTO_DL_REPO_DIR}/runs -> ${LOCAL_RUNS_DIR}/"
  "${SSHWRAP[@]}" scp "${SCP_ARGS[@]}" -r \
    "${REMOTE}:${AUTO_DL_REPO_DIR}/runs" \
    "${LOCAL_RUNS_DIR}/"
  echo
  echo "Local TensorBoard (no GPU):"
  echo "  tensorboard --logdir '${LOCAL_RUNS_DIR}/runs' --bind_all 127.0.0.1 --port 6006"
  echo "Videos (if any): find '${LOCAL_RUNS_DIR}/runs' -name '*.mp4'"
}

cmd_remote_play() {
  local phase="${1:?phase argument: phase1_trajectory or phase2_peel_surrogate}"
  case "${phase}" in
    phase1_trajectory | phase2_peel_surrogate) ;;
    *)
      echo "phase must be phase1_trajectory or phase2_peel_surrogate"
      exit 1
      ;;
  esac
  local q_repo
  q_repo=$(printf '%q' "${AUTO_DL_REPO_DIR}")
  local q_phase
  q_phase=$(printf '%q' "${phase}")
  "${SSHWRAP[@]}" ssh "${SSH_ARGS[@]}" "${REMOTE}" bash <<EOF
set -euo pipefail
AUTO_DL_REPO_DIR=${q_repo}
source /etc/network_turbo 2>/dev/null || true
cd "\${AUTO_DL_REPO_DIR}"
source .venv/bin/activate
exec python scripts/play_trained_peeler.py --profile ${q_phase} --headless true --capture-video True --num-envs 4
EOF
}

main() {
  case "${1:-}" in
    pull-train)
      shift
      cmd_pull_train "$@"
      ;;
    fetch-runs)
      cmd_fetch_runs
      ;;
    remote-play)
      shift
      cmd_remote_play "${1:?}"
      ;;
    ""|-h|--help|help)
      usage
      ;;
    *)
      echo "Unknown: $1"; usage; exit 1
      ;;
  esac
}

main "$@"
