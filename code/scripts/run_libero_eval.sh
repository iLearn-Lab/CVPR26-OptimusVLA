#!/usr/bin/env bash

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero}"
POLICY_DIR="${POLICY_DIR:-}"
ACTION_NORM_STATS_PATH="${ACTION_NORM_STATS_PATH:-}"

SERVER_CUDA_VISIBLE_DEVICES="${SERVER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
CLIENT_CUDA_VISIBLE_DEVICES="${CLIENT_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
OPENPI_TORCH_COMPILE="${OPENPI_TORCH_COMPILE:-0}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-180}"

NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
SEED="${SEED:-7}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
MUJOCO_BACKEND="${MUJOCO_GL:-egl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/libero_eval_${RUN_ID}}"
RESULTS_TXT="${RESULTS_TXT:-${LOG_DIR}/results.txt}"

MEMORY_TOP_K="${MEMORY_TOP_K:-8}"
MEMORY_REFRESH_EVERY="${MEMORY_REFRESH_EVERY:-1}"
ALIGN_MODE="${ALIGN_MODE:-hybrid}"
MIXTURE_MODE="${MIXTURE_MODE:-gaussian}"
TEMPERATURE="${TEMPERATURE:-10.0}"
SIGMA_MIN="${SIGMA_MIN:-0.05}"
NOISE_MIN="${NOISE_MIN:-0.20}"
NOISE_MAX="${NOISE_MAX:-1.00}"
NFE_MIN="${NFE_MIN:-1}"
NFE_MAX="${NFE_MAX:-10}"
USE_LCM="${USE_LCM:-1}"
LCM_SCALE="${LCM_SCALE:-0.10}"

if [[ "$#" -gt 0 ]]; then
  SUITES=("$@")
else
  SUITES=(libero_spatial libero_object libero_goal libero_10)
fi

server_pid=""
client_pids=()
client_names=()

is_true() {
  case "${1}" in
    1 | true | TRUE | yes | YES | on | ON) return 0 ;;
    *) return 1 ;;
  esac
}

wait_for_server() {
  local waited=0
  while ((waited < SERVER_WAIT_SECONDS)); do
    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "Policy server exited before it became ready. See ${server_log}" >&2
      tail -n 80 "${server_log}" >&2 || true
      return 1
    fi
    if (exec 3<>"/dev/tcp/${HOST}/${PORT}") >/dev/null 2>&1; then
      exec 3<&- || true
      exec 3>&- || true
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "Timed out waiting for policy server at ${HOST}:${PORT}. See ${server_log}" >&2
  tail -n 80 "${server_log}" >&2 || true
  return 1
}

cleanup() {
  local pid
  for pid in "${client_pids[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
  if [[ -n "${server_pid:-}" ]]; then
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap cleanup EXIT

cd "${OPENPI_ROOT}" || exit 1

if [[ -z "${POLICY_DIR}" ]]; then
  echo "Set POLICY_DIR to the pi0.5 PyTorch checkpoint directory." >&2
  exit 1
fi

if [[ -z "${ACTION_NORM_STATS_PATH}" ]]; then
  ACTION_NORM_STATS_PATH="${POLICY_DIR}/assets/physical-intelligence/libero/norm_stats.json"
fi

if [[ ! -f "examples/libero/.venv/bin/activate" ]]; then
  echo "Missing examples/libero/.venv. Create the LIBERO client environment first." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}/stdout"
server_log="${LOG_DIR}/server.log"
status_file="${LOG_DIR}/suite_status.tsv"
: >"${status_file}"

server_cmd=(
  uv run scripts/serve_policy.py
  --env LIBERO
  --port "${PORT}"
  --use-memory
  --action-norm-stats-path "${ACTION_NORM_STATS_PATH}"
  --action-use-quantile-norm
  --memory-top-k "${MEMORY_TOP_K}"
  --memory-refresh-every "${MEMORY_REFRESH_EVERY}"
  --align-mode "${ALIGN_MODE}"
  --mixture-mode "${MIXTURE_MODE}"
  --temperature "${TEMPERATURE}"
  --sigma-min "${SIGMA_MIN}"
  --noise-min "${NOISE_MIN}"
  --noise-max "${NOISE_MAX}"
  --nfe-min "${NFE_MIN}"
  --nfe-max "${NFE_MAX}"
)

if is_true "${USE_LCM}"; then
  server_cmd+=(--use-lcm --lcm-scale "${LCM_SCALE}")
fi

server_cmd+=(policy:checkpoint --policy.config "${POLICY_CONFIG}" --policy.dir "${POLICY_DIR}")

(
  export OPENPI_TORCH_COMPILE
  export CUDA_VISIBLE_DEVICES="${SERVER_CUDA_VISIBLE_DEVICES}"
  "${server_cmd[@]}"
) >"${server_log}" 2>&1 &
server_pid="$!"

echo "Started policy server: pid=${server_pid} log=${server_log}"
wait_for_server || exit 1
echo "Policy server is ready at ${HOST}:${PORT}"

source examples/libero/.venv/bin/activate

PYTHONPATH_VALUE="${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${OPENPI_ROOT}/third_party/libero"
if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH_VALUE="${PYTHONPATH_VALUE}:${PYTHONPATH}"
fi

for suite in "${SUITES[@]}"; do
  jsonl_file="${LOG_DIR}/${suite}.jsonl"
  stdout_file="${LOG_DIR}/stdout/${suite}.log"
  (
    export PYTHONPATH="${PYTHONPATH_VALUE}"
    export MUJOCO_GL="${MUJOCO_BACKEND}"
    export CUDA_VISIBLE_DEVICES="${CLIENT_CUDA_VISIBLE_DEVICES}"
    python examples/libero/main.py \
      --args.host "${HOST}" \
      --args.port "${PORT}" \
      --args.task-suite-name "${suite}" \
      --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
      --args.replan-steps "${REPLAN_STEPS}" \
      --args.num-steps-wait "${NUM_STEPS_WAIT}" \
      --args.resize-size "${RESIZE_SIZE}" \
      --args.seed "${SEED}" \
      --args.log-file "${jsonl_file}"
  ) >"${stdout_file}" 2>&1 &
  client_pid="$!"
  client_pids+=("${client_pid}")
  client_names+=("${suite}")
  echo "Started ${suite}: pid=${client_pid} log=${jsonl_file} stdout=${stdout_file}"
done

status=0
for i in "${!client_pids[@]}"; do
  if wait "${client_pids[$i]}"; then
    exit_code=0
    echo "Finished ${client_names[$i]}"
  else
    exit_code="$?"
    echo "Failed ${client_names[$i]} with exit code ${exit_code}" >&2
    status=1
  fi
  printf "%s\t%s\n" "${client_names[$i]}" "${exit_code}" >>"${status_file}"
done

if python - "${RESULTS_TXT}" "${LOG_DIR}" "${server_log}" "${status_file}" "${client_names[@]}" <<'PY'
import datetime
import json
from pathlib import Path
import sys

results_path = Path(sys.argv[1])
log_dir = Path(sys.argv[2])
server_log = sys.argv[3]
status_path = Path(sys.argv[4])
suites = sys.argv[5:]

suite_status = {}
if status_path.exists():
    for line in status_path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            suite_status[fields[0]] = fields[1]


def load_run_summary(path):
    summary = None
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "run_summary":
                summary = record
    return summary


total_episodes = 0
total_successes = 0
summary_rows = []
for suite in suites:
    jsonl_path = log_dir / f"{suite}.jsonl"
    stdout_path = log_dir / "stdout" / f"{suite}.log"
    summary = load_run_summary(jsonl_path)
    exit_code = suite_status.get(suite, "unknown")
    if summary is None:
        summary_rows.append((suite, exit_code, None, None, None, jsonl_path, stdout_path))
        continue
    episodes = int(summary.get("total_episodes", 0))
    successes = int(summary.get("total_successes", 0))
    rate = float(summary.get("total_success_rate", 0.0))
    total_episodes += episodes
    total_successes += successes
    summary_rows.append((suite, exit_code, episodes, successes, rate, jsonl_path, stdout_path))

results_path.parent.mkdir(parents=True, exist_ok=True)
with results_path.open("w", encoding="utf-8") as f:
    f.write("OptimusVLA LIBERO Evaluation Results\n")
    f.write(f"generated_at: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    f.write(f"log_dir: {log_dir.as_posix()}\n")
    f.write(f"server_log: {server_log}\n\n")
    f.write("suite\texit_code\tepisodes\tsuccesses\tsuccess_rate\tjsonl\tstdout\n")
    for suite, exit_code, episodes, successes, rate, jsonl_path, stdout_path in summary_rows:
        if rate is None:
            f.write(
                f"{suite}\t{exit_code}\tNA\tNA\tNA\t"
                f"{jsonl_path.as_posix()}\t{stdout_path.as_posix()}\n"
            )
        else:
            f.write(
                f"{suite}\t{exit_code}\t{episodes}\t{successes}\t{rate:.4f}\t"
                f"{jsonl_path.as_posix()}\t{stdout_path.as_posix()}\n"
            )
    if total_episodes > 0:
        overall_rate = total_successes / total_episodes
        f.write(
            f"\noverall_success_rate: {overall_rate:.4f} "
            f"({total_successes}/{total_episodes})\n"
        )
PY
then
  echo "Wrote evaluation results to ${RESULTS_TXT}"
else
  echo "Failed to write evaluation results to ${RESULTS_TXT}" >&2
  status=1
fi

cleanup
trap - EXIT
exit "${status}"
