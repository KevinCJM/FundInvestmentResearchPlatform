#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$PROJECT_ROOT/.run"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

PYTHON_BIN="${PYTHON_BIN:-/Users/chenjunming/Desktop/myenv_312/bin/python3.12}"
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
BACKEND_RELOAD="${BACKEND_RELOAD:-0}"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-20}"
# 首次启动需要编译全部 NJIT 内核并预热 worker，可能耗时数分钟。
BACKEND_HEALTH_TIMEOUT_SECONDS="${BACKEND_HEALTH_TIMEOUT_SECONDS:-600}"
BACKEND_HEALTH_PATH="${BACKEND_HEALTH_PATH:-/api/health}"
BACKEND_READY_HOST="${BACKEND_READY_HOST:-127.0.0.1}"
FRONTEND_READY_HOST="${FRONTEND_READY_HOST:-127.0.0.1}"
BACKEND_URL="http://${BACKEND_READY_HOST}:${BACKEND_PORT}"
FRONTEND_URL="http://${FRONTEND_READY_HOST}:${FRONTEND_PORT}"

mkdir -p "$RUN_DIR"

BACKEND_PID_FILE="$RUN_DIR/backend.pid"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"
BACKEND_LOG_FILE="$RUN_DIR/backend.log"
FRONTEND_LOG_FILE="$RUN_DIR/frontend.log"

command_exists() {
  command -v "$1" >/dev/null 2>&1 || [[ -x "$1" ]]
}

is_port_listening() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

read_pid_from_file() {
  local pid_file="$1"
  if [[ ! -f "$pid_file" ]]; then
    return 1
  fi
  local pid
  pid="$(tr -dc '0-9' < "$pid_file" | tr -d '\n')"
  if [[ -n "${pid:-}" ]]; then
    echo "$pid"
    return 0
  fi
  return 1
}

is_pid_running() {
  local pid="$1"
  kill -0 "$pid" >/dev/null 2>&1
}

wait_for_port() {
  local port="$1"
  local start_pid="${2:-}"
  local timeout="${3:-$STARTUP_WAIT_SECONDS}"
  local waited=0
  while true; do
    if is_port_listening "$port"; then
      return 0
    fi
    if [[ -n "$start_pid" ]] && ! is_pid_running "$start_pid"; then
      return 1
    fi
    if [[ "$waited" -ge "$timeout" ]]; then
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
}

backend_is_ready() {
  local base_url="${1:-$BACKEND_URL}"
  curl --noproxy '*' -fsS --max-time 3 "${base_url}${BACKEND_HEALTH_PATH}" 2>/dev/null |
    "$PYTHON_BIN" -c '
import json, sys
try:
    data = json.load(sys.stdin)
    warmup = data.get("numba_warmup", {})
    ready = (data.get("ok") is True and warmup.get("complete") is True
             and warmup.get("workers", {}).get("fully_warmed") is True)
except (ValueError, AttributeError, TypeError):
    ready = False
sys.exit(0 if ready else 1)
' 2>/dev/null
}

frontend_is_ready() {
  curl --noproxy '*' -fsS --max-time 3 "$FRONTEND_URL/" 2>/dev/null |
    "$PYTHON_BIN" -c 'import sys; sys.exit(0 if "id=\"root\"" in sys.stdin.read() else 1)' &&
    backend_is_ready "$FRONTEND_URL"
}

wait_for_backend_health() {

  local start_pid="$1"
  local started_at=$SECONDS
  local next_progress=0
  local timeout="$BACKEND_HEALTH_TIMEOUT_SECONDS"
  local last_msg="(not ready)"

  while true; do
    if ! is_pid_running "$start_pid"; then
      echo "[失败] 后端进程已退出，未完成启动。"
      return 1
    fi

    if is_port_listening "$BACKEND_PORT"; then
      if backend_is_ready; then
        return 0
      fi
      last_msg="端口已监听，等待健康检查确认 NJIT 和 worker 预热完成"
    fi

    local waited=$((SECONDS - started_at))
    if [[ "$waited" -ge "$timeout" ]]; then
      echo "[失败] 后端启动等待超过 ${timeout} 秒（${last_msg}）。"
      echo "冷启动可设置 BACKEND_HEALTH_TIMEOUT_SECONDS 增加等待时间。"
      return 1
    fi
    if [[ "$waited" -ge "$next_progress" ]]; then
      echo "[启动中] 后端正在加载 / Numba 预热，已等待 ${waited} 秒，最多等待 ${timeout} 秒……"
      next_progress=$((waited + 15))
    fi
    sleep 1
  done
}

show_log_tail() {
  local file="$1"
  if [[ -f "$file" ]]; then
    echo "--- $(basename "$file") tail ---"
    tail -n 60 "$file"
  fi
}

launch_detached() {
  # nohup 仅忽略 SIGHUP，仍会继承启动终端的进程组；独立 session 才能脱离其生命周期。
  "$PYTHON_BIN" - "$@" <<'PY'
import signal
import subprocess
import sys

signal.signal(signal.SIGHUP, signal.SIG_IGN)
with open(sys.argv[1], "ab", buffering=0) as log:
    process = subprocess.Popen(
        sys.argv[2:],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
print(process.pid)
PY
}

start_uvicorn() {
  local use_reload="$1"
  local args=(
    "$PYTHON_BIN" -m uvicorn app:app
    --host "$BACKEND_HOST"
    --port "$BACKEND_PORT"
  )
  if [[ "$use_reload" == "1" ]]; then
    args+=(--reload)
  fi
  local pid
  pid="$(launch_detached "$BACKEND_LOG_FILE" "${args[@]}")" || return 1
  echo "$pid" >"$BACKEND_PID_FILE"
  echo "$pid"
}

stop_process_by_pid_file() {
  local pid_file="$1"
  local pid
  pid="$(read_pid_from_file "$pid_file" || true)"
  if [[ -n "${pid:-}" ]] && is_pid_running "$pid"; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
}

stop_ports_if_busy() {
  local port="$1"
  if is_port_listening "$port"; then
    local stop_pids
    # The listener can exit after the first check; under pipefail this is normal,
    # not a reason to abort restart before printing a result or starting again.
    stop_pids="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | tail -n +2 | awk '{print $2}' || true)"
    for pid in $stop_pids; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 1
  fi
}

start_backend() {
  if is_port_listening "$BACKEND_PORT"; then
    if ! backend_is_ready; then
      echo "[失败] :${BACKEND_PORT} 已被占用，但后端健康检查未通过。"
      return 1
    fi
    echo "[就绪] 后端已运行，Numba 与 worker 预热完成。"
    return 0
  fi

  if ! command_exists "$PYTHON_BIN"; then
    echo "[失败] 找不到 Python：$PYTHON_BIN"
    return 1
  fi

  cd "$BACKEND_DIR"
  local attempts=()
  if [[ "$BACKEND_RELOAD" == "1" ]]; then
    attempts=("reload")
  else
    attempts=("plain")
  fi

  local attempt
  for attempt in "${attempts[@]}"; do
    echo > "$BACKEND_LOG_FILE"
    local start_pid
    if [[ "$attempt" == "reload" ]]; then
      echo "[启动中] 正在启动后端（热重载模式）……"
      start_pid="$(start_uvicorn "1")"
    else
      echo "[启动中] 正在启动后端（普通模式）……"
      start_pid="$(start_uvicorn "0")"
    fi

    if wait_for_backend_health "$start_pid"; then
      echo "[就绪] 后端启动成功，Numba 与 worker 预热完成（PID=$(cat "$BACKEND_PID_FILE")）。"
      return 0
    fi

    echo "[失败] 后端启动未完成（${attempt}），最近日志如下："
    show_log_tail "$BACKEND_LOG_FILE"
    stop_process_by_pid_file "$BACKEND_PID_FILE"
    rm -f "$BACKEND_PID_FILE"
    stop_ports_if_busy "$BACKEND_PORT"
    sleep 1
  done

  echo "[失败] 后端未能在 :$BACKEND_PORT 就绪。"
  show_log_tail "$BACKEND_LOG_FILE"
  rm -f "$BACKEND_PID_FILE"
  return 1
}

start_frontend() {
  if is_port_listening "$FRONTEND_PORT"; then
    if ! frontend_is_ready; then
      echo "[失败] :${FRONTEND_PORT} 已被占用，但前端页面或 API 转发检查未通过。"
      return 1
    fi
    echo "[就绪] 前端已运行，页面及 API 转发正常。"
    return 0
  fi

  if ! command_exists npm; then
    echo "[失败] 找不到 npm，请先安装 Node.js。"
    return 1
  fi

  echo "[启动中] 正在启动前端……"
  echo > "$FRONTEND_LOG_FILE"
  cd "$FRONTEND_DIR"
  local pid
  pid="$(launch_detached "$FRONTEND_LOG_FILE" npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT" --strictPort)" || return 1
  echo "$pid" > "$FRONTEND_PID_FILE"
  if ! wait_for_port "$FRONTEND_PORT" "$(read_pid_from_file "$FRONTEND_PID_FILE")"; then
    echo "[失败] 前端未能在 :$FRONTEND_PORT 就绪。"
    show_log_tail "$FRONTEND_LOG_FILE"
    rm -f "$FRONTEND_PID_FILE"
    return 1
  fi
  if ! frontend_is_ready; then
    echo "[失败] 前端已监听，但页面或 API 转发检查未通过。"
    show_log_tail "$FRONTEND_LOG_FILE"
    return 1
  fi
  echo "[就绪] 前端启动成功，页面及 API 转发正常（PID=$(cat "$FRONTEND_PID_FILE")）。"
}

status_services() {
  local ready=0
  if backend_is_ready; then
    echo "[就绪] 后端正常，Numba 与 worker 预热完成。"
  else
    echo "[未就绪] 后端未运行或仍在预热。"
    ready=1
  fi
  if frontend_is_ready; then
    echo "[就绪] 前端正常，页面及 API 转发可访问。"
  else
    echo "[未就绪] 前端页面或 API 转发不可用。"
    ready=1
  fi
  echo "前端访问地址：$FRONTEND_URL/"
  echo "后端 API 文档：$BACKEND_URL/docs"
  echo "后端日志：$BACKEND_LOG_FILE"
  echo "前端日志：$FRONTEND_LOG_FILE"
  return "$ready"
}

start_services() {
  local started_at=$SECONDS
  if ! command_exists curl || ! command_exists "$PYTHON_BIN"; then
    echo "[失败] 请确认 curl 和 Python 可用：$PYTHON_BIN"
    return 1
  fi
  if ! "$PYTHON_BIN" "$PROJECT_ROOT/scripts/manage_data_storage.py" startup; then
    echo "[启动阻止] 数据存储检查或迁移未完成。请处理上方提示，再重试；不会回退到本机。"
    return 1
  fi
  if ! start_backend || ! start_frontend || ! status_services; then
    echo "[启动失败] 前后端尚未全部就绪，请查看上方错误和 .run/ 中的日志。"
    return 1
  fi
  echo
  echo "========== 启动成功 / SUCCESS =========="
  echo "前后端均已就绪，Numba 预热已完成（耗时 $((SECONDS - started_at)) 秒）。"
  echo "请在浏览器打开：$FRONTEND_URL/"
  echo "服务已在后台运行，可以关闭当前终端。"
  echo "新版 ETL 下载由独立执行器运行，重启 API 后自动重连；下载进度请看运行记录。"
  echo "管理命令：./start_services.sh {status|stop|restart}"
}

stop_services() {
  stop_process_by_pid_file "$BACKEND_PID_FILE"
  stop_process_by_pid_file "$FRONTEND_PID_FILE"
  stop_ports_if_busy "$BACKEND_PORT"
  stop_ports_if_busy "$FRONTEND_PORT"
  for pid_file in "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE"; do
    rm -f "$pid_file"
  done
  echo "[停止] 已向前后端发送停止信号。"
  echo "[任务保留] 不终止独立 ETL 执行器；如需停止下载，请在网页运行记录中取消任务。"
}

restart_services() {
  stop_services
  sleep 1
  start_services
}

case "${1:-start}" in
  start)
    start_services
    ;;
  stop)
    stop_services
    ;;
  status)
    status_services
    ;;
  restart)
    restart_services
    ;;
  storage-status)
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/manage_data_storage.py" status
    ;;
  storage-cleanup)
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/manage_data_storage.py" cleanup-backup --confirm "${2:-}"
    ;;
  storage-cancel)
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/manage_data_storage.py" cancel-plan --confirm "${2:-}"
    ;;
  *)
    echo "用法：$0 {start|stop|status|restart|storage-status|storage-cleanup <迁移ID>|storage-cancel <迁移ID>}"
    exit 1
    ;;
esac
