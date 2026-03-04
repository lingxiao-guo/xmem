#!/usr/bin/env bash
set -euo pipefail

# ===== constants =====
ACCOUNT_ID="ac3355227b8f62bb52e732bc42368dcf"
BUCKET="behavior-cf"
PREFIX="${XFER_REPO_PREFIX:-LeCAR-xfer}"
ENDPOINT="https://${ACCOUNT_ID}.r2.cloudflarestorage.com"
RESTIC_REPO_URI="s3:${ENDPOINT}/${BUCKET}/${PREFIX}"

# R2 credentials (按你的要求写死)
AWS_ACCESS_KEY_ID_VAL="e64306563928427b781a93acd4e1193b"
AWS_SECRET_ACCESS_KEY_VAL="fb4a8cad67231fb374d752d0d05586cccf72af31d22a24fe77ed0208816d2353"

# Restic repo password
RESTIC_PASSWORD_VAL="LeCAR_restic_pw_3uQ7xgK2wL6s1Zp0Nf"

# ===== defaults =====
AUTO_INSTALL=1
TAG_BASE="XFER"                  # 不碰 LeCAR
RETRY_LOCK_DEFAULT="60m"
S3_CONN_DEFAULT=""               # e.g. 128
VERBOSE_DEFAULT="1"              # 0/1/2/3
DO_REMOTE_FORGET=1               # 默认远端成功后 forget 本次快照（不 prune）
SSH_RETRIES_DEFAULT="3"
SSH_RETRY_SLEEP_DEFAULT="8"
SSH_KEEPALIVE_INTERVAL_DEFAULT="15"
SSH_KEEPALIVE_COUNTMAX_DEFAULT="8"
SSH_CONNECT_TIMEOUT_DEFAULT="20"
FORGET_RETRY_LOCK_DEFAULT="10s"

# Pinned restic for both local & remote (avoid distro restic too old)
PIN_RESTIC_VER="${PIN_RESTIC_VER:-0.18.1}"

usage() {
  cat <<'USG'
Usage:
  ./xfer.sh --ssh <ssh_config_host> --src <path> --dest <path> [options]

Rsync-like dest semantics:
  - If --dest ends with "/" => treat as directory; result path is dest_dir/basename(src)
  - If --dest does NOT end with "/":
      * src is file => treat as file path (write exactly to dest)
      * src is dir  => treat as directory path (write/swap exactly to dest)

Required:
  --ssh   SSH config Host name, e.g. p1 (from ~/.ssh/config)
  --src   Push: local source path; Pull: remote source path
  --dest  Push: remote destination path; Pull: local destination path

Options:
  --pull                   pull from remote to local (default: push local to remote)
  --retry-lock <dur>       restic --retry-lock (default: 60m)  (auto-skip if unsupported)
  --s3-connections <n>     restic -o s3.connections=n (default: auto)
  --verbose <0|1|2|3>      restic verbosity (default: 1)
  --no-forget              do NOT forget the transfer snapshot after install
  --no-auto-install        disable auto install dependencies
  --help

Examples:
  # file into dir (rsync style):
  ./xfer.sh --ssh p1 --src ./test.txt --dest /mnt/.../B1K/

  # file to exact path:
  ./xfer.sh --ssh p1 --src ./test.txt --dest /mnt/.../B1K/test.txt

  # dir to exact dir path:
  ./xfer.sh --ssh p1 --src ./BigDir --dest /mnt/.../BigDir

  # pull remote file into local dir:
  ./xfer.sh --ssh p1 --pull --src /mnt/.../B1K/test.txt --dest ./downloads/

Notes:
  - Directory bundles require zstd.
  - Uses isolated XFER repo prefix by default: LeCAR-xfer (env override: XFER_REPO_PREFIX).
USG
}

die(){ echo "❌ $*" >&2; exit 2; }

# ---------------- args ----------------
SSH_HOST=""
SRC=""
DEST=""
RETRY_LOCK="$RETRY_LOCK_DEFAULT"
S3_CONN="$S3_CONN_DEFAULT"
VERBOSE="$VERBOSE_DEFAULT"
PULL_MODE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh) SSH_HOST="${2:-}"; shift 2;;
    --src) SRC="${2:-}"; shift 2;;
    --dest) DEST="${2:-}"; shift 2;;
    --retry-lock) RETRY_LOCK="${2:-}"; shift 2;;
    --s3-connections) S3_CONN="${2:-}"; shift 2;;
    --verbose) VERBOSE="${2:-}"; shift 2;;
    --pull) PULL_MODE=1; shift 1;;
    --no-forget) DO_REMOTE_FORGET=0; shift 1;;
    --no-auto-install) AUTO_INSTALL=0; shift 1;;
    --help|-h) usage; exit 0;;
    *) die "Unknown arg: $1";;
  esac
done

[[ -n "$SSH_HOST" ]] || die "Missing --ssh"
[[ -n "$SRC" ]] || die "Missing --src"
[[ -n "$DEST" ]] || die "Missing --dest"

SSH_RETRIES="${SSH_RETRIES:-$SSH_RETRIES_DEFAULT}"
SSH_RETRY_SLEEP="${SSH_RETRY_SLEEP:-$SSH_RETRY_SLEEP_DEFAULT}"
SSH_KEEPALIVE_INTERVAL="${SSH_KEEPALIVE_INTERVAL:-$SSH_KEEPALIVE_INTERVAL_DEFAULT}"
SSH_KEEPALIVE_COUNTMAX="${SSH_KEEPALIVE_COUNTMAX:-$SSH_KEEPALIVE_COUNTMAX_DEFAULT}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-$SSH_CONNECT_TIMEOUT_DEFAULT}"
FORGET_RETRY_LOCK="${FORGET_RETRY_LOCK:-$FORGET_RETRY_LOCK_DEFAULT}"

# ---------------- auto-install (local) ----------------
os_name() { uname -s; }

install_local_pkg() {
  local pkg="$1"
  case "$(os_name)" in
    Darwin)
      command -v brew >/dev/null 2>&1 || die "brew not found. Install Homebrew first."
      brew install "$pkg"
      ;;
    Linux)
      if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -y
        sudo apt-get install -y "$pkg"
      elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y "$pkg"
      else
        die "No supported package manager. Install '$pkg' manually."
      fi
      ;;
    *) die "Unsupported OS. Install '$pkg' manually." ;;
  esac
}

need_local() {
  local cmd="$1"
  local pkg="${2:-$1}"
  if command -v "$cmd" >/dev/null 2>&1; then return 0; fi
  [[ "$AUTO_INSTALL" = "1" ]] || die "Missing dependency: $cmd (install $pkg)"
  echo "==> [local] Auto-install missing: $cmd (pkg: $pkg)"
  install_local_pkg "$pkg"
  command -v "$cmd" >/dev/null 2>&1 || die "Auto-install failed for $cmd"
}

# ---------------- deps (local) ----------------
need_local ssh openssh
need_local tar tar
need_local zstd zstd
need_local sha256sum coreutils      # mac needs coreutils
need_local readlink coreutils || true
need_local hostname coreutils || true
need_local mktemp coreutils || true
need_local uname coreutils || true
need_local curl curl
need_local bzip2 bzip2

abs_path() {
  local p="$1"
  case "$p" in "~"|"~/"*) p="${p/#\~/$HOME}";; esac
  if [ "${p#/}" = "$p" ]; then p="$PWD/$p"; fi
  readlink -f "$p" 2>/dev/null || echo "$p"
}

join_dir_and_base() {
  local dir="$1"
  local base="$2"
  if [[ "$dir" == "/" ]]; then
    printf '/%s\n' "$base"
  else
    printf '%s/%s\n' "${dir%/}" "$base"
  fi
}

manifest_get_value() {
  local manifest_file="$1"
  local key="$2"
  awk -v key="$key" 'index($0, key "=") == 1 {print substr($0, length(key) + 2); exit}' "$manifest_file"
}

restic_vflag() {
  case "${VERBOSE}" in
    0) echo "";;
    1) echo "-v";;
    2) echo "-vv";;
    *) echo "-vvv";;
  esac
}
restic_s3opt_arr() {
  if [[ -n "${S3_CONN}" ]]; then
    echo "-o" "s3.connections=${S3_CONN}"
  else
    echo ""
  fi
}

# ---------------- restic env (local) ----------------
export AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID_VAL"
export AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY_VAL"
export AWS_DEFAULT_REGION="auto"
export RESTIC_REPOSITORY="$RESTIC_REPO_URI"
export RESTIC_PASSWORD="$RESTIC_PASSWORD_VAL"
export RESTIC_CACHE_DIR="${RESTIC_CACHE_DIR:-$HOME/.cache/restic}"
mkdir -p "$RESTIC_CACHE_DIR" >/dev/null 2>&1 || true

# ---------------- pinned restic-xfer (local) ----------------
LOCAL_RESTIC_BIN="${LOCAL_RESTIC_BIN:-$HOME/.local/bin/restic-xfer}"
if [[ ! -x "$LOCAL_RESTIC_BIN" ]]; then
  echo "==> [local] Installing restic-xfer v$PIN_RESTIC_VER -> $LOCAL_RESTIC_BIN"
  mkdir -p "$(dirname "$LOCAL_RESTIC_BIN")"
  tmpd="$(mktemp -d)"
  trap 'rm -rf "$tmpd"' EXIT
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) A="amd64" ;;
    aarch64|arm64) A="arm64" ;;
    *) die "Unsupported arch: $arch" ;;
  esac
  url="https://github.com/restic/restic/releases/download/v${PIN_RESTIC_VER}/restic_${PIN_RESTIC_VER}_linux_${A}.bz2"
  # if macOS, use darwin build
  if [[ "$(os_name)" == "Darwin" ]]; then
    url="https://github.com/restic/restic/releases/download/v${PIN_RESTIC_VER}/restic_${PIN_RESTIC_VER}_darwin_${A}.bz2"
  fi
  curl -fsSL -o "$tmpd/restic.bz2" "$url"
  bunzip2 -f "$tmpd/restic.bz2"
  chmod +x "$tmpd/restic"
  mv "$tmpd/restic" "$LOCAL_RESTIC_BIN"
fi
"$LOCAL_RESTIC_BIN" version >/dev/null 2>&1 || die "restic-xfer not runnable: $LOCAL_RESTIC_BIN"

RESTIC_BIN="$LOCAL_RESTIC_BIN"

restic_supports_retry_lock() {
  "$RESTIC_BIN" help backup 2>/dev/null | grep -q -- '--retry-lock'
}

extract_snapshot_from_output() {
  local out_file="$1"
  local snap_id=""

  snap_id="$(
    tr -d '\r' < "$out_file" \
      | grep -Eo 'snapshot[[:space:]]+[0-9a-f]{8,64}[[:space:]]+saved' \
      | tail -n 1 \
      | grep -Eo '[0-9a-f]{8,64}' || true
  )"
  [[ -n "$snap_id" ]] && { printf '%s\n' "$snap_id"; return 0; }

  snap_id="$(
    tr -d '\r' < "$out_file" \
      | grep -Eo '"snapshot_id"[[:space:]]*:[[:space:]]*"[0-9a-f]{8,64}"' \
      | tail -n 1 \
      | grep -Eo '[0-9a-f]{8,64}' || true
  )"
  [[ -n "$snap_id" ]] && { printf '%s\n' "$snap_id"; return 0; }

  return 1
}

lookup_snapshot_by_tag() {
  local snap_id=""

  snap_id="$(
    "$RESTIC_BIN" snapshots --compact --tag "$JOBID" --host "$HOSTTAG" \
      "${S3OPT[@]:+"${S3OPT[@]}"}" 2>/dev/null \
      | awk 'NF && $1 ~ /^[0-9a-f]+$/ && length($1) >= 8 {id=$1} END {print id}' || true
  )"
  [[ -n "$snap_id" ]] && { printf '%s\n' "$snap_id"; return 0; }

  snap_id="$(
    "$RESTIC_BIN" snapshots --compact --tag "$JOBID" \
      "${S3OPT[@]:+"${S3OPT[@]}"}" 2>/dev/null \
      | awk 'NF && $1 ~ /^[0-9a-f]+$/ && length($1) >= 8 {id=$1} END {print id}' || true
  )"
  [[ -n "$snap_id" ]] && { printf '%s\n' "$snap_id"; return 0; }

  return 1
}

# ---------------- pull mode (remote -> local) ----------------
if [[ "$PULL_MODE" == "1" ]]; then
  JOBID="$(date +%Y%m%d_%H%M%S)_$RANDOM"
  HOSTTAG="$(hostname | tr -cd '[:alnum:]_.-')"

  WORK_PULL="$(mktemp -d "${TMPDIR:-/tmp}/xfer_pull_local_${USER}_${JOBID}_XXXX")"
  cleanup_pull(){ rm -rf "$WORK_PULL"; }
  trap cleanup_pull EXIT

  VFLAG="$(restic_vflag)"
  S3OPT=()
  read -r -a S3OPT <<<"$(restic_s3opt_arr)"

  echo "==> [local] Pull mode: remote src -> local dest"
  echo "     repo=$RESTIC_REPOSITORY"
  echo "     tags=$TAG_BASE,$JOBID,$HOSTTAG"
  echo "     src_remote=$SRC"
  echo "     dest_input=$DEST"

  SSH_COMMON_OPTS=(
    -o "ServerAliveInterval=${SSH_KEEPALIVE_INTERVAL}"
    -o "ServerAliveCountMax=${SSH_KEEPALIVE_COUNTMAX}"
    -o "TCPKeepAlive=yes"
    -o "ConnectTimeout=${SSH_CONNECT_TIMEOUT}"
    -o "ConnectionAttempts=2"
    -o "BatchMode=no"
  )

  S3_CONN_ARG="${S3_CONN:-__XFER_EMPTY_S3_CONN__}"
  OUT_REMOTE="$WORK_PULL/remote_backup.out"

  REMOTE_OK=0
  trap 'die "Interrupted by user"' INT TERM
  for ((attempt=1; attempt<=SSH_RETRIES; attempt++)); do
    echo "==> [local] Remote backup attempt ${attempt}/${SSH_RETRIES} ..."
    if ssh "${SSH_COMMON_OPTS[@]}" "$SSH_HOST" "bash -s" -- \
      "$SRC" "$DEST" "$JOBID" "$RETRY_LOCK" "$S3_CONN_ARG" "$VERBOSE" "$AUTO_INSTALL" "$PIN_RESTIC_VER" "$RESTIC_REPO_URI" "$TAG_BASE" <<'REMOTE_PULL_SCRIPT' | tee "$OUT_REMOTE"
set -euo pipefail

SRC_REMOTE_INPUT="${1:?src_remote}"
DEST_INPUT_LOCAL="${2:?dest_input_local}"
JOBID="${3:?jobid}"
RETRY_LOCK="${4:?retry_lock}"
S3_CONN="${5:-}"
VERBOSE="${6:-1}"
AUTO_INSTALL="${7:-1}"
WANT_VER="${8:-0.18.1}"
RESTIC_REPO_URI_REMOTE="${9:?restic_repo_uri}"
TAG_BASE_REMOTE="${10:-XFER}"
if [[ "$S3_CONN" == "__XFER_EMPTY_S3_CONN__" ]]; then
  S3_CONN=""
fi

export AWS_ACCESS_KEY_ID="e64306563928427b781a93acd4e1193b"
export AWS_SECRET_ACCESS_KEY="fb4a8cad67231fb374d752d0d05586cccf72af31d22a24fe77ed0208816d2353"
export AWS_DEFAULT_REGION="auto"
export RESTIC_REPOSITORY="$RESTIC_REPO_URI_REMOTE"
export RESTIC_PASSWORD="LeCAR_restic_pw_3uQ7xgK2wL6s1Zp0Nf"
export RESTIC_CACHE_DIR="${RESTIC_CACHE_DIR:-$HOME/.cache/restic}"
mkdir -p "$RESTIC_CACHE_DIR" >/dev/null 2>&1 || true

RESTIC_BIN="$HOME/.local/bin/restic-xfer"
die(){ echo "❌ $*" >&2; exit 2; }

install_remote_pkg() {
  local pkg="$1"
  local runner=()
  if [[ "$(id -u)" -ne 0 ]]; then
    command -v sudo >/dev/null 2>&1 || die "sudo not found; cannot auto-install $pkg"
    runner=(sudo)
  fi
  if command -v apt-get >/dev/null 2>&1; then
    "${runner[@]}" apt-get update -y
    "${runner[@]}" apt-get install -y "$pkg"
  elif command -v yum >/dev/null 2>&1; then
    "${runner[@]}" yum install -y "$pkg"
  else
    die "No supported package manager on remote. Install '$pkg' manually."
  fi
}

need_remote() {
  local cmd="$1"
  local pkg="${2:-$1}"
  if command -v "$cmd" >/dev/null 2>&1; then return 0; fi
  [[ "$AUTO_INSTALL" = "1" ]] || die "Missing on remote: $cmd (install $pkg)"
  echo "==> [remote] Auto-install missing: $cmd (pkg: $pkg)"
  install_remote_pkg "$pkg"
  command -v "$cmd" >/dev/null 2>&1 || die "Auto-install failed for $cmd"
}

vflag() {
  case "$VERBOSE" in
    0) echo "";;
    1) echo "-v";;
    2) echo "-vv";;
    *) echo "-vvv";;
  esac
}

s3opt() {
  if [[ -n "${S3_CONN}" ]]; then
    echo "-o" "s3.connections=${S3_CONN}"
  else
    echo ""
  fi
}

abs_path_remote() {
  local p="$1"
  case "$p" in "~"|"~/"*) p="${p/#\~/$HOME}";; esac
  if [ "${p#/}" = "$p" ]; then p="$PWD/$p"; fi
  readlink -f "$p" 2>/dev/null || echo "$p"
}

join_dir_and_base_remote() {
  local dir="$1"
  local base="$2"
  if [[ "$dir" == "/" ]]; then
    printf '/%s\n' "$base"
  else
    printf '%s/%s\n' "${dir%/}" "$base"
  fi
}

extract_snapshot_from_output_remote() {
  local out_file="$1"
  local snap_id=""

  snap_id="$(
    tr -d '\r' < "$out_file" \
      | grep -Eo 'snapshot[[:space:]]+[0-9a-f]{8,64}[[:space:]]+saved' \
      | tail -n 1 \
      | grep -Eo '[0-9a-f]{8,64}' || true
  )"
  [[ -n "$snap_id" ]] && { printf '%s\n' "$snap_id"; return 0; }

  snap_id="$(
    tr -d '\r' < "$out_file" \
      | grep -Eo '"snapshot_id"[[:space:]]*:[[:space:]]*"[0-9a-f]{8,64}"' \
      | tail -n 1 \
      | grep -Eo '[0-9a-f]{8,64}' || true
  )"
  [[ -n "$snap_id" ]] && { printf '%s\n' "$snap_id"; return 0; }

  return 1
}

restic_supports_retry_lock_remote() {
  "$RESTIC_BIN" help backup 2>/dev/null | grep -q -- '--retry-lock'
}

lookup_snapshot_by_tag_remote() {
  local snap_id=""
  snap_id="$(
    "$RESTIC_BIN" snapshots --compact --tag "$JOBID" \
      ${S3OPT[@]:+"${S3OPT[@]}"} 2>/dev/null \
      | awk 'NF && $1 ~ /^[0-9a-f]+$/ && length($1) >= 8 {id=$1} END {print id}' || true
  )"
  [[ -n "$snap_id" ]] && { printf '%s\n' "$snap_id"; return 0; }
  return 1
}

need_remote curl curl
need_remote bzip2 bzip2
need_remote tar tar
need_remote zstd zstd
need_remote sha256sum coreutils || true
need_remote readlink coreutils || true
need_remote hostname coreutils || true
need_remote mktemp coreutils || true
need_remote uname coreutils || true

if [[ ! -x "$RESTIC_BIN" ]]; then
  echo "==> [remote] Installing restic-xfer v$WANT_VER -> $RESTIC_BIN"
  mkdir -p "$(dirname "$RESTIC_BIN")"
  tmpd="$(mktemp -d)"
  trap 'rm -rf "$tmpd"' EXIT
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) A="amd64" ;;
    aarch64|arm64) A="arm64" ;;
    *) die "Unsupported arch: $arch" ;;
  esac
  url="https://github.com/restic/restic/releases/download/v${WANT_VER}/restic_${WANT_VER}_linux_${A}.bz2"
  curl -fsSL -o "$tmpd/restic.bz2" "$url"
  bunzip2 -f "$tmpd/restic.bz2"
  chmod +x "$tmpd/restic"
  mv "$tmpd/restic" "$RESTIC_BIN"
fi
"$RESTIC_BIN" version >/dev/null 2>&1 || die "restic-xfer not runnable: $RESTIC_BIN"

WORK_REMOTE="$(mktemp -d "${TMPDIR:-/tmp}/xfer_pull_remote_${USER}_${JOBID}_XXXX")"
cleanup_remote(){ rm -rf "$WORK_REMOTE"; }
trap cleanup_remote EXIT
BUNDLE="$WORK_REMOTE/bundle"
mkdir -p "$BUNDLE"

SRC_REMOTE_ABS="$(abs_path_remote "$SRC_REMOTE_INPUT")"
MODE=""
PAYLOAD=""
COMPRESSION="none"

if [[ -d "$SRC_REMOTE_ABS" ]]; then
  MODE="dir"
  COMPRESSION="zstd"
  PAYLOAD="payload.tar.zst"
  SRCPARENT="$(cd "$(dirname "$SRC_REMOTE_ABS")" && pwd)"
  SRCNAME="$(basename "$SRC_REMOTE_ABS")"
  echo "==> [remote] Packing directory (zstd): $SRC_REMOTE_ABS"
  ( cd "$SRCPARENT" && tar -cf - "$SRCNAME" | zstd -T0 -19 -o "$BUNDLE/$PAYLOAD" )
elif [[ -f "$SRC_REMOTE_ABS" ]]; then
  MODE="file"
  PAYLOAD="$(basename "$SRC_REMOTE_ABS")"
  echo "==> [remote] Preparing file: $SRC_REMOTE_ABS"
  cp -f "$SRC_REMOTE_ABS" "$BUNDLE/$PAYLOAD"
else
  die "Remote SRC not found: $SRC_REMOTE_INPUT"
fi

DEST_LOCAL_CANON="$DEST_INPUT_LOCAL"
if [[ "$DEST_INPUT_LOCAL" == */ ]]; then
  dest_dir="${DEST_INPUT_LOCAL%/}"
  [[ -n "$dest_dir" ]] || dest_dir="/"
  DEST_LOCAL_CANON="$(join_dir_and_base_remote "$dest_dir" "$(basename "$SRC_REMOTE_ABS")")"
fi

echo "==> [remote] Writing checksum + manifest"
( cd "$BUNDLE" && sha256sum "$PAYLOAD" > "$PAYLOAD.sha256" )
cat > "$BUNDLE/manifest.txt" <<EOF
payload=$PAYLOAD
sha256=$PAYLOAD.sha256
mode=$MODE
compression=$COMPRESSION
dest=$DEST_LOCAL_CANON
dest_input=$DEST_INPUT_LOCAL
src_base=$(basename "$SRC_REMOTE_ABS")
jobid=$JOBID
EOF

if ! "$RESTIC_BIN" snapshots >/dev/null 2>&1; then
  echo "==> [remote] Repo not initialized; restic init"
  "$RESTIC_BIN" init
fi
"$RESTIC_BIN" unlock >/dev/null 2>&1 || true

VFLAG="$(vflag)"
S3OPT=( $(s3opt) ) 2>/dev/null || true
EXTRA_LOCK=()
if restic_supports_retry_lock_remote; then
  EXTRA_LOCK=(--retry-lock "$RETRY_LOCK")
fi

echo "==> [remote] Uploading pull bundle to R2 via restic-xfer"
echo "     repo=$RESTIC_REPOSITORY"
echo "     tags=$TAG_BASE_REMOTE,$JOBID,$(hostname | tr -cd '[:alnum:]_.-')"
echo "     mode=$MODE compression=$COMPRESSION jobid=$JOBID s3.connections=${S3_CONN:-auto}"
echo "     src_remote=$SRC_REMOTE_INPUT"
echo "     dest_local_input=$DEST_INPUT_LOCAL"
echo "     dest_local_canon=$DEST_LOCAL_CANON"

OUT="$WORK_REMOTE/backup.out"
"$RESTIC_BIN" $VFLAG backup "$BUNDLE" \
  --tag "$TAG_BASE_REMOTE" --tag "$JOBID" --tag "$(hostname | tr -cd '[:alnum:]_.-')" \
  "${EXTRA_LOCK[@]}" \
  ${S3OPT[@]:+"${S3OPT[@]}"} | tee "$OUT"

SNAP_ID="$(extract_snapshot_from_output_remote "$OUT" || true)"
if [[ -z "${SNAP_ID:-}" ]]; then
  SNAP_ID="$(lookup_snapshot_by_tag_remote || true)"
fi
[[ -n "${SNAP_ID:-}" ]] || die "Cannot parse snapshot id from remote backup output"
echo "==> [remote] Snapshot id: $SNAP_ID"
echo "XFER_PULL_SNAPSHOT_ID=$SNAP_ID"
REMOTE_PULL_SCRIPT
    then
      REMOTE_OK=1
      break
    fi
    if [[ "$attempt" -lt "$SSH_RETRIES" ]]; then
      echo "==> [local] Remote backup failed/disconnected; retry in ${SSH_RETRY_SLEEP}s..."
      sleep "$SSH_RETRY_SLEEP"
    fi
  done
  trap - INT TERM
  [[ "$REMOTE_OK" = "1" ]] || die "Remote backup step failed after ${SSH_RETRIES} attempts"

  SNAP_ID="$(tr -d '\r' < "$OUT_REMOTE" | awk -F= '/^XFER_PULL_SNAPSHOT_ID=[0-9a-f]{8,64}$/ {id=$2} END {print id}' || true)"
  if [[ -z "${SNAP_ID:-}" ]]; then
    SNAP_ID="$(extract_snapshot_from_output "$OUT_REMOTE" || true)"
  fi
  [[ -n "${SNAP_ID:-}" ]] || die "Cannot parse snapshot id for pull mode"
  echo "==> [local] Snapshot id: $SNAP_ID"

  "$RESTIC_BIN" unlock >/dev/null 2>&1 || true
  TMPBASE="$WORK_PULL/restore"
  mkdir -p "$TMPBASE/restore"

  echo "==> [local] Restoring by snapshot id=$SNAP_ID ..."
  MAX_TRIES="${RESTORE_TRIES:-60}"
  SLEEP_SEC="${RESTORE_SLEEP:-2}"
  RESTORE_CONN_FALLBACKS="${RESTORE_CONN_FALLBACKS:-4,2,1}"
  CONN_FALLBACK_IDX=0
  CONN_FALLBACK_LIST=()
  if [[ -z "$S3_CONN" ]]; then
    IFS=',' read -r -a CONN_FALLBACK_LIST <<< "$RESTORE_CONN_FALLBACKS"
  fi
  ok=0
  for i in $(seq 1 "$MAX_TRIES"); do
    RESTORE_LOG="$TMPBASE/restore_${i}.log"
    CUR_CONN="auto"
    if [[ -n "$S3_CONN" ]]; then
      CUR_CONN="$S3_CONN"
    elif [[ "${#S3OPT[@]}" -ge 2 && "${S3OPT[0]}" == "-o" && "${S3OPT[1]}" == s3.connections=* ]]; then
      CUR_CONN="${S3OPT[1]#s3.connections=}"
    fi
    echo "==> [local] restore attempt ${i}/${MAX_TRIES} (s3.connections=${CUR_CONN})"
    if "$RESTIC_BIN" $VFLAG restore "$SNAP_ID" --target "$TMPBASE/restore" ${S3OPT[@]:+"${S3OPT[@]}"} 2>&1 | tee "$RESTORE_LOG"; then
      ok=1; break
    fi
    if [[ -z "$S3_CONN" ]] && grep -qi 'unexpected EOF' "$RESTORE_LOG"; then
      while [[ "$CONN_FALLBACK_IDX" -lt "${#CONN_FALLBACK_LIST[@]}" ]]; do
        NEXT_CONN="$(echo "${CONN_FALLBACK_LIST[$CONN_FALLBACK_IDX]}" | tr -d '[:space:]')"
        CONN_FALLBACK_IDX=$((CONN_FALLBACK_IDX + 1))
        [[ -n "$NEXT_CONN" ]] || continue
        if [[ "$CUR_CONN" != "$NEXT_CONN" ]]; then
          echo "==> [local] Detected unexpected EOF; switching restore to s3.connections=${NEXT_CONN} and retrying..."
          S3OPT=(-o "s3.connections=${NEXT_CONN}")
          break
        fi
      done
    fi
    "$RESTIC_BIN" unlock >/dev/null 2>&1 || true
    echo "==> [local] restore failed ($i/$MAX_TRIES), retry in ${SLEEP_SEC}s..."
    sleep "$SLEEP_SEC"
  done
  [[ "$ok" = "1" ]] || die "pull restore still failing after retries"

  manifest_path="$(find "$TMPBASE/restore" -type f -name manifest.txt -print -quit 2>/dev/null || true)"
  [[ -n "${manifest_path:-}" ]] || die "Cannot find manifest.txt in restored content"
  BUNDLE_DIR="$(dirname "$manifest_path")"

  MANIFEST_FILE="$BUNDLE_DIR/manifest.txt"
  sed -i 's/\r$//' "$MANIFEST_FILE" 2>/dev/null || true

  PAYLOAD="$(manifest_get_value "$MANIFEST_FILE" payload)"
  SHAFILE="$(manifest_get_value "$MANIFEST_FILE" sha256)"
  MODE="$(manifest_get_value "$MANIFEST_FILE" mode)"
  COMPRESSION="$(manifest_get_value "$MANIFEST_FILE" compression)"
  DEST_LOCAL="$(manifest_get_value "$MANIFEST_FILE" dest)"
  DEST_INPUT_LOCAL="$(manifest_get_value "$MANIFEST_FILE" dest_input)"
  SRC_BASE="$(manifest_get_value "$MANIFEST_FILE" src_base)"

  [[ -n "${PAYLOAD:-}" ]] || die "manifest missing payload=... (file: $MANIFEST_FILE)"
  [[ -n "${SHAFILE:-}" ]] || die "manifest missing sha256=... (file: $MANIFEST_FILE)"
  [[ -n "${MODE:-}" ]] || die "manifest missing mode=... (file: $MANIFEST_FILE)"
  [[ -n "${DEST_LOCAL:-}" ]] || die "manifest missing dest=... (file: $MANIFEST_FILE)"
  [[ -n "${DEST_INPUT_LOCAL:-}" ]] || die "manifest missing dest_input=... (file: $MANIFEST_FILE)"
  [[ -n "${SRC_BASE:-}" ]] || die "manifest missing src_base=... (file: $MANIFEST_FILE)"

  [[ -f "$BUNDLE_DIR/$PAYLOAD" ]] || die "Payload missing after restore: $BUNDLE_DIR/$PAYLOAD"

  echo "==> [local] Verifying sha256..."
  ( cd "$BUNDLE_DIR" && sha256sum -c "$SHAFILE" )

  if [[ "$MODE" == "file" ]]; then
    if [[ "$DEST_INPUT_LOCAL" == */ ]]; then
      dest_dir="${DEST_INPUT_LOCAL%/}"
      [[ -n "$dest_dir" ]] || dest_dir="/"
      mkdir -p "$dest_dir"
      FINAL_DEST="$(join_dir_and_base "$dest_dir" "$SRC_BASE")"
    else
      FINAL_DEST="$DEST_LOCAL"
      [[ -d "$FINAL_DEST" ]] && die "Refusing to write file into existing directory path (missing trailing /?): dest=$FINAL_DEST"
      mkdir -p "$(dirname "$FINAL_DEST")"
    fi

    echo "==> [local] Installing file -> $FINAL_DEST"
    TMPFILE="${FINAL_DEST}.part.$(date +%s)"
    mv "$BUNDLE_DIR/$PAYLOAD" "$TMPFILE"
    mv -f "$TMPFILE" "$FINAL_DEST"
    echo "✅ [local] File updated -> $FINAL_DEST"
  elif [[ "$MODE" == "dir" ]]; then
    FINAL_DEST="${DEST_LOCAL%/}"
    [[ -n "$FINAL_DEST" ]] || FINAL_DEST="/"
    parent="$(dirname "$FINAL_DEST")"
    mkdir -p "$parent"

    stamp="$(date +%s)"
    base_name="$(basename "$FINAL_DEST")"
    STAGE_ROOT="${parent}/.${base_name}.stage.${stamp}"
    BAK="${parent}/.${base_name}.bak.${stamp}"
    mkdir -p "$STAGE_ROOT"

    [[ "$COMPRESSION" == "zstd" ]] || die "Unsupported compression in manifest: $COMPRESSION (expect zstd)"
    echo "==> [local] Extracting archive (zstd) -> $STAGE_ROOT"
    zstd -dc "$BUNDLE_DIR/$PAYLOAD" | tar -xf - -C "$STAGE_ROOT"
    EXTRACTED_DIR="${STAGE_ROOT}/${SRC_BASE}"
    [[ -d "$EXTRACTED_DIR" ]] || die "Extracted directory missing after untar: $EXTRACTED_DIR"

    echo "==> [local] Swapping into place..."
    if [[ -e "$FINAL_DEST" ]]; then mv "$FINAL_DEST" "$BAK"; fi
    mv "$EXTRACTED_DIR" "$FINAL_DEST"
    rm -rf "$STAGE_ROOT" || true
    rm -rf "$BAK" || true
    echo "✅ [local] Directory updated -> $FINAL_DEST"
  else
    die "MODE must be dir|file"
  fi

  if [[ "$DO_REMOTE_FORGET" == "1" ]]; then
    echo "==> [local] Forget snapshot id=$SNAP_ID (NO PRUNE, does not affect LeCAR)"
    "$RESTIC_BIN" unlock >/dev/null 2>&1 || true
    FORGET_EXTRA=()
    if "$RESTIC_BIN" help forget 2>/dev/null | grep -q -- '--retry-lock'; then
      FORGET_EXTRA=(--retry-lock "$FORGET_RETRY_LOCK")
    fi
    if "$RESTIC_BIN" forget "${FORGET_EXTRA[@]}" "$SNAP_ID" >/dev/null 2>&1; then
      echo "✅ [local] Forgot snapshot id=$SNAP_ID"
    else
      echo "⚠️  [local] Could not forget snapshot now (repo lock busy). Transfer is already complete."
      echo "    You can forget later: $RESTIC_BIN forget $SNAP_ID"
    fi
  else
    echo "⚠️  [local] Keeping snapshot (you used --no-forget)"
  fi

  echo "✅ Done."
  echo "   Snapshot=$SNAP_ID jobid=$JOBID mode=$MODE direction=pull"
  echo "   Note: no automatic prune is run (to avoid impacting LeCAR)."
  exit 0
fi

# ---------------- build bundle ----------------
JOBID="$(date +%Y%m%d_%H%M%S)_$RANDOM"
HOSTTAG="$(hostname | tr -cd '[:alnum:]_.-')"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/xfer_push_${USER}_${JOBID}_XXXX")"
cleanup(){ rm -rf "$WORK"; }
trap cleanup EXIT

BUNDLE="$WORK/bundle"
mkdir -p "$BUNDLE"

SRC_ABS="$(abs_path "$SRC")"
MODE=""
PAYLOAD=""
COMPRESSION="none"

if [[ -d "$SRC_ABS" ]]; then
  MODE="dir"
  COMPRESSION="zstd"
  PAYLOAD="payload.tar.zst"
  SRCPARENT="$(cd "$(dirname "$SRC_ABS")" && pwd)"
  SRCNAME="$(basename "$SRC_ABS")"
  echo "==> [local] Packing directory (zstd): $SRC_ABS"
  ( cd "$SRCPARENT" && tar -cf - "$SRCNAME" | zstd -T0 -19 -o "$BUNDLE/$PAYLOAD" )
elif [[ -f "$SRC_ABS" ]]; then
  MODE="file"
  PAYLOAD="$(basename "$SRC_ABS")"
  echo "==> [local] Preparing file: $SRC_ABS"
  cp -f "$SRC_ABS" "$BUNDLE/$PAYLOAD"
else
  die "SRC not found: $SRC"
fi

# ---- rsync-like dest semantics (LOCAL canonicalization) ----
DEST_INPUT="$DEST"
if [[ "$DEST_INPUT" == */ ]]; then
  dest_dir="${DEST_INPUT%/}"
  [[ -n "$dest_dir" ]] || dest_dir="/"
  DEST="$(join_dir_and_base "$dest_dir" "$(basename "$SRC_ABS")")"
else
  DEST="$DEST_INPUT"
fi

echo "==> [local] Writing checksum + manifest"
# IMPORTANT: write sha file with RELATIVE filename only (avoid absolute path)
( cd "$BUNDLE" && sha256sum "$PAYLOAD" > "$PAYLOAD.sha256" )
cat > "$BUNDLE/manifest.txt" <<EOF
payload=$PAYLOAD
sha256=$PAYLOAD.sha256
mode=$MODE
compression=$COMPRESSION
dest=$DEST
dest_input=$DEST_INPUT
src_base=$(basename "$SRC_ABS")
jobid=$JOBID
EOF

# ---------------- init repo if needed ----------------
if ! "$RESTIC_BIN" snapshots >/dev/null 2>&1; then
  echo "==> [local] Repo not initialized; restic init"
  "$RESTIC_BIN" init
fi
"$RESTIC_BIN" unlock >/dev/null 2>&1 || true

# ---------------- upload snapshot & capture snapshot id ----------------
VFLAG="$(restic_vflag)"
S3OPT=()
read -r -a S3OPT <<<"$(restic_s3opt_arr)"

EXTRA_LOCK=()
if restic_supports_retry_lock; then
  EXTRA_LOCK=(--retry-lock "$RETRY_LOCK")
fi

echo "==> [local] Uploading to R2 via restic-xfer"
echo "     repo=$RESTIC_REPOSITORY"
echo "     tags=$TAG_BASE,$JOBID,$HOSTTAG"
echo "     mode=$MODE compression=$COMPRESSION jobid=$JOBID s3.connections=${S3_CONN:-auto}"
echo "     dest_input=$DEST_INPUT"
echo "     dest_canon=$DEST"

OUT="$WORK/backup.out"
"$RESTIC_BIN" $VFLAG backup "$BUNDLE" \
  --tag "$TAG_BASE" --tag "$JOBID" --tag "$HOSTTAG" \
  "${EXTRA_LOCK[@]}" \
  "${S3OPT[@]:+"${S3OPT[@]}"}" | tee "$OUT"

SNAP_ID="$(extract_snapshot_from_output "$OUT" || true)"
if [[ -z "${SNAP_ID:-}" ]]; then
  echo "==> [local] Snapshot id not found in backup log; querying repo by tag..."
  SNAP_ID="$(lookup_snapshot_by_tag || true)"
fi
[[ -n "${SNAP_ID:-}" ]] || die "Cannot parse snapshot id from restic output"
echo "==> [local] Snapshot id: $SNAP_ID"

# ---------------- remote: install restic-xfer + restore-by-id + verify + safe install + forget ----------------
echo "==> [local] SSH remote ($SSH_HOST): restore-by-id + verify + safe install + forget"
echo "     ssh_retries=$SSH_RETRIES keepalive=${SSH_KEEPALIVE_INTERVAL}s x ${SSH_KEEPALIVE_COUNTMAX}"
echo "     forget_retry_lock=$FORGET_RETRY_LOCK"

SSH_COMMON_OPTS=(
  -o "ServerAliveInterval=${SSH_KEEPALIVE_INTERVAL}"
  -o "ServerAliveCountMax=${SSH_KEEPALIVE_COUNTMAX}"
  -o "TCPKeepAlive=yes"
  -o "ConnectTimeout=${SSH_CONNECT_TIMEOUT}"
  -o "ConnectionAttempts=2"
  -o "BatchMode=no"
)

REMOTE_OK=0
trap 'die "Interrupted by user"' INT TERM
S3_CONN_ARG="${S3_CONN:-__XFER_EMPTY_S3_CONN__}"
for ((attempt=1; attempt<=SSH_RETRIES; attempt++)); do
  echo "==> [local] Remote attempt ${attempt}/${SSH_RETRIES} ..."
  if ssh "${SSH_COMMON_OPTS[@]}" "$SSH_HOST" "bash -s" -- \
    "$SNAP_ID" "$JOBID" "$RETRY_LOCK" "$S3_CONN_ARG" "$VERBOSE" "$AUTO_INSTALL" "$PIN_RESTIC_VER" "$DO_REMOTE_FORGET" "$FORGET_RETRY_LOCK" "$RESTIC_REPO_URI" <<'REMOTE_SCRIPT'
set -euo pipefail

SNAP_ID="${1:?snap_id}"
JOBID="${2:?jobid}"
RETRY_LOCK="${3:?retry_lock}"
S3_CONN="${4:-}"
VERBOSE="${5:-1}"
AUTO_INSTALL="${6:-1}"
WANT_VER="${7:-0.18.1}"
DO_FORGET="${8:-1}"
FORGET_RETRY_LOCK="${9:-10s}"
RESTIC_REPO_URI_REMOTE="${10:?restic_repo_uri}"
if [[ "$S3_CONN" == "__XFER_EMPTY_S3_CONN__" ]]; then
  S3_CONN=""
fi

export AWS_ACCESS_KEY_ID="e64306563928427b781a93acd4e1193b"
export AWS_SECRET_ACCESS_KEY="fb4a8cad67231fb374d752d0d05586cccf72af31d22a24fe77ed0208816d2353"
export AWS_DEFAULT_REGION="auto"
export RESTIC_REPOSITORY="$RESTIC_REPO_URI_REMOTE"
export RESTIC_PASSWORD="LeCAR_restic_pw_3uQ7xgK2wL6s1Zp0Nf"
export RESTIC_CACHE_DIR="${RESTIC_CACHE_DIR:-$HOME/.cache/restic}"
mkdir -p "$RESTIC_CACHE_DIR" >/dev/null 2>&1 || true

RESTIC_BIN="$HOME/.local/bin/restic-xfer"
die(){ echo "❌ $*" >&2; exit 2; }

install_remote_pkg() {
  local pkg="$1"
  local runner=()
  if [[ "$(id -u)" -ne 0 ]]; then
    command -v sudo >/dev/null 2>&1 || die "sudo not found; cannot auto-install $pkg"
    runner=(sudo)
  fi
  if command -v apt-get >/dev/null 2>&1; then
    "${runner[@]}" apt-get update -y
    "${runner[@]}" apt-get install -y "$pkg"
  elif command -v yum >/dev/null 2>&1; then
    "${runner[@]}" yum install -y "$pkg"
  else
    die "No supported package manager on remote. Install '$pkg' manually."
  fi
}
need_remote() {
  local cmd="$1"
  local pkg="${2:-$1}"
  if command -v "$cmd" >/dev/null 2>&1; then return 0; fi
  [[ "$AUTO_INSTALL" = "1" ]] || die "Missing on remote: $cmd (install $pkg)"
  echo "==> [remote] Auto-install missing: $cmd (pkg: $pkg)"
  install_remote_pkg "$pkg"
  command -v "$cmd" >/dev/null 2>&1 || die "Auto-install failed for $cmd"
}
vflag() {
  case "$VERBOSE" in
    0) echo "";;
    1) echo "-v";;
    2) echo "-vv";;
    *) echo "-vvv";;
  esac
}
s3opt() {
  if [[ -n "${S3_CONN}" ]]; then
    echo "-o" "s3.connections=${S3_CONN}"
  else
    echo ""
  fi
}

need_remote curl curl
need_remote bzip2 bzip2
need_remote tar tar
need_remote zstd zstd
need_remote sha256sum coreutils || true
need_remote find findutils || true
need_remote mktemp coreutils || true
need_remote uname coreutils || true

if [[ ! -x "$RESTIC_BIN" ]]; then
  echo "==> [remote] Installing restic-xfer v$WANT_VER -> $RESTIC_BIN"
  mkdir -p "$(dirname "$RESTIC_BIN")"
  tmpd="$(mktemp -d)"
  trap 'rm -rf "$tmpd"' EXIT
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) A="amd64" ;;
    aarch64|arm64) A="arm64" ;;
    *) die "Unsupported arch: $arch" ;;
  esac
  url="https://github.com/restic/restic/releases/download/v${WANT_VER}/restic_${WANT_VER}_linux_${A}.bz2"
  curl -fsSL -o "$tmpd/restic.bz2" "$url"
  bunzip2 -f "$tmpd/restic.bz2"
  chmod +x "$tmpd/restic"
  mv "$tmpd/restic" "$RESTIC_BIN"
fi
"$RESTIC_BIN" version >/dev/null 2>&1 || die "restic-xfer not runnable: $RESTIC_BIN"

VFLAG="$(vflag)"
S3OPT=( $(s3opt) ) 2>/dev/null || true
"$RESTIC_BIN" unlock >/dev/null 2>&1 || true

TMPBASE="$(mktemp -d "${TMPDIR:-/tmp}/xfer_pull_${USER}_${JOBID}_XXXX")"
cleanup(){ rm -rf "$TMPBASE"; }
trap cleanup EXIT
mkdir -p "$TMPBASE/restore"

echo "==> [remote] Restoring by snapshot id=$SNAP_ID ..."
MAX_TRIES="${RESTORE_TRIES:-60}"
SLEEP_SEC="${RESTORE_SLEEP:-2}"
RESTORE_CONN_FALLBACKS="${RESTORE_CONN_FALLBACKS:-4,2,1}"
CONN_FALLBACK_IDX=0
CONN_FALLBACK_LIST=()
if [[ -z "$S3_CONN" ]]; then
  IFS=',' read -r -a CONN_FALLBACK_LIST <<< "$RESTORE_CONN_FALLBACKS"
fi
ok=0
for i in $(seq 1 "$MAX_TRIES"); do
  RESTORE_LOG="$TMPBASE/restore_${i}.log"
  CUR_CONN="auto"
  if [[ -n "$S3_CONN" ]]; then
    CUR_CONN="$S3_CONN"
  elif [[ "${#S3OPT[@]}" -ge 2 && "${S3OPT[0]}" == "-o" && "${S3OPT[1]}" == s3.connections=* ]]; then
    CUR_CONN="${S3OPT[1]#s3.connections=}"
  fi
  echo "==> [remote] restore attempt ${i}/${MAX_TRIES} (s3.connections=${CUR_CONN})"
  if "$RESTIC_BIN" $VFLAG restore "$SNAP_ID" --target "$TMPBASE/restore" ${S3OPT[@]:+"${S3OPT[@]}"} 2>&1 | tee "$RESTORE_LOG"; then
    ok=1; break
  fi
  if [[ -z "$S3_CONN" ]] && grep -qi 'unexpected EOF' "$RESTORE_LOG"; then
    while [[ "$CONN_FALLBACK_IDX" -lt "${#CONN_FALLBACK_LIST[@]}" ]]; do
      NEXT_CONN="$(echo "${CONN_FALLBACK_LIST[$CONN_FALLBACK_IDX]}" | tr -d '[:space:]')"
      CONN_FALLBACK_IDX=$((CONN_FALLBACK_IDX + 1))
      [[ -n "$NEXT_CONN" ]] || continue
      if [[ "$CUR_CONN" != "$NEXT_CONN" ]]; then
        echo "==> [remote] Detected unexpected EOF; switching restore to s3.connections=${NEXT_CONN} and retrying..."
        S3OPT=(-o "s3.connections=${NEXT_CONN}")
        break
      fi
    done
  fi
  "$RESTIC_BIN" unlock >/dev/null 2>&1 || true
  echo "==> [remote] restore failed ($i/$MAX_TRIES), retry in ${SLEEP_SEC}s..."
  sleep "$SLEEP_SEC"
done
[[ "$ok" = "1" ]] || die "restore-by-id still failing after retries"

BUNDLE_DIR="$(find "$TMPBASE/restore" -type f -name manifest.txt -print -quit | xargs -r dirname || true)"
[[ -n "${BUNDLE_DIR:-}" ]] || die "Cannot find manifest.txt in restored content"

sed -i 's/\r$//' "$BUNDLE_DIR/manifest.txt" 2>/dev/null || true
# shellcheck disable=SC1090
source <(grep -E '^(payload|sha256|mode|compression|dest|dest_input|src_base|jobid)=' "$BUNDLE_DIR/manifest.txt" || true)

PAYLOAD="${payload:?manifest missing payload=...}"
SHAFILE="${sha256:?manifest missing sha256=...}"
MODE="${mode:?manifest missing mode=...}"
COMPRESSION="${compression:-}"
DEST="${dest:?manifest missing dest=...}"
DEST_INPUT="${dest_input:?manifest missing dest_input=...}"
SRC_BASE="${src_base:?manifest missing src_base=...}"

[[ -f "$BUNDLE_DIR/$PAYLOAD" ]] || die "Payload missing after restore: $BUNDLE_DIR/$PAYLOAD"

echo "==> [remote] Verifying sha256..."
( cd "$BUNDLE_DIR" && sha256sum -c "$SHAFILE" )

# ---- safe rsync-like install ----
if [[ "$MODE" == "file" ]]; then
  if [[ "$DEST_INPUT" == */ ]]; then
    dest_dir="${DEST_INPUT%/}"
    mkdir -p "$dest_dir"
    FINAL_DEST="${dest_dir}/${SRC_BASE}"
  else
    FINAL_DEST="$DEST"
    [[ -d "$FINAL_DEST" ]] && die "Refusing to write file into existing directory path (missing trailing /?): dest=$FINAL_DEST"
    mkdir -p "$(dirname "$FINAL_DEST")"
  fi

  echo "==> [remote] Installing file -> $FINAL_DEST"
  TMPFILE="${FINAL_DEST}.part.$(date +%s)"
  mv "$BUNDLE_DIR/$PAYLOAD" "$TMPFILE"
  mv -f "$TMPFILE" "$FINAL_DEST"
  echo "✅ [remote] File updated -> $FINAL_DEST"

elif [[ "$MODE" == "dir" ]]; then
  FINAL_DEST="${DEST%/}"
  [[ -n "$FINAL_DEST" ]] || FINAL_DEST="/"
  parent="$(dirname "$FINAL_DEST")"
  mkdir -p "$parent"

  stamp="$(date +%s)"
  base_name="$(basename "$FINAL_DEST")"
  STAGE_ROOT="${parent}/.${base_name}.stage.${stamp}"
  BAK="${parent}/.${base_name}.bak.${stamp}"
  mkdir -p "$STAGE_ROOT"

  if [[ -z "$COMPRESSION" ]]; then
    case "$PAYLOAD" in
      *.tar.zst) COMPRESSION="zstd" ;;
      *) die "Cannot infer compression from payload (expect *.tar.zst): $PAYLOAD" ;;
    esac
  fi

  [[ "$COMPRESSION" == "zstd" ]] || die "Unsupported compression in manifest: $COMPRESSION (expect zstd)"
  echo "==> [remote] Extracting archive (zstd) -> $STAGE_ROOT"
  zstd -dc "$BUNDLE_DIR/$PAYLOAD" | tar -xf - -C "$STAGE_ROOT"
  EXTRACTED_DIR="${STAGE_ROOT}/${SRC_BASE}"
  [[ -d "$EXTRACTED_DIR" ]] || die "Extracted directory missing after untar: $EXTRACTED_DIR"

  echo "==> [remote] Swapping into place..."
  if [[ -e "$FINAL_DEST" ]]; then mv "$FINAL_DEST" "$BAK"; fi
  mv "$EXTRACTED_DIR" "$FINAL_DEST"
  rm -rf "$STAGE_ROOT" || true
  rm -rf "$BAK" || true
  echo "✅ [remote] Directory updated -> $FINAL_DEST"
else
  die "MODE must be dir|file"
fi

if [[ "$DO_FORGET" == "1" ]]; then
  echo "==> [remote] Forget snapshot id=$SNAP_ID (NO PRUNE, does not affect LeCAR)"
  "$RESTIC_BIN" unlock >/dev/null 2>&1 || true
  FORGET_EXTRA=()
  if "$RESTIC_BIN" help forget 2>/dev/null | grep -q -- '--retry-lock'; then
    FORGET_EXTRA=(--retry-lock "$FORGET_RETRY_LOCK")
  fi
  if "$RESTIC_BIN" forget "${FORGET_EXTRA[@]}" "$SNAP_ID" >/dev/null 2>&1; then
    echo "✅ [remote] Forgot snapshot id=$SNAP_ID"
  else
    echo "⚠️  [remote] Could not forget snapshot now (repo lock busy). Transfer is already complete."
    echo "    You can forget later: $RESTIC_BIN forget $SNAP_ID"
  fi
else
  echo "⚠️  [remote] Keeping snapshot (you used --no-forget)"
fi
REMOTE_SCRIPT
  then
    REMOTE_OK=1
    break
  fi
  if [[ "$attempt" -lt "$SSH_RETRIES" ]]; then
    echo "==> [local] Remote step failed/disconnected; retry in ${SSH_RETRY_SLEEP}s..."
    sleep "$SSH_RETRY_SLEEP"
  fi
done
trap - INT TERM
[[ "$REMOTE_OK" = "1" ]] || die "Remote step failed after ${SSH_RETRIES} attempts"

echo "✅ Done."
echo "   Snapshot=$SNAP_ID jobid=$JOBID mode=$MODE"
echo "   Note: no automatic prune is run (to avoid impacting LeCAR)."