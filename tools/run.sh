#!/usr/bin/env bash
# Run a Python script with the project's conda env, regardless of PATH.
# Usage: ./tools/run.sh <script.py> [args...]
#
# The plain `python` on Windows resolves to a Microsoft Store stub, so this
# wrapper always uses the project env's interpreter. The env's MKL/OpenBLAS
# runtime lives in `Library/bin` and must be on PATH or numpy.linalg crashes
# with a Windows delay-load failure (0xc06d007f). We prepend the env dirs to
# PATH ourselves using MSYS-style (/c/...) paths, because the Windows DLL
# loader receives mangled entries when Windows-style (C:/...) paths with
# colons are passed through git-bash.
set -euo pipefail

ENV_NAME="complex-networks-env"
PY=""

if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/python.exe" ]; then
  PY="${CONDA_PREFIX}/python.exe"
elif [ -x "/c/Users/lcdit/anaconda3/envs/${ENV_NAME}/python.exe" ]; then
  PY="/c/Users/lcdit/anaconda3/envs/${ENV_NAME}/python.exe"
else
  echo "ERROR: could not locate python.exe for env '${ENV_NAME}'" >&2
  exit 1
fi

ENV_ROOT="$(dirname "$PY")"
export PATH="${ENV_ROOT}:${ENV_ROOT}/Library/mingw-w64/bin:${ENV_ROOT}/Library/usr/bin:${ENV_ROOT}/Library/bin:${ENV_ROOT}/Scripts:${ENV_ROOT}/bin:$PATH"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <script.py> [args...]" >&2
  exit 2
fi

SCRIPT="$1"
shift
exec "$PY" "$SCRIPT" "$@"
