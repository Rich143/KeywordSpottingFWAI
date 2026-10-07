#!/usr/bin/env bash
# Set up the host-side Python environment for this repo on a new machine/account.
#
#   1. Creates the conda env from HostSW/environment.yml (or updates it if it exists)
#   2. Builds the C preprocessing library (HostSW/build/audioproc-build/libAudioPreprocessing.dylib)
#      that the training notebooks and run_model_on_wav.py load via ctypes
#   3. Smoke-tests the main imports
#
# Usage: HostSW/setup_env.sh [env-name]     (default env name: kws)
#
# Prerequisites: conda (Miniconda), cmake, Xcode command line tools, and the repo
# cloned with submodules (git clone --recurse-submodules ...).
set -euo pipefail

ENV_NAME="${1:-kws}"
HOSTSW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$HOSTSW_DIR")"

# Prefer the conda the user's shell is initialised with (CONDA_EXE) over whatever
# `conda` is first on PATH, which may be a different install (e.g. Homebrew's).
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
[ -n "$CONDA" ] || { echo "error: conda not found (install Miniconda and run 'conda init')" >&2; exit 1; }
echo "==> Using conda: $CONDA ($("$CONDA" --version))"
command -v cmake >/dev/null || { echo "error: cmake not found on PATH" >&2; exit 1; }
[ -f "$REPO_DIR/Middlewares/ARM/CMSIS-DSP/Source/CMakeLists.txt" ] || {
    echo "error: CMSIS-DSP submodule missing; run: git submodule update --init --recursive" >&2; exit 1; }

echo "==> Creating/updating conda env '$ENV_NAME'"
if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    "$CONDA" env update -n "$ENV_NAME" -f "$HOSTSW_DIR/environment.yml" --prune
else
    "$CONDA" env create -n "$ENV_NAME" -f "$HOSTSW_DIR/environment.yml"
fi

echo "==> Building libAudioPreprocessing.dylib"
cmake -S "$HOSTSW_DIR" -B "$HOSTSW_DIR/build"
cmake --build "$HOSTSW_DIR/build" -j

echo "==> Smoke test"
# Call the env's python by path: `conda run`/PATH can pick up another python (e.g. pyenv shims).
ENV_PREFIX="$("$CONDA" env list | awk -v n="$ENV_NAME" '$1 == n {print $NF}')"
cd "$HOSTSW_DIR"
"$ENV_PREFIX/bin/python" - <<'EOF'
import ctypes, numpy, pandas, sklearn, matplotlib, seaborn, soundfile, sounddevice, serial, optuna
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
ctypes.CDLL("build/audioproc-build/libAudioPreprocessing.dylib")
print("python", __import__("sys").version.split()[0], "| tensorflow", tf.__version__,
      "| GPU devices:", tf.config.list_physical_devices("GPU"))
print("OK")
EOF

echo "==> Done. Activate with: conda activate $ENV_NAME"
