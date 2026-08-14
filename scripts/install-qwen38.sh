#!/usr/bin/env bash
# One-shot installer: Qwen3.8-27B Q4_K_M with embedded MTP on a single RTX
# 4090. It shares the upstream llama.cpp CUDA build used by Qwen3.6, while
# keeping its model, slot cache, runtime config, and service separate.
#
# Model source of truth: bartowski/Qwen3.8-27B-GGUF. The conversion requires
# llama.cpp b10419 or newer and enables embedded MTP with --spec-type draft-mtp.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------- tunables --------------------------------------------------------
PREFIX="${PREFIX:-$HOME/Dev/qwen36}"
LLAMA_DIR="$PREFIX/llama.cpp"
MODEL_DIR="$PREFIX/models/qwen38-27b-mtp"
MODEL_REPO="bartowski/Qwen3.8-27B-GGUF"
MODEL_FILE="Qwen3.8-27B-Q4_K_M.gguf"
HF_ENV="${HF_ENV:-qwen36-hf}"
BUILD_ENV="${BUILD_ENV:-qwen36-build}"
CUDA_LABEL="${CUDA_LABEL:-cuda-12.4.1}"
CUDA_ARCH="${CUDA_ARCH:-89}"
CTX_SIZE="${CTX_SIZE:-65536}"
PARALLEL="${PARALLEL:-1}"
PORT="${PORT:-13838}"
JOBS="$(nproc)"

LLAMA_UPSTREAM_URL="${LLAMA_UPSTREAM_URL:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_REMOTE_NAME="${LLAMA_REMOTE_NAME:-upstream}"
LLAMA_MIN_TAG="b10419"

# ---------- helpers ---------------------------------------------------------
log()  { printf '\033[1;32m[+] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[x] %s\033[0m\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
conda_env_prefix() { conda env list | awk -v name="$1" '$1 == name {print $NF; exit}'; }

# ---------- 0. sanity + shared-checkout preflight --------------------------
log "Sanity checks"
have git || die "git not found"
have cmake || die "cmake not found"
have nvidia-smi || die "nvidia-smi not found (NVIDIA driver missing?)"
have conda || die "conda not found. Install Miniconda/Mamba first."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Fail before any conda, package, or model work when the shared checkout has
# tracked/staged changes. Ignored build artifacts are deliberately allowed.
if git -C "$LLAMA_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 && \
  { ! git -C "$LLAMA_DIR" diff --quiet || ! git -C "$LLAMA_DIR" diff --cached --quiet; }; then
  die "Existing llama.cpp checkout at $LLAMA_DIR has local changes. Commit/stash them or set PREFIX to a fresh path."
fi
mkdir -p "$PREFIX"

# ---------- 1. current upstream llama.cpp ----------------------------------
log "Preparing current upstream llama.cpp master"
if ! git -C "$LLAMA_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git clone "$LLAMA_UPSTREAM_URL" "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"
if git remote get-url "$LLAMA_REMOTE_NAME" >/dev/null 2>&1; then
  git remote set-url "$LLAMA_REMOTE_NAME" "$LLAMA_UPSTREAM_URL"
else
  git remote add "$LLAMA_REMOTE_NAME" "$LLAMA_UPSTREAM_URL"
fi

# Fetch the minimum release tag explicitly before ancestry checking. A clone
# may not have fetched historical tags, so relying on an existing local tag
# would reject an otherwise compatible current master checkout.
git fetch --prune "$LLAMA_REMOTE_NAME" \
  "+refs/heads/master:refs/remotes/$LLAMA_REMOTE_NAME/master"
git fetch "$LLAMA_REMOTE_NAME" \
  "refs/tags/$LLAMA_MIN_TAG:refs/tags/$LLAMA_MIN_TAG"

TARGET_REF="refs/remotes/$LLAMA_REMOTE_NAME/master"
git rev-parse --verify --quiet "refs/tags/$LLAMA_MIN_TAG^{commit}" >/dev/null \
  || die "failed to fetch upstream tag $LLAMA_MIN_TAG for Qwen3.8 compatibility validation"
git merge-base --is-ancestor "refs/tags/$LLAMA_MIN_TAG^{commit}" "$TARGET_REF" \
  || die "upstream master predates $LLAMA_MIN_TAG, which Qwen3.8 GGUF requires"
git checkout --detach "$TARGET_REF"
log "  using llama.cpp commit $(git rev-parse --short HEAD), validated >= $LLAMA_MIN_TAG"

# Fail early if the selected upstream ref lacks the launch features this profile
# requires. The source checks make the later command line deterministic.
grep -Rqs '"draft-mtp"' common/ \
  || die "llama.cpp checkout lacks draft-mtp support. Update upstream master."
grep -Rqs -- '--slot-save-path' common/ tools/ \
  || die "llama.cpp checkout lacks --slot-save-path support. Update upstream master."
grep -Rqs -- '--checkpoint-min-step' common/ tools/ \
  || die "llama.cpp checkout lacks --checkpoint-min-step support. Update upstream master."
grep -Rqs -- '--kv-unified' common/ tools/ \
  || die "llama.cpp checkout lacks --kv-unified support. Update upstream master."
grep -Rqs -- '--cache-idle-slots' common/ tools/ \
  || die "llama.cpp checkout lacks --cache-idle-slots support. Update upstream master."

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/lib/conda-strict.sh"

# ---------- 2. CUDA toolkit env + shared build -----------------------------
log "Conda env for CUDA toolkit: $BUILD_ENV ($CUDA_LABEL)"
if ! conda env list | awk '{print $1}' | grep -qx "$BUILD_ENV"; then
  conda create -y -n "$BUILD_ENV" -c "nvidia/label/$CUDA_LABEL" cuda-toolkit
elif [[ ! -x "$(conda_env_prefix "$BUILD_ENV")/bin/nvcc" ]]; then
  log "  nvcc missing; installing cuda-toolkit into existing env"
  conda install -y -n "$BUILD_ENV" -c "nvidia/label/$CUDA_LABEL" cuda-toolkit
fi
conda_activate "$BUILD_ENV"
NVCC="$CONDA_PREFIX/bin/nvcc"
[[ -x "$NVCC" ]] || die "nvcc not found in $BUILD_ENV after installing cuda-toolkit"
"$NVCC" --version | tail -1

log "Building shared llama.cpp (CUDA arch $CUDA_ARCH, $JOBS jobs)"
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
  -DCUDAToolkit_ROOT="$CONDA_PREFIX" \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib"
cmake --build build --config Release -j "$JOBS" \
  --target llama-server llama-cli llama-quantize
[[ -x build/bin/llama-server ]] || die "build failed: build/bin/llama-server missing"
log "  build OK -> $LLAMA_DIR/build/bin/"
conda_deactivate

# ---------- 3. Qwen3.8 GGUF download ---------------------------------------
log "Conda env for huggingface-hub: $HF_ENV"
if ! conda env list | awk '{print $1}' | grep -qx "$HF_ENV"; then
  conda create -y -n "$HF_ENV" python=3.11
fi
conda_activate "$HF_ENV"
pip install -q --upgrade "huggingface-hub[hf_xet]"

mkdir -p "$MODEL_DIR"
log "Downloading $MODEL_REPO -> $MODEL_DIR"
# Always invoke hf download for this exact file. The Hugging Face client handles
# cache metadata and validates/reuses a matching local file without inventing a
# separate checksum policy here.
HF_XET_HIGH_PERFORMANCE=1 \
  hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODEL_DIR"
SIZE_BYTES=$(stat -c%s "$MODEL_DIR/$MODEL_FILE")
(( SIZE_BYTES > 15*1024*1024*1024 )) \
  || die "GGUF only $SIZE_BYTES bytes — likely corrupt (don't use aria2c for multi-GB models)"
log "  GGUF OK ($((SIZE_BYTES/1024/1024/1024)) GiB)"
conda_deactivate

# ---------- 4. Qwen3.8 config + launcher -----------------------------------
LAUNCH="$REPO_DIR/scripts/run-qwen38.sh"
[[ -x "$LAUNCH" ]] || die "missing repo launcher: $LAUNCH"

CONF_DIR="$HOME/.config/qwen36-mtp"
CONF="$CONF_DIR/qwen38.env"
mkdir -p "$CONF_DIR"
log "Writing config: $CONF"
cat > "$CONF" <<EOF
# Generated by scripts/install-qwen38.sh — edit and \`systemctl --user restart qwen38\`
LLAMA_BIN=$LLAMA_DIR/build/bin/llama-server
MODEL_PATH=$MODEL_DIR/$MODEL_FILE
CTX_SIZE=$CTX_SIZE
PORT=$PORT
HOST=0.0.0.0
ALIAS=qwen3.8-27b
SLOT_CACHE_DIR=$PREFIX/slot-cache-qwen38
CACHE_REUSE=256
SPEC_DRAFT_N_MAX=3
CHECKPOINT_MIN_STEP=2048
PARALLEL=$PARALLEL
EOF

# ---------- 5. switchable systemd user unit --------------------------------
SVC="$HOME/.config/systemd/user/qwen38.service"
mkdir -p "$(dirname "$SVC")"
cat > "$SVC" <<EOF
[Unit]
Description=Qwen3.8-27B MTP llama-server (RTX 4090, parallel=$PARALLEL)
After=network-online.target qwen36.service qwen36-multi.service
Conflicts=qwen36.service qwen36-multi.service

[Service]
Type=simple
EnvironmentFile=$CONF
Environment=CUDA_VISIBLE_DEVICES=0
Environment=GGML_CUDA_ENABLE_UNIFIED_MEMORY=0
ExecStart=$LAUNCH
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
log "Wrote systemd user unit: $SVC"
rm -f "$HOME/.config/systemd/user/default.target.wants/qwen38.service"
log "Left qwen38 disabled; start it manually with: systemctl --user start qwen38"

# ---------- 6. summary ------------------------------------------------------
cat <<EOF

========================================================================
DONE.

Model:    $MODEL_DIR/$MODEL_FILE
Binary:   $LLAMA_DIR/build/bin/llama-server (shared with Qwen3.6)
Launcher: $LAUNCH
Endpoint: http://localhost:$PORT/v1 (OpenAI-compatible, model alias "qwen3.8-27b")

Manual systemd service:
  systemctl --user daemon-reload
  systemctl --user start qwen38   # or: $LAUNCH

The qwen38 and qwen36/qwen36-multi services conflict, and their ordering
stops the other dense 27B service before starting this one. qwen38 is disabled
by installer; do not enable it unless you want login startup.

Smoke test:
  curl -s http://localhost:$PORT/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"reply: pong"}]}' | jq .

Caveats:
  * The 65,536 context / one-slot default is deliberately conservative for a
    24 GB RTX 4090. Raise context or parallelism only after measuring VRAM.
  * Thinking is on by default. --reasoning-format deepseek keeps <think> in
    response.reasoning_content. Disable per request with
    chat_template_kwargs={"enable_thinking":false}.
========================================================================
EOF
