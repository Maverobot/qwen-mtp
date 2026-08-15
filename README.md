# qwen-mtp

High-performance local serving of **Qwen3.6 with Multi-Token Prediction (MTP)**
on a single 24 GB consumer GPU (tested on RTX 4090; should also work on RTX 3090
with `CUDA_ARCH=86`).

The point of this repo is to capture the *exact* recipe — upstream
[llama.cpp](https://github.com/ggml-org/llama.cpp) with merged MTP support,
MTP-preserving GGUF, CUDA-toolkit-via-conda, shared-cudart link fix, sane launch
flags — so that a fresh box can go from zero to a working OpenAI-compatible
endpoint with one command.

## Why this stack?

| Need | What this repo gives you |
| --- | --- |
| Real Qwen3.6 quality on 24 GB | dense 27B (or 35B-class with the right GGUF), q4 weights, q4 KV cache |
| 100+ tok/s decode | NextN/MTP speculative decoding (~3× over no-MTP) |
| Large context with safer 24 GB headroom | 131K default, configurable to ≥196K thanks to hybrid GatedDeltaNet + GatedAttention layout |
| Strong prefix caching | upstream `--cache-idle-slots`, `--ctx-checkpoints`, KV-slot save/restore |
| OpenAI-compatible | `llama-server` with `--jinja` and `--reasoning-format deepseek` |
| Tool-calling | `--jinja` enables Qwen's tool-call template; LiteLLM in front recommended for heavy agents |

## Measured numbers (RTX 4090, selected profiles)

The default is the quality-first/headroom profile: `CTX_SIZE=131072` and
`PARALLEL=2`, now using Unsloth's MTP-preserving `IQ4_NL` GGUF. A local smoke
test with this quant loaded and answered at 131K/parallel=2 with 20.7 GB
nvidia-smi used after a short chat completion. Re-run throughput benchmarks
locally before relying on exact tok/s for this quant.

Historical selected-profile numbers from the previous `IQ4_XS` default:

| ctx | parallel | decode | prefill | MTP accept | VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 768 | 4 | 124 tok/s | 310 tok/s | 76.8 % | 18.8 GB |
| 131 072 | 2 | 47 tok/s | 1 741 tok/s | 45.3 % | 20.7 GB |
| 196 608 | 4 | 142 tok/s | 196 tok/s | 4/4 | 22.2 GB |
| 262 144 | 4 | 159 tok/s | 342 tok/s | 4/4 | 23.8 GB |

131K row: `/completion`, `cache_prompt=false`, 120 071-token cold prompt,
512-token decode, VRAM sampled immediately after the request.

(The previous `IQ4_XS` model card reported 100.3 tok/s short-ctx on a 3090 Ti
with the same launcher recipe.)

### Prefill-cache TTFT (RTX 4090, 196 608 ctx, 3 628-token prompt)

| call | wall | `prompt_n` re-prefilled | `cache_n` reused via KV-shift |
| --- | ---: | ---: | ---: |
| cold | **3.11 s** | 3628 | 0 |
| warm (same preamble, different tail) | **0.42 s** | 515 | 3112 |
| warm again | **0.38 s** | 514 | 3112 |

These cache timings were also measured with the previous `IQ4_XS` default.
~7.5× faster TTFT on subsequent calls thanks to `--cache-reuse 256`,
`--cache-idle-slots`, and upstream llama.cpp's KV-slot persistence.

### Tuning `--cache-reuse` (RTX 4090)

Swept 64 / 128 / 256 / 512 on this stack. **Differences are within noise** for
both pure prefix-match prompts (warm TTFT 0.43-0.45 s across all values) and
prompts with a varying middle (mid-prompt KV-shift fires roughly the same way).
The default of `256` is fine. If your workload has many small differing
fragments and you want to squeeze a few percent more reuse, try 64; otherwise
don't bother.

The bigger wins were already on by default in this repo:

| flag | effect |
| --- | --- |
| `--cache-idle-slots` | keep slot KV warm across requests within a server lifetime |
| `--slot-save-path` | auto-save/restore slot KV to disk; survives `systemctl restart qwen36` |
| `--ctx-checkpoints 8 --checkpoint-min-step 2048` | context snapshots for recurrent/hybrid KV reuse without overly dense checkpoints |
| `--cache-ram -1 --kv-unified` | unlimited RAM-side overflow + the unified-KV mode required by idle-slot caching |

## Install

Prereqs: NVIDIA driver (CUDA 12-capable), `git`, `cmake`, `conda` (Miniconda or
Mamba). Nothing else. The installer fetches its own CUDA toolkit into a conda
env, so a "driver-only" Linux box works.

```bash
git clone https://github.com/Maverobot/qwen-mtp.git
cd qwen-mtp
./scripts/install.sh
```

### Qwen3.8-27B alongside Qwen3.6

Install Qwen3.8 without replacing the Qwen3.6 profile:

```bash
./scripts/install-qwen38.sh
```

The installer reuses `$HOME/Dev/qwen36/llama.cpp`, its CUDA build environment,
and the `qwen36-hf` download environment. It first rejects tracked or staged
changes in the shared checkout, then updates that checkout to current upstream
`master`, verifies it is at least llama.cpp `b10419`, and rebuilds it before
downloading Bartowski's `Qwen3.8-27B-Q4_K_M.gguf` into a separate model
directory. The Qwen3.8 GGUF includes its MTP layers; the launcher enables them
with `--spec-type draft-mtp`.

The initial RTX 4090 profile is deliberately conservative:

| Setting | Default |
| --- | ---: |
| Local Pi endpoint | `http://127.0.0.1:13838/v1` |
| LAN endpoint | `http://<pc-lan-ip>:13838/v1` |
| Model alias | `qwen3.8-27b` |
| Context | 65,536 |
| Parallel slots | 1 |
| MTP draft tokens | 3 |
| KV cache | Q4_0 |

Both dense 27B models are installed but cannot run simultaneously on the 24 GB
GPU. The `qwen36`, `qwen36-multi`, and `qwen38` units conflict and are ordered
so that systemd stops the active model before allocating memory for the other:

```bash
systemctl --user daemon-reload
systemctl --user start qwen38  # stops qwen36/qwen36-multi if needed
systemctl --user start qwen36  # stops qwen38 if needed
```

`qwen38` is left disabled after installation, just like `qwen36`. Its separate
runtime settings are in `~/.config/qwen-mtp/qwen38.env`; edit that file and
restart `qwen38` only after a local VRAM fit test proves that a larger context
or more parallel slots is safe.

The service intentionally uses `HOST=0.0.0.0`. Port 13838 is
**unauthenticated** and therefore reachable from permitted LAN interfaces and
firewall rules at `http://<pc-lan-ip>:13838/v1`. Keep local Pi configured with
`127.0.0.1`, and do not expose this port outside a trusted network without
adding access controls separately.

For direct use with Pi, add the distinct `llamacpp-qwen38` provider to
`~/.pi/agent/models.json` (merge it with any existing providers, including a
Qwen3.6 `llamacpp` provider):

```json
{
  "providers": {
    "llamacpp-qwen38": {
      "baseUrl": "http://127.0.0.1:13838/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "models": [
        {
          "id": "qwen3.8-27b",
          "name": "Qwen 3.8 27B (Q4_K_M, llama.cpp)",
          "reasoning": true,
          "thinkingLevelMap": {
            "minimal": null,
            "low": "low",
            "medium": "medium",
            "high": "xhigh",
            "xhigh": "xhigh",
            "max": null
          },
          "input": ["text"],
          "contextWindow": 65536,
          "maxTokens": 16384,
          "compat": {
            "thinkingFormat": "chat-template",
            "chatTemplateKwargs": {
              "enable_thinking": {"$var": "thinking.enabled"},
              "preserve_thinking": true,
              "reasoning_effort": {
                "$var": "thinking.effort",
                "omitWhenOff": true
              }
            },
            "supportsDeveloperRole": false
          }
        }
      ]
    }
  }
}
```

Pi's `off` level disables thinking. Its `low` and `medium` levels map directly
to Qwen3.8, while `high` and `xhigh` both request Qwen's valid `xhigh` effort.
Unsupported `minimal` and `max` are hidden. The model rejects a `developer`
message role, so the Pi entry explicitly marks developer-role support as false.

The full endpoint remains available for scripts and other OpenAI-compatible
clients:

```bash
curl -s http://127.0.0.1:13838/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"reply: pong"}]}'
```

If you previously installed the old `crucible-mtp` fork, stop the service and
rerun the installer. It fetches `ggml-org/llama.cpp` into the existing checkout,
leaves old fork branches intact, and builds detached upstream `master`:

```bash
systemctl --user disable --now qwen36-multi 2>/dev/null || true
rm -f ~/.config/systemd/user/qwen36-multi.service
systemctl --user stop qwen36 2>/dev/null || true
./scripts/install.sh
systemctl --user daemon-reload
systemctl --user start qwen36   # when needed
```

Set `LLAMA_REF=<branch-or-tag>` if you want to pin a specific upstream revision.

Override knobs via env vars. Defaults are quality-first with safer 24 GB
headroom: `CTX_SIZE=131072`, `PARALLEL=2`.

```bash
PREFIX=$HOME/llm/qwen36 \
CTX_SIZE=196608 \
PARALLEL=4 \
PORT=8081 \
./scripts/install.sh
```

If you're on a different GPU, override `CUDA_ARCH` to match it:

| GPU | `CUDA_ARCH` |
| --- | --- |
| RTX 4090 / 4080 / 4070 (Ada)        | `89`  *(default)* |
| RTX 3090 / 3090 Ti / 3080 (Ampere)  | `86` |
| RTX 5090 (Blackwell)                | `120` |

The script creates:

- `$PREFIX/llama.cpp/build/bin/llama-server` — upstream-built binary
- `$PREFIX/models/qwen36-27b-mtp/Qwen3.6-27B-IQ4_NL.gguf` — ~15.2 GiB
- `$PREFIX/slot-cache/` — on-disk slot KV cache (auto-save/restore across restarts)
- `~/.config/qwen-mtp/env` — runtime config (paths, ctx size, port, …)
- `~/.config/systemd/user/qwen36.service` — MTP parallel unit; ExecStart points
  at `scripts/run.sh` *in this repo* so `git pull` updates the launcher.

`qwen36-multi.service` is deprecated and is no longer created by fresh installs.
Old units still work through `scripts/run-multi.sh`, which now shims to the same
MTP + `PARALLEL=4` path and prints a deprecation warning.

## Architecture: single source of truth

```
~/.config/systemd/user/qwen36.service
        │ ExecStart=
        ▼
$REPO/scripts/run.sh                  ← versioned, edit here
        │ reads
        ▼
~/.config/qwen-mtp/env              ← per-host paths/tunables
        │ pointing at
        ▼
llama-server binary + GGUF
```

Edit the launcher? `git pull` and `systemctl --user restart qwen36`.
Change ctx size or port? Edit `~/.config/qwen-mtp/env` and restart.

## Run

```bash
# ~/.config/systemd/user/qwen36.service is installed disabled; start it when needed:
rm -f ~/.config/systemd/user/default.target.wants/qwen36.service
systemctl --user daemon-reload
systemctl --user start qwen36

# manual one-shot:
LLAMA_BIN=...llama-server MODEL_PATH=...gguf ./scripts/run.sh
```

Endpoint: `http://localhost:13636/v1` (model alias `qwen3.6-27b`).

```bash
curl -s http://localhost:13636/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"reply: pong"}]}'
```

### Parallel MTP is the default

`llama.cpp` merged MTP in `ggml-org/llama.cpp#22673`, and upstream now supports
MTP with parallel slots. This repo therefore uses **one service**:

```bash
--spec-type draft-mtp --spec-draft-n-max 4 --parallel 2 --kv-unified
```

`PARALLEL=2` is written to `~/.config/qwen-mtp/env` by the installer and is
also the default in `scripts/run.sh`. The OpenAI-compatible API is available
at `http://127.0.0.1:13636/v1`.

Install the unit template once (idempotent):

```bash
ln -sf "$PWD/scripts/systemd/qwen36.service" ~/.config/systemd/user/qwen36.service
rm -f ~/.config/systemd/user/default.target.wants/qwen36.service
systemctl --user daemon-reload
systemctl --user start qwen36
```

Tune slots by editing `PARALLEL` in `~/.config/qwen-mtp/env` and restarting:

```bash
PARALLEL=1   # lowest memory / old single-slot behavior
PARALLEL=2   # default: light concurrency with safer 24 GB headroom
PARALLEL=4   # higher concurrency for agents/subagents; costs more VRAM
systemctl --user restart qwen36
```

`qwen36-multi.service` is deprecated. If an older install already has that unit,
it will still start `scripts/run-multi.sh`, but the wrapper now unsets stale
`DISABLE_MTP`, forces `PARALLEL=4`, preserves host-specific `PORT` / `ALIAS`, and
prints a deprecation warning. You can remove it after migrating:

```bash
systemctl --user disable --now qwen36-multi 2>/dev/null || true
rm -f ~/.config/systemd/user/qwen36-multi.service
systemctl --user daemon-reload
```

Measured no-MTP multi-slot numbers from the old fork-era profile were useful for
the crossover decision, but upstream parallel MTP replaces that split. Re-run
your local benchmark after migration if exact per-slot throughput matters.

## Laptop profile (RTX 5070 Ti, 12 GB)

For a Blackwell laptop GPU with 12 GB VRAM, `scripts/install-laptop.sh` +
`scripts/run-laptop.sh` set up **Qwen3.6-35B-A3B** (40 layers, MoE, 3B active)
from `mradermacher/Qwen3.6-35B-A3B-i1-GGUF` (`i1-Q4_K_S`, ~20 GiB):

- builds llama.cpp with `CUDA_ARCH=120` (Blackwell sm_120) against
  conda's `cuda-12.8.1` toolkit (12.4 doesn't ship sm_120 PTX),
- runs upstream master. The Qwen3.6-35B-A3B base model *does* ship an MTP head
  per Alibaba's model card (vLLM and SGLang both expose
  `--speculative-config '{"method":"mtp",...}'` for it), and upstream
  llama.cpp now has MTP support. The specific i1-Q4_K_S GGUF we use
  (`mradermacher/Qwen3.6-35B-A3B-i1-GGUF`) drops that head during conversion —
  `gguf-py` reports 0 tensors named `mtp*`/`nextn*` out of 733 total — so the
  laptop launcher intentionally omits `--spec-type draft-mtp` and focuses on
  upstream's `--n-cpu-moe` flag,
- `--n-cpu-moe 40` keeps every layer's MoE expert tensors on CPU (the experts
  are ~17 GiB and won't fit in 12 GiB VRAM); attention/router/embeddings stay
  on the GPU,
- **262 144 context** (the model's native max), `-fa on`, `-ctk q8_0 -ctv q8_0`
  (Q8_0 KV is near-lossless and halves the KV size vs fp16), `--cache-reuse
  256`, slot save/restore. The KV cache occupies ~2 720 MiB on the GPU at full
  context, leaving ~6.5 GiB free for compute buffers on a 12 GiB card.

#### Measured numbers (RTX 5070 Ti laptop, Qwen3.6-35B-A3B i1-Q4_K_S)

Server response timings via `llama-server`'s `/completion` endpoint
(`temperature=0`, `cache_prompt=false` unless noted):

| prompt | decode | prefill (cold) | wall |
| ---: | ---: | ---: | ---: |
| ~50 tok | **56 tok/s** | 85 tok/s | 4.6 s for 256 toks |
| ~8 K  | **58 tok/s** | 1 173 tok/s | 7.9 s incl. prefill |
| ~32 K | **53 tok/s** | 1 306 tok/s | 24.7 s incl. prefill |
| ~128 K | **39 tok/s** | 1 027 tok/s | 125 s incl. prefill |

Decode stays well above the 25–45 tok/s estimate up to 32 K context; even at
128 K it is still usable for one-shot retrieval queries (a 13 K-token prompt
returns its answer in ~3.1 s end-to-end). Decode is RAM-bandwidth-bound by the
MoE-on-CPU split, so it scales with DDR5 speed/channels, not GPU clocks. Needs
≥ 32 GiB system RAM (48 GiB+ comfortable for full context).

### Why MoE-on-CPU is the right answer at 12 GB

Qwen3.6-35B-A3B is a sparse mixture-of-experts: 35 B total parameters, but only
**3 B active per token** (a router picks 8 of 256 experts per layer). Naive
quantized weight footprint at i1-Q4_K_S is ~20 GiB; ~17 GiB of that is expert
tensors that are touched sparsely, and ~3 GiB is the always-active stuff
(attention, embeddings, router, output head).

Trying to fit everything on a 12 GB card means heavy quantization (IQ2/IQ3)
that visibly hurts coding quality. The much better trade-off:

| Tensor class | Size | Where | Why |
|---|---:|---|---|
| Attention + router + embeddings | ~3 GiB | GPU | Touched every token; latency-sensitive |
| 262 K KV cache (Q8_0) | ~2.7 GiB | GPU | Touched every step; latency-sensitive |
| MoE experts (256 × 40 layers) | ~17 GiB | CPU RAM | Only 3 B params active per token; RAM-bandwidth bound |

The 3 B active params per token is small enough that DDR5 main-memory bandwidth
can deliver them in time, and the GPU only does the dense parts. Net effect:
you get 35B-quality output at ~30 tok/s on a laptop GPU you couldn't otherwise
fit even a 27B-Q4 model into.

### Walkthrough

**1. Install once on the laptop:**

```bash
git clone https://github.com/Maverobot/qwen-mtp ~/Dev/qwen-mtp
cd ~/Dev/qwen-mtp
bash scripts/install-laptop.sh
```

This installs miniconda's CUDA 12.8.1 toolkit into a dedicated env, builds
llama.cpp with `CUDA_ARCH=120`, downloads the GGUF (~20 GiB) via `hf` +
`hf_transfer`, and writes `~/.config/qwen-mtp/laptop.env` with `LLAMA_BIN`
and `MODEL_PATH` filled in.

**2. Run the server:**

```bash
set -a; source ~/.config/qwen-mtp/laptop.env; set +a
~/Dev/qwen-mtp/scripts/run-laptop.sh
```

Listens on `127.0.0.1:13636`, OpenAI-compatible at `/v1/chat/completions`. Test:

```bash
curl -s http://127.0.0.1:13636/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"pong?"}]}'
```

**3. Common overrides** (set the env var before running):

| Var | Default | When to change |
|---|---|---|
| `CTX_SIZE` | `131072` | Drop to `65536` if you OOM on KV; bump to `262144` only with ≥ 64 GiB RAM |
| `N_CPU_MOE` | `40` | **Lower** (e.g. 36, 32) if you have spare VRAM — moves more experts onto GPU = faster decode |
| `PORT` | `13636` | If 13636 is busy |
| `SLOT_CACHE_DIR` | unset | Set to e.g. `~/.cache/qwen36-laptop/slots` to persist KV across restarts |
| `PARALLEL` | `1` | Bump to 2+ for concurrent agent slots (costs VRAM) |

Tuning loop: watch `nvidia-smi` while serving — if VRAM stays well below 11 GiB
during steady-state decode you're leaving performance on the table; lower
`N_CPU_MOE` by 4 and rerun until you're at ~10.5 GiB.

**4. Optional — run as a systemd user service:**

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/qwen36-laptop.service <<'EOF'
[Unit]
Description=Qwen3.6-35B-A3B (laptop)
After=network.target

[Service]
EnvironmentFile=%h/.config/qwen-mtp/laptop.env
ExecStart=%h/Dev/qwen-mtp/scripts/run-laptop.sh
Restart=on-failure

EOF
systemctl --user daemon-reload
systemctl --user start qwen36-laptop
journalctl --user -fu qwen36-laptop
```

## Thinking mode

Qwen3.6 thinks by default. The launcher sets `--reasoning-format deepseek`, so
`<think>` blocks land in the response's separate `reasoning_content` field —
clients that don't render reasoning still see clean `content`. To disable
thinking per request:

```json
{ "chat_template_kwargs": { "enable_thinking": false } }
```

For most coding agent workflows, leave thinking on; MTP keeps the latency cost
modest.

## Caveats

- The build links against the conda CUDA env via rpath. If you delete or rename
  that env, set `LD_LIBRARY_PATH` to a CUDA 12 lib dir before launching.
- Don't use `aria2c` for the GGUF download — silent corruption has been
  reported for multi-GB transfers. The installer uses `hf` (huggingface-hub) +
  Xet instead.
- 262 144 ctx was already tight on the previous `IQ4_XS` profile. `IQ4_NL` is
  larger, so treat 196 608 / 262 144 as fit-test-only on 24 GB; 131 072 is the
  recommended default.
- `llama-server` tool-calling via `--jinja` works for simple agents. For
  multi-tool / strict-schema workloads, put **LiteLLM** in front of it.
- This recipe is dense-27B-specific. A 35B variant would need its own GGUF and
  possibly a different `CTX_SIZE`.
- Upstream MTP uses `--spec-type draft-mtp`. `scripts/run.sh` enables it with
  `PARALLEL=2` by default. `scripts/run-multi.sh` / `qwen36-multi.service` are
  deprecated compatibility shims for old installs.

## Credits

- Upstream MTP support: [`ggml-org/llama.cpp#22673`](https://github.com/ggml-org/llama.cpp/pull/22673)
- MTP-preserving GGUF: [`unsloth/Qwen3.6-27B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) (`Qwen3.6-27B-IQ4_NL.gguf`)
- Reference recipe + 100 tok/s claim: noonghunna's `qwen36-27b-single-3090` writeup

## License

MIT
