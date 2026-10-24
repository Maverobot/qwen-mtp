# Run NInfer

Use the existing Docker container named `ninfer` to serve Qwen3.8-27B with
text/image input and tool support. The container name is independent of the
client: Pi, RikkaHub, and other OpenAI-compatible clients use the same service.

This guide operates an already provisioned container. The installers in this
repository still install the separate llama.cpp profiles; they do not create
NInfer. `docker start` reuses the container's saved arguments and mounts, so you
don't need to run `ninfer serve` separately on the host.

## Start, monitor, and stop

Before starting NInfer:

- Stop any other large model server on the GPU. The `qwen36`, `qwen36-multi`, and
  `qwen38` systemd units only coordinate with each other; they do not stop this
  Docker container. Likewise, stop NInfer before starting a llama.cpp profile.
- Ensure the saved Tailscale and LAN bind addresses are available on the host.
  A changed LAN address requires updating the container's published bindings.

### Start

Start the retained container:

```sh
docker start ninfer
```

### Start automatically

Keep the container's restart policy at `no`. With a Docker restart policy, the
container can start before Tailscale assigns its address, and the port binding
fails. Instead, use a user systemd service that waits until both bind addresses
exist and Docker responds, then runs `docker start ninfer`. Give the service no
stop command, so logging out does not stop the model.

A user service starts when you log in. To start it at boot without logging in,
enable lingering once:

```sh
sudo loginctl enable-linger "$USER"
```

If the LAN address changes, for example after a new DHCP lease, the start fails
until the container's bindings are updated. Reserve the address on the router.

### Monitor

Follow its logs:

```sh
docker logs -f ninfer
```

Press Ctrl+C to leave the log viewer. This does not stop the server.

### Stop

When you want to stop serving, stop the container. This interrupts active requests:

```sh
docker stop ninfer
```

## Connect and check readiness

1. Inspect the container state and its published addresses:

   ```sh
   docker inspect ninfer --format '{{.State.Status}}'
   docker port ninfer 8080/tcp
   ```

2. Use a published host address, or its Tailscale DNS name, as `SERVER_HOST` in
   the following commands. Don't assume `127.0.0.1` is published. After startup,
   check the service and model metadata:

   ```sh
   curl -fsS "http://SERVER_HOST:13838/health"
   curl -fsS "http://SERVER_HOST:13838/v1/models"
   ```

3. Configure your OpenAI Chat Completions client with base URL
   `http://SERVER_HOST:13838/v1` and the model ID in the profile below.

The deployment publishes only its selected Tailscale and LAN addresses, not all
host interfaces. It does not validate API keys; a client that requires a value
can use the nonsecret placeholder `local`. Any device that can reach either
published endpoint can invoke the model. Tailscale encrypts traffic on its route;
ordinary LAN HTTP does not. Do not expose this unauthenticated service publicly.

### Ready state

The service is ready when the container is `running`, `/health` returns HTTP 200
with status `ok`, and `/v1/models` matches the profile's model ID, context limit,
and vision capability. A running container alone does not prove the model has
finished loading.

## Saved RTX 4090 profile

The validated profile uses these settings:

| Setting | Value |
| --- | --- |
| Model | Standard Qwen3.8-27B GSQ-RCO IQ3_S, not the heretic variant |
| Model ID | `qwen3.8-27b` |
| Per-request context | `229376` tokens (224K) |
| Shared KV pool | About 474K tokens, sized at startup to leave 1 GiB of free GPU memory |
| Active generation requests | `2`; further requests wait |
| Vision | `--vision --vision-residency overlay` |
| Default reasoning effort | `medium`; overridable per request |
| Speculative decoding | MTP, draft tokens `3` |
| KV format / Gated DeltaNet state | `rk4v4` / FP16 |
| Prefill | Concurrent prefill, 512-token chunks |
| Host cache | `2048` MiB |

Input, image tokens, reasoning, and generated text share the context window.
224K is not an additional output allowance. The server has no fixed 16K output
cap; clients can impose their own limits. The configured Pi client reserves
4096 tokens beyond its estimated input when calculating the output allowance.

### Concurrent requests

When a request starts, the server reserves KV space for its prompt plus its full
output allowance until the request finishes. Two requests run together only when
both reservations fit in the shared pool.

Pi sets its output allowance to the remaining window, so each Pi request reserves
about 225K tokens regardless of prompt length. Two Pi requests fit in the pool
and generate concurrently. A third request waits for a free slot, and a request
that waits longer than 10 minutes fails with HTTP 503 `request_queue_timeout`.
A client that omits its output limit receives about half the pool per request.

The retained container is the source of truth for its image and launch arguments:

```sh
docker inspect ninfer --format '{{.Config.Image}}'
docker inspect ninfer --format '{{json .Config.Cmd}}'
```

## Validation scope

On 2026-10-24, two independent requests using Pi's output allowance generated
concurrently with these inputs per request: 8K, 128K, 192K, and about 212K tokens
(near the full 224K window, twice), plus one long and one short request together.
The answers passed their coding and retrieval checks, except for one of four cold
requests near 210K tokens, which retrieved a wrong value. Small-image,
default-reasoning, and reasoning-override checks passed on this profile. The
minimum sampled free GPU memory was **1178 MiB**; the core temperature peaked at
82°C at the stock 450 W power limit.

Pi image attachments, native `read` image results, and RikkaHub image recognition
were confirmed on the previous single-request profile on 2026-10-23.

These checks do not establish arbitrary image resolutions, video, full-window
output, or sustained-load reliability. Memory was sampled, not measured
continuously, and no permanent guard enforces the 1 GiB desktop reserve. GPU-heavy
desktop applications can fail to allocate memory while long requests run. Overlay
was configured, but its internal memory-borrowing branch was not independently
observed. Changes to capacity, concurrency, or vision residency need a new fit
check rather than an assumption that this result still applies.

## Return to the single-request profile

The previous profile runs one request at a time with a 224K pool, `rk8v4`, and
DFlash2, and leaves about 2 GiB of GPU memory free. When you replace a profile,
rename the old container instead of removing it. To switch back, replace
`PREVIOUS_CONTAINER` with the retained container's name:

```sh
docker stop ninfer
docker rename ninfer ninfer-dual-224k
docker rename PREVIOUS_CONTAINER ninfer
docker start ninfer
```

See the [upstream serving documentation for the deployed source revision](https://github.com/iamwavecut/ninfer-all/blob/8d8aa48411f618d92806bca6641bc1429ca80089/docs/serving.md)
for argument semantics.
