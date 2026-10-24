# Left off

A flag stayed when it beat the current best by more than about 3% on the metric it was meant to move, on this host, with the other flags held still. Tail skip off and prefill `w4a8` were kept on purpose. Everything else in this list was slower, flat, or the wrong machine.

## Tried on the speed stack, 0.1.45, and not kept

| Tried | Result on this host |
| --- | --- |
| `--max-context 204800`, `--prefill auto:16384`, `--vram-reserve-mib 1024` | Fewer expert slots and 16k prompt chunks. A 32k-class prefill was 7,038 tok/s and a 228,244-token read was 6,601 tok/s. Moving to 262,144, reserve 512, and 32,768 chunks produced the 7,846 / 8,046 bench rows. |
| `--pcie-frac 0.25` locked, with `--spec-min-p 0.70` | Not faster than auto, which settles near 0.30, and `--spec-min-p 0.5`. |
| `--spec 6` | Slower long decode. The verify window is already up to 6. Draft depth stays 4. |
| `--pool-workers` 5, 7, 10, 13 | Not faster than 19 on the pinned arena. |
| `--pool-affinity measure` | Not kept. |
| `--mtp-q4 all` | Quantizing the MTP head was not faster. |
| `--adapt-swaps 80` | 160 was kept. |
| Adaptive swaps off | The adaptive pass stayed. |
| GPU power limit 500 W | 32k prefill 8,046 to 7,666 tok/s. Left at 600 W. |
| Huge pages | `nr_hugepages=0`. Not turned on. |

## Tried on the quality stack and not kept

| Flag | Why it is off |
| --- | --- |
| `--max-context 1048576` and YaRN scale 4 | Agent decode on 100k–200k prompts 135.1 to 119.7 tok/s. Slots 9,079 to 8,556. Reverted. See [comparisons.md](comparisons.md). |
| `--pool-workers 13` on 0.2.0, 2026-10-19 20:07 | Bench, same greedy 138,235-token prompt: cold decode +2.4%, warm decode +1.8%, prefill flat, drafts identical. Under 3%. Stayed at 19 that evening. |
| `--pool-workers 19` after 2026-10-21 | Five runs, same prompts, served binary. Warm long prompts were fastest at 13. The 24,267-token sampled cell was about 4% slower at 19. Prefill stayed flat. Nsight's kernel-only GPU fraction was 43% at both counts. Configs moved to 13. The 20:07 bench stays the record of that evening. |
| Raising `--vram-reserve-mib` to 883 or 896 | The boot suggests it because about 141 MiB is free. Raising it deletes expert-cache slots. A 300 MiB reserve failed to start. 512 MiB is the floor that still boots. No request in these logs fell to a few tok/s. |
| `STRATA_PREFILL_NVFP4=w4a4` | The speed lever. First-token KL against fp16 is 0.0080 against 0.0018 for w4a8. Stays off. |
| `STRATA_ROUTE_TAIL_SKIP=7` | The other speed lever. Against Q8_0, top-1 88.59% and KL 0.118, against 91.79% and 0.069 with the skip off. On this host it cost about 11% decode. Stays at 0. Unset is off on this NVFP4 binary. The config still sets 0 so a later binary cannot turn it on. |
| `--layer-split auto` | One GPU. The engine clears auto and prints that there is no split. |
| `--pipeline-windows 2` | Needs a two-GPU layer split. On this host the engine turns it off. The automatic path is AMD gfx12 only. |
| `--mmap-experts` | Leaves the pinned 63 GiB arena and reads through the file cache. That is the low-RAM path. The flag text records 71.97 ms/token cold and 34.78 ms/token once the pages are warm. |
| `--kv q4_0` or `k8v4` | `--kv int8` is already set. `q4_0` is the smaller cache, and the notes put its long-document perplexity 8–12% worse. `k8v4` stores the values at 4 bits. |
| `STRATA_PCIE_BALANCE=1` | The engine comment measures an RTX 5090 at 12.2 ms/round either way once 16 workers are running. This host uses 19. |
| `STRATA_EXCHANGE_ROTATE` | File-backed expert source. This server pins the arena. |
| `STRATA_RAM_ADAPT` | Helps when experts miss RAM. The whole 63 GiB arena is pinned. |
| `STRATA_PREFILL_BF16X2` | Default 0. Turning it on is the slower exact path. |
| `STRATA_ROUTE_PRIOR=0.5` | Opt-in. KL 0.0126 against a run-to-run floor of 0.0012. |
| `STRATA_PREFILL_HOST_PLAN` | Pinned to 0. It is a no-op on the pinned arena. |
| w4a4x2 | A doc microbench. This binary accepts only `w4a8`, `w4a4`, and `fp16`. |
| `STRATA_PF_FUSED` | The fused prompt experts cover IQ2_S, IQ3_XXS, IQ3_S, and IQ4_XS. This pack is NVFP4. Setting `=1` does not add an NVFP4 kernel. |
| `STRATA_SM86_PERF`, `STRATA_SM60_PERF`, HIP and Intel switches, GGUF prefetch | Not this card, and not an `experts.bin` arena. |
| `--vision` | The GPU reader holds on the order of 1.5 GiB and would shrink the expert cache. Pictures stay off. FP4-Local's vision tower was not converted. |
