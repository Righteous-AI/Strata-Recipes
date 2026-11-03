# Strata recipes — Qwen3.8-Flash-Next on an RTX 5090

**Hardware:** ASUS ProArt Z890 · Intel Core Ultra 7 265K · 128 GB DDR5-5200 · RTX 5090 32 GB · Samsung 9100 PRO
**Engine:** [Strata](https://github.com/Niko1221/Strata) v0.2.10, engine `0.1.56`, built for sm_120
**What is running:** [recipe 5](recipes/05-dflash2.md), NVIDIA's retrained FP4-Local weights, DFlash2, quality flags
**Measured:** 2026-10-17 through 2026-11-03

Five serve recipes on one machine. Two are the 0.1.45 pair (fastest, then the quality cut). Two are 0.2.0 (plain NVIDIA NVFP4, then FP4-Local with MTP). Recipe 5 is v0.2.10, the same FP4-Local pack, with DFlash2. The unit in [`services/strata-engine-dflash2.service`](services/strata-engine-dflash2.service) starts recipe 5.

The decode and long-prefill rates in the table below are from a real agentic coding pipeline: multi-turn sessions, tool calls, and prompts from a few thousand tokens up past 200k. They are not llama-benchy runs, not aiperf runs, and not a repeated pangram. Each row is a finished `strata serve:` line from those sessions. A separate [bench section](comparisons.md#controlled-benches) holds the greedy prompts that were sent on purpose. Those benches are marked, and they are not the headline.

Weighted means `1000 * tokens / milliseconds` inside the bucket. A live monitor window is noisier than the finished line for the same request. Short cached reads sit near 1,000 tok/s because they still pay startup, so a session-wide prefill mean looks slower than a long read. Compare long reads with long reads, and compare decode inside the same prompt band. Every table is also in [comparisons.md](comparisons.md).

A flag stayed when it beat the current best by more than about 3% on the metric it was meant to move, with every other flag held still. Two changes were kept because the quality gain was the point: tail skip off, and prefill `w4a8`.

---

## What to run

Recipe 5. Strata v0.2.10, NVIDIA FP4-Local (quantization-aware distillation), DFlash2, context 262,144, prefill `w4a8`, route tail skip off, reserve 192, 13 CPU workers.

```
STRATA_PREFILL_NVFP4=w4a8
STRATA_ROUTE_TAIL_SKIP=0
STRATA_GDN_TC=0
STRATA_GDN_CONVL2=0
STRATA_GR_R2=0
```

Config: [`configs/strata-fp4-local-dflash2.json`](configs/strata-fp4-local-dflash2.json). Unit: [`services/strata-engine-dflash2.service`](services/strata-engine-dflash2.service).

On the 2026-11-03 08:54 UTC boot, real agent traffic decoded at **160.0 tok/s** (gen ≥ 200, median 155.9, n=710) and read fresh prompts of 100k tokens and up at **7,251 tok/s** (n=13, best 7,456). A cold 249,908-token prompt on that same boot, nothing reused, prefilled at 6,942 tok/s and decoded at 179.4 tok/s. Draft acceptance on the agent set was 67.1%. Recipe 4, the 0.2.0 MTP boot, remains the previous record at 147.7 tok/s.

NVIDIA's published card for these weights, against the BF16 model: GPQA Diamond 91.5% versus 92.2%, ToolEvalBench 88.7% versus 89.4%. Strata ranks their own GPTQ NVFP4 pack as the best quality of the three NVFP4 versions, and describes QAD as slightly faster on short prompts when every expert fits in VRAM. This card holds 9,075 of 24,576 experts. The table below is this host, not that 72 GB card.

---

## The four recipes

| | [1 Speed](recipes/01-speed.md) | [2 Quality](recipes/02-quality.md) | [3 NVFP4 0.2.0](recipes/03-nvfp4-0.2.0.md) | [4 FP4-Local](recipes/04-fp4-local.md) | [5 DFlash2](recipes/05-dflash2.md) |
| --- | --- | --- | --- | --- | --- |
| Engine | 0.1.45 | 0.1.45 | 0.2.0-beta | 0.2.0-beta | v0.2.10 (`0.1.56`) |
| Weights | NVIDIA NVFP4 | NVIDIA NVFP4 | NVIDIA NVFP4 | NVIDIA FP4-Local (QAD) | NVIDIA FP4-Local (QAD) |
| Drafter | MTP, spec 4 | MTP, spec 4 | MTP, spec 4 | MTP, spec 4 | DFlash2, spec 6, conf 0.5 |
| Prefill | `w4a4` | `w4a8` | `w4a8` | `w4a8` | `w4a8` |
| Tail skip | 7 | 0 | 0 | 0 | 0 |
| Decode, gen ≥ 200 | **155.8** (n=149) | 134.3 (n=6,721) | **146.5** (n=54) | **147.7** (n=26) | **160.0** (n=710) |
| Prompts 100k–200k | 154.6 | 132.8 | 143.3 | 142.9 | **157.7** |
| Prompts 50k–90k | | 137.3 | 151.5 | 154.8 | **165.7** |
| Fresh reads ≥ 100k | **7,487** | 6,935 | 6,846 (n=6) | **6,951** (n=9) | **7,251** (n=13) |
| Drafts, gen ≥ 200 | 68.8% | 70.2% | 66.7% | 61.7% | **67.1%** |
| Where the rate comes from | Real agent sessions, 178 requests | Real agent sessions, 8,118 requests on the long boot | Real agent sessions, 72 requests | Real agent sessions, 40 finished requests | Real agent sessions, 852 requests on the 08:54 boot, bench excluded |
| Role | Fastest stack on 0.1.45 | Quality on 0.1.45 | Same flags, 0.2.0 engine | 0.2.0 FP4-Local record | Running server |

Recipe 1 is the speed ceiling on 0.1.45. Tail skip 7 drops a rank-7 expert and does not renormalize. Upstream, against Q8_0, skip off is top-1 91.79% and KL 0.069; skip 7 is 88.59% and KL 0.118. w4a4 is FP4 activations into the prompt GEMM. Upstream first-token KL against fp16 is 0.0080 for w4a4 and 0.0018 for w4a8. Both stayed off after they were measured.

Recipe 3 is the same quality flags on 0.2.0 with the plain NVFP4 pack. Decode rose about 8% on 100k–200k prompts and about 10% on 50k–90k prompts versus the long 0.1.45 quality boot. Long prefill was about 1% lower, on 6 reads against 857.

Recipe 4 swaps in the retrained weights. Real traffic did not give back the 0.2.0 decode gain. A matched greedy bench (138,235 tokens, temperature 0) was the one place FP4-Local was slower: cold prefill 3.5%, warm decode 9.8%, and that warm gap followed draft acceptance on a repetitive prompt (81% down to 70%). Cold decode on that bench was 135.2 against 134.6.

---

## Shared command line

Recipes 1–4 use this shape. Only the pack paths and the two environment variables change. Recipe 5 keeps the block, drops `--mtp`, and adds the DFlash2 flags and the 8,192 MiB conversation cache. Its reserve is 192, and its spec is 6.

```
--expert-cache auto
--prefill auto:32768
--max-context 262144
--kv int8
--kv-resident 32768
--vram-reserve-mib 512
--pool-workers 13
--spec 4
--spec-min-p 0.5
--adapt-every 1
--adapt-swaps 160
--adapt-decay 0.97
```

No YaRN, no locked `--pcie-frac`, no `--mtp-q4`, no vision. Pictures were left off. The GPU reader would hold about 1.5 GiB and shrink the expert cache.

Boot shape on this card, both 0.2.0 packs, as measured on 2026-10-19: 19 workers on CPUs 1–19, host thread on CPU 0, prompt chunk 32,768, **9,075** expert slots, 512 MiB reserve, about **141 MiB** CUDA free, KV 32,768 of 262,144 (3.09 GiB pinned). The 63.28 GiB expert arena is page-locked. The engine warns that 13 workers is its recommendation and that 141 MiB is low. The reserve stayed at 512 on those boots. Workers are **13** in the configs as of 2026-10-21: a five-run sweep on this host found 19 no faster, and a Nsight capture showed the same kernel-only GPU fraction at both counts. The 2026-10-19 session rates above were measured at 19. See [rejected.md](rejected.md).

The v0.2.10 boot on 2026-11-03 08:54 UTC is recipe 5: 13 workers, the same prompt chunk, **9,078** expert slots, 192 MiB reserve, 141 MiB free at request start. The draft head takes another 441 MiB, and 256 MiB is allocated after the cache.

PCIe probe is about 46 GB/s. The engine starts at `pcie_frac` 0.25 and refines to about 0.30.

---

## Install

The paths in the configs, the unit, and the monitor assume an install root of `/opt/strata` (engine tree at `/opt/strata/Strata`, service user `strata`); adjust them to your layout. Change `User`, `Group`, `WorkingDirectory`, the engine, the pack, and the API key before using them anywhere else. The key here is the LAN token the clients already send (`local`). `/health` is open. `/metrics`, `/slots`, and `/v1` want `Authorization: Bearer local`. The bind is `0.0.0.0:12345`. Keep that off the public internet.

```
sudo cp services/strata-engine-dflash2.service /etc/systemd/system/strata-engine-dflash2.service
sudo systemctl daemon-reload
sudo systemctl enable --now strata-engine-dflash2.service
curl -sS http://127.0.0.1:12345/health
```

`services/strata-engine.service` is the 0.2.0 MTP unit. Only one unit can bind port 12345.

The first load takes the better part of a minute while the 63 GiB arena pins. Health should report `"loaded": true`, `"max_context": 262144`, and `"model": "qwen3.8-flash-next-fp4-local"`.

Confirm the engine child, not the Python parent:

```
STRATA_PREFILL_NVFP4=w4a8
STRATA_ROUTE_TAIL_SKIP=0
STRATA_GDN_TC=0
STRATA_GDN_CONVL2=0
STRATA_GR_R2=0
```

The read-only panel in [`monitor/strata_monitor.py`](monitor/strata_monitor.py) is the one this host's `monitor` alias starts. It polls `/health`, `/metrics`, and `/slots` on port 12345 with the LAN token, tails `strata-fp4-local-dflash2.log` (override the path with `STRATA_LOG`) for finished `strata serve:` lines, and reads the live prompt and thinking lines from `journalctl -u strata-engine-dflash2.service`. Quitting it does not stop the server.

```
STRATA_API_KEY=local python3 monitor/strata_monitor.py
```

On this machine that command is the `monitor` alias. Open a new shell, or `source ~/.bashrc`, after the alias change. Keys `1`–`6` switch pages. `q` detaches.

Strata also ships its own panel. It is a different program:

```
cd /opt/strata/Strata
STRATA_API_KEY=local .venv/bin/python strata.py monitor --url http://127.0.0.1:12345
```

---

## Files

```
README.md                         this page
comparisons.md                    every measured table, agent traffic and benches
hardware.md                       the machine
rejected.md                       flags that were tried or read, and left off
recipes/01-speed.md               0.1.45, w4a4, tail skip 7
recipes/02-quality.md             0.1.45, w4a8, tail skip 0
recipes/03-nvfp4-0.2.0.md         0.2.0, plain NVIDIA NVFP4
recipes/04-fp4-local.md           0.2.0, FP4-Local, MTP
recipes/05-dflash2.md             v0.2.10, FP4-Local, DFlash2, the running server
configs/strata-fp4-local.json     recipe 4
configs/strata-fp4-local-dflash2.json  recipe 5
configs/strata-nvfp4-nvidia.json  recipe 3
configs/strata-nvfp4-speed.json   recipe 1, written so the two env lines are visible
services/strata-engine.service    recipe 4
services/strata-engine-dflash2.service  recipe 5
monitor/strata_monitor.py         read-only panel for the running server
```

Weights stay on the machine that converted them. This repo does not ship a model.

Sibling case study, same host, the FreeToken engine: FreeToken NVFP4 on this 5090 (separate write-up). FreeToken's sustained decode on that study was about 56–67 tok/s. These Strata recipes are a different engine and a later measurement.

---

## License

Measurements and write-up: [CC BY 4.0](LICENSE). Engine: Strata, as published by its author. Model weights: NVIDIA Open Model Agreement and the Qwen license, as published with each checkpoint.
