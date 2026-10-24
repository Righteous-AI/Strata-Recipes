# Recipe 4 — FP4-Local, the running server

Strata 0.2.0-beta, NVIDIA Qwen3.8-Flash-Next-FP4-Local. The weights were retrained with quantization-aware distillation (ModelOpt 0.47). The serve flags are the quality set from [recipe 3](03-nvfp4-0.2.0.md): `w4a8`, tail skip 0, context 262,144, `auto:32768`, kv int8, kv-resident 32,768, reserve 512, 13 workers, spec 4, spec-min-p 0.5, adapt 1 / 160 / 0.97. The table below is the 20:44 UTC boot, which used 19 workers.

**Source of the rates:** real agent sessions on the systemd boot that started 2026-10-19 20:44 UTC. Forty requests had finished by 21:04 UTC (46,203 generated tokens). One more was still thinking at the cutoff. The matched greedy bench against the plain pack is in [comparisons.md](../comparisons.md#controlled-benches) and is not this table.

| | Plain NVFP4, recipe 3, 72 agent requests | This boot, 40 agent requests |
| --- | --- | --- |
| gen ≥ 200 | 146.5, median 143.9, n=54 | **147.7**, median **145.2**, n=26, 123.9–169.4 |
| Prompts 100k–200k, gen ≥ 200 | 143.3, median 142.4, n=33 | 142.9, median 135.4, n=9 |
| Prompts 50k–90k, gen ≥ 200 | 151.5, median 150.4, n=10 | **154.8**, median 157.5, n=7 |
| Fresh reads ≥ 100k | 6,845.8, best 6,939.4, n=6 | **6,950.6**, best **7,084.6**, n=9 |
| Drafts, gen ≥ 200 | 66.7% | 61.7% (24,601 / 39,865) |
| Expert-cache hit, gen ≥ 200 | 91.4% | **92.6%** |

Decode is level with the plain pack. Long reads are about 1.5% higher. Draft acceptance is about 5 points lower. The cache hit rate is a point higher, on the expert profile that was ranked for the plain NVFP4 pack. This session did not show a stale cache.

The config now has `"expert_profile_save": "data/expert-profile-fp4-learned.bin"`. The engine still starts from `data/expert-profile.bin` until that learned file exists and its header is the same 48 layers and 512 experts. It writes the file after 10 minutes between requests, and on a clean stop. The start after that is the one that fills the cache from the learned order. `--adapt-every 1` still moves experts during a conversation, so this changes the starting rank, not the weights.

24 of the 40 prompts were fresh and 16 reused a checkpoint. Fresh gen ≥ 200 was 146.8 tok/s (n=18, drafts 65.6%). Cached gen ≥ 200 was 148.5 tok/s (n=8, drafts 58.1%). Four prompts past 200k tokens decoded at 140.6 tok/s. No finished decode fell under 50 tok/s.

Against the long 0.1.45 quality boot, gen ≥ 200 is 147.7 against 134.3.

## Pack

Nothing was downloaded. `tools/nvfp4_local_pack.py prepare` ran on this machine from the 0.2.0 tree, about 430 seconds, exit 0. The checkpoint was left in place.

| Output | Size |
| --- | --- |
| `ple-nvfp4.gguf` | 26.82 GiB. 320,001,536 NVFP4 rows. The plain pack's `ple-fp8.gguf` is the other model. |
| `dense.gguf` | 5.58 GiB. MXFP8 attention, GDN, shared experts, and indexer, dequantized, then Q8_0. Small projections the BF16 kernels read stay BF16. |
| `pack/experts.bin` | 63.28 GiB. CSF scales decoded to E4M3, then the same 36-byte blocks as the plain pack. |
| `pack/dense.bin` | 1.43 GiB |
| `mtp/rt` | experts 0.66 GiB, dense 0.11 GiB. Q2_0 draft of this checkpoint's MTP head. |

```
python tools/nvfp4_local_pack.py prepare \
  --model /mnt/models/qwen/nvidia-Qwen3.8-Flash-Next-FP4 \
  --out /mnt/models/qwen/Strata-Qwen3.8-Flash-FP4-Local
```

The 0.2.0 serving venv has no torch. The prepare step used the 0.1.45 venv's interpreter and the 0.2.0 script. Pictures are not converted. The server does not pass `--vision`.

Config: [`configs/strata-fp4-local.json`](../configs/strata-fp4-local.json). Unit: [`services/strata-engine.service`](../services/strata-engine.service). Model name: `qwen3.8-flash-next-fp4-local`.

Boot shape of the 20:44 session matches recipe 3: 19 workers, chunk 32,768, 9,075 slots, 141 MiB CUDA free, KV 32,768 of 262,144, arena 63.28 GiB pinned (4.13 GiB/s on this boot). PLE direct-read probe 153,705 rows/s. The engine warned about 13 workers and about 141 MiB free. The reserve stayed at 512. Workers moved to 13 on 2026-10-21. The running config is 13. The rates above were measured at 19.

NVIDIA's published card for these weights, against BF16: GPQA Diamond 91.5% versus 92.2%, ToolEvalBench 88.7% versus 89.4%. Strata does not rank QAD as the highest-quality of the three NVFP4 packs. Their note, and the bench that did not match the agent result, are in [comparisons.md](../comparisons.md).
