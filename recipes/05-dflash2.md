# Recipe 5 — FP4-Local, DFlash2, the running server

Strata v0.2.10, engine `0.1.56`, NVIDIA Qwen3.8-Flash-Next-FP4-Local. The quality flags are the ones from [recipe 4](04-fp4-local.md): `w4a8`, tail skip 0, context 262,144, `auto:32768`, kv int8, kv-resident 32,768, 13 workers, spec-min-p 0.5, adapt 1 / 160 / 0.97. The drafter is DFlash2 instead of MTP. The reserve is 192 MiB instead of 512.

There is no `"drafting"` key. On this model name that key strips the draft flags and starts MTP. DFlash2 is `--draft dflash2` in the args.

**Source of the rates:** the systemd boot that started 2026-11-03 08:54 UTC. 852 requests had finished when this page was written (936,067 generated tokens). Five of those requests are the cold bench below and are not in the agent table. No finished decode fell under 50 tok/s.

| | Recipe 4, 0.2.0 MTP, 40 agent requests | This boot, 710 answers of gen ≥ 200 |
| --- | --- | --- |
| gen ≥ 200 | 147.7, median 145.2, n=26 | **160.0**, median **155.9**, 115.6–219.2, n=710, 916,328 tokens |
| Prompts 100k–200k, gen ≥ 200 | 142.9, median 135.4, n=9 | **157.7**, median 154.9, n=413, 501,872 tokens |
| Prompts 50k–90k, gen ≥ 200 | 154.8, median 157.5, n=7 | **165.7**, median 163.0, n=143, 177,463 tokens |
| Prompts ≥ 200k, gen ≥ 200 | 140.6 on four prompts | **153.2**, median 151.9, n=82, 122,625 tokens |
| Fresh reads ≥ 100k | 6,950.6, best 7,084.6, n=9 | **7,250.8**, median 7,281.9, best **7,456.1**, n=13, 2,391,487 tokens |
| Fresh reads ≥ 20k | 6,809.5, n=20 | **7,061.6**, median 7,110.8, n=26, 3,139,934 tokens |
| Drafts, gen ≥ 200 | 61.7% | **67.1%** (576,407 / 858,507) |

Decode on gen ≥ 200 is about 8% above the 0.2.0 MTP boot. The 100k–200k band is about 10% above it. Long fresh reads are about 4% above it. Draft acceptance is about 5 points higher, on a different drafter.

The same boot, before that agent traffic, ran the five cold prompts. Temperature 0, 256 tokens out, nothing reused. These are the engine's finished lines.

| Prompt tokens | Prefill | Decode | Drafts |
| --- | --- | --- | --- |
| 8,244 | 4,672.3 tok/s | 180.0 tok/s | 122/171 |
| 32,820 | 7,010.4 tok/s | 179.6 tok/s | 110/168 |
| 131,124 | 7,468.9 tok/s | 181.2 tok/s | 114/164 |
| 196,660 | 7,105.8 tok/s | 169.4 tok/s | 108/170 |
| 249,908 | 6,941.9 tok/s | 179.4 tok/s | 120/172 |

The 249,908-token row is 95% of the 262,144 context. The client clock on that same request was 6,922 tok/s prefill and 179.9 tok/s decode. The table is the engine line.

Boot line: `expert cache auto: 24.25 GiB free, 192 MiB reserved (+441 MiB for the draft head, +256 MiB for what is allocated after the cache) -> 9078 slots`. Request-start free was 141 MiB. The engine asked for reserve 563, then 576. That was not applied. Raising the reserve gives the expert slots back.

Serve starts from `data/expert-profile-fp4-learned.bin` when its header matches the seed `data/expert-profile.bin`. This boot did.

## Flags that changed from recipe 4

```
--vram-reserve-mib 192
--spec 6
--draft dflash2
--draft-model /mnt/models/incoai/Qwen3.8-Flash-Next-DFlash2/qwen3.8-flash-next-dflash2.q4_0-selq4.pack
--draft-fallback none
--mtp-draft-vocab /opt/strata/Strata/data/draft_vocab.bin
--draft-conf 0.5
--conversation-cache-mib 8192
--conversation-cache-slots 4
```

No `--mtp` line. The selector pack is 339,694,592 bytes, sha256 `4a42b98220ea7880664002c8edda1563aecbae754b762a2e2f31b2b91ee93ecd`.

Environment added on this tag, and set to 0 so the fast kernels stay off: `STRATA_GDN_TC=0`, `STRATA_GDN_CONVL2=0`, `STRATA_GR_R2=0`. `STRATA_GDN_CHUNKED=1` stays. `STRATA_ROUTE_PRIOR` stays unset.

Config: [`configs/strata-fp4-local-dflash2.json`](../configs/strata-fp4-local-dflash2.json). Unit: [`services/strata-engine-dflash2.service`](../services/strata-engine-dflash2.service). Model name: `qwen3.8-flash-next-fp4-local`. The paths in the repo use `/opt/strata`. This host runs the same flags from `/home/righteous/inference/Strata`.

Pictures stay off this server. One unit binds port 12345. The MTP rollback is recipe 4's unit, with its reserve at 288 on the host that measured this boot.
