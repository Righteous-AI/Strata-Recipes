# Recipe 1 — speed, Strata 0.1.45

The fastest stack measured on this host. It is not what is running. Tail skip 7 and prefill `w4a4` are the two differences from the quality recipe, and both were rejected once the quality cost was measured.

**Source of the rates:** real agent sessions, 2026-10-17 11:17–11:50, 178 finished requests. The cold rows in [comparisons.md](../comparisons.md#controlled-benches) are a separate bench.

| Bucket | Agent result |
| --- | --- |
| All generated tokens | **156.0 tok/s** over 193,301 tokens, median 157.0, max 210.5 |
| gen ≥ 200 | **155.8**, median **155.9**, max 199.3, n=149, 189,070 tokens |
| Prompts 100k–200k, gen ≥ 200 | **154.6**, median 153.1, max 181.3, n=97 |
| Fresh reads ≥ 20k | **7,193 tok/s**, best **8,028.9**, 2,490,063 tokens |
| Fresh reads ≥ 100k | **7,487 tok/s**, best 8,028.9, 13 reads |
| Drafts | 117,108 / 170,172 = **68.8%** |

Real requests sat about 149–165 tok/s once the cache was warm. Expert-cache hit was about 91–95%. KV block reads were about 96–99% from VRAM.

The config with these two environment lines is [`configs/strata-nvfp4-speed.json`](../configs/strata-nvfp4-speed.json). It points at the 0.1.45 tree.

```
STRATA_PREFILL_NVFP4=w4a4
STRATA_ROUTE_TAIL_SKIP=7
```

`STRATA_ROUTE_TAIL_SKIP=7` has to be set. On an NVFP4 pack the engine leaves tail skip off unless it is exactly 7. The boot line is `STRATA_ROUTE_TAIL_SKIP=7 is ON`. w4a4 does not print its own line. The check is the child environment.

Shared args, and the flags that were tried and dropped on the way to this recipe, are in the [README](../README.md) and [rejected.md](../rejected.md). Turning the skip off cost about 11% decode and did not slow long prefill. Switching prefill from w4a4 to w4a8 cost about 5–7% on long prefill. Together, the quality recipe decodes about 11–14% slower and prefills long prompts about 8% slower. That trade is [recipe 2](02-quality.md).
