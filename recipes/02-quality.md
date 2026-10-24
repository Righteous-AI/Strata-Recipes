# Recipe 2 — quality, Strata 0.1.45

Same command line and the same plain NVIDIA NVFP4 pack as [recipe 1](01-speed.md). The environment is `w4a8` and tail skip `0`. This is the long-running 0.1.45 server. It served until 2026-10-19 18:21 UTC, when the 0.2.0 tree took the port.

**Source of the rates:** real agent sessions. The 11:59 boot is the day the quality flags were switched on (519 requests before the 1M experiment). The 18:50 boot is the long sample, 8,118 finished requests, including the next day.

| Bucket | 11:59 boot | 18:50 boot |
| --- | --- | --- |
| gen ≥ 200 | **138.1**, median 135.1, max 183.2, n=431, 520,495 tokens | **134.3**, median **132.4**, max 193.0, n=6,721 |
| Prompts 100k–200k, gen ≥ 200 | **135.1**, median 132.7, n=269, drafts 69.5% | **132.8**, median 130.7, n=3,765 |
| Prompts 50k–90k, gen ≥ 200 | **143.3**, median 139.5, n=88 | **137.3**, median 136.0, n=1,417 |
| Fresh reads ≥ 100k | **6,916.8**, best 7,379.4, n=55 | **6,934.9**, best **7,469.0**, n=857 |
| Drafts, gen ≥ 200 | **70.3%** (318,782 / 453,573) | **70.2%** |

`0` is set for the skip. Deleting the variable is not the same thing on every build: an unset CUDA serve can turn the skip on. On this NVFP4 binary, unset means off. The config still sets `0`. The boot must not print `STRATA_ROUTE_TAIL_SKIP=7 is ON`.

w4a8 is also the default int8 MMQ path. Setting it in the config makes the choice obvious. The binary accepts only `w4a8`, `w4a4`, and `fp16`. Confirm `STRATA_PREFILL_NVFP4=w4a8` on the engine child, not on the Python parent.

The step-by-step cost of each quality change, and the 1,048,576-token YaRN trial that was reverted, are in [comparisons.md](../comparisons.md). YaRN slowed 100k–200k agent decode from 135.1 to 119.7 tok/s.

0.2.0 with these same flags is [recipe 3](03-nvfp4-0.2.0.md). On real agent traffic that build decoded the 100k–200k band at 143.3 tok/s against 132.8 here.
