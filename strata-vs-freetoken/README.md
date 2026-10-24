# Strata vs FreeToken, 2026-10-24 (Strata side)

The full head-to-head (both engines, method, TTFT breakdown, per-run CSV, chart, and the benchmark client) is in the FreeToken repo: [FreeToken-Recipes/strata-vs-freetoken](https://github.com/Righteous-AI/FreeToken-Recipes/tree/main/strata-vs-freetoken). This folder keeps the Strata-only raw files from that run.

Same host and GPU (RTX 5090 32607 MiB, power cap 600 W), same client, same prompts, one engine at a time. One warmup discarded, then the median of three measured runs. Temperature 0, seed 1, `max_tokens` 512. Prompt is the repeated token `" fox"`. Client on the host, `http://127.0.0.1:12345/v1`.

![Decode tok/s vs prompt length](https://raw.githubusercontent.com/Righteous-AI/FreeToken-Recipes/main/strata-vs-freetoken/decode-vs-prompt.png)

## Median of the three measured runs

| Prompt tokens | Strata TTFT s | Strata prefill tok/s | Strata decode tok/s | FreeToken TTFT s | FreeToken prefill tok/s | FreeToken decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2100 | 1.288 | 1630.7 | 176.8 | 1.350 | 1555.8 | 52.9 |
| 32820 | 4.614 | 7112.7 | 163.1 | 5.532 | 5933.2 | 53.8 |
| 131124 | 17.642 | 7432.3 | 159.7 | 18.596 | 7051.4 | 52.8 |
| 187912 | 25.776 | 7290.2 | 157.8 | 27.220 | 6903.5 | 55.0 |

Client prefill tok/s is `prompt_tokens / TTFT`. Client decode tok/s is `512 / (total − TTFT)`. Strata `cached_tokens` was 0 on every row. Strata's config enables `--spec 4`; the FreeToken unit does not enable speculative decoding, and the weights are different packs.

## Strata version and launch

Recipe 4 ([`recipes/04-fp4-local.md`](../recipes/04-fp4-local.md)), already running. Tree `v0.2.2` at `011c9e600a93bbc9fd451c563501b5831c32bbca`. Engine binary `strata 0.1.47`. Unit [`strata-engine.service`](../services/strata-engine.service):

```
/opt/strata/Strata/.venv/bin/python -m serve.server --engine strata --config /opt/strata/Strata/strata-fp4-local.json --host 0.0.0.0 --port 12345
```

Config engine args: `--pack /mnt/models/qwen/Strata-Qwen3.8-Flash-FP4-Local/pack`, `--native` and `--native-dense-gguf` that tree's `dense.gguf`, `--ple-gguf` `ple-nvfp4.gguf`, `--expert-profile data/expert-profile.bin`, `--expert-cache auto`, `--prefill auto:32768`, `--max-context 262144`, `--kv int8`, `--kv-resident 32768`, `--vram-reserve-mib 512`, `--pool-workers 13`, `--mtp .../mtp/rt`, `--spec 4`, `--spec-min-p 0.5`. Served id `qwen3.8-flash-next-fp4-local`.

Strata's own log for these requests shows drafts accepted: at 2100 tokens, 235/325, 250/329, and 254/341 on the three measured runs (server decode 168.0, 176.2, 178.3 tok/s). At 187912 tokens the server rates were 156.7, 153.9, 160.2, and 157.2 tok/s, with expert-cache hit rate 95.2–95.8%.

One 131124-token Strata request logged `only 17 MiB of VRAM free` and decoded at 149.7 tok/s server-side (client 150.2). That is the low sample in the 159.7 median. It did not fall to a few tok/s.

## Strata against the earlier 157.5 figure

[`comparisons.md`](../comparisons.md) records 157.5 tok/s as the median of 7 agent requests, prompts 50k–90k, gen ≥ 200, on the 2026-10-19 FP4-Local boot. This bench's 187912-token median is 157.8 tok/s. The 2100-token median is 176.8. The same notes say decode slows as the prompt grows, and that a repetitive bench lands around 175–200 while agent text is lower. This prompt is the repeated token `" fox"`. The 2K warmup's server line was 161.2 tok/s (237 of 293 drafts). The three measured 2K runs after it were 168.0, 176.2, and 178.3. A logged 161.7 peak was not in those notes. The nearest short warm samples there are 161.3 and 162.3 tok/s.

## Files

```
README.md                    this page
strata-harness-console.txt   harness console for the Strata head-to-head, incl. prompt generation
strata-service-status.txt    unit state, tree commit/tag, engine version, GPU snapshot at bench time
```
