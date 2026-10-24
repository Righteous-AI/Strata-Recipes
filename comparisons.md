# Comparisons

Every rate below is a finished `strata serve:` line. Weighted means `1000 * tokens / milliseconds` over the requests in that bucket. Median is the median of the per-request tok/s figures.

**Agent** means a real agentic coding pipeline: multi-turn sessions, tool calls, and prompts from a few thousand tokens up past 200k. Those sessions are the headline. They are not llama-benchy, not aiperf, and not a repeated pangram.

**Bench** means a prompt sent on purpose, greedy, `temperature` 0, `seed` 1. Benches are marked and kept out of the agent tables.

Decode gets slower as the prompt grows, with no flag change in between. On the long 0.1.45 quality boot, 50k–90k prompts decoded at about 137 tok/s and 100k–200k prompts at about 133 tok/s. Compare a bucket with the same bucket.

A session-wide prefill mean looks slow because cached reads under about 5,000 fresh tokens still pay startup and land near 1,000 tok/s. Long-read rows are the ones to compare.

---

## Agent traffic, all four recipes

| | 1 Speed, 0.1.45 | 2 Quality, 0.1.45 | 3 NVFP4, 0.2.0 | 4 FP4-Local, 0.2.0 |
| --- | --- | --- | --- | --- |
| When | 2026-10-17 11:17–11:50 | 2026-10-17 18:50 boot, through 2026-10-18 | 2026-10-19 18:37 | 2026-10-19 20:44–21:04 |
| Source | real agent sessions | real agent sessions | real agent sessions | real agent sessions |
| Requests finished | 178 | 8,118 | 72 | 40 (one more still running at the cutoff) |
| Prefill / tail skip | `w4a4` / 7 | `w4a8` / 0 | `w4a8` / 0 | `w4a8` / 0 |
| Weights | NVIDIA NVFP4 | NVIDIA NVFP4 | NVIDIA NVFP4 | NVIDIA FP4-Local (QAD) |
| gen ≥ 200 | **155.8**, median 155.9, max 199.3, n=149, 189,070 tokens | **134.3**, median 132.4, max 193.0, n=6,721 | **146.5**, median 143.9, 116.5–171.0, n=54, 80,598 tokens | **147.7**, median 145.2, 123.9–169.4, n=26, 44,420 tokens |
| All generated tokens | **156.0** over 193,301, median 157.0, max 210.5 | the gen ≥ 200 row is the long-boot figure | 82,003 generated tokens in the boot | 46,203 generated tokens in the 40 requests |
| Prompts 100k–200k, gen ≥ 200 | **154.6**, median 153.1, max 181.3, n=97 | **132.8**, median 130.7, n=3,765 | **143.3**, median 142.4, n=33, 41,702 tokens | **142.9**, median 135.4, n=9 |
| Prompts ≥ 100k, gen ≥ 200 | | | | **141.6**, median 135.7, n=13, drafts 11,441/18,861 (60.7%) |
| Prompts 50k–90k, gen ≥ 200 | | **137.3**, median 136.0, n=1,417 | **151.5**, median 150.4, n=10, 20,128 tokens | **154.8**, median 157.5, n=7 |
| Fresh reads ≥ 100k | **7,487**, best 8,028.9, 13 reads | **6,934.9**, best 7,469.0, n=857 | **6,845.8**, median 6,860.2, best 6,939.4, n=6, 773,285 tokens | **6,950.6**, median 7,028.7, best 7,084.6, n=9, 1,295,598 tokens |
| Fresh reads ≥ 20k | **7,193**, best 8,028.9, 2,490,063 tokens | | **6,718.0**, n=13, 1,321,552 tokens | **6,809.5**, median 6,830.0, n=20 |
| Drafts, gen ≥ 200 | 117,108 / 170,172 = **68.8%** | **70.2%** on the gen ≥ 200 set | 45,270 / 67,829 = **66.7%** | 24,601 / 39,865 = **61.7%** |
| Expert-cache hit, gen ≥ 200 | about 91–95% once warm | about 93% after warmup | **91.4%** | **92.6%** (26,187,287 / 28,295,077) |
| Decode under 50 tok/s | none in the kept logs | none | none | none |

Recipe 4's 40 requests split 24 fresh and 16 reused. Fresh gen ≥ 200 was 146.8 tok/s (n=18, drafts 65.6%). Cached gen ≥ 200 was 148.5 tok/s (n=8, drafts 58.1%). Four prompts ran past 200k tokens: gen ≥ 200 there was 140.6 tok/s, drafts 60.5%.

One real FP4 request landed on the direct process before the systemd boot and is not inside the 40: a 4,809-token prompt, 1,722 generated at 139.7 tok/s, drafts 929/1,367 (68.0%).

---

## Old engine, 0.1.45: speed, then the two quality cuts

Same binary, same plain NVIDIA NVFP4 pack, same command line. One variable moved at a time. All three rows are real agent sessions.

### Tail skip 7 to 0, w4a4 held still

Skip 7 drops a missed expert that every token of a verify window routes at rank 7 or lower, and it does not renormalize. Upstream, against Unsloth Q8_0: skip off is top-1 91.79% and KL 0.069; skip 7 is 88.59% and KL 0.118.

| | Skip 7, 11:17–11:50 | Skip 0, 11:50–11:59 |
| --- | --- | --- |
| Decode, gen ≥ 200 | 155.8 weighted, median 155.9, 189,070 tokens | **138.7** weighted, median **140.7**, 53,415 tokens |
| Drafts | **68.8%** | **72.1%** (34,053 / 47,222 on gen ≥ 200) |
| Fresh reads ≥ 20k | 7,193 tok/s | **7,239** tok/s over 454,433 tokens, best 7,801 |
| Fresh reads ≥ 100k | 7,487 tok/s, 13 reads | 7,493 tok/s on 2 reads, best 7,801 |

Decode got about 11% slower. Long prefill did not move. Draft acceptance went up.

### Prefill w4a4 to w4a8, skip 0 held still

w4a4 is FP4 activations into the prompt GEMM. w4a8 is FP8 activations. Decode was already w4a8. Upstream first-token KL against fp16 is 0.0080 for w4a4 (7 of 8 rows inside their gate) and 0.0018 for w4a8 (8 of 8). Real-row error: gate/up 7.2% versus 0.46%, down 8.4% versus 0.90%.

The early matched slice, same day:

| | w4a4, skip 0 | w4a8, skip 0 |
| --- | --- | --- |
| Fresh reads ≥ 20k | **7,239.3** tok/s over 454,433 tokens | **6,723.4** tok/s over 373,977 tokens, about 7% slower |
| One matched ~137k read | 7,140 tok/s | 6,980.3 tok/s, about 2% slower |
| Decode | 138.9 tok/s | 133.6 tok/s over 18,067 tokens, median 135.1 |
| Drafts | 72.3% on the fuller skip-off sample | 67.6% on that early slice |

The rest of the 11:59 quality boot (519 requests, still w4a8, skip 0, before the 1M experiment) settled here:

| Bucket | 11:59 quality boot, agent |
| --- | --- |
| gen ≥ 200 | **138.1** tok/s, median **135.1**, max 183.2, n=431, 520,495 tokens |
| Prompts 100k–200k, gen ≥ 200 | **135.1**, median 132.7, max 159.2, n=269, drafts **69.5%** |
| Prompts 50k–90k, gen ≥ 200 | **143.3**, median 139.5, n=88 |
| Fresh reads ≥ 20k | **6,862**, best 7,379.4, 8,775,489 tokens |
| Fresh reads ≥ 100k | **6,916.8**, median 6,913, best **7,379.4**, n=55 |
| Drafts, gen ≥ 200 | 318,782 / 453,573 = **70.3%** |

Against the speed recipe's live traffic, the quality stack that day was decode 156.0 to 138.1 tok/s (about 11% slower) and long prefill 7,487 to 6,917 tok/s (about 8% slower). Most of the prefill gap is the w4a4 to w4a8 step. Drafts went from 68.8% to 70.3%.

The 18:50 boot is the long sample of that same quality stack, 8,118 finished requests, including traffic after the 4.5 GHz CPU cap. It sits in the same band: gen ≥ 200 at 134.3, 100k–200k at 132.8, 50k–90k at 137.3, reads ≥ 100k at 6,934.9. The CPU cap did not show up as a separate slowdown (uncapped 50k–90k weighted 137.7 tok/s, n=656; capped 140.0, n=13).

On the 11:59 boot, 333 requests and 4,847,256 tokens read averaged 5,423.6 tok/s only because 286 reads under 5,000 tokens ran at 1,058.9 tok/s. Reads ≥ 5k were 6,779. Reads ≥ 50k were 6,887. The longest read was 190,182 tokens at 7,379.4.

---

## New engine against the old one, same quality flags, plain NVFP4

0.2.0-beta, sm_120 kernels left on, w4a8, tail skip 0, 19 workers, `auto:32768`, context 262,144. The 0.1.45 column is the long 18:50 quality boot. Both columns are real agent sessions.

| Bucket | 0.1.45 quality | 0.2.0 NVFP4, 18:37 |
| --- | --- | --- |
| gen ≥ 200 | 134.3, median 132.4, max 193.0, n=6,721 | **146.5**, median **143.9**, 116.5–171.0, n=54, 80,598 tokens |
| Prompts 100k–200k, gen ≥ 200 | 132.8, median 130.7, n=3,765 | **143.3**, median **142.4**, n=33, 41,702 tokens |
| Prompts 50k–90k, gen ≥ 200 | 137.3, median 136.0, n=1,417 | **151.5**, median **150.4**, n=10, 20,128 tokens |
| Fresh reads ≥ 100k | 6,934.9, best 7,469.0, n=857 | **6,845.8**, median 6,860.2, best **6,939.4**, n=6, 773,285 tokens |
| Fresh reads ≥ 20k | | 6,718.0, n=13, 1,321,552 tokens |
| Drafts, gen ≥ 200 | 70.2% | **66.7%** (45,270 / 67,829) |
| Expert-cache hit, gen ≥ 200 | | **91.4%** |

Decode is about 8% faster on 100k–200k prompts and about 10% faster on 50k–90k prompts. Long prefill is about 1% lower, on 6 reads against 857. Draft acceptance is a few points lower. The 0.2.0 sample is one evening, not the 8,118-request boot. The direction is the same in both prompt bands.

The sm_120 decode switches (`STRATA_SM120_PERF` left unset, so on) are bitwise with the kernels they replace. The fused prompt path does not apply to NVFP4 experts. This host's number for the new engine is the table above.

---

## NVFP4 against FP4-Local, same 0.2.0 binary, same quality flags

### Agent sessions

| | Plain NVFP4, 18:37, 72 requests | FP4-Local, 20:44, 40 requests |
| --- | --- | --- |
| gen ≥ 200 | 146.5, median 143.9, n=54, 116.5–171.0, 80,598 tokens | 147.7, median 145.2, n=26, 123.9–169.4, 44,420 tokens |
| Prompts 100k–200k, gen ≥ 200 | 143.3, median 142.4, n=33 | 142.9, median 135.4, n=9 |
| Prompts 50k–90k, gen ≥ 200 | 151.5, median 150.4, n=10 | 154.8, median 157.5, n=7 |
| Fresh reads ≥ 100k | 6,845.8, median 6,860.2, best 6,939.4, n=6, 773,285 tokens | 6,950.6, median 7,028.7, best 7,084.6, n=9, 1,295,598 tokens |
| Fresh reads ≥ 20k | 6,718.0, n=13 | 6,809.5, median 6,830.0, n=20 |
| Drafts, gen ≥ 200 | 66.7% | 61.7% |
| Expert-cache hit, gen ≥ 200 | 91.4% | 92.6% |

On this traffic decode is about 1% higher, flat in the 100k–200k band, and about 2% higher at 50k–90k. Long fresh reads are about 1.5% higher. Draft acceptance is about 5 points lower. The expert profile loaded for FP4-Local is still the ranking built on the plain NVFP4 pack. This session hit the cache a point more often than the plain pack did, so the old ranking was not what held this traffic back.

### Quality notes that are not this host

NVIDIA's card for FP4-Local, against the BF16 model: GPQA Diamond 91.5% versus 92.2%, ToolEvalBench 88.7% versus 89.4%. That card is not a comparison with the plain NVFP4 checkpoint.

Strata ranks their GPTQ NVFP4 pack as the best quality of the three, and calls QAD slightly faster on short prompts when every expert is in VRAM. On an RTX PRO 5000 72 GB they measured QAD at 159 tok/s decode, a 32k prompt at 4,916 tok/s, and an agent session at 151 tok/s. Against Unsloth Q8_0, their `quant_quality.py` set v1 gave QAD top-1 90.5% and mean KL 0.085, and the plain NVFP4 path top-1 91.79% and KL 0.069 with tail skip off. They also say the QAD distance includes the retrained weights, so KL to Q8_0 is a weak score for a model that was deliberately changed. Those two tables are not the same test, and neither is this 32 GB card.

---

## Controlled benches

These prompts were sent on purpose. They are not the agent rates above. The client message body was empty on the 0.2.0 greedy runs because the 220 tokens were still inside the thinking block (`finish_reason` length). That is a speed measurement, not a comparison of answer text.

### Speed recipe, 0.1.45, bench on the 10:18 boot

| Prompt | Prefill | Decode |
| --- | --- | --- |
| 261,012 fresh tokens | **7,846.3** tok/s (33,266 ms) | 128 tokens at 145.2 tok/s, drafts 84/115, expert hit 91.0% |
| same prompt, warm (7 fresh tokens) | | 110 tokens at 162.3; 188 tokens at 161.3 |
| 32,779 fresh tokens | **8,046.3** tok/s (4,074 ms) | 32 tokens at 151.5 |
| 67-token pangram-style | | 200 tokens at **206.8** tok/s, drafts 148/154 |

The 8,046 row was at the 600 W limit. The next 32,792-token request under a 500 W cap prefilled at 7,666.1 tok/s. The cap was removed. Repeated short pangrams accept drafts nearer 80% and decode in the 175–200 tok/s range. Real tool-calling text does not.

### 19 workers against 13, plain NVFP4, 0.2.0, 20:07 UTC

Same greedy prompt both times. 138,235 tokens, nothing reused, then a follow-up that added "Again." and reused 131,072. w4a8, tail skip 0, context 262,144, `auto:32768` stayed put.

| | 19 workers | 13 workers |
| --- | --- | --- |
| Cold read 138,235 | 7,245.8 tok/s (19,078 ms) | 7,251.7 tok/s (19,063 ms) |
| Cold decode, 220 tokens | 134.6 tok/s, drafts 101/142 | 137.8 tok/s, drafts 101/142 |
| Warm follow-up, 7,166 read | 4,067.0 tok/s | 4,070.2 tok/s |
| Warm decode, 220 tokens | 181.9 tok/s, drafts 122/150 | 185.2 tok/s, drafts 122/150 |

13 workers was 2.4% faster on the cold decode and 1.8% on the warm one. Prefill did not move. Draft counts matched. Under the 3% bar, so the config stayed at 19 that evening. The warm 181 tok/s is this repetitive bench, not the 146.5 tok/s agent rate. A later five-run sweep on 2026-10-21, different prompts, kept 13. This table is the 20:07 bench and is not relabeled.

### FP4-Local against plain NVFP4, same greedy prompt, 19 workers

| | Plain NVFP4, 20:07 | FP4-Local, 20:30 |
| --- | --- | --- |
| Cold read 138,235 | 7,245.8 tok/s (19,078 ms) | 6,991.8 tok/s (19,771 ms), 3.5% slower |
| Cold decode, 220 tokens | 134.6 tok/s, drafts 101/142 (71.1%) | 135.2 tok/s, drafts 112/153 (73.2%) |
| Cold expert-cache hit | 91.5% | 91.5% |
| Warm read 7,166 | 4,067.0 tok/s (1,762 ms) | 3,678.9 tok/s (1,948 ms), 9.5% slower |
| Warm decode, 220 tokens | 181.9 tok/s, drafts 122/150 (81.3%) | 164.0 tok/s, drafts 115/164 (70.1%), 9.8% slower |
| Warm expert-cache hit | 95.7% | 95.7% |

Cold decode is level. The warm gap follows draft acceptance on this repetitive prompt. The agent table above is the one that decided the running server: on real sessions the two packs decode at the same rate.

---

## 1,048,576 context with YaRN, not kept

Real agent traffic on the quality stack, 2026-10-17 13:44–14:20. Flags added: `--max-context 1048576 --rope-scaling yarn --rope-scale 4`. w4a8, skip 0, prefill 32768, and kv-resident 32768 stayed. YaRN applies to every position, including inside the trained 262,144 window. Host K/V went from 3.09 GiB to 12.38 GiB. Expert slots fell from 9,079 to 8,556.

| | 262,144 quality (11:59) | 1,048,576 YaRN (13:44) |
| --- | --- | --- |
| Prompts 100k–200k, gen ≥ 200 | **135.1** tok/s, median 132.7, n=269, drafts 69.5% | **119.7** tok/s, median 120.9, n=30, drafts 67.3% |
| Fresh reads ≥ 100k | **6,916.8** (n=55, best 7,379.4) | **6,568.9** (n=11, best 7,498.2) |

Decode on the prompts this host actually serves was about 11% slower. A 717,131-token prompt (618,827 fresh) prefilled at 5,580 tok/s and decoded at 100.2 tok/s. Reverted. The running context is 262,144.
