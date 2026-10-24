# The machine

```
HOST RAM  128 GB DDR5-5200 (4×32). MemTotal about 123 GiB. Swap 32 GB.
├── 63.28 GiB   pinned expert arena (experts.bin), both NVFP4 packs
├── 47.68 GiB   FP8 PLE, plain NVFP4 pack, read from the GGUF
├── 26.82 GiB   NVFP4 PLE, FP4-Local pack, read from the GGUF
├── 3.09 GiB    pinned K/V for the 32,768-cell resident window
└── OS, page cache, the server

RTX 5090  32,607 MiB, compute capability 12.0, power limit 600 W
├── expert cache auto → 9,075 slots on 0.2.0 (9,079 on 0.1.45), about 23.4 GiB
├── native projections and the MTP draft head
├── prompt path borrows about 5,218 cache slots (about 13.4 GiB) during prefill
└── about 141 MiB CUDA free after load, with --vram-reserve-mib 512
```

- Board: ASUS ProArt Z890. CPU: Intel Core Ultra 7 265K, 8 P-cores (CPUs 0–7) and 12 E-cores (CPUs 8–19). No AVX-512. Expert kernels run AVX2.
- GPU driver: open kernel module 615.71.09, the Ubuntu 26.04 apt package. CUDA toolkit 13.4.3, nvcc 13.4.92. CUDA 13.2.0 and 13.2.1 are known-bad for sm_120 in this Strata tree.
- Kernel: `7.0.0-38-generic`. `vm.nr_hugepages=0`, so the arena is `cudaHostRegister` of ordinary pages. `MAP_HUGETLB` is unavailable.
- PCIe: boot probe about 46 GB/s host-to-device. `pcie_frac` starts at 0.25 and settles near 0.30.
- `cpu-freq-limit.service` caps every policy at 4.5 GHz. Package temperature under decode stays about 73–83°C. On 50k–90k prompts the cap did not change decode. It is a thermal limit, not one of the recipe flags.
- P-core energy preference is `performance` on CPUs 0–7 only. CPUs 8–19 stay `balance_performance`.
- A 500 W GPU power cap cut a 32,779-token prefill from 8,046 tok/s to 7,666 tok/s on the speed recipe. The card stays at 600 W. Decode of a few hundred tokens drew about 220–370 W, so the cap was not a decode problem.

The 0.1.45 tree is tag v0.1.45, commit `477ac13`. The 0.2.0 tree is tag v0.2.0-beta, commit `3d1fb480`, engine package string `0.1.45.3`, `BUILD.json` architectures `[120]`. llama.cpp is the pin both trees name: `3cf03257f219afbe7334045ff7c6a06ac68c627d`. The configs and the unit refer to the active tree as `/opt/strata/Strata`.
