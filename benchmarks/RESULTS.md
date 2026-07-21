# sr_solve GPU optimization — benchmark results

Microbenchmark of `qilisdk::gpu::sr_solve` (the device-resident stochastic
reconfiguration solve called once per ODE RHS evaluation in `lindblad_rhs_gpu`).

## Setup

- **GPU:** NVIDIA GeForce RTX 3050 Laptop (GA107, consumer — FP64 at ~1/64 of FP32).
- **CPU baseline:** Eigen `LLT`, `-O3 -march=x86-64-v3` (AVX2+FMA), double precision.
- **Harness:** `benchmarks/bench_sr_solve.cpp`, 100 timed GPU iters + 10 warmup per size,
  each size validated against the Eigen CPU reference (all **PASS**, max err < 1e-9).
- **Method:** identical harness built against `gpu.cpp` at `HEAD~1` (before, per-call
  `DeviceBuffer`) vs `HEAD` (after, persistent `DevicePool`).

## Result 1 — wall-clock latency (median µs per solve)

| N_s | p | before (µs) | after (µs) | Δ | GPU vs CPU (after) |
|----:|--:|----:|----:|--:|--:|
| 1000 | 32  | 975.9  | 958.4  | −1.8% | 0.07× (CPU wins) |
| 1000 | 64  | 546.5  | 652.1  | noise | 0.29× |
| 1000 | 128 | 1403.0 | 1123.9 | −20% (mostly noise) | 0.67× |
| 1000 | 256 | 3164.6 | 3072.0 | −2.9% | 0.84× |
| 1000 | 512 | 9271.4 | 9013.3 | −2.8% | 1.36× |
| 4000 | 256 | 8602.8 | 8370.4 | −2.7% | 1.64× |
| 4000 | 512 | 28266  | 28113  | −0.5% | 2.09× |

**Opt #1 gives a consistent but small (~1–3%) latency improvement** — and the best-case
(min) times at 1000:128 are near-identical (1109 → 1101 µs), i.e. <1%. The GPU only beats
the CPU at large sizes; for small/medium problems the FP64 consumer GPU loses to AVX2.

## Result 2 — the point of opt #1: CUDA API call counts (nsys, N_s=1000 p=128, 111 solves)

| API | before (calls) | after (calls) | per-solve |
|-----|---:|---:|----|
| `cudaMalloc` | **1,006** | **16** | ~9 → ~0 |
| `cudaFree`   | **1,002** | **3**  | ~9 → ~0 |

Opt #1 does exactly what it set out to: the per-call allocation churn is gone
(steady-state solves allocate nothing). **But it barely moves wall-clock**, because on
this driver the CUDA caching allocator already served repeated same-size `cudaMalloc`
in ~1 µs (before: median `cudaMalloc` 1.0 µs, `cudaFree` 0.9 µs → ~17 µs/solve of
alloc/free, small next to a ~1 ms+ solve). Opt #1 is still worth keeping: it removes
~2,000 driver round-trips, is robust to allocators *without* caching / under
fragmentation, avoids mid-evolution allocation stalls, and is a prerequisite for pinned
host buffers (opt #4).

## Result 3 — where the time actually goes (nsys, after, per solve)

GPU kernel time is dominated by the two big FP64 kernels:

| Kernel | % GPU time | µs/solve |
|--------|---:|---:|
| Cholesky factor (`getrf_wo_pivot` / potrf) | **48.0%** | 684 |
| GEMM `OᵀO` (`cutlass d884gemm`, FP64) | **40.2%** | 573 |
| triangular solves (potrs `trsm`) | 7.5% | 108 |
| gemv / scal / axpy / syr (level-1/2) | ~4% | ~45 |

On the CPU-side API timeline, **`cudaMemcpy` is 62% of API time** (777 calls; blocking
H2D of `O` up to 1.3 ms from pageable memory + blocking readbacks).

## Conclusion & reprioritization

The measurement reorders the optimization plan from the original review:

1. **FP64 compute is the bottleneck** (Cholesky 48% + GEMM 40% = ~88% of GPU time),
   because this is a consumer GPU with 1/64 FP64. This is why the GPU only wins at large
   sizes.
2. **Opt #2 (`cublasDsyrk` instead of `Dgemm`)** — halves the GEMM flops (~40% → ~20% of
   GPU time) and matches the "lower-triangle only" usage. Best low-risk compute win. **Do next.**
3. **Opt #4 (pinned host memory + async, upload int8 not double)** — attacks the 62% API
   memcpy cost. Now cheap to add since buffers are persistent.
4. **Opt #5 (size-gating)** — for small p the GPU loses to CPU; gate the GPU path on a
   size threshold to avoid regressions.
5. **Opt #1 (this change)** — architecturally correct and kept, but not the perf lever on
   this hardware. Its win would be larger on allocators without caching or under memory pressure.

---

# Optimization #2 — `cublasDsyrk` for the Gram matrix

Replaced `cublasDgemm`(full `OᵀO`) + `Dscal` with a single `cublasDsyrk` (lower triangle,
`alpha=1/N_s`). Min-of-median GPU latency, 3 interleaved rounds (to cancel laptop-GPU
clock drift), µs/solve:

| N_s | p | before (dgemm) | after (dsyrk) | change |
|----:|--:|---:|---:|---:|
| 1000 | 64  | 440.5 | 698.3 | **+58.5%** ⚠ |
| 1000 | 128 | 1104.9 | 1105.2 | +0.0% |
| 1000 | 256 | 3060.4 | 3042.4 | −0.6% |
| 1000 | 512 | 8961.1 | 8947.5 | −0.2% |
| 4000 | 256 | 8330.7 | 8301.4 | −0.4% |
| 4000 | 512 | 27612.5 | 19470.0 | **−29.5%** ✅ |

nsys (both use FP64 **tensor-core** cutlass kernels, so this is a kernel-tuning effect):

| size | `d884gemm` | `d884syrk_lower` |
|---|---:|---:|
| 4000:512 | 21.66 ms | **13.51 ms** (−38%) |
| 1000:64  | 149 µs   | **574 µs** (3.85× slower) |

**Verdict:** cuBLAS FP64 `syrk` is well-tuned for large matrices (halved flops → −29.5% at
4000:512) but poorly tuned for small/thin shapes (+58% at 1000:64). The regression is
confined to small p — a regime where the GPU is already ~3.5× slower than the CPU and will
be routed to the CPU by opt #5 (size-gating). Kept, because it is neutral-to-large-win in
the regime where the GPU path is actually worthwhile.

---

# Optimization #3 — fuse RHS into one gemm + ger — **TRIED, REVERTED**

Replaced the force-vector assembly's 2 `gemv` + 2 `scal` + 2 `axpy` (6 launches) with one
`gemm` (`B = -(1/N_s) Oᵀ[El_re|El_im]`, both columns) + one `ger` (mean correction).
Min-of-median µs/solve (3 interleaved rounds):

| N_s | p | opt#2 | opt#3 | change |
|----:|--:|---:|---:|---:|
| 1000 | 64  | 696.0 | 710.7 | +2.1% |
| 1000 | 128 | 1101.2 | 1118.9 | +1.6% |
| 1000 | 256 | 3044.8 | 3049.4 | +0.2% |
| 1000 | 512 | 8938.9 | 8922.3 | −0.2% |
| 4000 | 256 | 8317.9 | 8325.7 | +0.1% |
| 4000 | 512 | 19468.1 | 20571.6 | **+5.9%** (confirmed, 5 rounds, non-overlapping ranges) |

**Verdict: reverted.** No benefit at any size and a stable +5.9% regression at 4000:512.
Two reasons, both matching the profile: (1) kernel launches are only ~3% of API time here,
so cutting them can't help; (2) the fused `gemm` has `n=2` — far too thin to fill the FP64
tensor-core tile — so it is *slower* than the two purpose-built `gemv` kernels it replaced.
Reducing launch count is not a useful lever on this compute-bound workload.

---

# Optimization #4 — pinned host staging for the O upload — **TRIED, REVERTED**

Staged `O` through a pinned (page-locked) host buffer (`cudaMallocHost`) so the dominant H2D
DMA runs at full PCIe bandwidth. Min-of-median µs/solve (4 interleaved rounds):

| N_s | p | opt#2 (pageable) | opt#4 (pinned) | change |
|----:|--:|---:|---:|---:|
| 1000 | 128 | 1103.1 | 1170.5 | +6.1% |
| 1000 | 512 | 8932.0 | 9285.4 | +4.0% |
| 4000 | 256 | 8345.0 | 8975.7 | +7.6% |
| 4000 | 512 | 19462.0 | 20203.8 | +3.8% |

**Verdict: reverted.** A consistent regression. The staging approach does `memcpy`
(Eigen→pinned) then `cudaMemcpy` (pinned→device) *serially*, whereas the driver's pageable
path already overlaps its internal staging copy with the DMA in chunks — so explicit staging
just adds a serial CPU copy the driver was hiding. A genuine transfer win needs one of:
(a) uploading `O` as `int8` (8× less data — but needs a device-side conversion kernel, which
breaks the dlopen-only / no-nvcc design), (b) the caller handing `O` already in pinned
memory, or (c) cross-`sr_solve` async pipelining (needs integrator restructuring). None are
available within this PR's constraints.

---

# Optimization #5 — size-gating — **KEPT (biggest practical win)**

`sr_solve` now declines problems below a work threshold (`~ N_s·p²`, the Gram-matrix flop
count; default `1.5e8`, calibrated on the RTX 3050, override via `QILISDK_GPU_MIN_WORK`) so
the caller falls back to Eigen where the CPU is faster. Effective solve time (µs), gate OFF
(GPU forced for all sizes) vs gate ON (default):

| N_s | p | N_s·p² | gate OFF (GPU) | gate ON (effective) | result |
|----:|--:|---:|---:|---:|---|
| 1000 | 64  | 4.1e6  | 1071 (0.19×) | **215 (→CPU)** | **5.0× faster** |
| 1000 | 128 | 1.6e7  | 1492 (0.48×) | **712 (→CPU)** | **2.1× faster** |
| 1000 | 256 | 6.5e7  | 4111 (0.62×) | **2441 (→CPU)** | **1.7× faster** |
| 1000 | 512 | 2.6e8  | 9028 (1.33×) | 9043 (GPU) | unchanged (GPU wins) |
| 4000 | 256 | 2.6e8  | 8393 (1.09×) | 8418 (GPU) | unchanged (GPU wins) |
| 4000 | 512 | 1.05e9 | 19608 (1.80×) | 19676 (GPU) | unchanged (GPU wins) |

**Verdict: kept.** The gate makes the GPU backend a *strict* improvement — it is used only
where it beats the CPU, so small problems (which were 2–5× slower on the GPU) now run on the
CPU at full speed, and large problems keep the GPU win. It also neutralizes opt #2's small-p
`syrk` regression (those sizes are now on the CPU). The `test_gpu.cpp` correctness test sets
`QILISDK_GPU_MIN_WORK=0` to keep exercising the GPU compute path at small sizes.

---

# Overall summary (opt #1–#5)

| # | Optimization | Result | Kept? |
|---|--------------|--------|-------|
| 1 | Persistent device workspace (no per-call malloc/free) | Eliminates ~2000 API calls; ~1–3% latency (driver already cached allocs) | ✅ kept |
| 2 | `cublasDsyrk` for `OᵀO` | −29.5% at 4000:512; +58% at 1000:64 (cuBLAS syrk poorly tuned for thin shapes) | ✅ kept |
| 3 | Fuse RHS into one `gemm`+`ger` | No gain; +5.9% at 4000:512 (thin n=2 gemm underfills tensor core) | ❌ reverted |
| 4 | Pinned host staging for `O` | +4–8% regression (serial staging vs driver's overlapped bounce) | ❌ reverted |
| 5 | Size-gating (small p → CPU) | 1.7–5× faster on small problems; large unchanged | ✅ kept (biggest practical win) |

**Key lesson from the data:** on this consumer laptop GPU the SR solve is *FP64-compute-bound*
(Cholesky 48% + GEMM 40% of GPU time; consumer FP64 runs at 1/64 rate). Only two things
helped: reducing FP64 flops for large matrices (**#2**) and *not using the GPU* where it
loses (**#5**). The peripheral overheads (allocations, launches, transfer staging) that the
initial review flagged turned out to be negligible-to-counterproductive here — a reminder to
profile before optimizing. On a datacenter GPU (full-rate FP64) the crossover would move down
substantially; lower `QILISDK_GPU_MIN_WORK` there.
