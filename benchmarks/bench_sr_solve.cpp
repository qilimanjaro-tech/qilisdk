// Copyright 2026 Qilimanjaro Quantum Tech
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ---------------------------------------------------------------------------
// Standalone microbenchmark for qilisdk::gpu::sr_solve.
//
// Measures the wall-clock latency of the device-resident stochastic
// reconfiguration solve (the hot RHS-evaluation kernel in lindblad_rhs_gpu)
// across a sweep of problem sizes (N_s = number of samples, p = number of
// variational parameters), and contrasts it against the Eigen CPU reference
// it falls back to. Every timed size is first validated for correctness
// against that same reference, so a single run confirms both correctness and
// performance.
//
// Because sr_solve loads CUDA lazily via dlopen, this binary has NO link-time
// CUDA dependency: it builds and runs unchanged on a machine without a GPU
// (where it simply reports the GPU path as unavailable and times only the CPU
// reference).
//
// Configuration (all optional, via environment variables):
//   BENCH_SIZES   comma-separated "N_s:p" pairs
//                 (default: 1000:32,1000:64,1000:128,1000:256,1000:512,
//                           4000:256,4000:512)
//   BENCH_ITERS   timed iterations per size for the GPU path (default: 100)
//   BENCH_WARMUP  warmup iterations excluded from timing (default: 10)
//   BENCH_CPU_ITERS  timed iterations for the CPU reference (default: 20)
//   BENCH_CSV     if set, append machine-readable results to this CSV path
//   BENCH_LABEL   free-form label recorded in the CSV (e.g. "before"/"after")
//
// For an API-level view (e.g. counting cudaMalloc/cudaFree calls to see the
// per-call allocation churn), profile the steady-state loop with:
//   nsys profile --stats=true ./bench_sr_solve
// ---------------------------------------------------------------------------

#include "gpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

using cx = std::complex<double>;
using Clock = std::chrono::steady_clock;

// Reference stochastic-reconfiguration solve on the CPU. This is a verbatim
// mirror of sr_reference in tests/unit_cpp/qilisim/test_gpu.cpp and of the
// Eigen fallback in lindblad.cpp:
//   adot = M^{-1} V  with  M = OᵀO/N_s - ōōᵀ + εI  and
//   V = -(Oᵀ El/N_s - ō Ēl).
Eigen::VectorXcd sr_reference(const Eigen::MatrixXd& O, const Eigen::VectorXcd& El, double eps) {
    const int N_s = static_cast<int>(O.rows());
    const int p = static_cast<int>(O.cols());
    const Eigen::VectorXd om = O.colwise().mean();
    const cx Elm = El.mean();
    Eigen::MatrixXd M = O.transpose() * O / static_cast<double>(N_s) - om * om.transpose();
    M += eps * Eigen::MatrixXd::Identity(p, p);
    const Eigen::VectorXcd V = -((O.transpose().cast<cx>() * El) / static_cast<double>(N_s) - om.cast<cx>() * Elm);
    Eigen::LLT<Eigen::MatrixXd> llt(M);
    Eigen::VectorXcd adot(p);
    adot.real() = llt.solve(V.real());
    adot.imag() = llt.solve(V.imag());
    return adot;
}

// Deterministic ±1 sample-operator matrix (stand-in for samples.O_mat), matching
// the generator used by the unit tests so results are reproducible run-to-run.
Eigen::MatrixXd make_pm1(int N_s, int p) {
    Eigen::MatrixXd O(N_s, p);
    for (int i = 0; i < N_s; ++i) {
        for (int j = 0; j < p; ++j) {
            O(i, j) = (((i * 31 + j * 17) % 2) == 0) ? 1.0 : -1.0;
        }
    }
    return O;
}

// Deterministic complex local-energy vector.
Eigen::VectorXcd make_El(int N_s) {
    Eigen::VectorXcd El(N_s);
    for (int i = 0; i < N_s; ++i) {
        El(i) = cx(0.3 * i - 5.0, 0.1 * i + 1.0);
    }
    return El;
}

struct Stats {
    double mean = 0.0;
    double median = 0.0;
    double min = 0.0;
    double p90 = 0.0;
};

// Summarize a set of per-iteration durations (microseconds). Mutates (sorts) v.
Stats summarize(std::vector<double>& v) {
    Stats s;
    if (v.empty()) {
        return s;
    }
    std::sort(v.begin(), v.end());
    double sum = 0.0;
    for (double x : v) {
        sum += x;
    }
    const std::size_t n = v.size();
    s.mean = sum / static_cast<double>(n);
    s.median = v[n / 2];
    s.min = v.front();
    s.p90 = v[static_cast<std::size_t>(0.9 * static_cast<double>(n - 1))];
    return s;
}

std::vector<std::pair<int, int>> parse_sizes(const std::string& spec) {
    std::vector<std::pair<int, int>> out;
    std::size_t i = 0;
    while (i < spec.size()) {
        std::size_t comma = spec.find(',', i);
        if (comma == std::string::npos) {
            comma = spec.size();
        }
        const std::string tok = spec.substr(i, comma - i);
        const std::size_t colon = tok.find(':');
        if (colon != std::string::npos) {
            const int ns = std::atoi(tok.substr(0, colon).c_str());
            const int p = std::atoi(tok.substr(colon + 1).c_str());
            if (ns > 0 && p > 0) {
                out.emplace_back(ns, p);
            }
        }
        i = comma + 1;
    }
    return out;
}

const char* env_or(const char* key, const char* fallback) {
    const char* v = std::getenv(key);
    return (v && *v) ? v : fallback;
}

int env_int(const char* key, int fallback) {
    const char* v = std::getenv(key);
    return (v && *v) ? std::atoi(v) : fallback;
}

}  // namespace

int main() {
    const std::string sizes_spec =
        env_or("BENCH_SIZES", "1000:32,1000:64,1000:128,1000:256,1000:512,4000:256,4000:512");
    const int iters = env_int("BENCH_ITERS", 100);
    const int warmup = env_int("BENCH_WARMUP", 10);
    const int cpu_iters = env_int("BENCH_CPU_ITERS", 20);
    const std::string csv_path = env_or("BENCH_CSV", "");
    const std::string label = env_or("BENCH_LABEL", "");

    const std::vector<std::pair<int, int>> sizes = parse_sizes(sizes_spec);

    const bool gpu = qilisdk::gpu::cuda_available();

    std::printf("=====================================================================\n");
    std::printf(" qilisdk::gpu::sr_solve microbenchmark\n");
    std::printf("   GPU path : %s\n", gpu ? "AVAILABLE" : "UNAVAILABLE (timing CPU reference only)");
    if (!label.empty()) {
        std::printf("   label    : %s\n", label.c_str());
    }
    std::printf("   iters    : %d (gpu), %d (cpu)   warmup: %d\n", iters, cpu_iters, warmup);
    std::printf("=====================================================================\n\n");

    std::printf("%7s %6s | %10s %10s %10s %10s | %10s | %8s | %-9s\n", "N_s", "p", "gpu_mean",
                "gpu_med", "gpu_min", "gpu_p90", "cpu_mean", "speedup", "correct");
    std::printf("%7s %6s | %10s %10s %10s %10s | %10s | %8s | %-9s\n", "", "", "(us)", "(us)", "(us)",
                "(us)", "(us)", "(x cpu)", "");
    std::printf("--------------------------------------------------------------------------------------------------\n");

    std::ofstream csv;
    if (!csv_path.empty()) {
        const bool exists = std::ifstream(csv_path).good();
        csv.open(csv_path, std::ios::app);
        if (csv && !exists) {
            csv << "label,gpu_available,N_s,p,correct,max_abs_err,gpu_mean_us,gpu_median_us,"
                   "gpu_min_us,gpu_p90_us,cpu_mean_us,speedup_median\n";
        }
    }

    // Sink to defeat dead-code elimination of the solve results.
    double sink = 0.0;

    for (const auto& dim : sizes) {
        const int N_s = dim.first;
        const int p = dim.second;
        const Eigen::MatrixXd O = make_pm1(N_s, p);
        const Eigen::VectorXcd El = make_El(N_s);
        const double eps = 0.1 / std::sqrt(static_cast<double>(N_s));

        // Correctness: compare a single GPU solve against the CPU reference.
        const Eigen::VectorXcd ref = sr_reference(O, El, eps);
        bool correct = false;
        double max_err = 0.0;
        Eigen::VectorXcd adot;
        if (gpu) {
            if (qilisdk::gpu::sr_solve(O, El, eps, adot) && adot.size() == p) {
                max_err = (adot - ref).cwiseAbs().maxCoeff();
                correct = max_err < 1e-9;
            }
        }

        // Warmup (also pays any first-call cost that is not part of steady state).
        for (int w = 0; w < warmup; ++w) {
            if (gpu) {
                qilisdk::gpu::sr_solve(O, El, eps, adot);
            } else {
                adot = sr_reference(O, El, eps);
            }
        }

        // Timed GPU loop.
        Stats gs;
        if (gpu) {
            std::vector<double> samples;
            samples.reserve(iters);
            for (int it = 0; it < iters; ++it) {
                const auto t0 = Clock::now();
                const bool ok = qilisdk::gpu::sr_solve(O, El, eps, adot);
                const auto t1 = Clock::now();
                if (!ok) {
                    break;  // fell back mid-run; leave gs empty -> reported as n/a
                }
                samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                sink += adot(0).real();
            }
            gs = summarize(samples);
        }

        // Timed CPU reference loop.
        std::vector<double> cpu_samples;
        cpu_samples.reserve(cpu_iters);
        for (int it = 0; it < cpu_iters; ++it) {
            const auto t0 = Clock::now();
            const Eigen::VectorXcd r = sr_reference(O, El, eps);
            const auto t1 = Clock::now();
            cpu_samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            sink += r(0).real();
        }
        const Stats cs = summarize(cpu_samples);

        const double speedup = (gpu && gs.median > 0.0) ? cs.median / gs.median : 0.0;
        const char* correct_str = !gpu ? "n/a" : (correct ? "PASS" : "FAIL");

        if (gpu && gs.median > 0.0) {
            std::printf("%7d %6d | %10.2f %10.2f %10.2f %10.2f | %10.2f | %7.2fx | %-9s\n", N_s, p,
                        gs.mean, gs.median, gs.min, gs.p90, cs.mean, speedup, correct_str);
        } else {
            std::printf("%7d %6d | %10s %10s %10s %10s | %10.2f | %8s | %-9s\n", N_s, p, "n/a", "n/a",
                        "n/a", "n/a", cs.mean, "n/a", correct_str);
        }

        if (csv) {
            csv << label << ',' << (gpu ? 1 : 0) << ',' << N_s << ',' << p << ','
                << (correct ? 1 : 0) << ',' << max_err << ',' << gs.mean << ',' << gs.median << ','
                << gs.min << ',' << gs.p90 << ',' << cs.mean << ',' << speedup << '\n';
        }
    }

    std::printf("\n(checksum: %.6f)\n", sink);
    if (!gpu) {
        std::printf(
            "\nNOTE: No CUDA device/driver reachable at runtime, so sr_solve took the CPU\n"
            "      fallback. Re-run on a machine with a working NVIDIA driver to measure the\n"
            "      GPU path. (Hardware present but driver down? check `nvidia-smi`.)\n");
    }
    return 0;
}
