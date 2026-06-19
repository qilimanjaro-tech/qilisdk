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
#pragma once

#include "../../../libs/eigen.h"  // adjust if eigen typedefs live elsewhere

// ---------------------------------------------------------------------------
// Optional GPU acceleration shim.
//
// DESIGN: this header intentionally exposes NO CUDA types. The implementation
// (cuda_solver.cpp) loads libcudart / libcublas / libcusolver lazily via
// dlopen at runtime, so the wheel has ZERO link-time CUDA dependency and runs
// unchanged on machines without a GPU. Every entry point degrades gracefully:
// if the GPU path is unavailable or fails for any reason, the function returns
// false and the caller MUST fall back to the existing Eigen (CPU) path.
//
// This is the single seam through which all GPU offloading flows. Keep the
// surface tiny: just the two ops the variational solve needs today.
// ---------------------------------------------------------------------------

namespace qilisim::gpu {

// Returns true if a usable CUDA device + the cublas/cusolver shared libraries
// were found at runtime. Result is probed once and cached (thread-safe).
// Cheap to call repeatedly; safe to call when no GPU is present.
bool cuda_available();

// Compute C = A * B for real double matrices on the GPU (cuBLAS dgemm).
//   A: m x k, B: k x n, C: m x n (resized by callee).
// Returns false (and leaves C untouched) if the GPU path could not run, in
// which case the caller should compute C = A * B with Eigen instead.
bool gemm(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& C);

// Compute the Gram matrix G = Aᵀ * A (G: n x n) for a real m x n matrix A on the
// GPU. Unlike gemm(Aᵀ, A, .), this uploads A only ONCE (cuBLAS dgemm reads the
// same device buffer as both operands, with the transpose flag on the first),
// halving host->device traffic and avoiding a host-side transpose. G is fully
// populated (both triangles) and resized by the callee. Returns false (G
// untouched) if the GPU path could not run; caller should fall back to Eigen.
bool gram_ata(const Eigen::MatrixXd& A, Eigen::MatrixXd& G);

// Solve the SPD system A * X = B via Cholesky on the GPU (cuSOLVER potrf+potrs).
//   A: n x n symmetric positive-definite, B: n x nrhs, X: n x nrhs (resized).
// `A` is assumed already regularised by the caller. Returns false (X untouched)
// if the GPU path could not run; caller should fall back to Eigen::LLT.
bool cholesky_solve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& X);

// Fused, fully device-resident stochastic-reconfiguration step. Given the
// sample-operator matrix O (N_s x p, real ±1 cast to double, column-major) and
// the per-sample local energies El (length N_s, complex), compute
//   adot = M^{-1} V   with   M = OᵀO/N_s - ōōᵀ + εI,   V = -(Oᵀ El/N_s - ō Ēl)
// keeping every intermediate (means ō, Gram OᵀO, M, V, Cholesky factor) on the
// device: only O and El are uploaded, only adot (length p) is read back. This
// avoids the host round-trips of calling gram_ata + cholesky_solve separately.
// `adot` is resized to length p. Returns false (adot untouched) if the GPU path
// could not run (missing symbols, allocation failure, or non-SPD M), in which
// case the caller must fall back to the CPU implementation.
bool sr_solve(const Eigen::MatrixXd& O, const Eigen::VectorXcd& El, double epsilon, Eigen::VectorXcd& adot);

}  // namespace qilisim::gpu
