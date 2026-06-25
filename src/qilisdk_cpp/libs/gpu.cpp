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
#include "gpu.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

// For now no Windows GPU support because god knows that will cause problems
#if !defined(_WIN32)
#include <dirent.h>
#include <dlfcn.h>
#endif

// GCOVR_EXCL_START

namespace qilisdk::gpu {

// CUDA C-ABI function-pointer typedefs
// We declare ONLY the symbols we call, so we never need the CUDA headers
namespace {

// Status constants
constexpr int kCudaSuccess = 0;
constexpr int kCublasStatusSuccess = 0;
constexpr int kCusolverStatusSuccess = 0;
constexpr int kMemcpyHostToDevice = 1;
constexpr int kMemcpyDeviceToHost = 2;
constexpr int kCublasOpN = 0;
constexpr int kCublasOpT = 1;
constexpr int kFillModeLower = 0;

// libcudart
using cudaGetDeviceCount_t = int (*)(int*);
using cudaMalloc_t = int (*)(void**, std::size_t);
using cudaFree_t = int (*)(void*);
using cudaMemcpy_t = int (*)(void*, const void*, std::size_t, int);
using cudaDeviceSynchronize_t = int (*)();

// libcublas
using cublasCreate_t = int (*)(void**);
using cublasDestroy_t = int (*)(void*);
using cublasDgemm_t = int (*)(void* handle, int transa, int transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc);
using cublasDgemv_t = int (*)(void* handle, int trans, int m, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy);
using cublasDscal_t = int (*)(void* handle, int n, const double* alpha, double* x, int incx);
using cublasDaxpy_t = int (*)(void* handle, int n, const double* alpha, const double* x, int incx, double* y, int incy);
using cublasDsyr_t = int (*)(void* handle, int uplo, int n, const double* alpha, const double* x, int incx, double* A, int lda);

// libcusolver
using cusolverDnCreate_t = int (*)(void**);
using cusolverDnDestroy_t = int (*)(void*);
using cusolverDnDpotrf_bufferSize_t = int (*)(void* handle, int uplo, int n, double* A, int lda, int* lwork);
using cusolverDnDpotrf_t = int (*)(void* handle, int uplo, int n, double* A, int lda, double* workspace, int lwork, int* dev_info);
using cusolverDnDpotrs_t = int (*)(void* handle, int uplo, int n, int nrhs, const double* A, int lda, double* B, int ldb, int* dev_info);

// Structure containing all the function pointers and handles we need
struct CudaApi {
    bool loaded = false;
    bool has_device = false;

    void* h_cudart = nullptr;
    void* h_cublas = nullptr;
    void* h_cusolver = nullptr;

    void* cublas_handle = nullptr;
    void* cusolver_handle = nullptr;

    cudaGetDeviceCount_t cudaGetDeviceCount = nullptr;
    cudaMalloc_t cudaMalloc = nullptr;
    cudaFree_t cudaFree = nullptr;
    cudaMemcpy_t cudaMemcpy = nullptr;
    cudaDeviceSynchronize_t cudaDeviceSynchronize = nullptr;

    cublasCreate_t cublasCreate = nullptr;
    cublasDestroy_t cublasDestroy = nullptr;
    cublasDgemm_t cublasDgemm = nullptr;
    cublasDgemv_t cublasDgemv = nullptr;
    cublasDscal_t cublasDscal = nullptr;
    cublasDaxpy_t cublasDaxpy = nullptr;
    cublasDsyr_t cublasDsyr = nullptr;

    cusolverDnCreate_t cusolverDnCreate = nullptr;
    cusolverDnDestroy_t cusolverDnDestroy = nullptr;
    cusolverDnDpotrf_bufferSize_t cusolverDnDpotrf_bufferSize = nullptr;
    cusolverDnDpotrf_t cusolverDnDpotrf = nullptr;
    cusolverDnDpotrs_t cusolverDnDpotrs = nullptr;
};

// Again, no Windows GPU support for now
#if !defined(_WIN32)

void add_subdirs(std::vector<std::string>& out, const std::string& root, const char* prefix, const char* suffix) {
    /*
    Append subdirectory paths to a vector, optionally filtering by prefix and suffix.

    Args:
        out: vector to append the results to
        root: directory to scan
        prefix: optional prefix filter for subdirectory names (nullptr = no filter)
        suffix: optional suffix to append to each result (nullptr = no suffix)
    */
    DIR* d = ::opendir(root.c_str());
    if (!d) {
        return;
    }
    const std::size_t prefix_len = prefix ? std::strlen(prefix) : 0;
    while (const dirent* e = ::readdir(d)) {
        if (e->d_name[0] == '.') {  // skip ".", ".." and hidden entries
            continue;
        }
        if (prefix && std::strncmp(e->d_name, prefix, prefix_len) != 0) {
            continue;
        }
        out.push_back(root + "/" + e->d_name + suffix);
    }
    ::closedir(d);
}

std::vector<std::string> cuda_search_dirs() {
    /*
    Build the list of extra directories to probe for the CUDA shared libraries, in priority order

    Returns:
        std::vector<std::string>: list of directories to probe for CUDA shared libraries
    */
    std::vector<std::string> dirs;

    // Just in case we want to manually set it
    if (const char* p = std::getenv("QILISDK_CUDA_LIB_DIR")) {
        dirs.emplace_back(p);
    }

    // Standard CUDA toolkit environment variables
    for (const char* var : {"CUDA_HOME", "CUDA_PATH"}) {
        if (const char* base = std::getenv(var)) {
            dirs.emplace_back(std::string(base) + "/lib64");
            dirs.emplace_back(std::string(base) + "/lib");
        }
    }

    // Check the venv
    if (const char* venv = std::getenv("VIRTUAL_ENV")) {
        std::vector<std::string> py_dirs;
        add_subdirs(py_dirs, std::string(venv) + "/lib", "python", "");
        for (const std::string& py : py_dirs) {
            add_subdirs(dirs, py + "/site-packages/nvidia", nullptr, "/lib");
        }
    }

    // Other system locations
    dirs.emplace_back("/usr/local/cuda/lib64");
    dirs.emplace_back("/usr/lib/x86_64-linux-gnu");

    return dirs;
}

void* dlopen_first(std::initializer_list<const char*> candidates, const std::vector<std::string>& dirs) {
    /*
    Try a list of candidate sonames, returning the first that loads

    Args:
        candidates: list of sonames to try
        dirs: extra directories to search if the bare sonames don't resolve

    Returns:
        void*: handle to the first library that loaded successfully, or nullptr if none did
    */
    for (const char* name : candidates) {
        if (void* h = ::dlopen(name, RTLD_NOW | RTLD_GLOBAL)) {
            return h;
        }
    }
    for (const std::string& dir : dirs) {
        for (const char* name : candidates) {
            const std::string full = dir + "/" + name;
            if (void* h = ::dlopen(full.c_str(), RTLD_NOW | RTLD_GLOBAL)) {
                return h;
            }
        }
    }
    return nullptr;
}
template <typename Fn>
bool resolve(void* handle, const char* name, Fn& out) {
    /*
    Resolve a symbol from a shared library handle

    Args:
        handle: shared library handle (from dlopen)
        name: symbol name to resolve
        out: reference to a function pointer to populate

    Returns:
        bool: true if the symbol was found and resolved, false otherwise
    */
    out = reinterpret_cast<Fn>(::dlsym(handle, name));
    return out != nullptr;
}

#endif

// Check if CUDA is available and load the symbols once, lazily
const CudaApi& probe() {
    static CudaApi api = [] {
        CudaApi a;

// If Windows, just return an empty CudaApi
#if !defined(_WIN32)

        // Load the libaries, searching the loader's default path first and then the
        // extra locations (pip nvidia wheels, CUDA_HOME, common system dirs).
        const std::vector<std::string> dirs = cuda_search_dirs();
        a.h_cudart = dlopen_first({"libcudart.so", "libcudart.so.13", "libcudart.so.12"}, dirs);
        a.h_cublas = dlopen_first({"libcublas.so", "libcublas.so.13", "libcublas.so.12"}, dirs);
        a.h_cusolver = dlopen_first({"libcusolver.so", "libcusolver.so.12", "libcusolver.so.11"}, dirs);
        if (!a.h_cudart || !a.h_cublas || !a.h_cusolver) {
            // TODO(luke): this should be a logger warning, but for that I need the logger introduced in the stab state PR
            std::cout << "[QiliSim GPU] CUDA libraries unavailable -> using CPU (Eigen)." << std::endl;
            return a;
        }

        // If everything was fine, load the symbols we need
        bool ok = true;
        ok &= resolve(a.h_cudart, "cudaGetDeviceCount", a.cudaGetDeviceCount);
        ok &= resolve(a.h_cudart, "cudaMalloc", a.cudaMalloc);
        ok &= resolve(a.h_cudart, "cudaFree", a.cudaFree);
        ok &= resolve(a.h_cudart, "cudaMemcpy", a.cudaMemcpy);
        ok &= resolve(a.h_cudart, "cudaDeviceSynchronize", a.cudaDeviceSynchronize);
        ok &= resolve(a.h_cublas, "cublasCreate_v2", a.cublasCreate);
        ok &= resolve(a.h_cublas, "cublasDestroy_v2", a.cublasDestroy);
        ok &= resolve(a.h_cublas, "cublasDgemm_v2", a.cublasDgemm);
        ok &= resolve(a.h_cublas, "cublasDgemv_v2", a.cublasDgemv);
        ok &= resolve(a.h_cublas, "cublasDscal_v2", a.cublasDscal);
        ok &= resolve(a.h_cublas, "cublasDaxpy_v2", a.cublasDaxpy);
        ok &= resolve(a.h_cublas, "cublasDsyr_v2", a.cublasDsyr);
        ok &= resolve(a.h_cusolver, "cusolverDnCreate", a.cusolverDnCreate);
        ok &= resolve(a.h_cusolver, "cusolverDnDestroy", a.cusolverDnDestroy);
        ok &= resolve(a.h_cusolver, "cusolverDnDpotrf_bufferSize", a.cusolverDnDpotrf_bufferSize);
        ok &= resolve(a.h_cusolver, "cusolverDnDpotrf", a.cusolverDnDpotrf);
        ok &= resolve(a.h_cusolver, "cusolverDnDpotrs", a.cusolverDnDpotrs);
        if (!ok) {
            std::cout << "[QiliSim GPU] Failed to resolve one or more CUDA symbols -> using CPU (Eigen)." << std::endl;
            return a;
        }

        // Must have at least one usable device.
        int count = 0;
        if (a.cudaGetDeviceCount(&count) != kCudaSuccess || count < 1) {
            std::cout << "[QiliSim GPU] No CUDA device detected (count=" << count << ") -> using CPU (Eigen)." << std::endl;
            return a;
        }
        a.has_device = true;

        // Create the shared cuBLAS / cuSOLVER handles
        if (a.cublasCreate(&a.cublas_handle) != kCublasStatusSuccess) {
            std::cout << "[QiliSim GPU] cublasCreate failed -> using CPU (Eigen)." << std::endl;
            return a;
        }
        if (a.cusolverDnCreate(&a.cusolver_handle) != kCusolverStatusSuccess) {
            std::cout << "[QiliSim GPU] cusolverDnCreate failed -> using CPU (Eigen)." << std::endl;
            a.cublasDestroy(a.cublas_handle);
            a.cublas_handle = nullptr;
            return a;
        }

        // All was good
        a.loaded = true;

#endif
        return a;
    }();

    return api;
}

std::mutex& gpu_mutex() {
    /*
    Some handles are not safe for concurrent use, so make a mutex
    (i.e. mutex is called to reserve GPU resources before any GPU operation)

    Returns:
        std::mutex&: reference to a static mutex for GPU operations
    */
    static std::mutex m;
    return m;
}

// RAII wrapper for a device buffer, automatically freed on destruction
// (RAII = Resource Acquisition Is Initialization, basicially it's a way of alllocating the GPU memory
// so that it is automatically freed when the object goes out of scope)
struct DeviceBuffer {
    const CudaApi& api;
    void* ptr = nullptr;
    explicit DeviceBuffer(const CudaApi& a) : api(a) {}
    bool alloc(std::size_t bytes) { return api.cudaMalloc(&ptr, bytes) == kCudaSuccess; }
    ~DeviceBuffer() {
        if (ptr) {
            api.cudaFree(ptr);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

}  // namespace

bool cuda_available() {
    /*
    Check if we have everything loaded already

    Setting the QILISDK_DISABLE_GPU environment variable forces this to report
    "unavailable" so callers fall back to the CPU path, regardless of the hardware.
    This is re-checked on every call (it is not cached with the probe), so it can be
    toggled at runtime, e.g. to exercise the CPU fallback on a machine that has a GPU.

    Returns:
        bool: true if a usable CUDA device + the cublas/cusolver shared libraries was found
    */
    if (std::getenv("QILISDK_DISABLE_GPU")) {
        return false;
    }
    const CudaApi& api = probe();
    return api.loaded && api.has_device;
}

bool sr_solve(const Eigen::MatrixXd& O, const Eigen::VectorXcd& El, double epsilon, Eigen::VectorXcd& adot) {
    /*
    Solve the linear system (Oᵀ O / N_s - ō ōᵀ + ε I) adot = -(Oᵀ El / N_s - El_mean ō) for adot, where
    ō = (1/N_s) Oᵀ 1 is the column mean of O, and El_mean is the mean of El.

    Args:
        O: Eigen::MatrixXd, input matrix (N_s x p)
        El: Eigen::VectorXcd, complex vector (N_s)
        epsilon: double, regularization parameter
        adot: Eigen::VectorXcd, output vector (p)

    Returns:
        bool: true if the GPU path was taken and succeeded, false otherwise
    */

    // Make sure we have a GPU and that the dimensions are compatible
    if (!cuda_available()) {
        return false;
    }
    const CudaApi& api = probe();
    const int N_s = static_cast<int>(O.rows());
    const int p = static_cast<int>(O.cols());
    if (N_s <= 0 || p <= 0 || static_cast<int>(El.size()) != N_s) {
        return false;
    }
    if (!api.cublasDgemv || !api.cublasDscal || !api.cublasDaxpy || !api.cublasDsyr) {
        return false;  // resident path needs the level-1/2 BLAS symbols
    }
    std::lock_guard<std::mutex> guard(gpu_mutex());

    // Some precomputations on the CPU side
    const Eigen::VectorXd El_re = El.real();
    const Eigen::VectorXd El_im = El.imag();
    const double El_mean_re = El_re.mean();
    const double El_mean_im = El_im.mean();
    const int ones_len = (N_s > p) ? N_s : p;  // serves both the mean (N_s) and diagonal (p)
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(ones_len);
    const double one = 1.0, zero = 0.0, neg_one = -1.0;
    const double inv_Ns = 1.0 / static_cast<double>(N_s);
    const double neg_inv_Ns = -inv_Ns;
    const std::size_t szd = sizeof(double);

    // Copy everything to GPU
    DeviceBuffer dO(api), dOnes(api), dOmean(api), dM(api), dElr(api), dEli(api), dB(api), dWork(api), dInfo(api);
    if (!dO.alloc(szd * static_cast<std::size_t>(N_s) * p) || !dOnes.alloc(szd * static_cast<std::size_t>(ones_len)) || !dOmean.alloc(szd * static_cast<std::size_t>(p)) || !dM.alloc(szd * static_cast<std::size_t>(p) * p) || !dElr.alloc(szd * static_cast<std::size_t>(N_s)) || !dEli.alloc(szd * static_cast<std::size_t>(N_s)) || !dB.alloc(szd * static_cast<std::size_t>(p) * 2) || !dInfo.alloc(sizeof(int))) {
        return false;
    }
    if (api.cudaMemcpy(dO.ptr, O.data(), szd * static_cast<std::size_t>(N_s) * p, kMemcpyHostToDevice) != kCudaSuccess || api.cudaMemcpy(dOnes.ptr, ones.data(), szd * static_cast<std::size_t>(ones_len), kMemcpyHostToDevice) != kCudaSuccess || api.cudaMemcpy(dElr.ptr, El_re.data(), szd * static_cast<std::size_t>(N_s), kMemcpyHostToDevice) != kCudaSuccess || api.cudaMemcpy(dEli.ptr, El_im.data(), szd * static_cast<std::size_t>(N_s), kMemcpyHostToDevice) != kCudaSuccess) {
        return false;
    }

    // Run the various routines
    auto* O_d = static_cast<double*>(dO.ptr);
    auto* ones_d = static_cast<double*>(dOnes.ptr);
    auto* omean_d = static_cast<double*>(dOmean.ptr);
    auto* M_d = static_cast<double*>(dM.ptr);
    auto* B_d = static_cast<double*>(dB.ptr);
    void* H = api.cublas_handle;
    bool ok = true;
    // ō = (1/N_s) Oᵀ·1
    ok &= api.cublasDgemv(H, kCublasOpT, N_s, p, &inv_Ns, O_d, N_s, ones_d, 1, &zero, omean_d, 1) == kCublasStatusSuccess;
    // M = OᵀO ; M /= N_s ; M -= ōōᵀ (lower) ; M += εI
    ok &= api.cublasDgemm(H, kCublasOpT, kCublasOpN, p, p, N_s, &one, O_d, N_s, O_d, N_s, &zero, M_d, p) == kCublasStatusSuccess;
    ok &= api.cublasDscal(H, p * p, &inv_Ns, M_d, 1) == kCublasStatusSuccess;
    ok &= api.cublasDsyr(H, kFillModeLower, p, &neg_one, omean_d, 1, M_d, p) == kCublasStatusSuccess;
    ok &= api.cublasDaxpy(H, p, &epsilon, ones_d, 1, M_d, p + 1) == kCublasStatusSuccess;  // diagonal stride p+1
    // V_re = -(Oᵀ El_re / N_s - El_mean_re·ō)  -> B column 0
    ok &= api.cublasDgemv(H, kCublasOpT, N_s, p, &one, O_d, N_s, static_cast<double*>(dElr.ptr), 1, &zero, B_d, 1) == kCublasStatusSuccess;
    ok &= api.cublasDscal(H, p, &neg_inv_Ns, B_d, 1) == kCublasStatusSuccess;
    ok &= api.cublasDaxpy(H, p, &El_mean_re, omean_d, 1, B_d, 1) == kCublasStatusSuccess;
    // V_im -> B column 1
    ok &= api.cublasDgemv(H, kCublasOpT, N_s, p, &one, O_d, N_s, static_cast<double*>(dEli.ptr), 1, &zero, B_d + p, 1) == kCublasStatusSuccess;
    ok &= api.cublasDscal(H, p, &neg_inv_Ns, B_d + p, 1) == kCublasStatusSuccess;
    ok &= api.cublasDaxpy(H, p, &El_mean_im, omean_d, 1, B_d + p, 1) == kCublasStatusSuccess;
    if (!ok) {
        return false;
    }

    // Cholesky factor + solve, in place on M_d / B_d (lower triangle).
    int lwork = 0;
    if (api.cusolverDnDpotrf_bufferSize(api.cusolver_handle, kFillModeLower, p, M_d, p, &lwork) != kCusolverStatusSuccess) {
        return false;
    }
    if (lwork > 0 && !dWork.alloc(szd * static_cast<std::size_t>(lwork))) {
        return false;
    }
    int host_info = 0;
    if (api.cusolverDnDpotrf(api.cusolver_handle, kFillModeLower, p, M_d, p, static_cast<double*>(dWork.ptr), lwork, static_cast<int*>(dInfo.ptr)) != kCusolverStatusSuccess) {
        return false;
    }
    if (api.cudaMemcpy(&host_info, dInfo.ptr, sizeof(int), kMemcpyDeviceToHost) != kCudaSuccess || host_info != 0) {
        return false;  // M not SPD -> caller falls back to Eigen
    }
    if (api.cusolverDnDpotrs(api.cusolver_handle, kFillModeLower, p, 2, M_d, p, B_d, p, static_cast<int*>(dInfo.ptr)) != kCusolverStatusSuccess) {
        return false;
    }
    if (api.cudaMemcpy(&host_info, dInfo.ptr, sizeof(int), kMemcpyDeviceToHost) != kCudaSuccess || host_info != 0) {
        return false;
    }
    if (api.cudaDeviceSynchronize() != kCudaSuccess) {
        return false;
    }

    // Read back only adot (the two solution columns are the real/imag parts).
    Eigen::MatrixXd Bres(p, 2);
    if (api.cudaMemcpy(Bres.data(), dB.ptr, szd * static_cast<std::size_t>(p) * 2, kMemcpyDeviceToHost) != kCudaSuccess) {
        return false;
    }
    adot.resize(p);
    adot.real() = Bres.col(0);
    adot.imag() = Bres.col(1);
    return true;
}

}  // namespace qilisdk::gpu

// GCOVR_EXCL_STOP
