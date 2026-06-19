// Copyright 2025 Qilimanjaro Quantum Tech
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

#include "lindblad.h"

#include <chrono>
#include <iostream>

#include "../gpu/cuda_solver.h"

// GCOV_EXCL_BR_START

const std::complex<double> imag(0.0, 1.0);

SparseMatrix create_superoperator(const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators) {
    /*
    Form the Lindblad superoperator for the given Hamiltonian and jump operators.

    Args:
        currentH (SparseMatrix): The current Hamiltonian.
        jump_operators (std::vector<SparseMatrix>): The list of jump operators.

    Returns:
        SparseMatrix: The Lindblad superoperator.
    */

    // The superoperator dimension
    long dim = long(currentH.rows());
    long dim_rho = dim * dim;
    SparseMatrix L(dim_rho, dim_rho);

    // The identity
    Triplets iden_entries;
    for (int i = 0; i < dim; ++i) {
        iden_entries.emplace_back(Triplet(i, i, 1.0));
    }
    SparseMatrix iden(dim, dim);
    iden.setFromTriplets(iden_entries.begin(), iden_entries.end());

    // The contribution from the Hamiltonian
    SparseMatrix iden_H = Eigen::KroneckerProductSparse<SparseMatrix, SparseMatrix>(iden, currentH);
    SparseMatrix H_T_iden = Eigen::KroneckerProductSparse<SparseMatrix, SparseMatrix>(currentH.transpose(), iden);
    L += iden_H * std::complex<double>(0, -1);
    L += H_T_iden * std::complex<double>(0, 1);

    // The contribution from the jump operators
    for (const auto& L_k : jump_operators) {
        SparseMatrix L_k_dag = L_k.adjoint();
        SparseMatrix L_k_L_k_dag = L_k_dag * L_k;

        SparseMatrix term1 = Eigen::KroneckerProductSparse<SparseMatrix, SparseMatrix>(L_k, L_k.conjugate());
        SparseMatrix term2 = Eigen::KroneckerProductSparse<SparseMatrix, SparseMatrix>(L_k_L_k_dag, iden) * 0.5;
        SparseMatrix term3 = Eigen::KroneckerProductSparse<SparseMatrix, SparseMatrix>(iden, L_k_L_k_dag.transpose()) * 0.5;

        L += term1;
        L -= term2;
        L -= term3;
    }
    return L;
}

void lindblad_rhs(DenseMatrix& drho, const DenseMatrix& rho, const SparseMatrix& H, const std::vector<SparseMatrix>& jumps, bool is_unitary_on_statevector) {
    /*
    Evaluate the right-hand side of the Lindblad master equation.

    Args:
        drho (DenseMatrix&): The output derivative of the density matrix.
        rho (DenseMatrix): The current density matrix.
        H (SparseMatrix): The Hamiltonian.
        jumps (std::vector<SparseMatrix>): The list of jump operators.
        is_unitary_on_statevector (bool): Whether the evolution is unitary on a state vector.
    */
    if (is_unitary_on_statevector) {
        drho = H * rho;
        drho *= -imag;
    } else {
        DenseMatrix Hrho = H * rho;
        drho = Hrho;
        drho -= Hrho.adjoint();
        drho *= -imag;
        for (const auto& J : jumps) {
            SparseMatrix Jdag = J.adjoint();
            SparseMatrix JdagJ = Jdag * J;
            drho += J * rho * Jdag;
            drho -= 0.5 * (JdagJ * rho + rho * JdagJ);
        }
    }
}

void lindblad_rhs(DenseMatrix& drho, const DenseMatrix& rho, const MatrixFreeHamiltonian& H, const std::vector<SparseMatrix>& jumps, bool is_unitary_on_statevector) {
    /*
    Evaluate the right-hand side of the Lindblad master equation.

    Args:
        drho (DenseMatrix&): The output derivative of the density matrix.
        rho (DenseMatrix): The current density matrix.
        H (MatrixFreeHamiltonian): The Hamiltonian.
        jumps (std::vector<SparseMatrix>): The list of jump operators.
        is_unitary_on_statevector (bool): Whether the evolution is unitary on a state vector.
    */
    if (is_unitary_on_statevector) {
        H.apply(rho, MatrixFreeApplicationType::Left, drho);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < drho.size(); ++i) {
            drho(i) *= -imag;
        }
    } else {
        DenseMatrix Hrho(rho.rows(), rho.cols());
        DenseMatrix rhoH(rho.rows(), rho.cols());
        H.apply(rho, MatrixFreeApplicationType::Left, Hrho);
        H.apply(rho, MatrixFreeApplicationType::Right, rhoH);
        drho = -imag * (Hrho - rhoH);
        for (const auto& J : jumps) {
            SparseMatrix Jdag = J.adjoint();
            SparseMatrix JdagJ = Jdag * J;
            drho += J * rho * Jdag;
            drho -= 0.5 * (JdagJ * rho + rho * JdagJ);
        }
    }
}

namespace {

using Clock = std::chrono::steady_clock;
inline double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// Coarse per-section timer shared by the CPU and GPU variational paths.
// Accumulates across calls and prints a breakdown every kReportEvery calls.
// Remove or gate behind the `verbose` define once profiling is done.
class PhaseTimer {
   public:
    explicit PhaseTimer(const char* tag) : tag_(tag) {}
    void add(int section, double s) { acc_[section] += s; }  // 0=sample 1=energy 2=linalg 3=drho
    void tick(int p, int N_s) {
        if (++calls_ % kReportEvery != 0) {
            return;
        }
        const double tot = acc_[0] + acc_[1] + acc_[2] + acc_[3];
        if (tot <= 0.0) {
            return;
        }
        auto pct = [tot](double t) { return 100.0 * t / tot; };
        std::cout << "[QiliSim timing " << tag_ << "] calls=" << calls_ << " p=" << p << " N_s=" << N_s
                  << " | total=" << tot << "s"
                  << " sample=" << acc_[0] << "s (" << pct(acc_[0]) << "%)"
                  << " energy=" << acc_[1] << "s (" << pct(acc_[1]) << "%)"
                  << " linalg=" << acc_[2] << "s (" << pct(acc_[2]) << "%)"
                  << " drho=" << acc_[3] << "s (" << pct(acc_[3]) << "%)" << std::endl;
    }

   private:
    static constexpr long kReportEvery = 100;
    const char* tag_;
    long calls_ = 0;
    double acc_[4] = {0.0, 0.0, 0.0, 0.0};
};

// Stochastic-reconfiguration linear algebra on the CPU: given the (N_s x p)
// sample-operator matrix O and per-sample local energies El, return
//   adot = M^{-1} V,  M = OᵀO/N_s - ōōᵀ + εI,  V = -(Oᵀ El/N_s - ō Ēl).
// This is the single source of truth for the SR math; the GPU path
// (qilisim::gpu::sr_solve) mirrors it and falls back here on any failure.
Eigen::VectorXcd sr_adot_cpu(const Eigen::MatrixXd& O, const Eigen::VectorXcd& El, double epsilon) {
    const int N_s = static_cast<int>(O.rows());
    const int p = static_cast<int>(O.cols());
    const Eigen::VectorXd O_mean = O.colwise().mean();
    const std::complex<double> El_mean = El.mean();

    Eigen::MatrixXd M = O.transpose() * O / static_cast<double>(N_s) - O_mean * O_mean.transpose();
    M += epsilon * Eigen::MatrixXd::Identity(p, p);

    const Eigen::VectorXcd V =
        -((O.transpose().cast<std::complex<double>>() * El) / static_cast<double>(N_s) -
          O_mean.cast<std::complex<double>>() * El_mean);

    Eigen::LLT<Eigen::MatrixXd> llt(M);
    Eigen::VectorXcd adot(p);
    adot.real() = llt.solve(V.real());
    adot.imag() = llt.solve(V.imag());
    return adot;
}

// Shared front/back end of the variational RHS: draw samples + local energy
// (CPU), then `compute_adot` turns (O, El) into adot, then write into drho.
template <typename ComputeAdot>
void variational_rhs(ExponentialAnsatz& drho, const ExponentialAnsatz& rho, const MatrixFreeHamiltonian& H,
                     PhaseTimer& timer, ComputeAdot&& compute_adot) {
    const auto& ops = rho.get_terms().get_operators();
    const int p = static_cast<int>(ops.size());
    std::vector<std::pair<PauliString, std::complex<double>>> terms_vec(ops.begin(), ops.end());

    auto t0 = Clock::now();
    SampleSet samples = rho.draw_samples();
    auto t1 = Clock::now();
    const int N_s = static_cast<int>(samples.configs.size());
    Eigen::VectorXcd El = rho.local_energy(samples, H);
    auto t2 = Clock::now();

    // Cast int8 ±1 storage to doubles so we can use BLAS routines.
    Eigen::MatrixXd O_mat_d = samples.O_mat.cast<double>();
    const double epsilon = 0.1 / std::sqrt(static_cast<double>(N_s));
    Eigen::VectorXcd adot = compute_adot(O_mat_d, El, epsilon);
    auto t3 = Clock::now();

    drho *= 0.0;
    for (int k = 0; k < p; ++k) {
        drho.get_terms().add(adot(k), terms_vec[k].first);
    }
    auto t4 = Clock::now();

    timer.add(0, secs(t0, t1));
    timer.add(1, secs(t1, t2));
    timer.add(2, secs(t2, t3));
    timer.add(3, secs(t3, t4));
    timer.tick(p, N_s);
}

}  // namespace

void lindblad_rhs(ExponentialAnsatz& drho, const ExponentialAnsatz& rho, const MatrixFreeHamiltonian& H) {
    /*
    Evaluate the right-hand side of the variational equations for the ExponentialAnsatz (CPU).

    The ansatz is |Ψ⟩ = exp(∑_k a_k P_k)|+⟩ where P_k are Z-type Pauli strings.
    Uses stochastic reconfiguration (VMC) to compute ȧ_k = (M⁻¹ V)_k where:
      M_{kk'} = <O_k* O_k'> - <O_k*><O_k'>        (quantum geometric tensor)
      V_k     = -(<O_k* E_loc> - <O_k*><E_loc>)   (imaginary-time force / SR)
      O_k(σ)  = P_k(σ) = ∏_{i∈S_k} σᵢ             (log-derivative)
    Samples are drawn from |Ψ(σ)|² via Metropolis-Hastings.

    Args:
        drho (ExponentialAnsatz&): Output: coefficients set to ȧ_k.
        rho (ExponentialAnsatz): Current ansatz with parameters a_k.
        H (MatrixFreeHamiltonian): The Hamiltonian.
    */
    static PhaseTimer timer("cpu");
    variational_rhs(drho, rho, H, timer,
                    [](const Eigen::MatrixXd& O, const Eigen::VectorXcd& El, double eps) {
                        return sr_adot_cpu(O, El, eps);
                    });
}

void lindblad_rhs_gpu(ExponentialAnsatz& drho, const ExponentialAnsatz& rho, const MatrixFreeHamiltonian& H) {
    /*
    GPU counterpart of lindblad_rhs. Sampling and local energy stay on the CPU;
    the entire stochastic-reconfiguration linear algebra (means, Gram, M
    assembly, V, Cholesky solve) runs device-resident via qilisim::gpu::sr_solve,
    uploading only O and El and reading back only adot. On any GPU failure it
    falls back to the identical CPU computation (sr_adot_cpu), so results match.
    Dispatch (this vs lindblad_rhs) happens once per RK4 step in iter_rk4.
    */
    static PhaseTimer timer("gpu");
    variational_rhs(drho, rho, H, timer,
                    [](const Eigen::MatrixXd& O, const Eigen::VectorXcd& El, double eps) {
                        Eigen::VectorXcd adot;
                        if (!qilisim::gpu::sr_solve(O, El, eps, adot)) {
                            adot = sr_adot_cpu(O, El, eps);  // resident path failed -> CPU
                        }
                        return adot;
                    });
}

// GCOV_EXCL_BR_STOP