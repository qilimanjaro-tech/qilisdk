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

#include <complex>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "circuit_sampling/gate.h"

// Eigen specfic type defs
typedef Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> SparseMatrix;
typedef Eigen::SparseMatrix<std::complex<double>, Eigen::ColMajor> SparseMatrixCol;
typedef Eigen::MatrixXcd DenseMatrix;
typedef Eigen::Triplet<std::complex<double>> Triplet;
typedef std::vector<Eigen::Triplet<std::complex<double>>> Triplets;

// Shorthand
namespace py = pybind11;

// Tolerance for numerical comparisons
const double atol_ = 1e-12;

// Identity matrix constant
const SparseMatrix I = []() {
    Triplets entries;
    entries.emplace_back(Triplet(0, 0, 1.0));
    entries.emplace_back(Triplet(1, 1, 1.0));
    SparseMatrix I_mat(2, 2);
    I_mat.setFromTriplets(entries.begin(), entries.end());
    return I_mat;
}();

// Needed for _a literals
using namespace pybind11::literals;

// Get the Python functional classes
const py::object Sampling = py::module_::import("qilisdk.functionals.sampling").attr("Sampling");
const py::object TimeEvolution = py::module_::import("qilisdk.functionals.time_evolution").attr("TimeEvolution");
const py::object SamplingResult = py::module_::import("qilisdk.functionals.sampling").attr("SamplingResult");
const py::object TimeEvolutionResult = py::module_::import("qilisdk.functionals.time_evolution").attr("TimeEvolutionResult");
const py::object numpy_array = py::module_::import("numpy").attr("array");
const py::object spmatrix = py::module_::import("scipy.sparse").attr("spmatrix");
const py::object QTensor = py::module_::import("qilisdk.core.qtensor").attr("QTensor");
const py::object Hamiltonian = py::module_::import("qilisdk.analog.hamiltonian").attr("Hamiltonian");
const py::object PauliOperator = py::module_::import("qilisdk.analog.hamiltonian").attr("PauliOperator");

// The main QiliSim C++ class
class QiliSimCpp {
   private:
    // utils/numpy.cpp
    SparseMatrix from_numpy(const py::buffer& matrix_buffer) const;
    SparseMatrix from_spmatrix(const py::object& matrix) const;
    py::array_t<double> to_numpy(const std::vector<double>& vec) const;
    py::array_t<double> to_numpy(const std::vector<std::vector<double>>& vecs) const;
    py::array_t<std::complex<double>> to_numpy(const SparseMatrix& matrix) const;

    // utils/parsers.cpp
    std::vector<SparseMatrix> parse_hamiltonians(const py::object& Hs) const;
    std::vector<SparseMatrix> parse_jump_operators(const py::object& jumps) const;
    std::vector<SparseMatrix> parse_observables(const py::object& observables, long nqubits) const;
    std::vector<std::vector<double>> parse_parameters(const py::object& coeffs) const;
    std::vector<double> parse_time_steps(const py::object& steps) const;
    SparseMatrix parse_initial_state(const py::object& initial_state) const;
    std::vector<Gate> parse_gates(const py::object& circuit) const;
    std::vector<bool> parse_measurements(const py::object& circuit) const;

    // utils/matrix_utils.cpp
    SparseMatrix exp_mat_action(const SparseMatrix& H, std::complex<double> dt, const SparseMatrix& e1) const;
    SparseMatrix exp_mat(const SparseMatrix& H, std::complex<double> dt) const;
    std::complex<double> dot(const SparseMatrix& v1, const SparseMatrix& v2) const;
    std::complex<double> dot(const DenseMatrix& v1, const DenseMatrix& v2) const;
    std::complex<double> trace(const SparseMatrix& matrix) const;
    SparseMatrix vectorize(const SparseMatrix& matrix) const;
    SparseMatrix devectorize(const SparseMatrix& vec_matrix) const;

    // circuit_sampling/circuit_optimizations.cpp
    void combine_single_qubit_gates(std::vector<Gate>& gates) const;
    std::vector<std::vector<Gate>> compress_gate_layers(std::vector<Gate>& gates) const;
    SparseMatrix layer_to_matrix(const std::vector<Gate>& gate_layer, int n_qubits) const;

    // time_evolution/lindblad.cpp
    SparseMatrix create_superoperator(const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators) const;
    void lindblad_rhs(DenseMatrix& drho, const DenseMatrix& rho, const SparseMatrix& H, const std::vector<SparseMatrix>& jumps, bool is_unitary_on_statevector) const;

    // utils/random.cpp
    std::map<std::string, int> sample_from_probabilities(const std::vector<std::tuple<int, double>>& prob_entries, int n_qubits, int n_shots, int seed) const;
    std::map<std::string, int> sample_from_probabilities(const std::vector<double>& probabilities, int n_qubits, int n_shots, int seed) const;
    SparseMatrix sample_from_density_matrix(const SparseMatrix& rho, int n_trajectories, int seed) const;
    SparseMatrix trajectories_to_density_matrix(const SparseMatrix& trajectories) const;
    SparseMatrix get_vector_from_density_matrix(const SparseMatrix& rho_t) const;

    // time_evolution/integrate.cpp
    SparseMatrix iter_integrate(const SparseMatrix& rho_0, double dt, const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators, int num_substeps, bool is_unitary_on_statevector)
        const;

    // time_evolution/arnoldi.cpp
    void arnoldi(const SparseMatrix& L, const SparseMatrix& v0, int m, std::vector<SparseMatrix>& V, SparseMatrix& H) const;
    void arnoldi_mat(const SparseMatrix& Hsys, const SparseMatrix& rho0, int m, std::vector<SparseMatrix>& V, SparseMatrix& Hk) const;
    SparseMatrix iter_arnoldi(const SparseMatrix& rho_0,
                              double dt,
                              const SparseMatrix& currentH,
                              const std::vector<SparseMatrix>& jump_operators,
                              int arnoldi_dim,
                              int num_substeps,
                              bool is_unitary_on_statevector) const;

    // time_evolution/direct.cpp
    SparseMatrix iter_direct(const SparseMatrix& rho_0, double dt, const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators, bool is_unitary_on_statevector) const;

   public:
    // circuit_sampling/execute_sampling.cpp
    py::object execute_sampling(const py::object& functional, const py::dict& solver_params);

    // time_evolution/execute_time_evolution.cpp
    py::object execute_time_evolution(const py::object& initial_state,
                                      const py::object& Hs,
                                      const py::object& coeffs,
                                      const py::object& steps,
                                      const py::object& observables,
                                      const py::object& jumps,
                                      bool store_intermediate_results,
                                      const py::dict& solver_params);
};
