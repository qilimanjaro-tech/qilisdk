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

#include <algorithm>
#include <complex>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/src/KroneckerProduct/KroneckerTensorProduct.h>
#include <unsupported/Eigen/MatrixFunctions>

// Eigen specfic type defs
typedef Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> SparseMatrix;
typedef Eigen::SparseMatrix<std::complex<double>, Eigen::ColMajor> SparseMatrixCol;
typedef Eigen::MatrixXcd DenseMatrix;
typedef Eigen::Triplet<std::complex<double>> Triplet;
typedef std::vector<Eigen::Triplet<std::complex<double>>> Triplets;

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
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

class Gate {
    /*
    A quantum gate with type, control qubits and target qubits.
    */
   private:
    std::string gate_type;
    SparseMatrix base_matrix;
    std::vector<int> control_qubits;
    std::vector<int> target_qubits;
    std::vector<std::pair<std::string, double>> parameters;

    int permute_bits(int index, const std::vector<int>& perm) const {
        /*
        Permute the bits of an index according to a given permutation.
        This works by taking each bit from the original index and placing it in the new position
        specified by the permutation vector.
        Also note, the qubit versus bit ordering is reversed (i.e. {1, 0, 2} swaps the
        first bits (0 and 1), not the last/least-significant).

        Args:
            index (int): The original index.
            perm (std::vector<int>): The permutation vector.

        Returns:
            int: The permuted index.
        */
        int n = int(perm.size());
        int out = 0;
        for (int old_q = 0; old_q < n; old_q++) {
            int new_q = perm[old_q];
            int old_bit = n - 1 - old_q;
            int new_bit = n - 1 - new_q;
            int bit = (index >> old_bit) & 1;
            out |= (bit << new_bit);
        }
        return out;
    }

    Triplets tensor_product(Triplets& A, Triplets& B, int B_width) const {
        /*
        Compute the tensor product of two sets of (row, col, value) tuples.

        Args:
            A (Triplets&): First matrix entries.
            B (Triplets&): Second matrix entries.
            B_width (int): Width of the second matrix (assumed square).

        Returns:
            Triplets: The tensor product entries.
        */
        Triplets result;
        int row = 0;
        int col = 0;
        std::complex<double> val = 0.0;
        for (const auto& a : A) {
            for (const auto& b : B) {
                row = a.row() * B_width + b.row();
                col = a.col() * B_width + b.col();
                val = a.value() * b.value();
                result.emplace_back(row, col, val);
            }
        }
        return result;
    }

    SparseMatrix base_to_full(const SparseMatrix& base_gate, int num_qubits, const std::vector<int>& control_qubits, const std::vector<int>& target_qubits) const {
        /*
        Expand a base gate matrix to the full matrix on the entire qubit register.

        Args:
            base_gate (SparseMatrix): The base gate matrix.
            num_qubits (int): Total number of qubits in the register.
            control_qubits (std::vector<int>): List of control qubit indices.
            target_qubits (std::vector<int>): List of target qubit indices.

        Returns:
            SparseMatrix: The full matrix representation of the gate.
        */

        // Determine how many tensor products we need before and after
        int min_qubit = num_qubits;
        for (int q : control_qubits) {
            if (q < min_qubit)
                min_qubit = q;
        }
        for (int q : target_qubits) {
            if (q < min_qubit)
                min_qubit = q;
        }
        int base_gate_qubits = int(target_qubits.size()) + int(control_qubits.size());
        int needed_before = min_qubit;
        int needed_after = num_qubits - needed_before - base_gate_qubits;

        // Do everything in tuple form for easier manipulation
        Triplets out_entries;
        for (int k = 0; k < base_gate.outerSize(); ++k) {
            for (SparseMatrix::InnerIterator it(base_gate, k); it; ++it) {
                int row = int(it.row());
                int col = int(it.col());
                std::complex<double> val = it.value();
                out_entries.emplace_back(Triplet(row, col, val));
            }
        }

        // Make it controlled if needed
        // Reminder that control gates look like:
        // 1 0 0 0
        // 0 1 0 0
        // 0 0 G G
        // 0 0 G G
        int base_gate_actual_qubits = std::ceil(std::log2(base_gate.cols()));
        int missing_controls = base_gate_actual_qubits - base_gate_qubits;
        for (int i = 0; i < missing_controls; ++i) {
            int delta = 1 << (target_qubits.size());
            Triplets new_entries;
            for (const auto& entry : out_entries) {
                int row = int(entry.row());
                int col = int(entry.col());
                std::complex<double> val = entry.value();
                new_entries.emplace_back(Triplet(row + delta, col + delta, val));
            }
            for (int i = 0; i < delta; ++i) {
                new_entries.emplace_back(Triplet(i, i, 1.0));
            }
            out_entries = new_entries;
        }

        // Perform the tensor products with the identity
        Triplets identity_entries = {Triplet(0, 0, 1.0), Triplet(1, 1, 1.0)};
        int gate_size = 1 << (target_qubits.size() + control_qubits.size());
        for (int i = 0; i < needed_before; ++i) {
            out_entries = tensor_product(identity_entries, out_entries, gate_size);
            gate_size *= 2;
        }
        for (int i = 0; i < needed_after; ++i) {
            out_entries = tensor_product(out_entries, identity_entries, 2);
            gate_size *= 2;
        }

        // Get a list of all qubits involved
        std::vector<int> all_qubits;
        all_qubits.insert(all_qubits.end(), control_qubits.begin(), control_qubits.end());
        all_qubits.insert(all_qubits.end(), target_qubits.begin(), target_qubits.end());

        // Determine the permutation to map qubits to their correct positions
        // perm is initially 0 1 2
        // if we have a CNOT on qubits 0 and 2, we actually did it on 0 and 1 internally
        // so we have 0 2 1 and we need to map back
        std::vector<int> perm(num_qubits);
        for (int q = 0; q < num_qubits; ++q) {
            perm[q] = q;
        }
        for (int i = 0; i < int(all_qubits.size()); ++i) {
            if (perm[needed_before + i] != all_qubits[i]) {
                std::swap(perm[needed_before + i], perm[all_qubits[i]]);
            }
        }

        // Invert the perm
        std::vector<int> inv_perm(num_qubits);
        for (int q = 0; q < num_qubits; ++q) {
            inv_perm[perm[q]] = q;
        }
        perm = inv_perm;

        // Apply the permutation
        for (size_t i = 0; i < out_entries.size(); ++i) {
            int old_row = out_entries[i].row();
            int old_col = out_entries[i].col();
            int new_row = permute_bits(old_row, perm);
            int new_col = permute_bits(old_col, perm);
            out_entries[i] = Triplet(new_row, new_col, out_entries[i].value());
        }

        // Make sure the entries are sorted
        std::sort(out_entries.begin(), out_entries.end(), [](const Triplet& a, const Triplet& b) {
            if (a.row() != b.row()) {
                return a.row() < b.row();
            } else {
                return a.col() < b.col();
            }
        });

        // Form the matrix and return
        int full_size = 1 << num_qubits;
        SparseMatrix out_matrix(full_size, full_size);
        out_matrix.setFromTriplets(out_entries.begin(), out_entries.end());
        return out_matrix;
    }

   public:
    // Constructor
    Gate(const std::string& gate_type_,
         const SparseMatrix& base_matrix_,
         const std::vector<int>& controls_,
         const std::vector<int>& targets_,
         const std::vector<std::pair<std::string, double>>& parameters_)
        : gate_type(gate_type_), base_matrix(base_matrix_), control_qubits(controls_), target_qubits(targets_), parameters(parameters_) {}

    std::string get_name() const {
        /*
        Get the name of the gate, prefixed by c's for each control qubit.
        Returns:
            std::string: The gate name.
        */
        std::string control_string = "";
        for (int i = 0; i < int(control_qubits.size()); ++i) {
            control_string += "c";
        }
        return control_string + gate_type;
    }

    std::string get_id() const {
        /*
        Get a unique identifier for the gate based on its type and qubits.
        Returns:
            std::string: The gate identifier.
        */
        std::ostringstream oss;
        oss << get_name();
        if (!control_qubits.empty()) {
            oss << "_c_";
            for (const auto& q : control_qubits) {
                oss << q << "_";
            }
        }
        if (!target_qubits.empty()) {
            oss << "_t_";
            for (const auto& q : target_qubits) {
                oss << q << "_";
            }
        }
        if (!parameters.empty()) {
            oss << "p_";
            for (const auto& param : parameters) {
                oss << param.first << "_" << param.second << "_";
            }
        }
        return oss.str();
    }

    SparseMatrix get_full_matrix(int num_qubits) const {
        /*
        Get the full matrix representation of the gate on the entire qubit register.

        Args:
            num_qubits (int): Total number of qubits in the register.

        Returns:
            SparseMatrix: The full matrix representation of the gate.
        */
        return base_to_full(base_matrix, num_qubits, control_qubits, target_qubits);
    }
};

// Get the Python functional classes
const py::object Sampling = py::module_::import("qilisdk.functionals.sampling").attr("Sampling");
const py::object TimeEvolution = py::module_::import("qilisdk.functionals.time_evolution").attr("TimeEvolution");
const py::object SamplingResult = py::module_::import("qilisdk.functionals.sampling").attr("SamplingResult");
const py::object TimeEvolutionResult = py::module_::import("qilisdk.functionals.time_evolution").attr("TimeEvolutionResult");
const py::object numpy_array = py::module_::import("numpy").attr("array");
const py::object QTensor = py::module_::import("qilisdk.core.qtensor").attr("QTensor");
const py::object Hamiltonian = py::module_::import("qilisdk.analog.hamiltonian").attr("Hamiltonian");
const py::object PauliOperator = py::module_::import("qilisdk.analog.hamiltonian").attr("PauliOperator");

// Needed for _a literals
using namespace pybind11::literals;

class QiliSimCpp {
   private:
    SparseMatrix exp_mat_action(const SparseMatrix& H, std::complex<double> dt, const SparseMatrix& e1) const {
        /*
        Compute the action of the matrix exponential exp(H*dt) acting on a vector e1.

        Args:
            H (SparseMatrix): The upper Hessenberg matrix.
            dt (std::complex<double>): The time step. Can be complex if needed.
            e1 (SparseMatrix): The vector to apply the exponential to.

        Returns:
            SparseMatrix: The result of exp(H*dt) * e1.
        */
        DenseMatrix H_dense = dt * DenseMatrix(H);
        DenseMatrix exp_H = H_dense.exp() * e1;
        SparseMatrix exp_H_sparse = exp_H.sparseView();
        return exp_H_sparse;
    }

    SparseMatrix exp_mat(const SparseMatrix& H, std::complex<double> dt) const {
        /*
        Compute the matrix exponential exp(H*dt).

        Args:
            H (SparseMatrix): The matrix to exponentiate.
            dt (std::complex<double>): The time step. Can be complex if needed.

        Returns:
            SparseMatrix: The matrix exponential exp(H*dt).
        */
        DenseMatrix H_dense = dt * DenseMatrix(H);
        DenseMatrix exp_H = H_dense.exp();
        SparseMatrix exp_H_sparse = exp_H.sparseView();
        return exp_H_sparse;
    }

    std::complex<double> dot(const SparseMatrix& v1, const SparseMatrix& v2) const {
        /*
        Compute the inner product between two sparse matrices.
        Note that the first matrix is conjugated.

        Args:
            v1 (SparseMatrix): The first matrix.
            v2 (SparseMatrix): The second matrix.

        Returns:
            std::complex<double>: The inner product result.
        */
        return v1.conjugate().cwiseProduct(v2).sum();
    }

    std::complex<double> dot(const DenseMatrix& v1, const DenseMatrix& v2) const {
        /*
        Compute the inner product between two dense matrices.
        Note that the first matrix is conjugated.

        Args:
            v1 (DenseMatrix): The first matrix.
            v2 (DenseMatrix): The second matrix.

        Returns:
            std::complex<double>: The inner product result.
        */
        return v1.conjugate().cwiseProduct(v2).sum();
    }

    void arnoldi(const SparseMatrix& L, const SparseMatrix& v0, int m, std::vector<SparseMatrix>& V, SparseMatrix& H) {
        /*
        Perform the Arnoldi iteration to build the basis.

        Args:
            L (SparseMatrix): The Lindblad superoperator.
            v0 (SparseMatrix): The initial vectorized density matrix.
            m (int): The dimension of the subspace.
            V (SparseMatrix&): Output basis vectors.
            H (SparseMatrix&): Output upper Hessenberg matrix.
        */

        // Set up the outputs
        V.clear();
        H = SparseMatrix(m + 1, m);

        // Normalize the initial vector
        SparseMatrix v = v0;
        double beta = v.norm();
        v /= beta;

        // Add the first vector to the list
        V.push_back(v);

        // For each Arnoldi iteration
        for (int j = 0; j < m; ++j) {
            // Apply the Lindbladian to the previous vector
            SparseMatrix w = L * V[j];

            // Orthogonalize against previous vectors
            for (int i = 0; i <= j; ++i) {
                std::complex<double> prod = dot(V[i], w);
                H.coeffRef(i, j) = prod;
                w -= V[i] * prod;
            }

            // Update H and check for convergence
            double to_add = w.norm();
            H.coeffRef(j + 1, j) = to_add;
            if (to_add < atol_) {
                break;
            }

            // Normalize and add to V
            w /= to_add;
            V.push_back(w);
        }
    }

    void arnoldi_mat(const SparseMatrix& Hsys, const SparseMatrix& rho0, int m, std::vector<SparseMatrix>& V, SparseMatrix& Hk) {
        /*
        Arnoldi iteration for the unitary Liouvillian:
            L(rho) = -i (H rho - rho H)

        Args:
            Hsys (SparseMatrix): Hamiltonian (dim x dim)
            rho0 (SparseMatrix): Initial density matrix (dim x dim)
            m (int): Krylov dimension
            V (vector<SparseMatrix>): Orthonormal basis (output)
            Hk (SparseMatrix): Upper Hessenberg matrix (output)
        */

        // Set up outputs
        V.clear();
        Hk = SparseMatrix(m + 1, m);

        // Normalize initial matrix (Frobenius norm)
        SparseMatrix v = rho0;
        double beta = v.norm();
        if (beta < atol_) {
            return;
        }
        v /= beta;
        V.push_back(v);

        for (int j = 0; j < m; ++j) {
            // Apply reduced Liouvillian: w = -i (H v - v H)
            SparseMatrix w = -std::complex<double>(0.0, 1.0) * (Hsys * V[j] - V[j] * Hsys);

            // Modified Gram–Schmidt
            for (int i = 0; i <= j; ++i) {
                std::complex<double> hij = dot(V[i], w);  // Tr(V[i]† w)
                Hk.coeffRef(i, j) = hij;
                w -= V[i] * hij;
            }

            // Compute norm and check for convergence
            double norm_w = w.norm();
            Hk.coeffRef(j + 1, j) = norm_w;
            if (norm_w < atol_) {
                break;
            }

            // Normalize and add to V
            w /= norm_w;
            V.push_back(w);
        }
    }

    SparseMatrix from_numpy(const py::buffer& matrix_buffer) {
        /*
        Convert a numpy array buffer to a SparseMatrix.

        Args:
            matrix_buffer (py::buffer): The numpy array buffer.

        Returns:
            SparseMatrix: The converted sparse matrix.
        */
        py::buffer_info buf = matrix_buffer.request();
        if (buf.ndim != 2) {
            throw py::value_error("Input array must be 2D.");
        }
        int rows = int(buf.shape[0]);
        int cols = int(buf.shape[1]);
        auto ptr = static_cast<std::complex<double>*>(buf.ptr);
        Triplets entries;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                std::complex<double> val = ptr[r * cols + c];
                if (std::abs(val) > atol_) {
                    entries.emplace_back(Triplet(r, c, val));
                }
            }
        }
        SparseMatrix mat(rows, cols);
        mat.setFromTriplets(entries.begin(), entries.end());
        return mat;
    }

    std::vector<SparseMatrix> parse_hamiltonians(const py::object& Hs) {
        /*
        Extract Hamiltonian matrices from a list of QTensor objects.

        Args:
            Hs (py::object): A list of QTensor Hamiltonians.

        Returns:
            std::vector<SparseMatrix>: The list of Hamiltonian sparse matrices.
        */
        std::vector<SparseMatrix> hamiltonians;
        for (auto& hamiltonian : Hs) {
            py::buffer matrix = numpy_array(hamiltonian.attr("to_matrix")().attr("toarray")(), py::dtype("complex128"));
            py::buffer_info buf = matrix.request();
            SparseMatrix H = from_numpy(matrix);
            hamiltonians.push_back(H);
        }
        return hamiltonians;
    }

    std::vector<SparseMatrix> parse_jump_operators(const py::object& jumps) {
        /*
        Extract jump operator matrices from a list of QTensor objects.

        Args:
            jumps (py::object): A list of QTensor jump operators.

        Returns:
            std::vector<SparseMatrix>: The list of jump operator sparse matrices.
        */
        std::vector<SparseMatrix> jump_matrices;
        for (auto jump : jumps) {
            py::buffer matrix = numpy_array(jump.attr("dense")(), py::dtype("complex128"));
            py::buffer_info buf = matrix.request();
            SparseMatrix J = from_numpy(matrix);
            jump_matrices.push_back(J);
        }
        return jump_matrices;
    }

    std::vector<SparseMatrix> parse_observables(const py::object& observables, long nqubits) {
        /*
        Extract observable matrices from a list of QTensor objects.

        Args:
            observables (py::object): A list of QTensor observables.
            nqubits (long): The total number of qubits.

        Returns:
            std::vector<SparseMatrix>: The list of observable sparse matrices.
        */
        std::vector<SparseMatrix> observable_matrices;
        for (auto obs : observables) {
            // Depending on the type of observable given
            if (py::isinstance(obs, Hamiltonian)) {
                // Get the matrix
                py::buffer matrix = numpy_array(obs.attr("to_matrix")().attr("toarray")(), py::dtype("complex128"));
                py::buffer_info buf = matrix.request();
                SparseMatrix O = from_numpy(matrix);

                // Expand to full qubit count if needed
                int obs_qubits = obs.attr("nqubits").cast<int>();
                SparseMatrix O_global = O;
                for (long q = obs_qubits; q < nqubits; ++q) {
                    O_global = Eigen::kroneckerProduct(O_global, I).eval();
                }
                observable_matrices.push_back(O_global);

            } else if (py::isinstance(obs, PauliOperator)) {
                // Get the matrix
                py::buffer matrix = numpy_array(obs.attr("matrix"), py::dtype("complex128"));
                py::buffer_info buf = matrix.request();
                SparseMatrix O = from_numpy(matrix);

                // Expand to full qubit count
                int obs_qubit = obs.attr("qubit").cast<int>();
                SparseMatrix O_global(1, 1);
                O_global.coeffRef(0, 0) = 1.0;
                O_global.makeCompressed();
                for (long q = 0; q < nqubits; ++q) {
                    if (q != obs_qubit) {
                        O_global = Eigen::kroneckerProduct(O_global, I).eval();
                    } else {
                        O_global = Eigen::kroneckerProduct(O_global, O).eval();
                    }
                }
                observable_matrices.push_back(O_global);

            } else if (py::isinstance(obs, QTensor)) {
                py::buffer matrix = numpy_array(obs.attr("dense")(), py::dtype("complex128"));
                py::buffer_info buf = matrix.request();
                SparseMatrix O = from_numpy(matrix);
                observable_matrices.push_back(O);

            } else {
                throw py::value_error("Observable type not recognized.");
            }
        }
        return observable_matrices;
    }

    std::vector<std::vector<double>> parse_parameters(const py::object& coeffs) {
        /*
        Extract parameter lists from a list of coefficient objects.

        Args:
            coeffs (py::object): A list of coefficient objects.

        Returns:
            std::vector<std::vector<double>>: The list of parameter vectors.
        */
        std::vector<std::vector<double>> parameters_list;
        for (auto& param_set : coeffs) {
            std::vector<double> param_vector;
            for (auto& param : param_set) {
                param_vector.push_back(param.cast<double>());
            }
            parameters_list.push_back(param_vector);
        }
        return parameters_list;
    }

    std::vector<double> parse_time_steps(const py::object& steps) {
        /*
        Extract time steps from a list of step objects.

        Args:
            steps (py::object): A list of step objects.

        Returns:
            std::vector<double>: The list of time steps.
        */
        std::vector<double> step_list;
        for (auto step : steps) {
            step_list.push_back(step.cast<double>());
        }
        return step_list;
    }

    SparseMatrix parse_initial_state(const py::object& initial_state) {
        /*
        Extract the initial state from a QTensor object.

        Args:
            initial_state (py::object): The initial state as a QTensor.

        Returns:
            SparseMatrix: The initial state as a sparse matrix.
        */
        py::buffer init_state = numpy_array(initial_state.attr("dense")(), py::dtype("complex128"));
        py::buffer_info buf = init_state.request();
        if (buf.ndim != 2) {
            throw py::value_error("Initial state must be a 2D array.");
        }
        int rows = int(buf.shape[0]);
        int cols = int(buf.shape[1]);
        auto ptr = static_cast<std::complex<double>*>(buf.ptr);
        Triplets rho_0_entries;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                std::complex<double> val = ptr[r * cols + c];
                if (std::abs(val) > atol_) {
                    rho_0_entries.emplace_back(Triplet(r, c, val));
                }
            }
        }
        SparseMatrix rho_0(rows, cols);
        rho_0.setFromTriplets(rho_0_entries.begin(), rho_0_entries.end());
        return rho_0;
    }

    std::complex<double> trace(const SparseMatrix& matrix) {
        /*
        Compute the trace of a square matrix.

        Args:
            matrix (SparseMatrix): The input square matrix.

        Returns:
            std::complex<double>: The trace of the matrix.
        */
        if (matrix.rows() != matrix.cols()) {
            throw py::value_error("Matrix must be square to compute trace.");
        }
        std::complex<double> tr = 0.0;
        for (int i = 0; i < matrix.rows(); ++i) {
            tr += matrix.coeff(i, i);
        }
        return tr;
    }

    SparseMatrix vectorize(const SparseMatrix& matrix) {
        /*
        Vectorize a matrix by stacking its columns.

        Args:
            matrix (SparseMatrix): The input matrix.

        Returns:
            SparseMatrix: The vectorized matrix as a column vector.
        */
        int rows = int(matrix.rows());
        int cols = int(matrix.cols());
        Triplets vec_entries;
        for (int c = 0; c < cols; ++c) {
            for (int r = 0; r < rows; ++r) {
                std::complex<double> val = matrix.coeff(r, c);
                if (std::abs(val) > atol_) {
                    vec_entries.emplace_back(Triplet(r + c * rows, 0, val));
                }
            }
        }
        SparseMatrix vec_matrix(long(rows * cols), 1);
        vec_matrix.setFromTriplets(vec_entries.begin(), vec_entries.end());
        return vec_matrix;
    }

    SparseMatrix devectorize(const SparseMatrix& vec_matrix) {
        /*
        Devectorize a column vector back into a square matrix.

        Args:
            vec_matrix (SparseMatrix): The input vectorized matrix.

        Returns:
            SparseMatrix: The devectorized square matrix.
        */
        long dim = static_cast<long>(std::sqrt(vec_matrix.rows()));
        Triplets mat_entries;
        for (int c = 0; c < dim; ++c) {
            for (int r = 0; r < dim; ++r) {
                std::complex<double> val = vec_matrix.coeff(r + c * dim, 0);
                if (std::abs(val) > atol_) {
                    mat_entries.emplace_back(Triplet(r, c, val));
                }
            }
        }
        SparseMatrix mat(dim, dim);
        mat.setFromTriplets(mat_entries.begin(), mat_entries.end());
        return mat;
    }

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

    py::array_t<std::complex<double>> to_numpy(const SparseMatrix& matrix) {
        /*
        Convert a SparseMatrix to a NumPy array.

        Args:
            matrix (SparseMatrix): The input sparse matrix.

        Returns:
            py::array_t<std::complex<double>>: The corresponding NumPy array.
        */
        int rows = int(matrix.rows());
        int cols = int(matrix.cols());
        py::array_t<std::complex<double>> np_array({rows, cols});
        py::buffer_info buf = np_array.request();
        auto ptr = static_cast<std::complex<double>*>(buf.ptr);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                ptr[r * cols + c] = matrix.coeff(r, c);
            }
        }
        return np_array;
    }

    template <typename T>
    py::array_t<T> to_numpy(const std::vector<T>& vec) {
        /*
        Convert a vector of complex numbers to a NumPy array.

        Args:
            vec (std::vector<T>): The input vector.

        Returns:
            py::array_t<T>: The corresponding NumPy array.
        */
        int size = vec.size();
        py::array_t<T> np_array({size});
        py::buffer_info buf = np_array.request();
        auto ptr = static_cast<T*>(buf.ptr);
        for (int i = 0; i < size; ++i) {
            ptr[i] = vec[i];
        }
        return np_array;
    }

    template <typename T>
    py::array_t<T> to_numpy(const std::vector<std::vector<T>>& vecs) {
        /*
        Convert a vector of vectors of complex numbers to a 2D NumPy array.

        Args:
            vecs (std::vector<std::vector<T>>): The input vector of vectors.

        Returns:
            py::array_t<T>: The corresponding 2D NumPy array.
        */
        int rows = vecs.size();
        int cols = vecs[0].size();
        py::array_t<T> np_array({rows, cols});
        py::buffer_info buf = np_array.request();
        auto ptr = static_cast<T*>(buf.ptr);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                ptr[r * cols + c] = vecs[r][c];
            }
        }
        return np_array;
    }

    std::vector<Gate> parse_gates(const py::object& circuit) {
        /*
        Extract gates from a circuit object.

        Args:
            circuit (py::object): The circuit object.

        Returns:
            std::vector<Gate>: The list of Gate objects.
        */
        std::vector<Gate> gates;
        py::list py_gates = circuit.attr("gates");
        for (auto py_gate : py_gates) {
            // Get the name
            std::string gate_type_str = py_gate.attr("name").cast<std::string>();

            // If it's a measurement, skip it
            if (gate_type_str == "M") {
                continue;
            }

            // Get the matrix
            py::buffer matrix = py_gate.attr("_generate_matrix")();
            py::buffer_info buf = matrix.request();
            SparseMatrix base_matrix = from_numpy(matrix);

            // Get the controls
            std::vector<int> controls;
            py::list py_controls = py_gate.attr("control_qubits");
            for (auto py_control : py_controls) {
                controls.push_back(py_control.cast<int>());
            }

            // Get the targets
            std::vector<int> targets;
            py::list py_targets = py_gate.attr("target_qubits");
            for (auto py_target : py_targets) {
                targets.push_back(py_target.cast<int>());
            }

            // Get the parameter names
            std::vector<std::pair<std::string, double>> parameters;
            py::dict py_parameters = py_gate.attr("get_parameters")();
            for (auto item : py_parameters) {
                std::string name = item.first.cast<std::string>();
                double value = item.second.cast<double>();
                parameters.emplace_back(name, value);
            }

            // Add the gate
            gates.emplace_back(gate_type_str, base_matrix, controls, targets, parameters);
        }

        return gates;
    }

    std::vector<bool> parse_measurements(const py::object& circuit) {
        /*
        Extract measurement qubit information from a circuit object.

        Args:
            circuit (py::object): The circuit object.
            n_qubits (int): The total number of qubits.

        Returns:
            std::vector<bool>: A vector indicating which qubits are measured.
        */
        int n_qubits = circuit.attr("nqubits").cast<int>();
        std::vector<bool> qubits_to_measure(n_qubits, false);
        py::list py_gates = circuit.attr("gates");
        bool any_measurements = false;
        for (auto py_gate : py_gates) {
            // Get the name
            std::string gate_type_str = py_gate.attr("name").cast<std::string>();

            // If it's a measurement, mark the qubits
            if (gate_type_str == "M") {
                py::list py_targets = py_gate.attr("target_qubits");
                for (auto py_target : py_targets) {
                    int target = py_target.cast<int>();
                    qubits_to_measure[target] = true;
                    any_measurements = true;
                }
            }
        }

        // If we found no measurements, measure all
        if (!any_measurements) {
            qubits_to_measure = std::vector<bool>(n_qubits, true);
        }

        return qubits_to_measure;
    }

    std::map<std::string, int> sample_from_probabilities(const std::vector<std::tuple<int, double>>& prob_entries, int n_qubits, int n_shots) {
        /*
        Sample measurement outcomes from a probability distribution.

        Args:
            prob_entries (std::vector<std::tuple<int, double>>): List of (state index, probability) tuples.
            n_qubits (int): Number of qubits.
            n_shots (int): Number of measurement shots.

        Returns:
            std::map<std::string, int>: A map of bitstring outcomes to their counts.
        */
        std::map<int, std::string> binary_strings;
        std::map<std::string, int> counts;
        std::string current_bitstring = "";
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (int shot = 0; shot < n_shots; ++shot) {
            double random_value = distribution(generator);
            double cumulative_prob = 0.0;
            for (const auto& entry : prob_entries) {
                int state_index = std::get<0>(entry);
                double prob = std::get<1>(entry);
                cumulative_prob += prob;
                if (random_value <= cumulative_prob) {
                    if (binary_strings.find(state_index) == binary_strings.end()) {
                        std::string bitstring = "";
                        for (int b = n_qubits - 1; b >= 0; --b) {
                            bitstring += ((state_index >> b) & 1) ? '1' : '0';
                        }
                        binary_strings[state_index] = bitstring;
                    }
                    counts[binary_strings[state_index]]++;
                    break;
                }
            }
        }
        return counts;
    }

    SparseMatrix get_vector_from_density_matrix(SparseMatrix& rho_t) {
        /*
        Extract a state vector from a pure density matrix by finding a non-zero diagonal element.

        Args:
            rho_t (SparseMatrix): The density matrix.

        Returns:
            SparseMatrix: The extracted state vector.

        Raises:
            py::value_error: If the density matrix has no non-zero diagonal elements.
        */

        // Find a non-zero diagonal element
        int non_zero_col = -1;
        for (int r = 0; r < rho_t.rows(); ++r) {
            std::complex<double> val = rho_t.coeff(r, r);
            if (std::abs(val) > atol_) {
                non_zero_col = r;
                break;
            }
        }
        if (non_zero_col == -1) {
            throw py::value_error("Final density matrix has no non-zero diagonal elements.");
        }

        // Extract the corresponding state vector
        Triplets state_vec_entries;
        for (int r = 0; r < rho_t.rows(); ++r) {
            std::complex<double> val = rho_t.coeff(r, non_zero_col);
            if (std::abs(val) > atol_) {
                state_vec_entries.emplace_back(Triplet(r, 0, val));
            }
        }
        SparseMatrix final_state_vec(rho_t.rows(), 1);
        final_state_vec.setFromTriplets(state_vec_entries.begin(), state_vec_entries.end());
        final_state_vec /= final_state_vec.norm();

        return final_state_vec;
    }

    void lindblad_rhs(DenseMatrix& drho, const DenseMatrix& rho, const SparseMatrix& H, const std::vector<SparseMatrix>& jumps, bool is_unitary_on_statevector) {
        /*
        Compute the right-hand side of the Lindblad master equation.

        Args:
            drho (DenseMatrix&): The output derivative of the density matrix.
            rho (DenseMatrix): The current density matrix.
            H (SparseMatrix): The Hamiltonian.
            jumps (std::vector<SparseMatrix>): The list of jump operators.
            is_unitary_on_statevector (bool): Whether the evolution is unitary on a state vector.
        */
        const std::complex<double> I(0.0, 1.0);
        if (is_unitary_on_statevector) {
            drho = -I * H * rho;
        } else {
            SparseMatrix temp = (H * rho).sparseView();
            drho = -I * temp;
            temp = (rho * H).sparseView();
            drho += I * temp;
            for (const auto& J : jumps) {
                SparseMatrix Jdag = J.adjoint();
                SparseMatrix JdagJ = Jdag * J;
                DenseMatrix JdagJ_rho = JdagJ * rho;
                DenseMatrix rho_JdagJ = rho * JdagJ;
                drho += J * rho * Jdag;
                drho -= 0.5 * (JdagJ_rho + rho_JdagJ);
            }
        }
    }

    SparseMatrix iter_integrate(const SparseMatrix& rho_0, double dt, const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators, int num_substeps, bool is_unitary_on_statevector) {
        /*
        4th-order Runge–Kutta integration of the Lindblad master equation

        Args:
            rho_0 (SparseMatrix): The initial density matrix.
            dt (double): The total time step.
            currentH (SparseMatrix): The current Hamiltonian.
            jump_operators (std::vector<SparseMatrix>): The list of jump operators.
            is_unitary_on_statevector (bool): Whether the evolution is unitary on a state vector.
            num_substeps (int): Number of substeps to divide the time step into.

        Returns:
            SparseMatrix: The evolved density matrix after time dt.

        Raises:
            py::value_error: If num_substeps is non-positive.
            py::value_error: If currentH is not square.
            py::value_error: If rho_0 is not square (and evolution is not unitary on state vector).
            py::value_error: If Hamiltonian and initial density matrix dimensions do not match.
            py::value_error: If any jump operator dimension does not match Hamiltonian dimension.
        */

        // Sanity checks
        if (currentH.rows() != currentH.cols()) {
            throw py::value_error("Hamiltonian must be square.");
        }
        if (rho_0.rows() != rho_0.cols() && !is_unitary_on_statevector) {
            throw py::value_error("Initial density matrix must be square.");
        }
        long dim = long(currentH.rows());
        if (rho_0.rows() != dim) {
            throw py::value_error("Dimension mismatch.");
        }
        for (const auto& J : jump_operators) {
            if (J.rows() != dim || J.cols() != dim) {
                throw py::value_error("Jump operator dimension mismatch.");
            }
        }

        int rho_rows = int(rho_0.rows());
        int rho_cols = int(rho_0.cols());

        // Standard RK4 loop
        DenseMatrix rho = rho_0;
        DenseMatrix k1(rho_rows, rho_cols);
        DenseMatrix k2(rho_rows, rho_cols);
        DenseMatrix k3(rho_rows, rho_cols);
        DenseMatrix k4(rho_rows, rho_cols);
        DenseMatrix rho_tmp(rho_rows, rho_cols);
        double dt_sub = dt / static_cast<double>(num_substeps);
        for (int step = 0; step < num_substeps; ++step) {
            lindblad_rhs(k1, rho, currentH, jump_operators, is_unitary_on_statevector);
            rho_tmp = rho;
            rho_tmp += 0.5 * dt_sub * k1;
            lindblad_rhs(k2, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);
            rho_tmp = rho;
            rho_tmp += 0.5 * dt_sub * k2;
            lindblad_rhs(k3, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);
            rho_tmp = rho;
            rho_tmp += dt_sub * k3;
            lindblad_rhs(k4, rho_tmp, currentH, jump_operators, is_unitary_on_statevector);
            rho += (dt_sub / 6.0) * k1;
            rho += (dt_sub / 3.0) * k2;
            rho += (dt_sub / 3.0) * k3;
            rho += (dt_sub / 6.0) * k4;

            // Normalize the density matrix
            if (is_unitary_on_statevector) {
                rho /= rho.norm();
            } else {
                std::complex<double> tr = 0;
                for (int i = 0; i < dim; ++i) {
                    tr += rho(i, i);
                }
                rho /= tr;
            }
        }
        return rho.sparseView();
    }

    SparseMatrix iter_arnoldi(const SparseMatrix& rho_0,
                              double dt,
                              const SparseMatrix& currentH,
                              const std::vector<SparseMatrix>& jump_operators,
                              int arnoldi_dim,
                              int num_substeps,
                              bool is_unitary_on_statevector) {
        /*
        Perform time evolution using the Arnoldi iteration.

        Args:
            rho_0 (SparseMatrix): The initial density matrix.
            dt (double): The total time step.
            currentH (SparseMatrix): The current Hamiltonian.
            jump_operators (std::vector<SparseMatrix>): The list of jump operators.
            arnoldi_dim (int): Dimension of the subspace.
            num_substeps (int): Number of substeps to divide the time step into.
            is_unitary_on_statevector (bool): Whether the evolution is unitary on a state vector.

        Returns:
            SparseMatrix: The evolved density matrix after time dt.

        Raises:
            py::value_error: If arnoldi_dim is non-positive.
            py::value_error: If num_substeps is non-positive.
            py::value_error: If currentH is not square.
            py::value_error: If rho_0 is not square.
            py::value_error: If Hamiltonian and initial density matrix dimensions do not match.
            py::value_error: If any jump operator dimension does not match Hamiltonian dimension.
        */

        // Sanity checks
        if (arnoldi_dim <= 0) {
            throw py::value_error("Arnoldi dimension must be positive.");
        }
        if (num_substeps <= 0) {
            throw py::value_error("Number of substeps must be positive.");
        }
        if (currentH.rows() != currentH.cols()) {
            throw py::value_error("Hamiltonian must be square.");
        }
        if (rho_0.cols() != rho_0.rows() && !is_unitary_on_statevector) {
            throw py::value_error("Initial density matrix must be square.");
        }
        long dim = long(currentH.rows());
        if (rho_0.rows() != dim) {
            throw py::value_error("Initial density matrix dimension does not match Hamiltonian dimension.");
        }
        for (const auto& J : jump_operators) {
            if (J.rows() != dim || J.cols() != dim) {
                throw py::value_error("Jump operator dimension does not match Hamiltonian dimension.");
            }
        }

        // If we don't have jump operators, we can work directly with the density matrix
        bool is_unitary = (jump_operators.size() == 0);

        // Need to vectorize the density matrix if we're going to use the superoperator
        SparseMatrix rho_t;
        if (!is_unitary && !is_unitary_on_statevector) {
            rho_t = vectorize(rho_0);
        } else {
            rho_t = rho_0;
        }

        // Vars for the Arnoldi iteration
        std::vector<SparseMatrix> V;
        SparseMatrix A;
        int subspace_dim = 0;

        // Form the Lindblad superoperator if needed
        SparseMatrix L;
        if (!is_unitary) {
            L = create_superoperator(currentH, jump_operators);
        }

        // Divide into smaller timesteps if requested
        double dt_sub = dt / static_cast<double>(num_substeps);
        for (int substep_ind = 0; substep_ind < num_substeps; ++substep_ind) {
            // Run the Arnoldi iteration to build the basis
            // After this, we have our operator approximated in the basis as A
            // and the basis vectors in V
            if (is_unitary_on_statevector) {
                arnoldi(std::complex<double>(0, 1) * currentH, rho_t, arnoldi_dim, V, A);
                subspace_dim = int(V.size());
            } else if (!is_unitary) {
                arnoldi(L, rho_t, arnoldi_dim, V, A);
                subspace_dim = int(V.size()) - 1;
            } else {
                arnoldi_mat(currentH, rho_t, arnoldi_dim, V, A);
                subspace_dim = int(V.size());
            }
            A.conservativeResize(subspace_dim, subspace_dim);
            V.resize(subspace_dim);

            // If everything is zero then we're probably in an eigenstate and need to skip until we aren't
            if (subspace_dim == 0) {
                continue;
            }

            // Compute the action of the matrix exponential
            SparseMatrix e1(subspace_dim, 1);
            e1.coeffRef(0, 0) = 1;
            SparseMatrix y = exp_mat_action(A, dt_sub, e1);

            // Reconstruct the final density matrix using the basis vectors
            SparseMatrix rho_t_new(rho_t.rows(), rho_t.cols());
            for (int j = 0; j < int(V.size()); ++j) {
                rho_t_new += V[j] * y.coeff(j, 0);
            }
            rho_t = rho_t_new;

            // Normalize the density matrix
            if (is_unitary_on_statevector) {
                rho_t /= rho_t.norm();
            } else if (is_unitary) {
                rho_t /= trace(rho_t);
                continue;
            } else if (!is_unitary_on_statevector) {
                std::complex<double> tr = 0;
                for (long i = 0; i < dim; ++i) {
                    long vec_index = i * dim + i;
                    tr += rho_t.coeff(vec_index, 0);
                }
                rho_t /= tr;
            }
        }

        // If we vectorized, need to devectorize before returning
        if (!is_unitary && !is_unitary_on_statevector) {
            rho_t = devectorize(rho_t);
        }

        return rho_t;
    }

    SparseMatrix iter_direct(const SparseMatrix& rho_0, double dt, const SparseMatrix& currentH, const std::vector<SparseMatrix>& jump_operators, bool is_unitary_on_statevector) {
        /*
        Perform time evolution using direct matrix exponentiation.

        Args:
            rho_0 (SparseMatrix): The initial density matrix, which should be vectorized.
            dt (double): The total time step.
            currentH (SparseMatrix): The current Hamiltonian.
            jump_operators (std::vector<SparseMatrix>): The list of jump operators.
            is_unitary_on_statevector (bool): Whether to treat the Hamiltonian as unitary on a statevector.

        Returns:
            SparseMatrix: The evolved density matrix after time dt.

        Raises:
            py::value_error: If currentH is not square.
            py::value_error: If rho_0 is not square.
            py::value_error: If Hamiltonian and initial density matrix dimensions do not match.
            py::value_error: If any jump operator dimension does not match Hamiltonian dimension.
        */

        // Sanity checks
        if (currentH.rows() != currentH.cols()) {
            throw py::value_error("Hamiltonian must be square.");
        }
        if (rho_0.cols() != rho_0.rows() && !is_unitary_on_statevector) {
            throw py::value_error("Initial density matrix must be square.");
        }
        long dim = long(currentH.rows());
        if (rho_0.rows() != dim) {
            throw py::value_error("Initial density matrix dimension does not match Hamiltonian dimension.");
        }
        for (const auto& J : jump_operators) {
            if (J.rows() != dim || J.cols() != dim) {
                throw py::value_error("Jump operator dimension does not match Hamiltonian dimension.");
            }
        }

        // If we're just doing unitary evolution on a statevector, we can exponentiate the Hamiltonian directly
        if (is_unitary_on_statevector) {
            SparseMatrix U = exp_mat(currentH, std::complex<double>(0, -dt));
            return U * rho_0;

            // If we have jump operators, need to form the full superoperator and act on the vectorized density matrix
        } else if (jump_operators.size() > 0) {
            SparseMatrix rho_t = vectorize(rho_0);
            SparseMatrix L = create_superoperator(currentH, jump_operators);
            rho_t = exp_mat_action(L, dt, rho_t);
            return devectorize(rho_t);

            // Otherwise we just exponentiate the Hamiltonian
        } else {
            SparseMatrix U = exp_mat(currentH, std::complex<double>(0, -dt));
            return U * rho_0 * U.adjoint();
        }
    }

   public:
    py::object execute_sampling(const py::object& functional, const py::dict& solver_params) {
        /*
        Execute a sampling functional using a simple statevector simulator.

        Args:
            functional (py::object): The Sampling functional to execute.
            solver_params (py::dict): Solver parameters, including 'max_cache_size'.

        Returns:
            SamplingResult: A result object containing the measurement samples and computed probabilities.

        Raises:
            py::value_error: If nqubits is non-positive.
            py::value_error: If shots is non-positive.
        */

        // Get info from the functional
        int n_shots = functional.attr("nshots").cast<int>();
        int n_qubits = functional.attr("circuit").attr("nqubits").cast<int>();

        // Get parameters
        int max_cache_size = 100;
        if (solver_params.contains("max_cache_size")) {
            max_cache_size = solver_params["max_cache_size"].cast<int>();
        }
        int num_threads = 1;
        if (solver_params.contains("num_threads")) {
            num_threads = solver_params["num_threads"].cast<int>();
        }

        // Set the number of threads
        if (num_threads <= 0) {
            num_threads = 1;
        }
        Eigen::setNbThreads(num_threads);

        // Sanity checks
        if (n_qubits <= 0) {
            throw py::value_error("nqubits must be positive.");
        }
        if (n_shots <= 0) {
            throw py::value_error("nshots must be positive.");
        }

        // Get the gate
        std::vector<Gate> gates = parse_gates(functional.attr("circuit"));

        // Determine which qubits to measure
        std::vector<bool> qubits_to_measure = parse_measurements(functional.attr("circuit"));

        // Start with the zero state
        long dim = 1L << n_qubits;
        DenseMatrix state = DenseMatrix::Zero(dim, 1);
        state(0, 0) = 1.0;

        // Determine the start/end use of each gate
        std::map<std::string, std::pair<int, int>> gate_first_last_use;
        for (int i = 0; i < int(gates.size()); ++i) {
            std::string gate_id = gates[i].get_id();
            if (gate_first_last_use.find(gate_id) == gate_first_last_use.end()) {
                gate_first_last_use[gate_id] = std::make_pair(i, i);
            } else {
                gate_first_last_use[gate_id].second = i;
            }
        }

        // Apply each gate (partial cache)
        SparseMatrix gate_matrix;
        std::map<std::string, SparseMatrix> gate_cache;
        std::string gate_id = "";
        int gate_count = 0;
        for (const auto& gate : gates) {
            gate_id = gate.get_id();

            // If we already have it in the cache, use it
            if (gate_cache.find(gate_id) != gate_cache.end()) {
                gate_matrix = gate_cache[gate_id];

                // If it will be used again later and we have space, cache it
            } else if (gate_first_last_use[gate_id].second > gate_count && int(gate_cache.size()) < max_cache_size) {
                gate_cache[gate_id] = gate.get_full_matrix(n_qubits);
                gate_matrix = gate_cache[gate_id];

                // Otherwise just generate it on the fly
            } else {
                gate_matrix = gate.get_full_matrix(n_qubits);
            }

            // Apply the gate (Sparse-Dense multiplication, OpenMP parallel if enabled)
            state = gate_matrix * state;

            // Renormalize the state
            state /= state.norm();

            // Clear the gate from the cache if this was its last use
            if (gate_first_last_use[gate_id].second == gate_count) {
                gate_cache.erase(gate_id);
            }
            gate_count++;
        }

        // Get the probabilities
        std::vector<std::tuple<int, double>> prob_entries;
        double total_prob = 0.0;
        for (int row = 0; row < state.rows(); ++row) {
            std::complex<double> amp = state(row, 0);
            double prob = std::norm(amp);
            if (prob > atol_) {
                prob_entries.emplace_back(row, prob);
                total_prob += prob;
            }
        }

        // Make sure probabilities sum to 1
        // const double probability_tolerance = 1e-5;
        // if (std::abs(total_prob - 1.0) > probability_tolerance) {
        if (std::abs(total_prob - 1.0) > atol_) {
            std::stringstream ss;
            ss << std::setprecision(10) << total_prob;
            throw py::value_error("Probabilities do not sum to 1 (sum = " + ss.str() + ")");
        }

        // Sample from these probabilities
        std::map<std::string, int> counts = sample_from_probabilities(prob_entries, n_qubits, n_shots);

        // Only keep measured qubits in the counts
        std::map<std::string, int> filtered_counts;
        for (const auto& pair : counts) {
            std::string bitstring = pair.first;
            std::string filtered_bitstring = "";
            for (int i = 0; i < n_qubits; ++i) {
                if (qubits_to_measure[i]) {
                    filtered_bitstring += bitstring[i];
                }
            }
            filtered_counts[filtered_bitstring] += pair.second;
        }
        counts = filtered_counts;

        // Convert counts to samples dict
        py::dict samples;
        for (const auto& pair : counts) {
            samples[py::cast(pair.first)] = py::cast(pair.second);
        }

        return SamplingResult("nshots"_a = n_shots, "samples"_a = samples);
    }

    // Sample from a density matrix
    SparseMatrix sample_from_density_matrix(const SparseMatrix& rho, int n_trajectories) {
        /*
        Get statevector samples from a density matrix, using the eigendecomposition.

        Args:
            rho (SparseMatrix): The input density matrix.
            n_trajectories (int): Number of trajectories.

        Returns:
            SparseMatrix: A matrix who's columns are the sampled statevectors.
        */

        // Eigendecompose the density matrix
        Eigen::SelfAdjointEigenSolver<DenseMatrix> es(rho);
        DenseMatrix evals = es.eigenvalues();
        DenseMatrix evecs = es.eigenvectors();
        std::vector<std::tuple<int, double>> prob_entries;
        double total_prob = 0.0;
        for (int i = 0; i < evals.size(); ++i) {
            double prob = evals(i).real();
            if (prob > atol_) {
                prob_entries.emplace_back(i, prob);
                total_prob += prob;
            }
        }

        // Make sure probabilities sum to 1
        if (std::abs(total_prob - 1.0) > atol_) {
            throw py::value_error("Probabilities from state do not sum to 1 (sum = " + std::to_string(total_prob) + ")");
        }

        // Sample from these probabilities
        int n_qubits = static_cast<int>(std::log2(rho.rows()));
        std::map<std::string, int> counts = sample_from_probabilities(prob_entries, n_qubits, n_trajectories);

        // Construct the sampled states matrix
        long dim = 1L << n_qubits;
        Triplets new_mat_entries;
        int traj_index = 0;
        for (const auto& pair : counts) {
            // Get the eigenvector corresponding to this bitstring
            std::string bitstring = pair.first;
            int count = pair.second;
            int eigenvec_index = std::stoi(bitstring, nullptr, 2);
            SparseMatrix state_vec = evecs.col(eigenvec_index).sparseView();

            // Normalize the state vector
            double norm = std::sqrt(state_vec.squaredNorm());
            if (norm > atol_) {
                state_vec /= norm;
            }

            // Add this state vector count times to the new matrix
            for (int i = 0; i < count; ++i) {
                for (int k = 0; k < state_vec.outerSize(); ++k) {
                    for (SparseMatrix::InnerIterator it(state_vec, k); it; ++it) {
                        int row = int(it.row());
                        std::complex<double> val = it.value();
                        new_mat_entries.emplace_back(Triplet(row, traj_index, val));
                    }
                }
                traj_index++;
            }
        }

        // Form the matrix from the triplets
        SparseMatrix sampled_states(dim, traj_index);
        sampled_states.setFromTriplets(new_mat_entries.begin(), new_mat_entries.end());

        return sampled_states;
    }

    // Convert a matrix containing trajectories as columns to a density matrix
    SparseMatrix trajectories_to_density_matrix(const SparseMatrix& trajectories) {
        /*
        Convert a matrix containing statevector trajectories as columns to a density matrix.
        If we have N trajectories |psi_i>, the density matrix is given by
        rho = 1/N sum_i |psi_i><psi_i|. Or, in matrix form, if the trajectories are columns of a matrix T,
        rho = 1/N T T^dagger.

        Args:
            trajectories (SparseMatrix): The input matrix with statevectors as columns.

        Returns:
            SparseMatrix: The corresponding density matrix.
        */
        SparseMatrix rho = trajectories * trajectories.adjoint();
        rho /= static_cast<double>(trajectories.cols());
        rho /= trace(rho);
        return rho;
    }

    // Execute time evolution
    py::object execute_time_evolution(const py::object& initial_state,
                                      const py::object& Hs,
                                      const py::object& coeffs,
                                      const py::object& steps,
                                      const py::object& observables,
                                      const py::object& jumps,
                                      bool store_intermediate_results,
                                      const py::dict& solver_params) {
        /*
        Execute a time evolution functional.

        Args:
            initial_state (py::object): The initial state as a QTensor.
            Hs (py::object): A list of Hamiltonians for time-dependent Hamiltonians.
            coeffs (py::object): A list of coefficients for the Hamiltonians at each time step.
            steps (py::object): A list of time steps at which to evaluate the evolution.
            observables (py::object): A list of observables to measure at each time step.
            jumps (py::object): A list of jump operators for the Lindblad equation.
            store_intermediate_results (bool): Whether to store results at each time step.
            params (py::dict): Additional parameters for the method. See the Python wrapper for details.

        Returns:
            TimeEvolutionResult: The results of the evolution.

        Raises:
            py::value_error: If no Hamiltonians are provided.
            py::value_error: If no time steps are provided.
            py::value_error: If number of parameters for any Hamiltonian does not match number of time steps.
            py::value_error: If an unknown time evolution method is specified.
        */

        // Parse the info from the python objects
        std::vector<SparseMatrix> hamiltonians = parse_hamiltonians(Hs);
        if (hamiltonians.size() == 0) {
            throw py::value_error("At least one Hamiltonian must be provided");
        }
        int nqubits = static_cast<int>(std::log2(hamiltonians[0].rows()));
        std::vector<SparseMatrix> observable_matrices = parse_observables(observables, nqubits);
        std::vector<std::vector<double>> parameters_list = parse_parameters(coeffs);
        std::vector<SparseMatrix> jump_operators = parse_jump_operators(jumps);
        std::vector<double> step_list = parse_time_steps(steps);
        SparseMatrix rho_0 = parse_initial_state(initial_state);

        // Get parameters
        int arnoldi_dim = 10;
        if (solver_params.contains("arnoldi_dim")) {
            arnoldi_dim = solver_params["arnoldi_dim"].cast<int>();
        }
        int num_arnoldi_substeps = 1;
        if (solver_params.contains("num_arnoldi_substeps")) {
            num_arnoldi_substeps = solver_params["num_arnoldi_substeps"].cast<int>();
        }
        int num_integrate_substeps = 1;
        if (solver_params.contains("num_integrate_substeps")) {
            num_integrate_substeps = solver_params["num_integrate_substeps"].cast<int>();
        }
        std::string method = "integrate";
        if (solver_params.contains("evolution_method")) {
            method = solver_params["evolution_method"].cast<std::string>();
        }
        bool monte_carlo = false;
        if (solver_params.contains("monte_carlo")) {
            monte_carlo = solver_params["monte_carlo"].cast<bool>();
        }
        int num_monte_carlo_trajectories = 100;
        if (solver_params.contains("num_monte_carlo_trajectories")) {
            num_monte_carlo_trajectories = solver_params["num_monte_carlo_trajectories"].cast<int>();
        }
        int num_threads = 1;
        if (solver_params.contains("num_threads")) {
            num_threads = solver_params["num_threads"].cast<int>();
        }

        // Set the number of threads
        if (num_threads <= 0) {
            num_threads = 1;
        }
        Eigen::setNbThreads(num_threads);

        // Sanity checks
        if (step_list.size() == 0) {
            throw py::value_error("At least one time step must be provided");
        }
        if (hamiltonians.size() != parameters_list.size()) {
            throw py::value_error("Number of Hamiltonians does not match number of parameter lists");
        }
        for (size_t h_ind = 0; h_ind < hamiltonians.size(); ++h_ind) {
            if (parameters_list[h_ind].size() != step_list.size()) {
                throw py::value_error("Number of parameters for Hamiltonian " + std::to_string(h_ind) + " does not match number of time steps");
            }
        }
        if (method != "direct" && method != "arnoldi" && method != "integrate") {
            throw py::value_error("Unknown time evolution method: " + method);
        }
        if (arnoldi_dim <= 0) {
            throw py::value_error("arnoldi_dim must be a positive integer");
        }
        if (num_arnoldi_substeps <= 0) {
            throw py::value_error("num_arnoldi_substeps must be a positive integer");
        }
        if (num_integrate_substeps <= 0) {
            throw py::value_error("num_integrate_substeps must be a positive integer");
        }
        if (num_monte_carlo_trajectories <= 0) {
            throw py::value_error("num_monte_carlo_trajectories must be a positive integer");
        }

        // Dimensions of everything
        int dim = int(hamiltonians[0].rows());

        // Check if we have unitary dynamics
        bool is_unitary_dynamics = (jump_operators.size() == 0);

        // Determine if the input was a state vector
        bool input_was_vector = false;
        if (rho_0.rows() == 1 || rho_0.cols() == 1) {
            input_was_vector = true;
        }
        if (rho_0.rows() == 1 && rho_0.cols() > 1) {
            rho_0 = rho_0.adjoint();
        }

        // Determine if should treat it as unitary evolution on a statevector
        // Note that this can change if the input was a density matrix but is actually pure
        // Or similarly if we use monte-carlo, we treat it as statevector evolution
        bool is_unitary_on_statevector = is_unitary_dynamics && input_was_vector;

        // If we have unitary dynamics and the input was a pure state, convert to state vector
        if (is_unitary_dynamics && !input_was_vector) {
            double trace_rho2 = 0.0;
            for (int k = 0; k < rho_0.outerSize(); ++k) {
                for (SparseMatrix::InnerIterator it1(rho_0, k); it1; ++it1) {
                    trace_rho2 += std::pow(std::abs(it1.value()), 2);
                }
            }
            if (std::abs(trace_rho2 - 1.0) < atol_) {
                rho_0 = get_vector_from_density_matrix(rho_0);
                is_unitary_on_statevector = true;
            }
        }

        // If we were told to do monte carlo, but we already have unitary dynamics, don't bother
        if (is_unitary_on_statevector) {
            monte_carlo = false;
        }

        // If we have non-unitary dynamics and the input was a state vector, convert to density matrix
        if (!is_unitary_dynamics && input_was_vector) {
            if (rho_0.rows() == 1) {
                rho_0 = rho_0.adjoint() * rho_0;
                input_was_vector = true;
            } else if (rho_0.cols() == 1) {
                rho_0 = rho_0 * rho_0.adjoint();
                input_was_vector = true;
            }
        }

        // If monte carlo, sample from rho_0 to get initial states
        // Then rho should be a collection of state vectors as columns
        if (monte_carlo) {
            rho_0 = sample_from_density_matrix(rho_0, num_monte_carlo_trajectories);
            is_unitary_on_statevector = true;
        }

        // Init rho_0
        SparseMatrix rho_t = rho_0;
        std::vector<SparseMatrix> intermediate_rhos;

        // Precalculate the sparsity pattern of the combined Hamiltonians
        SparseMatrix combinedH(dim, dim);
        for (size_t h_ind = 0; h_ind < hamiltonians.size(); ++h_ind) {
            combinedH += hamiltonians[h_ind];
        }
        combinedH.makeCompressed();
        combinedH *= 0.0;

        // For each time step
        for (size_t step_ind = 0; step_ind < step_list.size(); ++step_ind) {
            // Get the current Hamiltonian
            SparseMatrix currentH = combinedH;
            for (size_t h = 0; h < hamiltonians.size(); ++h) {
                double c = parameters_list[h][step_ind];
                for (int k = 0; k < hamiltonians[h].outerSize(); ++k) {
                    for (SparseMatrix::InnerIterator it(hamiltonians[h], k); it; ++it) {
                        currentH.coeffRef(it.row(), it.col()) += c * it.value();
                    }
                }
            }

            // Determine the time step
            double dt = step_list[step_ind];
            if (step_ind > 0) {
                dt = (step_list[step_ind] - step_list[step_ind - 1]);
            }

            // Perform the iteration depending on the method
            if (method == "integrate") {
                rho_t = iter_integrate(rho_t, dt, currentH, jump_operators, num_integrate_substeps, is_unitary_on_statevector);
            } else if (method == "direct") {
                rho_t = iter_direct(rho_t, dt, currentH, jump_operators, is_unitary_on_statevector);
            } else if (method == "arnoldi") {
                rho_t = iter_arnoldi(rho_t, dt, currentH, jump_operators, arnoldi_dim, num_arnoldi_substeps, is_unitary_on_statevector);
            }

            // If we should store intermediates, do it here
            if (store_intermediate_results) {
                if (monte_carlo || (!input_was_vector && rho_t.cols() == 1)) {
                    intermediate_rhos.push_back(trajectories_to_density_matrix(rho_t));
                } else {
                    intermediate_rhos.push_back(rho_t);
                }
            }
        }

        // If we have statevector/s but we should return a density matrix
        if (monte_carlo || (!input_was_vector && rho_t.cols() == 1)) {
            rho_t = trajectories_to_density_matrix(rho_t);
        }

        // Apply the operators using the Born rule
        std::vector<double> expectation_values;
        for (const auto& O : observable_matrices) {
            if (rho_t.cols() == 1) {
                expectation_values.push_back(std::real(dot(rho_t, O * rho_t)));
            } else {
                expectation_values.push_back(std::real(dot(O, rho_t)));
            }
        }

        // If we have intermediates, process them too
        std::vector<std::vector<double>> intermediate_expectation_values;
        if (store_intermediate_results) {
            for (const auto& rho_intermediate : intermediate_rhos) {
                std::vector<double> step_expectation_values;
                for (const auto& O : observable_matrices) {
                    if (rho_intermediate.cols() == 1) {
                        DenseMatrix rho_dense(rho_intermediate);
                        step_expectation_values.push_back(std::real(dot(rho_dense, O * rho_dense)));
                    } else {
                        step_expectation_values.push_back(std::real(dot(O, rho_intermediate)));
                    }
                }
                intermediate_expectation_values.push_back(step_expectation_values);
            }
        }

        // Convert things to numpy arrays
        py::array_t<std::complex<double>> rho_numpy = to_numpy(rho_t);
        py::array_t<double> expect_numpy = to_numpy(expectation_values);

        // Also convert intermediates if needed
        py::list intermediate_rho_numpy;
        py::array_t<double> intermediate_expect_numpy;
        if (store_intermediate_results) {
            for (const auto& rho_intermediate : intermediate_rhos) {
                py::array_t<std::complex<double>> rho_step_numpy = to_numpy(rho_intermediate);
                intermediate_rho_numpy.append(QTensor(rho_step_numpy));
            }
            intermediate_expect_numpy = to_numpy(intermediate_expectation_values);
        }

        // Return a TimeEvolutionResult with these
        return TimeEvolutionResult("final_state"_a = QTensor(rho_numpy), "final_expected_values"_a = expect_numpy, "intermediate_states"_a = intermediate_rho_numpy,
                                   "expected_values"_a = intermediate_expect_numpy);
    }
};

PYBIND11_MODULE(qilisim_module, m) {
    py::class_<QiliSimCpp>(m, "QiliSimCpp").def(py::init<>()).def("execute_sampling", &QiliSimCpp::execute_sampling).def("execute_time_evolution", &QiliSimCpp::execute_time_evolution);
}