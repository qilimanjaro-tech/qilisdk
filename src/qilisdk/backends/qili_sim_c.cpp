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

#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <complex>
#include <sstream>
#include <tuple>
#include <map>
#include <random>
#include <stdexcept>
#include <algorithm>

#include <pybind11/pybind11.h>
namespace py = pybind11;

enum class GateType {
    H,
    X,
    Y,
    Z,
    RX,
    RY,
    RZ,
    U1,
    U2,
    U3,
    M,
    CNOT,
    UNKNOWN
};

const double atol_ = 1e-12;

class SparseMatrix {
    /*
    A simple sparse matrix representation using compressed sparse row (CSR) format.
    */
private:
    int nrows_ = 0;
    int ncols_ = 0;
    std::vector<int> rows_;
    std::vector<int> cols_;
    std::vector<std::complex<double>> values_;
public:
    SparseMatrix() {}
    SparseMatrix(int nrows, int ncols) : nrows_(nrows), ncols_(ncols) {
        /*
        Construct an empty sparse matrix of given dimensions.
        Args:
            nrows (int): Number of rows.
            ncols (int): Number of columns.
        */
        rows_.resize(nrows_ + 1, 0);
    }
    SparseMatrix(std::vector<std::vector<std::complex<double>>> dense) {
        /*
        Construct a sparse matrix from a dense 2D vector.
        Args:
            dense (std::vector<std::vector<std::complex<double>>>): The dense matrix.
        */
        nrows_ = dense.size();
        ncols_ = dense[0].size();
        rows_.resize(nrows_ + 1, 0);
        for (int r = 0; r < nrows_; ++r) {
            for (int c = 0; c < ncols_; ++c) {
                if (std::abs(dense[r][c]) > atol_) {
                    cols_.push_back(c);
                    values_.push_back(dense[r][c]);
                    rows_[r + 1]++;
                }
            }
        }
        // Convert counts to offsets
        for (int r = 1; r <= nrows_; ++r) {
            rows_[r] += rows_[r - 1];
        }
    }

    SparseMatrix(std::vector<std::tuple<int, int, std::complex<double>>> entries, int nrows, int ncols) {
        /*
        Construct a sparse matrix from a list of (row, col, value) tuples.
        The entries must be sorted by (row, col).

        Args:
            entries (std::vector<std::tuple<int, int, std::complex<double>>>): The non-zero entries.
            nrows (int): Number of rows.
            ncols (int): Number of columns.

        Throws:
            std::runtime_error: If entries are not sorted.
        */
        
        // Ensure entries are sorted
        bool sorted = true;
        for (size_t i = 1; i < entries.size(); ++i) {
            if (std::get<0>(entries[i]) < std::get<0>(entries[i - 1]) ||
                (std::get<0>(entries[i]) == std::get<0>(entries[i - 1]) &&
                 std::get<1>(entries[i]) < std::get<1>(entries[i - 1]))) {
                sorted = false;
                break;
            }
        }
        if (!sorted) {
            throw std::runtime_error("Entries must be sorted by (row, col)");
        }

        // Set the internal sizes
        nrows_ = nrows;
        ncols_ = ncols;

        // Size of the vectors
        rows_.resize(nrows_ + 2, 0);
        cols_.reserve(entries.size());
        values_.reserve(entries.size());

        // Fill CSR structure
        int currentRow = 0;
        int nnzCount = 0;
        for (auto &e : entries) {

            // Fill missing row offsets up to row r
            while (currentRow < std::get<0>(e)) {
                rows_[currentRow + 1] = nnzCount;
                currentRow++;
            }

            // Add value
            cols_.push_back(std::get<1>(e));
            values_.push_back(std::get<2>(e));
            nnzCount++;

        }

        // Finish remaining rows
        while (currentRow <= nrows_) {
            rows_[currentRow + 1] = nnzCount;
            currentRow++;
        }
    }

    // Get the vector of tuples of (row, col, value)
    std::vector<std::tuple<int, int, std::complex<double>>> to_tuples() const {
        /*
        Convert the sparse matrix to a list of (row, col, value) tuples.
        Returns:
            std::vector<std::tuple<int, int, std::complex<double>>>: The list of non-zero entries.
        */
        std::vector<std::tuple<int, int, std::complex<double>>> entries;
        for (int r = 0; r < nrows_; ++r) {
            for (int idx = rows_[r]; idx < rows_[r + 1]; ++idx) {
                entries.emplace_back(r, cols_[idx], values_[idx]);
            }
        }
        return entries;
    }

    // Iterators
    auto begin() const {
        return values_.begin();
    }
    auto end() const {
        return values_.end();
    }
    auto size() const {
        return values_.size();
    }

    int get_width() const {
        /*
        Get the number of columns in the matrix.
        Returns:
            int: The number of columns.
        */
        return ncols_;
    }

    std::complex<double> get(int row, int col) const {
        /*
        Get the value at (row, col). Returns 0 if not present.
        Args:
            row (int): The row index.
            col (int): The column index.
        Returns:
            std::complex<double>: The value at (row, col) or 0 if not present.
        */
        for (int idx = rows_[row]; idx < rows_[row + 1]; ++idx) {
            if (cols_[idx] == col) {
                return values_[idx];
            }
        }
        return 0.0;
    }

    void insert(int row, int col, std::complex<double> value) {
        /*
        Insert a non-zero value into the sparse matrix, maintaining sorted order.
        Args:
            row (int): The row index.
            col (int): The column index.
            value (std::complex<double>): The value to insert.
        */
        
        // Find the position to insert
        int start = rows_[row];
        int end = rows_[row + 1];
        int insertPos = start;
        while (insertPos < end && cols_[insertPos] < col) {
            insertPos++;
        }
        cols_.insert(cols_.begin() + insertPos, col);
        values_.insert(values_.begin() + insertPos, value);

        // Update row offsets
        for (size_t r = row + 1; r < rows_.size(); ++r) {
            rows_[r]++;
        }
        rows_[row + 1]++;
        
    }

    std::string get_dims() const {
        /*
        Get the dimensions of the matrix as a string.
        Returns:
            std::string: The dimensions in the format "nrows x ncols".
        */
        return std::to_string(nrows_) + "x" + std::to_string(ncols_);
    }

    SparseMatrix operator*(const SparseMatrix& other) const {
        /*
        Multiply two sparse matrices.
        Args:
            other (SparseMatrix): The other matrix to multiply.
        Returns:
            SparseMatrix: The result of the multiplication.
        Raises:
            std::runtime_error: If matrix dimensions do not match for multiplication.
        */
        // Check that they match
        if (ncols_ != other.nrows_) {
            throw std::runtime_error("Matrix dimensions do not match for multiplication: " +
                                        get_dims() + " * " + other.get_dims());
        }

        // std::cout << "First matrix:" << std::endl;
        // for (int r = 0; r < nrows_; ++r) {
        //     for (int idx = rows_[r]; idx < rows_[r + 1]; ++idx) {
        //         std::cout << "(" << r << ", " << cols_[idx] << ") = " << values_[idx] << std::endl;
        //     }
        // }
        // std::cout << "Second matrix:" << std::endl;
        // for (int r = 0; r < other.nrows_; ++r) {
        //     for (int idx = other.rows_[r]; idx < other.rows_[r + 1]; ++idx) {
        //         std::cout << "(" << r << ", " << other.cols_[idx] << ") = " << other.values_[idx] << std::endl;
        //     }
        // }
        
        // Perform multiplication
        std::map<std::tuple<int, int>, std::complex<double>> entries_map;
        for (size_t r = 0; r < rows_.size() - 1; ++r) {
            for (int idxA = rows_[r]; idxA < rows_[r + 1]; ++idxA) {
                int colA = cols_[idxA];
                std::complex<double> valA = values_[idxA];
                for (int idxB = other.rows_[colA]; idxB < other.rows_[colA + 1]; ++idxB) {
                    int colB = other.cols_[idxB];
                    std::complex<double> valB = other.values_[idxB];
                    entries_map[std::make_tuple(r, colB)] += valA * valB;
                }
            }
        }

        // std::cout << "Entries map:" << std::endl;
        // for (const auto& entry : entries_map) {
        //     std::cout << "(" << std::get<0>(entry.first) << ", " << std::get<1>(entry.first) << ") = " << entry.second << std::endl;
        // }

        // Convert map to vector
        std::vector<std::tuple<int, int, std::complex<double>>> entries;
        for (const auto& entry : entries_map) {
            if (std::abs(entry.second) > atol_) {
                entries.emplace_back(std::get<0>(entry.first), std::get<1>(entry.first), entry.second);
            }
        }

        SparseMatrix result(entries, nrows_, other.ncols_);
        return result;

    }

};

// Identity matrix constant
const SparseMatrix I({{1, 0}, {0, 1}});

class Gate {
    /*
    A quantum gate with type, control qubits and target qubits.
    */
private:
    GateType type;
    SparseMatrix base_matrix;
    std::vector<int> control_qubits;
    std::vector<int> target_qubits;

    std::vector<std::tuple<int, int, std::complex<double>>> tensor_product(
        const std::vector<std::tuple<int, int, std::complex<double>>>& A,
        const std::vector<std::tuple<int, int, std::complex<double>>>& B,
        int A_width,
        int B_width
    ) const {
        /*
        Compute the tensor product of two sets of (row, col, value) tuples.
        Args:
            A (std::vector<std::tuple<int, int, std::complex<double>>>): First matrix entries.
            B (std::vector<std::tuple<int, int, std::complex<double>>>): Second matrix entries.
            A_width (int): Width of the first matrix (assumed square).
            B_width (int): Width of the second matrix (assumed square).
        Returns:
            std::vector<std::tuple<int, int, std::complex<double>>>: The tensor product entries.
        */
        std::vector<std::tuple<int, int, std::complex<double>>> result;
        int row = 0;
        int col = 0;
        std::complex<double> val = 0.0;
        for (const auto& a : A) {
            for (const auto& b : B) {
                row = std::get<0>(a) * B_width + std::get<0>(b);
                col = std::get<1>(a) * B_width + std::get<1>(b);
                val = std::get<2>(a) * std::get<2>(b);
                result.emplace_back(row, col, val);
            }
        }
        return result;
    }

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
        int n = perm.size();
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

    SparseMatrix base_to_full(const SparseMatrix& base_gate,
                              int num_qubits,
                              const std::vector<int>& control_qubits,
                              const std::vector<int>& target_qubits) const {
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
            if (q < min_qubit) min_qubit = q;
        }
        for (int q : target_qubits) {
            if (q < min_qubit) min_qubit = q;
        }
        int base_gate_qubits = target_qubits.size() + control_qubits.size();
        int needed_before = min_qubit;
        int needed_after = num_qubits - needed_before - base_gate_qubits;
        std::vector<std::tuple<int, int, std::complex<double>>> out_entries = base_gate.to_tuples();

        // Make it controlled if needed
        // Reminder that control gates look like:
        // 1 0 0 0
        // 0 1 0 0
        // 0 0 G G
        // 0 0 G G
        for (int cq : control_qubits) {
            int delta = 1 << (target_qubits.size());
            std::vector<std::tuple<int, int, std::complex<double>>> new_entries;
            for (const auto& entry : out_entries) {
                int row = std::get<0>(entry);
                int col = std::get<1>(entry);
                std::complex<double> val = std::get<2>(entry);
                new_entries.emplace_back(row + delta, col + delta, val);
            }
            for (int i = 0; i < delta; ++i) {
                new_entries.emplace_back(i, i, 1.0);
            }
            out_entries = new_entries;
        }

        // Perform the tensor products with the identity
        std::vector<std::tuple<int, int, std::complex<double>>> identity_entries = {{0, 0, 1.0}, {1, 1, 1.0}};
        int gate_size = 1 << (target_qubits.size() + control_qubits.size());
        for (int i = 0; i < needed_before; ++i) {
            out_entries = tensor_product(identity_entries, out_entries, 2, gate_size);
            gate_size *= 2;
        }
        for (int i = 0; i < needed_after; ++i) {
            out_entries = tensor_product(out_entries, identity_entries, gate_size, 2);
            gate_size *= 2;
        }

        // Get a list of all qubits involved
        std::vector<int> all_qubits;
        for (auto q : control_qubits) all_qubits.push_back(q);
        for (auto q : target_qubits) all_qubits.push_back(q);

        // Determine the permutation to map qubits to their correct positions
        // perm is initially 0 1 2
        // if we have a CNOT on qubits 0 and 2, we actually did it on 0 and 1 internally
        // so we have 0 2 1 and we need to map back
        std::vector<int> perm(num_qubits);
        for (int q = 0; q < num_qubits; ++q) {
            perm[q] = q;
        }
        for (int i = 0; i < all_qubits.size(); ++i) {
            if (perm[needed_before + i] != all_qubits[i]) {
                std::swap(perm[needed_before + i], perm[all_qubits[i]]);
            }
        }
        // invert the perm
        std::vector<int> inv_perm(num_qubits);
        for (int q = 0; q < num_qubits; ++q) {
            inv_perm[perm[q]] = q;
        }
        perm = inv_perm;

        // Apply the permutation
        for (int i=0; i<out_entries.size(); ++i) {
            int old_row = std::get<0>(out_entries[i]);
            int old_col = std::get<1>(out_entries[i]);
            int new_row = permute_bits(old_row, perm);
            int new_col = permute_bits(old_col, perm);
            out_entries[i] = std::make_tuple(new_row, new_col, std::get<2>(out_entries[i]));
        }

        // Make sure the entries are sorted
        std::sort(out_entries.begin(), out_entries.end(),
                  [](const std::tuple<int, int, std::complex<double>>& a,
                     const std::tuple<int, int, std::complex<double>>& b) {
                      if (std::get<0>(a) != std::get<0>(b)) {
                          return std::get<0>(a) < std::get<0>(b);
                      } else {
                          return std::get<1>(a) < std::get<1>(b);
                      }
                  });

        // Form the matrix and return
        int full_size = 1 << num_qubits;
        return SparseMatrix(out_entries, full_size, full_size);

    }

public:

    // Constructor
    Gate(const SparseMatrix& base_matrix_,
         const std::vector<int>& controls_,
         const std::vector<int>& targets_) {
        control_qubits = controls_;
        target_qubits = targets_;
        base_matrix = base_matrix_;
    }

    SparseMatrix get_full_matrix(int num_qubits) const {
        /*
        Get the full matrix representation of the gate on the entire qubit register.
        Args:
            num_qubits (int): Total number of qubits in the register.
        Returns:
            SparseMatrix: The full matrix representation of the gate.
        */
        SparseMatrix full_gate = base_to_full(base_matrix, num_qubits, control_qubits, target_qubits);
        return full_gate;
    }
};

py::object Sampling = py::module_::import("qilisdk.functionals.sampling").attr("Sampling");
py::object TimeEvolution = py::module_::import("qilisdk.functionals.time_evolution").attr("TimeEvolution");
py::object SamplingResult = py::module_::import("qilisdk.functionals.sampling").attr("SamplingResult");
py::object TimeEvolutionResult = py::module_::import("qilisdk.functionals.time_evolution").attr("TimeEvolutionResult");

using namespace pybind11::literals;

class QiliSimC {
public:
    py::object execute_sampling(py::object functional) {

        // Get info from the functional
        int n_shots = functional.attr("nshots").cast<int>();
        int n_qubits = functional.attr("circuit").attr("nqubits").cast<int>();

        // Get the gates
        py::list py_gates = functional.attr("circuit").attr("gates");
        std::vector<Gate> gates;
        for (auto py_gate : py_gates) {
            
            // Get the name
            std::string gate_type_str = py_gate.attr("name").cast<std::string>();
            
            // Get the matrix
            py::buffer matrix = py_gate.attr("_generate_matrix")();
            py::buffer_info buf = matrix.request();
            int rows = buf.shape[0];
            int cols = buf.shape[1];
            auto ptr = static_cast<std::complex<double>*>(buf.ptr);
            std::vector<std::vector<std::complex<double>>> dense_matrix(rows, std::vector<std::complex<double>>(cols, 0.0));
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    dense_matrix[r][c] = ptr[r * cols + c];
                }
            }
            SparseMatrix base_matrix(dense_matrix);
            if (gate_type_str == "CNOT") {
                base_matrix = SparseMatrix({{0, 1},
                                            {1, 0}});
            }

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

            // Add the gate
            gates.emplace_back(base_matrix, controls, targets);

        }

        // Start with the zero state
        int dim = 1 << n_qubits;
        std::vector<std::tuple<int, int, std::complex<double>>> state_entries = {{0, 0, 1.0}};
        SparseMatrix state(state_entries, dim, dim);

        // Apply each gate
        for (const auto& gate : gates) {
            SparseMatrix gate_matrix = gate.get_full_matrix(n_qubits);
            state = gate_matrix * state;
        }

        // Get the probabilities
        std::vector<std::tuple<int, int, std::complex<double>>> amplitude_entries = state.to_tuples();
        std::vector<std::tuple<int, double>> prob_entries;
        double total_prob = 0.0;
        for (const auto& entry : amplitude_entries) {
            int row = std::get<0>(entry);
            std::complex<double> amp = std::get<2>(entry);
            double prob = std::norm(amp);
            if (prob > atol_) {
                prob_entries.emplace_back(row, prob);
                total_prob += prob;
            }
        }

        // Make sure probabilities sum to 1
        if (std::abs(total_prob - 1.0) > atol_) {
            throw std::runtime_error("Probabilities do not sum to 1 (sum = " + std::to_string(total_prob) + ")");
        }

        // Sample from the final state
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

        // Convert counts to samples dict
        py::dict samples;
        for (const auto& pair : counts) {
            samples[py::cast(pair.first)] = py::cast(pair.second);
        }

        return SamplingResult("nshots"_a=n_shots, "samples"_a=samples);

    }

};

PYBIND11_MODULE(qili_sim_c, m) {
    py::class_<QiliSimC>(m, "QiliSimC")
        .def(py::init<>())
        .def("execute_sampling", &QiliSimC::execute_sampling);
}