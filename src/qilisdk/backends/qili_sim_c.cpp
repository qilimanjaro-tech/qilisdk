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
        int current_row = 0;
        int nnz_count = 0;
        for (auto &e : entries) {

            // Fill missing row offsets up to row r
            while (current_row < std::get<0>(e)) {
                rows_[current_row + 1] = nnz_count;
                current_row++;
            }

            // Add value
            cols_.push_back(std::get<1>(e));
            values_.push_back(std::get<2>(e));
            nnz_count++;

        }

        // Finish remaining rows
        for (int r = current_row; r < nrows_; ++r) {
            rows_[r + 1] = nnz_count;
        }

    }

    void clean() {
        /*
        Remove entries with absolute value below the tolerance.
        */
        std::vector<std::tuple<int, int, std::complex<double>>> entries = to_tuples();
        entries.erase(std::remove_if(entries.begin(), entries.end(),
                                     [](const std::tuple<int, int, std::complex<double>>& e) {
                                         return std::abs(std::get<2>(e)) < atol_;
                                     }),
                      entries.end());
        *this = SparseMatrix(entries, nrows_, ncols_);
    }

    SparseMatrix dagger() const {
        /*
        Compute the conjugate transpose of the sparse matrix.
        Returns:
            SparseMatrix: The conjugate transpose.
        */
        std::vector<std::tuple<int, int, std::complex<double>>> entries;
        for (int r = 0; r < nrows_; ++r) {
            for (int idx = rows_[r]; idx < rows_[r + 1]; ++idx) {
                int c = cols_[idx];
                std::complex<double> val = std::conj(values_[idx]);
                entries.emplace_back(c, r, val);
            }
        }
        // Sort entries by (row, col)
        std::sort(entries.begin(), entries.end(),
                  [](const std::tuple<int, int, std::complex<double>>& a,
                     const std::tuple<int, int, std::complex<double>>& b) {
                      if (std::get<0>(a) != std::get<0>(b)) {
                          return std::get<0>(a) < std::get<0>(b);
                      } else {
                          return std::get<1>(a) < std::get<1>(b);
                      }
                  });
        SparseMatrix result(entries, ncols_, nrows_);
        return result;
    }

    std::vector<std::complex<double>>::const_iterator begin() const {
        /*
        Get an iterator to the beginning of the values.
        Returns:
            std::vector<std::complex<double>>::const_iterator: Iterator to the beginning.
        */
        return values_.begin();
    }
    std::vector<std::complex<double>>::const_iterator end() const {
        /*
        Get an iterator to the end of the values.
        Returns:
            std::vector<std::complex<double>>::const_iterator: Iterator to the end.
        */
        return values_.end();
    }

    double norm() const {
        /*
        Compute the Frobenius norm of the sparse matrix.
        Returns:
            double: The Frobenius norm.
        */
        double sum = 0.0;
        for (const auto& val : values_) {
            sum += std::norm(val);
        }
        return std::sqrt(sum);
    }

    void add_scaled(const SparseMatrix& other, std::complex<double> scale) {
        /*
        Add a scaled version of another sparse matrix to this one.
        Args:
            other (SparseMatrix): The other sparse matrix.
            scale (double): The scaling factor.
        Raises:
            std::runtime_error: If matrix dimensions do not match for addition.
        */
        if (nrows_ != other.nrows_ || ncols_ != other.ncols_) {
            throw std::runtime_error("Matrix dimensions do not match for addition: " +
                                        get_dims_string() + " + " + other.get_dims_string());
        }
        for (int r = 0; r < nrows_; ++r) {
            for (int idx = other.rows_[r]; idx < other.rows_[r + 1]; ++idx) {
                int c = other.cols_[idx];
                std::complex<double> val = other.values_[idx] * scale;
                this->at(r, c) += val;
            }
        }
    }

    SparseMatrix& operator+=(const SparseMatrix& other) {
        /*
        Add another sparse matrix to this one in place.
        Args:
            other (SparseMatrix): The other sparse matrix.
        Returns:
            SparseMatrix&: Reference to the modified matrix.
        Raises:
            std::runtime_error: If matrix dimensions do not match for addition.
        */
        if (nrows_ != other.nrows_ || ncols_ != other.ncols_) {
            throw std::runtime_error("Matrix dimensions do not match for addition: " +
                                        get_dims_string() + " + " + other.get_dims_string());
        }
        for (int r = 0; r < nrows_; ++r) {
            for (int idx = other.rows_[r]; idx < other.rows_[r + 1]; ++idx) {
                int c = other.cols_[idx];
                std::complex<double> val = other.values_[idx];
                this->at(r, c) += val;
            }
        }
        return *this;
    }

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

    size_t size() const {
        /*
        Get the number of non-zero entries in the matrix.
        Returns:
            int: The number of non-zero entries.
        */
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
    int get_height() const {
        /*
        Get the number of rows in the matrix.
        Returns:
            int: The number of rows.
        */
        return nrows_;
    }

    bool has(int row, int col) const {
        /*
        Check if there is a non-zero value at (row, col).
        Args:
            row (int): The row index.
            col (int): The column index.
        Returns:
            bool: True if there is a non-zero value at (row, col), False otherwise.
        */
        for (int idx = rows_[row]; idx < rows_[row + 1]; ++idx) {
            if (cols_[idx] == col) {
                return true;
            }
        }
        return false;
    }

    std::complex<double>& at(int row, int col) {
        /*
        Get a reference to the value at (row, col). Inserts 0 if not present.
        Args:
            row (int): The row index.
            col (int): The column index.
        Returns:
            std::complex<double>&: Reference to the value at (row, col).
        */
        
        int row_start = rows_[row];
        int row_end = rows_[row + 1];
        
        // Assuming it's there
        for (int idx = row_start; idx < row_end; ++idx) {
            if (cols_[idx] == col) {
                return values_[idx];
            }
        }

        // If not found, insert a zero value and return the reference
        int insert_pos = row_start;
        while (insert_pos < row_end && cols_[insert_pos] < col) {
            insert_pos++;
        }
        cols_.insert(cols_.begin() + insert_pos, col);
        values_.insert(values_.begin() + insert_pos, 0.0);
        for (size_t r = row + 1; r < rows_.size(); ++r) {
            rows_[r]++;
        }
        rows_[rows_.size() - 1] = values_.size();
        return values_[insert_pos];

    }

    // Output operator
    friend std::ostream& operator<<(std::ostream& os, const SparseMatrix& matrix) {
        /*
        Output the sparse matrix in a readable format.
        e.g. std::cout << matrix;
        Args:
            os (std::ostream&): The output stream.
            matrix (SparseMatrix): The sparse matrix to output.
        Returns:
            std::ostream&: The output stream.
        */
        for (int r = 0; r < matrix.nrows_; ++r) {
            for (int c = 0; c < matrix.ncols_; ++c) {
                std::complex<double> val = matrix.get(r, c);
                os << val << " ";
            }
            os << std::endl;
        }
        return os;
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

    std::complex<double> dot(const SparseMatrix& other) const {
        /*
        Compute the Frobenius inner product with another sparse matrix.
        Args:
            other (SparseMatrix): The other sparse matrix.
        Returns:
            double: The Frobenius inner product.
        */
        std::complex<double> sum = 0.0;
        for (int r = 0; r < nrows_; ++r) {
            int idxA = rows_[r];
            int idxB = other.rows_[r];
            while (idxA < rows_[r + 1] && idxB < other.rows_[r + 1]) {
                if (cols_[idxA] == other.cols_[idxB]) {
                    sum += std::conj(values_[idxA]) * other.values_[idxB];
                    idxA++;
                    idxB++;
                } else if (cols_[idxA] < other.cols_[idxB]) {
                    idxA++;
                } else {
                    idxB++;
                }
            }
        }
        return sum;
    }

    SparseMatrix& operator/=(std::complex<double> scalar) {
        /*
        Divide the sparse matrix by a scalar in place.
        Args:
            scalar (double): The scalar to divide by.
        Returns:
            SparseMatrix&: Reference to the modified matrix.
        */
        for (auto& val : values_) {
            val /= scalar;
        }
        return *this;
    }

    SparseMatrix& operator*=(std::complex<double> scalar) {
        /*
        Multiply the sparse matrix by a scalar in place.
        Args:
            scalar (double): The scalar to multiply by.
        Returns:
            SparseMatrix&: Reference to the modified matrix.
        */
        for (auto& val : values_) {
            val *= scalar;
        }
        return *this;
    }

    SparseMatrix operator/(std::complex<double> scalar) const {
        /*
        Divide the sparse matrix by a scalar.
        Args:
            scalar (double): The scalar to divide by.
        Returns:
            SparseMatrix: The result of the division.
        */
        SparseMatrix result = *this;
        for (auto& val : result.values_) {
            val /= scalar;
        }
        return result;
    }

    SparseMatrix operator*(std::complex<double> scalar) const {
        /*
        Multiply the sparse matrix by a scalar.
        Args:
            scalar (double): The scalar to multiply by.
        Returns:
            SparseMatrix: The result of the multiplication.
        */
        SparseMatrix result = *this;
        for (auto& val : result.values_) {
            val *= scalar;
        }
        return result;
    }

    std::string get_dims_string() const {
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
                                        get_dims_string() + " * " + other.get_dims_string());
        }

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

        // Convert map to vector
        std::vector<std::tuple<int, int, std::complex<double>>> entries;
        for (const auto& entry : entries_map) {
            if (std::abs(entry.second) > atol_) {
                entries.emplace_back(std::get<0>(entry.first), std::get<1>(entry.first), entry.second);
            }
        }

        // Form result matrix
        SparseMatrix result(entries, nrows_, other.ncols_);
        return result;

    }

};

std::vector<std::tuple<int, int, std::complex<double>>> tensor_product(
                const std::vector<std::tuple<int, int, std::complex<double>>>& A,
                const std::vector<std::tuple<int, int, std::complex<double>>>& B,
                int A_width,
                int B_width
            ) {
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

SparseMatrix tensor_product(const SparseMatrix& A, const SparseMatrix& B) {
    /*
    Compute the tensor product of two sparse matrices.
    Args:
        A (SparseMatrix): First matrix.
        B (SparseMatrix): Second matrix.
    Returns:
        SparseMatrix: The tensor product matrix.
    */
    auto entries = tensor_product(A.to_tuples(), B.to_tuples(), A.get_width(), B.get_width());
    std::sort(entries.begin(), entries.end(),
              [](const std::tuple<int, int, std::complex<double>>& a,
                 const std::tuple<int, int, std::complex<double>>& b) {
                  if (std::get<0>(a) != std::get<0>(b)) {
                      return std::get<0>(a) < std::get<0>(b);
                  } else {
                      return std::get<1>(a) < std::get<1>(b);
                  }
              });
    SparseMatrix result(entries, A.get_height() * B.get_height(), A.get_width() * B.get_width());
    return result;
}

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

// Get the Python functional classes
py::object Sampling = py::module_::import("qilisdk.functionals.sampling").attr("Sampling");
py::object TimeEvolution = py::module_::import("qilisdk.functionals.time_evolution").attr("TimeEvolution");
py::object SamplingResult = py::module_::import("qilisdk.functionals.sampling").attr("SamplingResult");
py::object TimeEvolutionResult = py::module_::import("qilisdk.functionals.time_evolution").attr("TimeEvolutionResult");

// Needed for _a literals
using namespace pybind11::literals;

class QiliSimC {
private:

    // TODO
    SparseMatrix exp_mat_action(const SparseMatrix& H,
                                double dt) const {
        /*
        Compute the action of the matrix exponential exp(-i*H*dt) on the first basis vector e1.
        For now this just does a simple Taylor expansion.
        Args:
            H (SparseMatrix): The upper Hessenberg matrix.
            dt (double): The time step.
        Returns:
            SparseMatrix: The result of exp(-i*H*dt) * e1.
        */

        // Initialize everything
        int m = H.get_width();
        SparseMatrix y(m, 1);
        SparseMatrix term(m, 1);
        y.at(0, 0) = 1.0;
        term.at(0, 0) = 1.0;
        const int max_terms = 20;

        // Taylor expansion
        for (int k = 1; k < max_terms; ++k) {
            SparseMatrix next(m, 1);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    next.at(i, 0) += dt * H.get(i, j) * term.get(j, 0) / std::complex<double>(k, 0);
                }
            }
            term = next;
            y += term;
        }
        y.clean();
        return y;

    }

    // TODO
    void arnoldi(const SparseMatrix& L,
                 const SparseMatrix& v0,
                 int m,
                 std::vector<SparseMatrix>& V,
                 SparseMatrix& H) {
        /*
        Perform the Arnoldi iteration to build the Krylov basis.
        Args:
            L (SparseMatrix): The Lindblad superoperator.
            v0 (SparseMatrix): The initial vectorized density matrix.
            m (int): The dimension of the Krylov subspace.
            V (SparseMatrix&): Output Krylov basis vectors.
            H (SparseMatrix&): Output upper Hessenberg matrix.
        */

        // Set up the outputs
        int n = v0.get_height();
        V.clear();
        H = SparseMatrix(m + 1, m);

        // Normalize the initial vector
        SparseMatrix v = v0;
        double beta = v.norm();
        v /= beta;

        // Add the first vector to the list
        V.push_back(v);

        // For as many iterations as needed
        for (int j = 0; j < m; ++j) {

            // Apply the Lindbladian to the previous vector
            SparseMatrix w = L * V[j];

            // Orthogonalize against previous vectors
            for (int i = 0; i <= j; ++i) {
                std::complex<double> prod = V[i].dot(w);
                H.at(i, j) = prod;
                w.add_scaled(V[i], -prod);
            }

            // Update H and check for convergence
            double to_add = w.norm();
            H.at(j + 1, j) = to_add;
            if (to_add < atol_) {
                break;
            }

            // Normalize and add to V
            w /= to_add;
            V.push_back(w);

        }

    }

public:

    py::object execute_sampling(py::object functional) {
        /*
        Execute a sampling functional using a simple statevector simulator.
        Args:
            functional (py::object): The Sampling functional to execute.
        Returns:
            SamplingResult: A result object containing the measurement samples and computed probabilities.
        */

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

    py::object execute_time_evolution(py::object functional, py::object Hs, py::object coeffs, py::object steps, int arnoldi_dim=10) {
        /*
        Execute a time evolution functional using a Krylov subspace method.
        Args:
            functional (py::object): The TimeEvolution functional to execute.
            Hs (py::object): A list of Hamiltonians for time-dependent Hamiltonians.
            coeffs (py::object): A list of coefficients for the Hamiltonians at each time step.
            steps (py::object): A list of time steps at which to evaluate the evolution.
            arnoldi_dim (int): The dimension of the Krylov subspace to use.
        Returns:
            TimeEvolutionResult: The results of the evolution.
        */

        // Get the list of Hamiltonians and parameters
        std::vector<SparseMatrix> hamiltonians;
        for (auto& hamiltonian : Hs) {

            // Get the Hamiltonian matrix
            py::buffer matrix = hamiltonian.attr("to_matrix")().attr("toarray")();
            py::buffer_info buf = matrix.request();
            int rows = buf.shape[0];
            int cols = buf.shape[1];
            auto ptr = static_cast<std::complex<double>*>(buf.ptr);
            std::vector<std::tuple<int, int, std::complex<double>>> entries;
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    std::complex<double> val = ptr[r * cols + c];
                    if (std::abs(val) > atol_) {
                        entries.emplace_back(r, c, val);
                    }
                }
            }
            SparseMatrix H(entries, rows, cols);
            hamiltonians.push_back(H);

        }

        // Get the parameters
        std::vector<std::vector<double>> parameters_list;
        for (auto& param_set : coeffs) {
            std::vector<double> param_vector;
            for (auto& param : param_set) {
                param_vector.push_back(param.cast<double>());
            }
            parameters_list.push_back(param_vector);
        }

        // Get the time step
        std::vector<double> step_list;
        for (auto step : steps) {
            step_list.push_back(step.cast<double>());
        }

        // Dimensions of everything
        int dim = hamiltonians[0].get_height();
        int dim_rho = dim * dim;

        // Get the initial state
        SparseMatrix rho0;
        py::buffer init_state = functional.attr("initial_state").attr("dense")();
        py::buffer_info buf = init_state.request();
        int rows = buf.shape[0];
        int cols = buf.shape[1];
        auto ptr = static_cast<std::complex<double>*>(buf.ptr);
        std::vector<std::tuple<int, int, std::complex<double>>> rho0_entries;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                std::complex<double> val = ptr[r * cols + c];
                if (std::abs(val) > atol_) {
                    rho0_entries.emplace_back(r, c, val);
                }
            }
        }
        rho0 = SparseMatrix(rho0_entries, rows, cols);

        // It's a statevector, make it a density matrix
        if (rho0.get_height() == 1) {
            rho0 = rho0.dagger() * rho0;

        // It's a column vector, make it a density matrix
        } else if (rho0.get_width() == 1) {
            rho0 = rho0 * rho0.dagger();
        }

        // Vectorize rho0
        std::vector<std::tuple<int, int, std::complex<double>>> rho0_vec_entries;
        for (const auto& entry : rho0.to_tuples()) {
            int r = std::get<0>(entry);
            int c = std::get<1>(entry);
            std::complex<double> val = std::get<2>(entry);
            int vec_index = r * rho0.get_width() + c;
            rho0_vec_entries.emplace_back(vec_index, 0, val);
        }
        rho0 = SparseMatrix(rho0_vec_entries, dim_rho, 1);
        SparseMatrix rho_t = rho0;

        // For each time step
        for (int step_ind = 0; step_ind < step_list.size(); ++step_ind) {
            
            // Determine the time step size
            double dt = step_list[step_ind];
            if (step_ind > 0) {
                dt = step_list[step_ind] - step_list[step_ind - 1];
            }

            // For now just use the first Hamiltonian, later to be time-dependent TODO
            SparseMatrix currentH(dim, dim);
            for (int h_ind = 0; h_ind < hamiltonians.size(); ++h_ind) {
                currentH.add_scaled(hamiltonians[h_ind], parameters_list[h_ind][step_ind]);
            }

            // Form the Lindblad superoperator
            // for now just unitary evolution TODO
            std::vector<std::tuple<int, int, std::complex<double>>> H_entries = currentH.to_tuples();
            std::vector<std::tuple<int, int, std::complex<double>>> iden_entries;
            for (int i = 0; i < dim; ++i) {
                iden_entries.emplace_back(i, i, 1.0);
            }
            SparseMatrix iden(iden_entries, dim, dim);
            SparseMatrix H_iden = tensor_product(currentH, iden);
            SparseMatrix iden_H = tensor_product(iden, currentH);
            SparseMatrix L(dim_rho, dim_rho);
            L.add_scaled(H_iden, std::complex<double>(0, -1));
            L.add_scaled(iden_H, std::complex<double>(0, 1));
            
            // Run the Arnoldi iteration to build the Krylov basis
            // After this, we have L approximated in the Krylov basis as H
            // and the basis vectors in V
            std::vector<SparseMatrix> V;
            SparseMatrix H;
            arnoldi(L, rho_t, arnoldi_dim, V, H);

            // Compute the action of the matrix exponential
            // i.e. estimating exp(-i*L*dt) * rho0 via exp(-i*H*dt) * e1
            SparseMatrix y = exp_mat_action(H, dt);

            // Reconstruct the final density matrix using the basis vectors
            // SparseMatrix rho_t = V * y;
            SparseMatrix rho_t_new(rho0.get_height(), rho0.get_width());
            for (int j = 0; j < int(V.size()); ++j) {
                std::complex<double> coeff = y.get(j, 0);
                if (std::abs(coeff) > atol_) {
                    rho_t_new.add_scaled(V[j], coeff);
                }
            }
            rho_t = rho_t_new;

        }

        return TimeEvolutionResult();

    }

};

PYBIND11_MODULE(qili_sim_c, m) {
    py::class_<QiliSimC>(m, "QiliSimC")
        .def(py::init<>())
        .def("execute_sampling", &QiliSimC::execute_sampling)
        .def("execute_time_evolution", &QiliSimC::execute_time_evolution);
}