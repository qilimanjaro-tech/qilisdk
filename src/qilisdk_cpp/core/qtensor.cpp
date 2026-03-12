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

#include "qtensor.h"
#include "../libs/numpy.h"
#include "../libs/pybind.h"

#include <random>
#include <sstream>

DenseMatrix _get_dense_eigenvectors(const std::vector<SparseMatrix>& evecs) {
    /*
    Convert a vector of sparse matrices representing eigenvectors into a single dense matrix where each column is an eigenvector.

    Args:
        evecs (std::vector<SparseMatrix>): A vector of sparse matrices representing the eigenvectors.

    Returns:
        DenseMatrix: A dense matrix where each column is an eigenvector from the input vector.
    */
    if (evecs.empty()) {
        return DenseMatrix();
    }
    int nrows = int(evecs[0].rows());
    int ncols = int(evecs.size());
    DenseMatrix evecs_dense(nrows, ncols);
    for (int i = 0; i < ncols; ++i) {
        evecs_dense.col(i) = DenseMatrix(evecs[i]);
    }
    return evecs_dense;
}

QTensorCpp _reconstruct_from_diag(const Eigen::VectorXcd& evals_scaled, const std::vector<SparseMatrix>& evecs) {
    /*
    Reconstruct a dense matrix from a diagonal matrix of eigenvalues and a set of eigenvectors.

    Args:
        evals_scaled (Eigen::VectorXcd): A vector containing the scaled eigenvalues.
        evecs (std::vector<SparseMatrix>): A vector of sparse matrices representing the eigenvectors.

    Returns:
        DenseMatrix: The reconstructed dense matrix.
    */
    DenseMatrix evecs_dense = _get_dense_eigenvectors(evecs);
    DenseMatrix result_dense = evecs_dense * evals_scaled.asDiagonal() * evecs_dense.adjoint();
    return QTensorCpp(result_dense.sparseView());
}

void _validate_shape(const SparseMatrix& data) {
    /*
    Given a matrix, check it has a valid quantum shape, that is:
     - all dimensions are powers of two
     - it is either square or a vector

    Args:
        data (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>): The matrix to validate.

    Raises:
        py::value_error: If the input matrix is empty.
        py::value_error: If the input matrix does not have dimensions that are powers of 2.
        py::value_error: If the input matrix is not square and not a vector.
    */
    if (data.rows() == 0 || data.cols() == 0) {
        throw py::value_error("A QTensor should be initialized with a non-empty 2D array; got shape (" + std::to_string(data.rows()) + ", " + std::to_string(data.cols()) + ")");
    }
    if (((data.rows() & (data.rows() - 1)) != 0) || ((data.cols() & (data.cols() - 1)) != 0)) {
        throw py::value_error("A QTensor should have dimensions that are powers of 2; got shape (" + std::to_string(data.rows()) + ", " + std::to_string(data.cols()) + ")");
    }
    if (data.rows() != data.cols() && data.rows() != 1 && data.cols() != 1) {
        throw py::value_error("A QTensor should be either square or a vector; got shape (" + std::to_string(data.rows()) + ", " + std::to_string(data.cols()) + ")");
    }
}

QTensorCpp::QTensorCpp(int rows, int cols) {
    /*
    Construct a QTensor from the given number of rows and columns, which must be powers of 2.

    Args:
        rows (int): The number of rows in the tensor, must be a power of 2.
        cols (int): The number of columns in the tensor, must be a power of 2.

    Raises:
        py::value_error: If rows or cols are not positive.
        py::value_error: If rows or cols are not powers of 2.
    */
    _data = SparseMatrix(rows, cols);
    _validate_shape(_data);
}

QTensorCpp::QTensorCpp(const SparseMatrix& data) : _data(data) {
    /*
    Construct a QTensor from the given Eigen::SparseMatrix.

    Args:
        data (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>): The sparse matrix data for the tensor.
    */
    _validate_shape(_data);
}

QTensorCpp::QTensorCpp(const py::object& data) {
    /*
    Construct a QTensor from a py::object, which can be one of the following:
        - Another QTensorCpp
        - A list of lists representing a 2D array
        - A scipy sparse matrix (csr, csc, coo, or sparse array)
        - A numpy array
    The data is converted to an Eigen::SparseMatrix and stored in the _data member.

    Args:
        data (py::object): The input data to construct the QTensor from.
    */
    if (py::isinstance<QTensorCpp>(data)) {
        _data = data.cast<QTensorCpp>().get_data();
    } else if (py::hasattr(data, "_qtensor_cpp")) {
        _data = data.attr("_qtensor_cpp").cast<QTensorCpp>().get_data();
    } else if (py::isinstance<py::list>(data)) {
        py::list data_list = data.cast<py::list>();
        int rows = int(data_list.size());
        if (rows == 0) {
            _data = SparseMatrix(0, 0);
            return;
        }
        py::object first_row = data_list[0];
        if (!py::isinstance<py::list>(first_row)) {
            throw py::value_error("Data object must be a list of lists");
        }
        int cols = int(first_row.cast<py::list>().size());
        _data = SparseMatrix(rows, cols);
        for (int i = 0; i < rows; ++i) {
            py::object row_obj = data_list[i];
            if (!py::isinstance<py::list>(row_obj)) {
                throw py::value_error("Data object must be a list of lists");
            }
            py::list row = row_obj.cast<py::list>();
            if (int(row.size()) != cols) {
                throw py::value_error("All rows must have the same number of columns");
            }
            for (int j = 0; j < cols; ++j) {
                std::complex<double> val = row[j].cast<std::complex<double>>();
                if (std::abs(val) > default_atol) {
                    _data.insert(i, j) = val;
                }
            }
        }
        _data.makeCompressed();
    } else if (py::isinstance(data, csrmatrix) || py::isinstance(data, cscmatrix) || py::isinstance(data, coomatrix) || py::isinstance(data, sparray)) {
        _data = from_spmatrix(data, default_atol);
    } else if (py::isinstance(data, numpy_array_type)) {
        _data = from_numpy(data.cast<py::buffer>(), default_atol);
    } else {
        throw py::value_error("Data object must be a QTensor, a list of lists, a scipy sparse matrix, or a numpy array, got: " + std::string(py::str(data)));
    }
    _validate_shape(_data);
}

py::object QTensorCpp::as_scipy() const {
    /*
    Convert the internal Eigen::SparseMatrix representation of the QTensor to a scipy sparse matrix in Python.

    Returns:
        py::object: A scipy sparse matrix (csr, csc, coo, or sparse array) representing the same data as the QTensor.
    */
    return to_spmatrix(_data);
}

py::object QTensorCpp::as_numpy() const {
    /*
    Convert the internal Eigen::SparseMatrix representation of the QTensor to a numpy array in Python.

    Returns:
        py::object: A numpy array representing the same data as the QTensor.
    */
    return to_numpy(_data);
}

int QTensorCpp::get_nqubits() const {
    /*
    Get the number of qubits represented by the QTensor.

    Returns:
        int: The number of qubits represented by the QTensor.
    */
    int max_dim = int(std::max(_data.rows(), _data.cols()));
    return static_cast<int>(std::ceil(std::log2(max_dim)));
}

std::pair<int, int> QTensorCpp::get_shape() const {
    /*
    Get the shape of the QTensor as a pair of integers (rows, cols).

    Returns:
        std::pair<int, int>: A pair of integers representing the number of rows and columns in the QTensor.
    */
    return std::make_pair(int(_data.rows()), int(_data.cols()));
}

bool QTensorCpp::is_ket() const {
    /*
    Check if the QTensor represents a ket vector, which is defined as having a single column.

    Returns:
        bool: True if the QTensor is a ket vector, False otherwise.
    */
    return _data.cols() == 1;
}

bool QTensorCpp::is_bra() const {
    /*
    Check if the QTensor represents a bra vector, which is defined as having a single row.

    Returns:
        bool: True if the QTensor is a bra vector, False otherwise.
    */
    return _data.rows() == 1;
}

bool QTensorCpp::is_operator() const {
    /*
    Check if the QTensor represents an operator, which is defined as having more than one row and more than one column.

    Returns:
        bool: True if the QTensor is an operator, False otherwise.
    */
    return _data.rows() == _data.cols() && (_data.rows() & (_data.rows() - 1)) == 0 && _data.rows() > 0;
}

bool QTensorCpp::is_scalar() const {
    /*
    Check if the QTensor represents a scalar, which is defined as having a single row and a single column.

    Returns:
        bool: True if the QTensor is a scalar, False otherwise.
    */
    return _data.rows() == 1 && _data.cols() == 1;
}

bool QTensorCpp::is_symmetric(double atol) {
    /*
    Check if the QTensor is symmetric within a given absolute tolerance.
    This means A == A^T within the specified tolerance, where A^T is the transpose of A.

    Args:
        atol (double): The absolute tolerance for checking symmetry.

    Returns:
        bool: True if the QTensor is symmetric within the given tolerance, False otherwise
    */
    if (_data.rows() != _data.cols()) {
        return false;
    }
    if (!_symmetric_diff_computed) {
        for (int k = 0; k < _data.outerSize(); ++k) {
            for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
                int i = int(it.row());
                int j = int(it.col());
                std::complex<double> val = it.value();
                std::complex<double> other_val = _data.coeff(j, i);
                _max_symmetric_diff = std::max(_max_symmetric_diff, std::abs(val - other_val));
            }
        }
        _symmetric_diff_computed = true;
    }
    if (_max_symmetric_diff > atol) {
        return false;
    }
    return true;

}

bool QTensorCpp::is_self_adjoint(double atol) {
    /*
    Check if the QTensor is self-adjoint (Hermitian) within a given absolute tolerance.
    This means A == A^† within the specified tolerance, where A^† is the conjugate transpose of A.

    Args:
        atol (double): The absolute tolerance for checking self-adjointness.

    Returns:
        bool: True if the QTensor is self-adjoint within the given tolerance, False otherwise
    */
    if (_data.rows() != _data.cols()) {
        return false;
    }
    if (!_max_adjoint_diff_computed) {
        for (int k = 0; k < _data.outerSize(); ++k) {
            for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
                int i = int(it.row());
                int j = int(it.col());
                std::complex<double> val = it.value();
                std::complex<double> conj_val = std::conj(val);
                std::complex<double> other_val = _data.coeff(j, i);
                _max_adjoint_diff = std::max(_max_adjoint_diff, std::abs(conj_val - other_val));
            }
        }
        _max_adjoint_diff_computed = true;
    }
    if (_max_adjoint_diff > atol) {
        return false;
    }
    return true;
}

bool QTensorCpp::is_positive_semidefinite(double atol) {
    /*
    Check if the QTensor is positive semidefinite within a given absolute tolerance.
    Caches the result for faster subsequent checks with the same tolerance.

    Args:
        atol (double): The absolute tolerance for checking positive semidefiniteness.

    Returns:
        bool: True if the QTensor is positive semidefinite within the given tolerance, False otherwise.
    */
    if (_data.rows() != _data.cols()) {
        return false;
    }
    if (!_is_positive_computed || atol != _atol_used_for_positive) {
        // If we have the eigenvalues cached, we can check them directly
        if (!_eigenvalues.empty()) {
            for (const auto& eval : _eigenvalues) {
                if (eval.real() < -atol) {
                    _is_positive = false;
                    _is_positive_computed = true;
                    _atol_used_for_positive = atol;
                    return _is_positive;
                }
            }
            _is_positive = true;
            _is_positive_computed = true;
            _atol_used_for_positive = atol;
            return _is_positive;
        }

        // Otherwise try an LDLT decomposition with a shift to check for positive semidefiniteness
        SparseMatrix sparse_identity(_data.rows(), _data.cols());
        sparse_identity.setIdentity();
        Eigen::SimplicialLDLT<SparseMatrix> chol(_data + atol * sparse_identity);
        _is_positive = (chol.info() == Eigen::Success);
        _is_positive_computed = true;
        _atol_used_for_positive = atol;
    }

    return _is_positive;
}

void QTensorCpp::compute_eigendecomposition() {
    /*
    Compute the eigendecomposition of the QTensor,
    storing the eigenvalues and eigenvectors in the _eigenvalues and _eigenvectors members.
    Uses Eigen's SelfAdjointEigenSolver if the matrix is self-adjoint, and ComplexEigenSolver otherwise.
    Caches the results for faster subsequent access.
    */
    if (!_eigenvalues.empty() && !_eigenvectors.empty()) {
        return;
    }
    if (is_ket()) {
        _eigenvalues.push_back(1.0);
        _eigenvectors.push_back(_data);
        return;
    }
    if (is_bra()) {
        _eigenvalues.push_back(1.0);
        _eigenvectors.push_back(_data.transpose());
        return;
    }
    if (_data.rows() != _data.cols()) {
        throw py::value_error("Eigendecomposition is only defined for square matrices or vectors: got shape (" + std::to_string(_data.rows()) + ", " + std::to_string(_data.cols()) + ")");
    }
    DenseMatrix dense_data(_data);
    if (is_self_adjoint(default_atol)) {
        Eigen::SelfAdjointEigenSolver<DenseMatrix> es(dense_data);
        _eigenvectors.clear();
        auto evecs = es.eigenvectors();
        for (int i = 0; i < evecs.cols(); ++i) {
            SparseMatrix vec = evecs.col(i).sparseView();
            _eigenvectors.emplace_back(vec);
        }
        _eigenvalues.clear();
        auto evals = es.eigenvalues();
        for (int i = 0; i < evals.size(); ++i) {
            _eigenvalues.push_back(evals[i]);
        }
    } else {
        Eigen::ComplexEigenSolver<DenseMatrix> es(dense_data);
        _eigenvectors.clear();
        auto evecs = es.eigenvectors();
        for (int i = 0; i < evecs.cols(); ++i) {
            SparseMatrix vec = evecs.col(i).sparseView();
            _eigenvectors.emplace_back(vec);
        }
        _eigenvalues.clear();
        auto evs = es.eigenvalues();
        for (int i = 0; i < evs.size(); ++i) {
            _eigenvalues.push_back(evs[i]);
        }
    }
}

bool QTensorCpp::is_density_matrix(double atol) {
    /*
    Check if the QTensor represents a density matrix, which is defined
    as being an operator that is self-adjoint, has trace 1, and is positive
    semidefinite, all within a given absolute tolerance.

    Args:
        atol (double): The absolute tolerance for checking the properties of a density matrix.

    Returns:
        bool: True if the QTensor is a density matrix within the given tolerance, False otherwise.
    */
    if (!is_operator() || !is_self_adjoint(atol) || std::abs(trace().real() - 1.0) > atol || !is_positive_semidefinite(atol)) {
        return false;
    }
    return true;
}

bool QTensorCpp::is_pure(double atol) {
    /*
    Check if the QTensor represents a pure state, which is defined as being either a ket or a bra,
    or being a density matrix that satisfies the purity condition Tr(rho^2) = (Tr(rho))^2 within a given absolute tolerance.

    Args:
        atol (double): The absolute tolerance for checking the properties of a pure state.

    Returns:
        bool: True if the QTensor is a pure state within the given tolerance, False otherwise
    */
    if (is_ket() || is_bra()) {
        return true;
    }
    if (!is_density_matrix(atol)) {
        return false;
    }
    double tr = trace().real();
    if (!_trace_squared_computed) {
        _trace_squared = purity();
        _trace_squared_computed = true;
    }
    if (std::abs(_trace_squared - tr * tr) < atol) {
        return true;
    }
    return false;
}

QTensorCpp QTensorCpp::conjugate() const {
    /*
    Return a new QTensor that is the element-wise complex conjugate of this QTensor.

    Returns:
        QTensorCpp: A new QTensor that is the complex conjugate of this QTensor.
    */
    return QTensorCpp(_data.conjugate());
}

QTensorCpp QTensorCpp::transpose() const {
    /*
    Return a new QTensor that is the transpose of this QTensor.

    Returns:
        QTensorCpp: A new QTensor that is the transpose of this QTensor.
    */
    return QTensorCpp(_data.transpose());
}

QTensorCpp QTensorCpp::adjoint() const {
    /*
    Return a new QTensor that is the adjoint (conjugate transpose) of this QTensor.

    Returns:
        QTensorCpp: A new QTensor that is the adjoint of this QTensor.
    */
    return QTensorCpp(_data.adjoint());
}

std::complex<double> QTensorCpp::trace() {
    /*
    Compute the trace of the QTensor, which is defined as the sum of the diagonal elements.
    Caches the result for faster subsequent access.

    Returns:
        std::complex<double>: The trace of the QTensor.
    */
    if (_trace_computed) {
        return _trace;
    }
    _trace = 0.0;
    for (int k = 0; k < _data.outerSize(); ++k) {
        for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
            if (it.row() == it.col()) {
                _trace += it.value();
            }
        }
    }
    _trace_computed = true;
    return _trace;
}

QTensorCpp QTensorCpp::partial_trace_python(const py::object& keep) const {
    /*
    Perform a partial trace over the qubits not in the keep set.

    Args:
        keep (py::object): An iterable of integers representing the qubits to keep.

    Returns:
        QTensorCpp: The reduced density matrix after tracing out the other qubits.
    */
    std::set<int> keep_set;
    for (const auto& item : keep) {
        if (!py::isinstance<py::int_>(item)) {
            throw py::value_error("Keep set must be an iterable of integers");
        }
        keep_set.insert(item.cast<int>());
    }
    return partial_trace(keep_set);
}

QTensorCpp QTensorCpp::partial_trace(const std::set<int>& keep) const {
    /*
    Perform a partial trace over the qubits not in the keep set.

    Args:
        keep (std::set<int>): A set of integers representing the qubits to keep.

    Returns:
        QTensorCpp: The reduced density matrix after tracing out the other qubits.
    */

    // Turn into a density matrix if needed
    if (is_ket()) {
        QTensorCpp bra = this->adjoint();
        QTensorCpp density = this->matmul(bra);
        return density.partial_trace(keep);
    } else if (is_bra()) {
        QTensorCpp ket = this->adjoint();
        QTensorCpp density = ket.matmul(*this);
        return density.partial_trace(keep);
    }

    // The dimension and number of qubits
    int total_dim = std::max(_data.rows(), _data.cols());
    int n = static_cast<int>(std::log2(total_dim));

    // Check the indices
    for (int q : keep) {
        if (q < 0 || q >= n) {
            throw py::value_error("Keep set contains invalid qubit index: " + std::to_string(q));
        }
    }

    // Which qubits to keep and which to trace out
    std::vector<bool> kept(n, false);
    for (int q : keep) {
        kept[q] = true;
    }
    std::vector<int> trace_out;
    for (int q = 0; q < n; q++) {
        if (!kept[q]) {
            trace_out.push_back(q);
        }
    }

    // The dimension of the reduced system is 2^(number of kept qubits)
    int keep_dim = 1 << keep.size();

    // Function computing the index in the reduced system by extracting the bits corresponding to the kept and traced-out qubits
    auto extract_bits = [&](int flat_idx, const std::vector<int>& qubits) -> int {
        int result = 0;
        for (int i = 0; i < (int)qubits.size(); i++) {
            int q = qubits[qubits.size() - 1 - i];
            int bit = (flat_idx >> (n - 1 - q)) & 1;
            result |= (bit << i);
        }
        return result;
    };

    // Precompute the mapping from full indices to reduced indices for both rows and columns.
    const std::vector<int> keep_vec(keep.begin(), keep.end());
    std::vector<int> idx_keep(total_dim), idx_trace(total_dim);
    for (int i = 0; i < total_dim; i++) {
        idx_keep[i] = extract_bits(i, keep_vec);
        idx_trace[i] = extract_bits(i, trace_out);
    }

    // Accumulate into a map keyed by (row, col) of the reduced matrix.
    using Index = std::pair<int, int>;
    struct PairHash {
        size_t operator()(const Index& p) const { return std::hash<long long>()((long long)p.first << 32 | p.second); }
    };
    std::unordered_map<Index, std::complex<double>, PairHash> accum;
    accum.reserve(_data.nonZeros());
    for (int k = 0; k < _data.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(_data, k); it; ++it) {
            int row = int(it.row());
            int col = int(it.col());
            if (idx_trace[row] == idx_trace[col]) {
                accum[{idx_keep[row], idx_keep[col]}] += it.value();
            }
        }
    }

    // Convert map to triplets and build sparse result
    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
    triplets.reserve(accum.size());
    for (auto& [idx, val] : accum) {
        if (std::abs(val) > 0.0) {
            triplets.emplace_back(idx.first, idx.second, val);
        }
    }
    Eigen::SparseMatrix<std::complex<double>> result(keep_dim, keep_dim);
    result.setFromTriplets(triplets.begin(), triplets.end());
    return QTensorCpp(result);
}

double QTensorCpp::norm(const std::string& norm_type) {
    /*
    Compute the specified norm of the QTensor. Supported norms include:
        - "auto": Automatically choose the norm based on the type of QTensor (trace norm for operators, Frobenius norm for vectors)
        - "frobenius": The Frobenius norm (L2 norm of all elements)
        - "l1", "l2", etc.: The Lp norm of all elements
        - "trace": The trace norm (sum of singular values)
        - "nuclear": The nuclear norm (sum of singular values, same as trace norm for operators)
        - "inf": The infinity norm (maximum absolute row sum)

    Args:
        norm_type (std::string): The type of norm to compute.

    Returns:
        double: The computed norm of the QTensor.
    */
    if (norm_type == "auto") {
        if (is_operator()) {
            return norm("trace");
        } else {
            return norm("frobenius");
        }
    }
    if (norm_type == "frobenius") {
        return norm("l2");
    }
    if (norm_type[0] == 'l' && std::isdigit(norm_type[1])) {
        int norm_order = std::stoi(norm_type.substr(1));
        double sum = 0.0;
        for (int k = 0; k < _data.outerSize(); ++k) {
            for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
                sum += std::pow(std::abs(it.value()), norm_order);
            }
        }
        return std::pow(sum, 1.0 / norm_order);
    } else if (norm_type == "trace") {
        return trace().real();
    } else if (norm_type == "nuclear") {
        DenseMatrix dense_data(_data);
        Eigen::BDCSVD<DenseMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(dense_data);
        double sum_singular_values = svd.singularValues().array().abs().sum();
        return sum_singular_values;
    } else if (norm_type == "inf") {
        double max_abs_val = 0.0;
        for (int k = 0; k < _data.outerSize(); ++k) {
            for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
                max_abs_val = std::max(max_abs_val, std::abs(it.value()));
            }
        }
        return max_abs_val;
    } else {
        throw py::value_error("Unsupported norm type: " + norm_type);
    }
}

QTensorCpp QTensorCpp::normalized(const std::string& norm_type) {
    /*
    Return a new QTensor that is the normalized version of this QTensor according to the specified norm type.

    Args:
        norm_type (std::string): The type of norm to use for normalization.

    Returns:
        QTensorCpp: A new QTensor that is the normalized version of this QTensor.
    */
    double nrm = norm(norm_type);
    if (std::abs(nrm) < default_atol) {
        throw py::value_error("Cannot normalize a tensor with zero norm");
    }
    return QTensorCpp(_data / nrm);
}

QTensorCpp QTensorCpp::ket(const std::vector<int>& qubit_values) {
    /*
    Construct a ket vector QTensor from a list of integers representing the qubit values (0 or 1).

    Args:
        qubit_values (std::vector<int>): A list of integers representing the qubit values, where each integer is either 0 or 1.

    Returns:
        QTensorCpp: A QTensor representing the ket vector corresponding to the given qubit values
    */
    if (qubit_values.empty()) {
        throw py::value_error("Ket state cannot be empty");
    }
    for (int bit : qubit_values) {
        if (bit != 0 && bit != 1) {
            throw py::value_error("Ket state must be a list of 0s and 1s");
        }
    }
    int n = int(qubit_values.size());
    int dim = 1 << n;
    QTensorCpp result(dim, 1);
    int one_index = 0;
    for (size_t i = 0; i < qubit_values.size(); ++i) {
        if (qubit_values[qubit_values.size() - 1 - i] == 1) {
            one_index |= (1 << i);
        }
    }
    result._data.reserve(1);
    result._data.insert(one_index, 0) = 1.0;
    result._data.makeCompressed();
    return result;
}

QTensorCpp QTensorCpp::zero(int nqubits, std::string qtensor_type) {
    /*
    Construct a QTensor of the given shape filled with zeros. The number of rows and columns must be powers of 2.

    Args:
        nqubits (int): The number of qubits, which determines the shape of the QTensor. The number of rows and columns will be 2^nqubits.
        qtensor_type (std::string): The type of QTensor to create, which can be "ket", "bra", or "operator".

    Returns:
        QTensorCpp: A QTensor of the specified shape filled with zeros.
    */
    if (nqubits < 0) {
        throw py::value_error("Number of qubits must be non-negative");
    }
    int dim = 1 << nqubits;
    if (qtensor_type == "ket") {
        return QTensorCpp(dim, 1);
    } else if (qtensor_type == "bra") {
        return QTensorCpp(1, dim);
    } else if (qtensor_type == "operator") {
        return QTensorCpp(dim, dim);
    } else {
        throw py::value_error("Invalid qtensor type: " + qtensor_type + ". Must be 'ket', 'bra', or 'operator'");
    }
}

QTensorCpp QTensorCpp::ket_python(const py::object& state) {
    /*
    Construct a ket vector QTensor from a Python object, which should be an iterable of integers representing the qubit values (0 or 1).

    Args:
        state (py::object): A Python iterable of integers representing the qubit values, where each integer is either 0 or 1.

    Returns:
        QTensorCpp: A QTensor representing the ket vector corresponding to the given qubit values
    */
    std::vector<int> qubit_values;
    for (const auto& item : state) {
        if (!py::isinstance<py::int_>(item)) {
            throw py::value_error("Ket state must be a list of integers");
        }
        qubit_values.push_back(item.cast<int>());
    }
    return ket(qubit_values);
}

QTensorCpp QTensorCpp::bra(const std::vector<int>& qubit_values) {
    /*
    Construct a bra vector QTensor from a list of integers representing the qubit values (0 or 1).

    Args:
        qubit_values (std::vector<int>): A list of integers representing the qubit values, where each integer is either 0 or 1.

    Returns:
        QTensorCpp: A QTensor representing the bra vector corresponding to the given qubit values
    */
    if (qubit_values.empty()) {
        throw py::value_error("Bra state cannot be empty");
    }
    for (int bit : qubit_values) {
        if (bit != 0 && bit != 1) {
            throw py::value_error("Bra state must be a list of 0s and 1s");
        }
    }
    int n = int(qubit_values.size());
    int dim = 1 << n;
    QTensorCpp result(1, dim);
    int one_index = 0;
    for (size_t i = 0; i < qubit_values.size(); ++i) {
        if (qubit_values[qubit_values.size() - 1 - i] == 1) {
            one_index |= (1 << i);
        }
    }
    result._data.reserve(1);
    result._data.insert(0, one_index) = 1.0;
    result._data.makeCompressed();
    return result;
}

QTensorCpp QTensorCpp::bra_python(const py::object& state) {
    /*
    Construct a bra vector QTensor from a Python object, which should be an iterable of integers representing the qubit values (0 or 1).

    Args:
        state (py::object): A Python iterable of integers representing the qubit values, where each integer is either 0 or 1.

    Returns:
        QTensorCpp: A QTensor representing the bra vector corresponding to the given qubit values
    */
    std::vector<int> qubit_values;
    for (const auto& item : state) {
        if (!py::isinstance<py::int_>(item)) {
            throw py::value_error("Bra state must be a list of integers");
        }
        qubit_values.push_back(item.cast<int>());
    }
    return bra(qubit_values);
}

QTensorCpp QTensorCpp::tensor_product_python(const py::list& others) const {
    /*
    Compute the tensor product of a list of QTensors provided as a Python list.

    Args:
        others (py::list): A Python list of QTensors to tensor.

    Returns:
        QTensorCpp: The resulting QTensor that is the tensor product of the QTensors in the list.
    */
    std::vector<QTensorCpp> other_tensors;
    for (const auto& other : others) {
        if (py::isinstance<QTensorCpp>(other)) {
            other_tensors.push_back(other.cast<QTensorCpp>());
        } else if (py::hasattr(other, "_qtensor_cpp")) {
            other_tensors.push_back(other.attr("_qtensor_cpp").cast<QTensorCpp>());
        } else {
            throw py::value_error("All elements in the list must be a QTensor");
        }
    }
    return tensor_product(other_tensors);
}

QTensorCpp QTensorCpp::tensor_product(const std::vector<QTensorCpp>& others) const {
    /*
    Compute the tensor product of a list of QTensors.

    Args:
        others (std::vector<QTensorCpp>): A vector of QTensors to tensor.

    Returns:
        QTensorCpp: The resulting QTensor that is the tensor product of the QTensors in the list.
    */
    if (others.empty()) {
        throw py::value_error("The tensor product requires at least one tensor");
    }
    Triplets triplets{{0, 0, 1.0}};
    int total_rows = 1;
    int total_cols = 1;
    for (const auto& other : others) {
        const SparseMatrix& B = other.get_data();
        total_rows *= int(B.rows());
        total_cols *= int(B.cols());
        Triplets new_triplets(triplets.size() * B.nonZeros());
        int index = 0;
        for (const auto& tA : triplets) {
            for (int k = 0; k < B.outerSize(); ++k) {
                for (typename SparseMatrix::InnerIterator it(B, k); it; ++it) {
                    int row = int(tA.row() * B.rows() + it.row());
                    int col = int(tA.col() * B.cols() + it.col());
                    std::complex<double> val = tA.value() * it.value();
                    new_triplets[index++] = Triplet(row, col, val);
                }
            }
        }
        triplets = std::move(new_triplets);
    }
    SparseMatrix result(total_rows, total_cols);
    result.setFromTriplets(triplets.begin(), triplets.end());
    return QTensorCpp(result);
}

QTensorCpp QTensorCpp::add_python(const py::object& other) const {
    /*
    Add another QTensor or a scalar to this QTensor, returning a new QTensor as the result.

    Args:
        other (py::object): The other QTensor or scalar to add to this QTensor.

    Returns:
        QTensorCpp: A new QTensor that is the result of adding the other QTensor or scalar to this QTensor.
    */
    if (py::isinstance<QTensorCpp>(other)) {
        return add(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return add(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else if (py::isinstance<py::int_>(other) || py::isinstance<py::float_>(other) || py::isinstance(other, py_complex)) {
        if (is_scalar()) {
            std::complex<double> scalar = other.cast<std::complex<double>>();
            SparseMatrix scalar_matrix(_data.rows(), _data.cols());
            for (int i = 0; i < std::min(_data.rows(), _data.cols()); ++i) {
                scalar_matrix.insert(i, i) = scalar;
            }
            scalar_matrix.makeCompressed();
            return QTensorCpp(_data + scalar_matrix);
        } else if (std::abs(other.cast<std::complex<double>>()) < default_atol) {
            return *this;
        } else {
            throw py::type_error("unsupported operand type(s) for addition: 'QTensor' and '" + std::string(py::str(py::type::handle_of(other))) + "'. Addition of a scalar is only supported for 1x1 QTensors.");
        }
    } else {
        throw py::type_error("Addition is only supported with another QTensors or a scalar");
    }
}

QTensorCpp QTensorCpp::add(const QTensorCpp& other) const {
    /*
    Add another QTensor to this QTensor, returning a new QTensor as the result.

    Args:
        other (QTensorCpp): The other QTensor to add to this QTensor.

    Returns:
        QTensorCpp: A new QTensor that is the result of adding the other QTensor to this QTensor.
    */
    return QTensorCpp(_data + other.get_data());
}

QTensorCpp QTensorCpp::sub_python(const py::object& other) const {
    /*
    Subtract another QTensor or a scalar from this QTensor, returning a new QTensor as the result.

    Args:
        other (py::object): The other QTensor or scalar to subtract from this QTensor.

    Returns:
        QTensorCpp: A new QTensor that is the result of subtracting the other QTensor or scalar from this QTensor.
    */
    if (py::isinstance<QTensorCpp>(other)) {
        return sub(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return sub(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else {
        throw py::type_error("Subtraction is only supported between QTensors");
    }
}

QTensorCpp QTensorCpp::sub(const QTensorCpp& other) const {
    /*
    Subtract another QTensor from this QTensor, returning a new QTensor as the result.

    Args:
        other (QTensorCpp): The other QTensor to subtract from this QTensor.

    Returns:
        QTensorCpp: A new QTensor that is the result of subtracting the other QTensor from this QTensor.
    */
    return QTensorCpp(_data - other.get_data());
}

QTensorCpp QTensorCpp::mul_python(const py::object& other) const {
    /*
    Multiply this QTensor by another QTensor or a scalar, returning a new QTensor as the result.

    Args:
        other (py::object): The other QTensor or scalar to multiply with this QTensor.

    Returns:
        QTensorCpp: A new QTensor that is the result of multiplying this QTensor by the other QTensor or scalar.
    */
    if (py::isinstance<QTensorCpp>(other)) {
        return mul(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return mul(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else if (py::isinstance<py::int_>(other) || py::isinstance<py::float_>(other) || py::isinstance(other, py_complex)) {
        std::complex<double> scalar = other.cast<std::complex<double>>();
        return mul(scalar);
    } else {
        throw py::type_error("Multiplication is only supported between QTensors");
    }
}

QTensorCpp QTensorCpp::mul(std::complex<double> scalar) const {
    /*
    Multiply this QTensor by a scalar, returning a new QTensor as the result.

    Args:
        scalar (std::complex<double>): The scalar to multiply with this QTensor.

    Returns:
        QTensorCpp: A new QTensor that is the result of multiplying this QTensor by
    */
    return QTensorCpp(_data * scalar);
}

QTensorCpp QTensorCpp::mul(const QTensorCpp& other) const {
    /*
    Multiply this QTensor element-wise by another QTensor, returning a new QTensor as the result.

    Args:
        other (QTensorCpp): The other QTensor to multiply element-wise with this QTensor.

    Returns:
        QTensorCpp: A new QTensor that is the result of element-wise multiplying this QTensor by the other QTensor.
    */
    return QTensorCpp(_data.cwiseProduct(other.get_data()));
}

QTensorCpp QTensorCpp::matmul_python(const py::object& other) const {
    /*
    Perform matrix multiplication between this QTensor and another QTensor, returning a new QTensor as the result.

    Args:
        other (py::object): The other QTensor to matrix multiply with this QTensor.

    Returns:
        QTensorCpp: A new QTensor that is the result of matrix multiplying this QTensor by the other QTensor.
    */
    if (py::isinstance<QTensorCpp>(other)) {
        return matmul(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return matmul(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else {
        throw py::type_error("Matrix multiplication is only supported between QTensors");
    }
}

QTensorCpp QTensorCpp::matmul(const QTensorCpp& other) const {
    /*
    Perform matrix multiplication between this QTensor and another QTensor, returning a new QTensor as the result.

    Args:
        other (QTensorCpp): The other QTensor to matrix multiply with this QTensor.

    Returns:
        QTensorCpp: A new QTensor that is the result of matrix multiplying this QTensor by the other QTensor.
    */
    return QTensorCpp(_data * other.get_data());
}

bool QTensorCpp::equals_python(const py::object& other) const {
    /*
    Check if this QTensor is approximately equal to another QTensor, within a given absolute tolerance.

    Args:
        other (py::object): The other QTensor to compare with this QTensor.

    Returns:
        bool: True if this QTensor is approximately equal to the other QTensor, False otherwise.
    */
    if (py::isinstance<QTensorCpp>(other)) {
        return equals(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return equals(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else {
        return false;
    }
}

bool QTensorCpp::equals(const QTensorCpp& other) const {
    /*
    Check if this QTensor is approximately equal to another QTensor, within a given absolute tolerance.

    Args:
        other (QTensorCpp): The other QTensor to compare with this QTensor.

    Returns:
        bool: True if this QTensor is approximately equal to the other QTensor, False otherwise.
    */
    if (_data.rows() != other.get_data().rows() || _data.cols() != other.get_data().cols()) {
        return false;
    }
    return other.get_data().isApprox(_data);
}

std::string QTensorCpp::as_string() const {
    /*
    Return a string representation of the QTensor, including its shape, number of non-zero elements, and a dense representation of its data.

    Returns:
        std::string: A string representation of the QTensor.
    */
    DenseMatrix dense = _data;
    std::stringstream ss;
    ss << "QTensor(shape=" << _data.rows() << "x" << _data.cols() << ", nnz=" << _data.nonZeros() << "):" << std::endl;
    ss << dense;
    return ss.str();
}

QTensorCpp QTensorCpp::identity(int nqubits) {
    /*
    Construct an identity operator QTensor of the given dimension.

    Args:
        nqubits (int): The number of qubits, which determines the size of the identity matrix (2^nqubits x 2^nqubits).

    Returns:
        QTensorCpp: A QTensor representing the identity operator of the specified dimension.
    */
    int dim = 1 << nqubits;
    SparseMatrix id(dim, dim);
    for (int i = 0; i < dim; ++i) {
        id.insert(i, i) = 1.0;
    }
    id.makeCompressed();
    return QTensorCpp(id);
}

QTensorCpp QTensorCpp::as_density_matrix(double atol, double max_relative_correction) {
    /*
    Convert this QTensor to a density matrix. If the QTensor is already a valid density matrix
    within the given absolute tolerance, it is returned as is.
    If it is a ket or a bra, it is converted to a density matrix in the standard way.

    If it is an operator that is not a valid density matrix, an attempt is made to "repair" it by making it
    self-adjoint and positive semidefinite, and normalizing it to have trace 1.
    If the required correction to make it a valid density matrix is larger than the specified
    maximum relative correction (measured by the Frobenius norm), an error is raised.

    Args:
        atol (double): The absolute tolerance for checking if the QTensor is already a valid density matrix.
        max_relative_correction (double): The maximum allowed relative correction (measured by the Frobenius norm) when repairing an operator to make it a valid density matrix.

    Returns:
        QTensorCpp: A QTensor that is a valid density matrix corresponding to this QTensor
    */
    if (is_scalar()) {
        throw py::value_error("Cannot convert a scalar to a density matrix");
    } else if (is_ket()) {
        QTensorCpp mat = QTensorCpp(_data * _data.adjoint());
        return mat / mat.trace();
    } else if (is_bra()) {
        QTensorCpp mat = QTensorCpp(_data.adjoint() * _data);
        return mat / mat.trace();
        // If an operator, try to repair it
    } else if (is_operator()) {
        // Make self-adjoint by averaging with its adjoint
        QTensorCpp self_adjoint = (QTensorCpp(_data) + QTensorCpp(_data.adjoint())) * 0.5;

        // Shift eigenvalues to make positive semidefinite if necessary
        if (!self_adjoint.is_positive_semidefinite(atol)) {
            if (_eigenvalues.empty()) {
                self_adjoint.compute_eigendecomposition();
            }
            std::complex<double> min_eval = *std::min_element(_eigenvalues.begin(), _eigenvalues.end(), [](const std::complex<double>& a, const std::complex<double>& b) { return a.real() < b.real(); });
            double shift = std::max(0.0, -min_eval.real() + atol);
            SparseMatrix shift_matrix(self_adjoint.get_data().rows(), self_adjoint.get_data().cols());
            for (int i = 0; i < std::min(self_adjoint.get_data().rows(), self_adjoint.get_data().cols()); ++i) {
                shift_matrix.insert(i, i) = shift;
            }
            shift_matrix.makeCompressed();
            self_adjoint = QTensorCpp(self_adjoint.get_data() + shift_matrix);
        }

        // Normalize to have trace 1
        QTensorCpp return_tensor = self_adjoint / self_adjoint.trace();

        // Check the correction
        double correction = (return_tensor - self_adjoint).norm("frobenius") / self_adjoint.norm("frobenius");
        if (correction > max_relative_correction) {
            throw py::value_error("Repairing the density matrix required a large correction (relative Frobenius correction=" + std::to_string(correction) + "). This likely indicates that the original tensor is not close to a valid density matrix, and the result may not be meaningful.");
        }

        return return_tensor;

    } else {
        throw py::value_error("Only kets, bras or operators can be converted to a density matrix");
    }
}

QTensorCpp QTensorCpp::exp() {
    /*
    Compute the matrix exponential of this QTensor, returning a new QTensor as the result.
    If the eigendecomposition of this QTensor has already been computed, the exponential
    is computed more efficiently using the eigenvalues and eigenvectors.

    Returns:
        QTensorCpp: A new QTensor that is the matrix exponential of this QTensor.

    Raises:
        ValueError: If this QTensor is not an operator, since the matrix exponential is only defined for operators.
    */
    if (!is_operator()) {
        throw py::value_error("Matrix exponential is only defined for operators");
    }
    if (!_eigenvalues.empty() && !_eigenvectors.empty() && is_symmetric()) {
        Eigen::VectorXcd new_evals(_eigenvalues.size());
        new_evals.setZero();
        for (size_t i = 0; i < _eigenvalues.size(); ++i) {
            if (std::abs(_eigenvalues[i]) > 0) {
                new_evals(i) = std::exp(_eigenvalues[i]);
            }
        }
        return _reconstruct_from_diag(new_evals, _eigenvectors);
    }
    Eigen::MatrixXcd dense = _data;
    Eigen::MatrixXcd exp_dense = dense.exp();
    SparseMatrix exp_sparse = exp_dense.sparseView();
    return QTensorCpp(exp_sparse);
}

QTensorCpp QTensorCpp::log() {
    /*
    Compute the matrix logarithm of this QTensor, returning a new QTensor as the result.
    If the eigendecomposition of this QTensor has already been computed, the logarithm
    is computed more efficiently using the eigenvalues and eigenvectors.

    Returns:
        QTensorCpp: A new QTensor that is the matrix logarithm of this QTensor.

    Raises:
        ValueError: If this QTensor is not an operator, since the matrix logarithm is only defined for operators.
    */
    if (!is_operator()) {
        throw py::value_error("Matrix logarithm is only defined for operators");
    }
    if (!_eigenvalues.empty() && !_eigenvectors.empty() && is_symmetric()) {
        Eigen::VectorXcd new_evals(_eigenvalues.size());
        new_evals.setZero();
        for (size_t i = 0; i < _eigenvalues.size(); ++i) {
            if (std::abs(_eigenvalues[i]) > 0) {
                new_evals(i) = std::log(_eigenvalues[i]);
            }
        }
        return _reconstruct_from_diag(new_evals, _eigenvectors);
    }
    Eigen::MatrixXcd dense = _data;
    Eigen::MatrixXcd log_dense = dense.log();
    SparseMatrix log_sparse = log_dense.sparseView();
    return QTensorCpp(log_sparse);
}

QTensorCpp QTensorCpp::sqrt() {
    /*
    Compute the matrix square root of this QTensor, returning a new QTensor as the result.
    If the eigendecomposition of this QTensor has already been computed, the square root
    is computed more efficiently using the eigenvalues and eigenvectors.

    Returns:
        QTensorCpp: A new QTensor that is the matrix square root of this QTensor.

    Raises:
        ValueError: If this QTensor is not an operator, since the matrix square root is only defined for operators.
    */
    if (!is_operator()) {
        throw py::value_error("Matrix square root is only defined for operators");
    }
    if (!_eigenvalues.empty() && !_eigenvectors.empty() && is_self_adjoint()) {
        Eigen::VectorXcd new_evals(_eigenvalues.size());
        new_evals.setZero();
        for (size_t i = 0; i < _eigenvalues.size(); ++i) {
            if (std::abs(_eigenvalues[i]) > 0) {
                new_evals(i) = std::sqrt(_eigenvalues[i]);
            }
        }
        return _reconstruct_from_diag(new_evals, _eigenvectors);
    }
    // Handle if all zero (would give NaN in Eigen's sqrt)
    if (_data.nonZeros() == 0) {
        return QTensorCpp(_data);
    }
    Eigen::MatrixXcd dense = _data;
    Eigen::MatrixXcd sqrt_dense = dense.sqrt();
    SparseMatrix sqrt_sparse = sqrt_dense.sparseView();
    return QTensorCpp(sqrt_sparse);
}

QTensorCpp QTensorCpp::pow(double n) {
    /*
    Compute the matrix power of this QTensor to the integer n, returning a new QTensor as the result.
    If the eigendecomposition of this QTensor has already been computed, the power is computed
    more efficiently using the eigenvalues and eigenvectors.

    Args:
        n (float): The power to which to raise this QTensor.

    Returns:
        QTensorCpp: A new QTensor that is this QTensor raised to the power of n.

    Raises:
        ValueError: If this QTensor is not an operator, since the matrix power is only defined for operators.
    */
    if (!is_operator()) {
        throw py::value_error("Matrix power is only defined for operators");
    }
    if (!_eigenvalues.empty() && !_eigenvectors.empty() && is_self_adjoint()) {
        Eigen::VectorXcd new_evals(_eigenvalues.size());
        new_evals.setZero();
        for (size_t i = 0; i < _eigenvalues.size(); ++i) {
            if (std::abs(_eigenvalues[i]) > 0) {
                new_evals(i) = std::pow(_eigenvalues[i], n);
            }
        }
        return _reconstruct_from_diag(new_evals, _eigenvectors);
    }
    Eigen::MatrixXcd dense = _data;
    Eigen::MatrixXcd pow_dense = dense.pow(n);
    SparseMatrix pow_sparse = pow_dense.sparseView();
    return QTensorCpp(pow_sparse);
}

std::vector<std::complex<double>> QTensorCpp::get_eigenvalues() const {
    /*
    Return the eigenvalues of this QTensor. If the eigenvalues have not been computed yet, an error is raised.

    Returns:
        std::vector<std::complex<double>>: A vector containing the eigenvalues of this QTensor.

    Raises:
        py::value_error: If the eigenvalues have not been computed yet.
    */
    if (_eigenvalues.empty() && _data.rows() > 0 && _data.cols() > 0) {
        throw py::value_error("Eigenvalues have not been computed yet. Call compute_eigendecomposition() first.");
    }
    return _eigenvalues;
}

py::object QTensorCpp::get_eigenvalues_python() const {
    /*
    Return the eigenvalues of this QTensor as a Python list. If the eigenvalues have not been computed yet, an error is raised.

    Returns:
        py::object: A Python list containing the eigenvalues of this QTensor.

    Raises:
        py::value_error: If the eigenvalues have not been computed yet.
    */
    py::list evals;
    for (const auto& eval : get_eigenvalues()) {
        evals.append(eval);
    }
    return evals;
}

std::vector<SparseMatrix> QTensorCpp::get_eigenvectors() const {
    /*
    Return the eigenvectors of this QTensor as a vector of sparse matrices. If the eigenvectors have not been computed yet, an error is raised.

    Returns:
        std::vector<SparseMatrix>: A vector containing the eigenvectors of this QTensor as sparse matrices.

    Raises:
        py::value_error: If the eigenvectors have not been computed yet.
    */
    if (_eigenvectors.empty() && _data.rows() > 0 && _data.cols() > 0) {
        throw py::value_error("Eigenvectors have not been computed yet. Call compute_eigendecomposition() first.");
    }
    return _eigenvectors;
}

py::object QTensorCpp::get_eigenvectors_python() const {
    /*
    Return the eigenvectors of this QTensor as a Python list of sparse matrices. If the
    eigenvectors have not been computed yet, an error is raised.

    Returns:
        py::object: A Python list containing the eigenvectors of this QTensor as sparse matrices.

    Raises:
        py::value_error: If the eigenvectors have not been computed yet.
    */
    py::list evecs;
    for (const auto& evec : get_eigenvectors()) {
        evecs.append(to_spmatrix(evec));
    }
    return evecs;
}

void QTensorCpp::clear_cache() {
    /*
    Clear the cached properties of this QTensor, such as whether it is self-adjoint, positive semidefinite, its trace, eigenvalues, etc.
    */
    _is_positive_computed = false;
    _trace_computed = false;
    _trace_squared_computed = false;
    _max_adjoint_diff_computed = false;
    _max_unitary_diff_computed = false;
    _rank_computed = false;
    _eigenvalues.clear();
    _eigenvectors.clear();
}

std::complex<double> QTensorCpp::expectation_value_python(const py::object& other, int nshots) const {
    /*
    Compute the expectation value of another QTensor with respect to this QTensor.
    If nshots <= 0, compute the exact expectation value using the trace formula.
    If nshots > 0, compute an estimated expectation value by sampling from the
    eigenvalues of the operator according to the probabilities given by the overlaps of the eigenvectors with this state.

    Args:
        other (py::object): The other QTensor for which to compute the expectation value with respect to this QTensor.
        nshots (int): The number of samples to use when estimating the expectation value. If nshots <= 0, the exact expectation value is computed using the trace formula.

    Returns:
        std::complex<double>: The expectation value of the other QTensor with respect to this QTensor.
    */
    if (py::isinstance<QTensorCpp>(other)) {
        return expectation_value(other.cast<QTensorCpp>(), nshots);
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return expectation_value(other.attr("_qtensor_cpp").cast<QTensorCpp>(), nshots);
    } else {
        throw py::value_error("Other object must be a QTensor");
    }
}

std::complex<double> QTensorCpp::expectation_value(const QTensorCpp& other, int nshots) const {
    /*
    Compute the expectation value of another QTensor with respect to this QTensor.
    If nshots <= 0, compute the exact expectation value using the trace formula.
    If nshots > 0, compute an estimated expectation value by sampling from the eigenvalues of
    the operator according to the probabilities given by the overlaps of the eigenvectors with this state.

    Args:
        other (QTensorCpp): The other QTensor for which to compute the expectation value with respect to this QTensor.
        nshots (int): The number of samples to use when estimating the expectation value. If nshots <= 0, the exact expectation value is computed using the trace formula.

    Returns:
        std::complex<double>: The expectation value of the other QTensor with respect to this QTensor.
    */

    // If nshots <= 0, compute the exact expectation value using the trace formula
    if (nshots <= 0) {
        if (is_ket()) {
            return (adjoint() * other * (*this)).trace();
        } else if (is_bra()) {
            return ((*this) * other * adjoint()).trace();
        } else {
            return (other * (*this)).trace();
        }
    } else {
        // Create the random device
        std::random_device rd;
        std::mt19937 gen(rd());

        // Get the eigenvalues and eigenvectors of the operator
        if (_eigenvalues.empty() || _eigenvectors.empty()) {
            throw py::value_error("Eigendecomposition must be computed before calling expectation_value with nshots > 0");
        }
        const std::vector<std::complex<double>>& evals = _eigenvalues;
        const std::vector<SparseMatrix>& evecs = _eigenvectors;

        // Compute the probabilities of each outcome
        std::vector<double> probs(evals.size(), 0.0);
        for (size_t i = 0; i < evals.size(); ++i) {
            const SparseMatrix& evec = evecs[i];
            std::complex<double> overlap = 0.0;
            for (int k = 0; k < evec.outerSize(); ++k) {
                for (typename SparseMatrix::InnerIterator it(evec, k); it; ++it) {
                    int row = int(it.row());
                    int col = int(it.col());
                    std::complex<double> val = it.value();
                    overlap += val * _data.coeff(row, col);
                }
            }
            probs[i] = std::norm(overlap);
        }

        // Normalize probabilities
        double sum_probs = std::accumulate(probs.begin(), probs.end(), 0.0);
        if (sum_probs <= 0) {
            throw py::value_error("Invalid state: non-positive measurement probability normalization");
        }
        for (double& p : probs) {
            p /= sum_probs;
        }

        // Sample from the distribution
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        double sampled_mean = 0.0;
        for (int i = 0; i < nshots; ++i) {
            int outcome = dist(gen);
            sampled_mean += evals[outcome].real();
        }
        sampled_mean /= nshots;
        return sampled_mean;
    }
}

QTensorCpp QTensorCpp::commutator_python(const py::object& other) const {
    /*
    Compute the commutator of this QTensor with another QTensor, returning a new QTensor as the result.

    Args:
        other (py::object): The other QTensor to compute the commutator with.

    Returns:
        QTensorCpp: A new QTensor that is the commutator of this QTensor with the other QTensor.
    */
    if (py::isinstance<QTensorCpp>(other)) {
        return commutator(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return commutator(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else {
        throw py::value_error("Other object must be a QTensor");
    }
}

QTensorCpp QTensorCpp::commutator(const QTensorCpp& other) const {
    /*
    Compute the commutator of this QTensor with another QTensor, returning a new QTensor as the result.

    Args:
        other (QTensorCpp): The other QTensor to compute the commutator with.

    Returns:
        QTensorCpp: A new QTensor that is the commutator of this QTensor with the other QTensor.
    */
    return (*this * other) - (other * (*this));
}

QTensorCpp QTensorCpp::anticommutator_python(const py::object& other) const {
    /*
    Compute the anticommutator of this QTensor with another QTensor, returning a new QTensor as the result.

    Args:
        other (py::object): The other QTensor to compute the anticommutator with.

    Returns:
        QTensorCpp: A new QTensor that is the anticommutator of this QTensor with the other QTensor.
    */
    if (py::isinstance<QTensorCpp>(other)) {
        return anticommutator(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return anticommutator(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else {
        throw py::value_error("Other object must be a QTensor");
    }
}

QTensorCpp QTensorCpp::anticommutator(const QTensorCpp& other) const {
    /*
    Compute the anticommutator of this QTensor with another QTensor, returning a new QTensor as the result.

    Args:
        other (QTensorCpp): The other QTensor to compute the anticommutator with.

    Returns:
        QTensorCpp: A new QTensor that is the anticommutator of this QTensor with the other QTensor.
    */
    return (*this * other) + (other * (*this));
}

py::list QTensorCpp::probabilities_python() const {
    /*
    Compute the probabilities associated with this QTensor, depending on whether it is a ket, bra, or operator, and return them as a Python list.

    For a ket, the probabilities are given by the squared magnitudes of the coefficients in the standard basis.
    For a bra, the probabilities are given by the squared magnitudes of the coefficients in the standard basis.
    For an operator, the probabilities are given by the real parts of the diagonal elements in the standard basis.

    Returns:
        py::list: A Python list of probabilities associated with this QTensor, depending on its type (ket, bra, or operator).

    Raises:
        py::value_error: If this QTensor is not a ket, bra, or operator.

    */
    std::vector<double> probs = probabilities();
    py::list py_probs;
    for (double p : probs) {
        py_probs.append(p);
    }
    return py_probs;
}

std::vector<double> QTensorCpp::probabilities() const {
    /*
    Compute the probabilities associated with this QTensor, depending on whether it is a ket, bra, or operator.

    For a ket, the probabilities are given by the squared magnitudes of the coefficients in the standard basis.
    For a bra, the probabilities are given by the squared magnitudes of the coefficients in the standard basis.
    For an operator, the probabilities are given by the real parts of the diagonal elements in the standard basis.

    Returns:
        std::vector<double>: A vector of probabilities associated with this QTensor, depending on its type (ket, bra, or operator).

    Raises:
        py::value_error: If this QTensor is not a ket, bra, or operator.

    */
    if (is_ket()) {
        std::vector<double> probs(_data.rows(), 0.0);
        for (int k = 0; k < _data.outerSize(); ++k) {
            for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
                int row = it.row();
                std::complex<double> val = it.value();
                probs[row] += std::norm(val);
            }
        }
        return probs;
    } else if (is_bra()) {
        std::vector<double> probs(_data.cols(), 0.0);
        for (int k = 0; k < _data.outerSize(); ++k) {
            for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
                int col = it.col();
                std::complex<double> val = it.value();
                probs[col] += std::norm(val);
            }
        }
        return probs;
    } else if (is_operator()) {
        std::vector<double> probs(_data.rows(), 0.0);
        for (int k = 0; k < _data.outerSize(); ++k) {
            for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
                if (it.row() == it.col()) {
                    std::complex<double> val = it.value();
                    probs[it.row()] += val.real();
                }
            }
        }
        return probs;
    } else {
        throw py::value_error("Probabilities can only be computed for kets, bras, or operators");
    }
}

bool QTensorCpp::is_unitary(double atol) {
    /*
    Check if this QTensor is unitary within a given absolute tolerance.

    For an operator to be unitary, it must satisfy U * U^† = I, where U^† is the adjoint of U and I is the identity operator.

    Args:
        atol (double): The absolute tolerance for checking if this QTensor is unitary.

    Returns:
        bool: True if this QTensor is unitary within the given absolute tolerance, False otherwise.
    */
    if (!is_operator()) {
        return false;
    }
    if (!_max_unitary_diff_computed) {
        QTensorCpp product = (*this * adjoint());
        _max_unitary_diff = 0.0;
        for (int k = 0; k < product.get_data().outerSize(); ++k) {
            for (typename SparseMatrix::InnerIterator it(product.get_data(), k); it; ++it) {
                int row = int(it.row());
                int col = int(it.col());
                std::complex<double> val = it.value();
                std::complex<double> identity_val = (row == col) ? 1.0 : 0.0;
                _max_unitary_diff = std::max(_max_unitary_diff, std::abs(val - identity_val));
            }
        }
        _max_unitary_diff_computed = true;
    }
    return _max_unitary_diff <= atol;
}

double QTensorCpp::purity() {
    /*
    Compute the purity of this QTensor, defined as Tr(ρ^2) for a density matrix ρ.

    Returns:
        double: The purity of this QTensor if it is a density matrix, or NaN if it is not a density matrix.
    */
    if (!_trace_squared_computed) {
        QTensorCpp squared = (*this * (*this));
        _trace_squared = squared.trace().real();
        _trace_squared_computed = true;
    }
    return _trace_squared;
}

std::complex<double> QTensorCpp::dot(const QTensorCpp& other) const {
    /*
    Compute the dot product (inner product) between this QTensor and another QTensor, returning a complex number as the result.

    For two kets |ψ⟩ and |ϕ⟩, the dot product is defined as ⟨ψ|ϕ⟩.
    For two bras ⟨ψ| and ⟨ϕ|, the dot product is defined as ⟨ψ|ϕ⟩.
    For two operators A and B, the dot product is defined as Tr(A^† B).

    Args:
        other (QTensorCpp): The other QTensor to compute the dot product with this QTensor.

    Returns:
        std::complex<double>: The dot product between this QTensor and the other QTensor.
    */
    std::complex<double> result = 0.0;
    for (int k = 0; k < _data.outerSize(); ++k) {
        for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
            int row = int(it.row());
            int col = int(it.col());
            std::complex<double> val = it.value();
            std::complex<double> other_val = other.get_data().coeff(row, col);
            result += std::conj(val) * other_val;
        }
    }
    return result;
}

std::complex<double> QTensorCpp::dot_python(const py::object& other) const {
    /*
    Compute the dot product (inner product) between this QTensor and another QTensor, returning a complex number as the result.

    For two kets |ψ⟩ and |ϕ⟩, the dot product is defined as ⟨ψ|ϕ⟩.
    For two bras ⟨ψ| and ⟨ϕ|, the dot product is defined as ⟨ψ|ϕ⟩.
    For two operators A and B, the dot product is defined as Tr(A^† B).

    Args:
        other (py::object): The other QTensor to compute the dot product with this QTensor.

    Returns:
        std::complex<double>: The dot product between this QTensor and the other QTensor.

    Raises:
        py::value_error: If the other object is not a QTensor.
    */
    if (py::isinstance<QTensorCpp>(other)) {
        return dot(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return dot(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else {
        throw py::value_error("Other object must be a QTensor");
    }
}

double QTensorCpp::fidelity(const QTensorCpp& other) {
    /*
    Compute the fidelity between this QTensor and another QTensor, which is a measure of how close the two QTensors are to each other.

    For two kets |ψ⟩ and |ϕ⟩, the fidelity is defined as F(|ψ⟩, |ϕ⟩) = |⟨ψ|ϕ⟩|^2.
    For two bras ⟨ψ| and ⟨ϕ|, the fidelity is defined as F(⟨ψ|, ⟨ϕ|) = |⟨ψ|ϕ⟩|^2.
    For two operators ρ and σ, the fidelity is defined as F(ρ, σ) = (Tr(√(√ρ σ √ρ)))^2.

    Args:
        other (QTensorCpp): The other QTensor to compute the fidelity with respect to this QTensor.

    Returns:
        double: The fidelity between this QTensor and the other QTensor.

    Raises:
        py::value_error: If the two QTensors are not of the same type (both kets, both bras, or both operators).
    */
    if ((is_ket() && other.is_ket()) || (is_bra() && other.is_bra())) {
        return std::pow(std::abs(dot(other)), 2);
    } else if (is_operator() && other.is_operator()) {
        QTensorCpp sqrt_self = sqrt();
        QTensorCpp product = sqrt_self * other * sqrt_self;
        QTensorCpp sqrt_product = product.sqrt();
        return std::pow(sqrt_product.trace().real(), 2);
    } else {
        throw py::value_error("Fidelity can only be computed between states of the same type (ket, bra, or operator)");
    }
}

double QTensorCpp::fidelity_python(const py::object& other) {
    /*
    Compute the fidelity between this QTensor and another QTensor, which is a measure of how close the two QTensors are to each other.

    For two kets |ψ⟩ and |ϕ⟩, the fidelity is defined as F(|ψ⟩, |ϕ⟩) = |⟨ψ|ϕ⟩|^2.
    For two bras ⟨ψ| and ⟨ϕ|, the fidelity is defined as F(⟨ψ|, ⟨ϕ|) = |⟨ψ|ϕ⟩|^2.
    For two operators ρ and σ, the fidelity is defined as F(ρ, σ) = (Tr(√(√ρ σ √ρ)))^2.

    Args:
        other (py::object): The other QTensor to compute the fidelity with respect to this QTensor.

    Returns:
        double: The fidelity between this QTensor and the other QTensor.

    Raises:
        py::value_error: If the other object is not a QTensor, or if the two QTensors are not of the same type.
    */
    if (py::isinstance<QTensorCpp>(other)) {
        return fidelity(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return fidelity(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else {
        throw py::value_error("Other object must be a QTensor");
    }
}

double QTensorCpp::entropy_von_neumann() {
    /*
    Compute the von Neumann entropy of this QTensor, defined as S(ρ) = -Tr(ρ log(ρ)) for a density matrix ρ.

    Returns:
        double: The von Neumann entropy of this QTensor if it is a density matrix, or NaN if it is not a density matrix.

    Raises:
        py::value_error: If this QTensor is not a density matrix.
    */
    if (!is_density_matrix()) {
        throw py::value_error("Von Neumann entropy can only be computed for density matrices");
    }
    compute_eigendecomposition();
    double entropy = 0.0;
    for (size_t i = 0; i < _eigenvalues.size(); ++i) {
        double p = std::abs(_eigenvalues[i]);
        if (p > 0) {
            entropy -= p * std::log(p);
        }
    }
    return entropy;
}

double QTensorCpp::entropy_renyi(double alpha) {
    /*
    Compute the Renyi entropy of this QTensor for a given order alpha, defined as S_alpha(ρ) = (1/(1-alpha)) * log(Tr(ρ^alpha)) for a density matrix ρ.

    Args:
        alpha (double): The order of the Renyi entropy, must be greater than 0 and not equal to 1.

    Returns:
        double: The Renyi entropy of this QTensor if it is a density matrix, or NaN if it is not a density matrix.

    Raises:
        py::value_error: If this QTensor is not a density matrix, or if alpha is not greater than 0 or is equal to 1.
    */
    if (!is_density_matrix()) {
        throw py::value_error("Renyi entropy can only be computed for density matrices");
    }
    compute_eigendecomposition();
    if (alpha <= 0) {
        throw py::value_error("Alpha must be greater than 0");
    }
    if (alpha == 1) {
        throw py::value_error("Alpha cannot be 1 for Renyi entropy (use von Neumann entropy instead)");
    }
    double sum = 0.0;
    for (size_t i = 0; i < _eigenvalues.size(); ++i) {
        double p = std::abs(_eigenvalues[i]);
        sum += std::pow(p, alpha);
    }
    return (1.0 / (1.0 - alpha)) * std::log(sum);
}

DenseMatrix QTensorCpp::as_dense() const {
    /*
    Convert this QTensor to a dense matrix representation, returning a new DenseMatrix as the result.

    Returns:
        DenseMatrix: A new DenseMatrix that is the dense matrix representation of this QTensor.
    */
    return DenseMatrix(_data);
}

int QTensorCpp::rank() {
    /*
    Compute the rank of this QTensor. If the rank has already been computed, it is returned from the cache.
    If this QTensor is a ket or a bra, the rank is 1.

    Returns:
        int: The rank of this QTensor.
    */
    if (!_rank_computed) {
        if (is_ket() || is_bra()) {
            _rank = 1;
        } else if (!_eigenvalues.empty()) {
            _rank = 0;
            for (const auto& eval : _eigenvalues) {
                if (std::abs(eval) > 0) {
                    _rank++;
                }
            }
        } else {
            Eigen::MatrixXcd dense = _data;
            Eigen::FullPivLU<Eigen::MatrixXcd> lu(dense);
            _rank = int(lu.rank());
        }
        _rank_computed = true;
    }
    return _rank;
}

QTensorCpp QTensorCpp::inverse() {
    /*
    Compute the matrix inverse of this QTensor, returning a new QTensor as the result.

    If the eigendecomposition of this QTensor has already been computed, the inverse
    is computed more efficiently using the eigenvalues and eigenvectors.

    If the matrix is singular, this function will return a pseudo-inverse where the 
    inverse of zero eigenvalues is treated as zero.

    Returns:
        QTensorCpp: A new QTensor that is the matrix inverse of this QTensor.

    Raises:
        py::value_error: If this QTensor is not an operator.
    */
    if (!is_operator()) {
        throw py::value_error("Inverse can only be computed for operators");
    }
    if (!_eigenvalues.empty() && !_eigenvectors.empty() && is_self_adjoint()) {
        Eigen::VectorXcd inv_evals(_eigenvalues.size());
        inv_evals.setZero();
        for (size_t i = 0; i < _eigenvalues.size(); ++i) {
            if (std::abs(_eigenvalues[i]) > 0) {
                inv_evals(i) = 1.0 / _eigenvalues[i];
            }
        }
        return _reconstruct_from_diag(inv_evals, _eigenvectors);
    }
    Eigen::MatrixXcd dense = _data;
    Eigen::MatrixXcd inv_dense = dense.inverse();
    SparseMatrix inv_sparse = inv_dense.sparseView();
    return QTensorCpp(inv_sparse);
}

QTensorCpp QTensorCpp::ghz(int nqubits) {
    /*
    Create a GHZ state for the specified number of qubits, returning a new QTensor as the result.

    Args:
        nqubits (int): The number of qubits in the GHZ state.

    Returns:
        QTensorCpp: A new QTensor that is the GHZ state for the specified number of qubits.
    */
    int dim = 1 << nqubits;
    QTensorCpp result(dim, 1);
    int zero_index = 0;
    int one_index = dim - 1;
    result._data.insert(zero_index, 0) = 1.0 / std::sqrt(2.0);
    result._data.insert(one_index, 0) = 1.0 / std::sqrt(2.0);
    result._data.makeCompressed();
    return result;
}

QTensorCpp QTensorCpp::reset_qubits_python(const py::object& qubits) {
    /*
    Reset the specified qubits in this QTensor, returning a new QTensor as the result.

    Args:
        qubits (py::object): A Python iterable containing the indices of the qubits to reset.

    Returns:
        QTensorCpp: A new QTensor that is the result of resetting the specified qubits in this QTensor.
    */
    std::set<int> qubits_set;
    for (const auto& item : qubits) {
        qubits_set.insert(item.cast<int>());
    }
    return reset_qubits(qubits_set);
}

QTensorCpp QTensorCpp::reset_qubits(const std::set<int>& qubits) {
    /*
    Reset the specified qubits in this QTensor, returning a new QTensor as the result.

    Args:
        qubits (std::set<int>): A set of integers specifying the indices of the qubits to reset.

    Returns:
        QTensorCpp: A new QTensor that is the result of resetting the specified qubits in this QTensor.
    */

    // If not resetting any qubits, just return the original state
    if (qubits.empty()) {
        return *this;
    }

    // Check to make sure the input state is a density matrix and the qubits to reset are valid
    if (!is_density_matrix()) {
        throw py::value_error("reset_qubits requires a density matrix input state.");
    }

    // Validate qubit indices
    for (int q : qubits) {
        if (q < 0 || q >= get_nqubits()) {
            throw py::value_error("Invalid qubit indices");
        }
    }

    // Determine which qubits to keep and which to trace out
    std::set<int> rest_qubits;
    int nqubits = get_nqubits();
    for (int q = 0; q < nqubits; ++q) {
        if (qubits.find(q) == qubits.end()) {
            rest_qubits.insert(q);
        }
    }

    // Perform the partial trace over the qubits to be reset
    QTensorCpp rest_state = partial_trace(rest_qubits);
    int out_dim = 1 << nqubits;

    // If the resulting state is zero, we can just return a zero matrix of the appropriate size
    if (rest_state.get_data().nonZeros() == 0) {
        return QTensorCpp(SparseMatrix(out_dim, out_dim));
    }

    // Now we need to embed the rest_state back into the full space of nqubits, filling in zeros for the reset qubits
    if (!rest_qubits.empty()) {
        std::vector<Eigen::Triplet<std::complex<double>>> triplets;
        const SparseMatrix& rest_data = rest_state.get_data();
        std::vector<int> rest_dims(rest_qubits.size(), 2);
        std::vector<int> full_dims(nqubits, 2);
        for (int k = 0; k < rest_data.outerSize(); ++k) {
            for (typename SparseMatrix::InnerIterator it(rest_data, k); it; ++it) {
                int row = int(it.row());
                int col = int(it.col());
                std::complex<double> val = it.value();

                // Convert row and col to multi-dimensional indices in the rest subsystem
                std::vector<int> row_digits_rest(rest_qubits.size());
                std::vector<int> col_digits_rest(rest_qubits.size());
                int temp_row = row;
                int temp_col = col;
                for (size_t i = 0; i < rest_qubits.size(); ++i) {
                    row_digits_rest[i] = temp_row % 2;
                    col_digits_rest[i] = temp_col % 2;
                    temp_row /= 2;
                    temp_col /= 2;
                }

                // Build full multi-dimensional indices with zeros for the reset qubits
                std::vector<int> row_digits_full(nqubits, 0);
                std::vector<int> col_digits_full(nqubits, 0);
                size_t rest_idx = 0;
                for (int q = 0; q < nqubits; ++q) {
                    if (rest_qubits.find(q) != rest_qubits.end()) {
                        row_digits_full[q] = row_digits_rest[rest_idx];
                        col_digits_full[q] = col_digits_rest[rest_idx];
                        rest_idx++;
                    }
                }

                // Convert back to flat indices in the full space
                int out_row = 0;
                int out_col = 0;
                for (int q = 0; q < nqubits; ++q) {
                    out_row |= (row_digits_full[q] << q);
                    out_col |= (col_digits_full[q] << q);
                }

                triplets.emplace_back(out_row, out_col, val);
            }
        }

        // Construct the final sparse matrix in the full space
        SparseMatrix result(out_dim, out_dim);
        result.setFromTriplets(triplets.begin(), triplets.end());
        return QTensorCpp(result);
    }

    // If all qubits are reset, we just return a state with a single entry of 1 at the (0, 0) position
    SparseMatrix result(out_dim, out_dim);
    result.insert(0, 0) = 1.0;
    result.makeCompressed();
    return QTensorCpp(result);
}

QTensorCpp QTensorCpp::div(std::complex<double> scalar) const {
    /*
    Divide this QTensor by a scalar, returning a new QTensor as the result.
    If the scalar is zero (within a given absolute tolerance), an error is raised.

    Args:
        scalar (std::complex<double>): The scalar by which to divide this QTensor.

    Returns:
        QTensorCpp: A new QTensor that is the result of dividing this QTensor by the given scalar.

    Raises:
        py::value_error: If the scalar is zero (within a given absolute tolerance).
    */
    if (std::abs(scalar) <= default_atol) {
        throw py::value_error("Cannot divide by zero");
    }
    return QTensorCpp(_data / scalar);
}