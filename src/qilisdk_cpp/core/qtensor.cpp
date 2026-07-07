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
#include "../backends/qilisim/utils/parsers.h"
#include "../libs/numpy.h"
#include "../libs/pybind.h"

#include <random>
#include <sstream>

// GCOV_EXCL_BR_START

#ifndef _WIN32
#if defined(_OPENMP)
#pragma omp declare reduction(complex_double_reduction : std::complex<double> : omp_out += omp_in) initializer(omp_priv = std::complex<double>(0.0, 0.0))
#endif
#endif

DenseMatrix _get_dense_eigenvectors(const std::vector<SparseMatrix>& evecs) {
    /*
    Convert a vector of sparse matrices representing eigenvectors into a single dense matrix where each column is an eigenvector.

    Args:
        evecs (std::vector<SparseMatrix>): A vector of sparse matrices representing the eigenvectors.

    Returns:
        DenseMatrix: A dense matrix where each column is an eigenvector from the input vector.
    */
    // if (evecs.empty()) {
    //     return DenseMatrix();
    // }
    int ncols = int(evecs.size());
    int nrows = ncols > 0 ? evecs[0].rows() : 0;
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

void _validate_shape(Eigen::Index rows, Eigen::Index cols) {
    /*
    Given a shape, check it is a valid quantum shape, that is:
     - all dimensions are powers of two
     - it is either square or a vector

    Args:
        rows (Eigen::Index): The number of rows.
        cols (Eigen::Index): The number of columns.

    Raises:
        py::value_error: If the shape is empty.
        py::value_error: If the dimensions are not powers of 2.
        py::value_error: If the shape is not square and not a vector.
    */
    if (rows == 0 || cols == 0) {
        throw py::value_error("A QTensor should be initialized with a non-empty 2D array; got shape (" + std::to_string(rows) + ", " + std::to_string(cols) + ")");
    }
    if (((rows & (rows - 1)) != 0) || ((cols & (cols - 1)) != 0)) {
        throw py::value_error("A QTensor should have dimensions that are powers of 2; got shape (" + std::to_string(rows) + ", " + std::to_string(cols) + ")");
    }
    if (rows != cols && rows != 1 && cols != 1) {
        throw py::value_error("A QTensor should be either square or a vector; got shape (" + std::to_string(rows) + ", " + std::to_string(cols) + ")");
    }
}

void _validate_shape(const SparseMatrix& data) {
    _validate_shape(data.rows(), data.cols());
}

StorageFormat QTensorCpp::_choose_format(Eigen::Index rows, Eigen::Index cols, Eigen::Index nnz, StorageFormat requested) {
    /*
    Pick the storage backend for a tensor of the given shape and non-zero count.

    When `requested` is not Auto it is honoured verbatim (letting callers force a backend, e.g. to
    override the row-sparse default for a square operator). Otherwise:
     - dense (nnz > dense_storage_threshold * rows * cols)  -> Dense
     - a ket (more rows than columns)                       -> ColSparse
     - a bra or square tensor                               -> RowSparse

    Args:
        rows (Eigen::Index): The number of rows.
        cols (Eigen::Index): The number of columns.
        nnz (Eigen::Index): The number of stored non-zero entries.
        requested (StorageFormat): A forced backend, or Auto to derive one.

    Returns:
        StorageFormat: The concrete backend to use (never Auto).
    */
    if (requested != StorageFormat::Auto) {
        return requested;
    }
    const Eigen::Index size = rows * cols;
    if (size > 0 && static_cast<double>(nnz) > dense_storage_threshold * static_cast<double>(size)) {
        return StorageFormat::Dense;
    }
    if (rows > cols) {
        return StorageFormat::ColSparse;
    }
    return StorageFormat::RowSparse;
}

void QTensorCpp::_set_from_row_sparse(const SparseMatrix& data, StorageFormat requested) {
    /*
    Store `data` (a row-major sparse matrix) in the backend chosen for its shape/sparsity, converting
    the representation as needed.

    Args:
        data (SparseMatrix): The row-major sparse matrix to store.
        requested (StorageFormat): A forced backend, or Auto to derive one from the data.
    */
    _format = _choose_format(data.rows(), data.cols(), data.nonZeros(), requested);
    switch (_format) {
        case StorageFormat::ColSparse: {
            SparseMatrixCol col_major;
            col_major = data;
            _data = std::move(col_major);
            break;
        }
        case StorageFormat::Dense: {
            _data = DenseMatrix(data);
            break;
        }
        case StorageFormat::RowSparse:
        default: {
            _data = data;
            _format = StorageFormat::RowSparse;
            break;
        }
    }
}

Eigen::Index QTensorCpp::rows() const {
    /*
    Return the number of rows, regardless of the active storage backend.

    Returns:
        Eigen::Index: The number of rows in the tensor.
    */
    return std::visit([](const auto& mat) { return Eigen::Index(mat.rows()); }, _data);
}

Eigen::Index QTensorCpp::cols() const {
    /*
    Return the number of columns, regardless of the active storage backend.

    Returns:
        Eigen::Index: The number of columns in the tensor.
    */
    return std::visit([](const auto& mat) { return Eigen::Index(mat.cols()); }, _data);
}

Eigen::Index QTensorCpp::nnz() const {
    /*
    Return the number of stored non-zero entries, regardless of the active storage backend.

    Returns:
        Eigen::Index: The number of stored non-zero entries in the tensor.
    */
    return std::visit(
        [](const auto& mat) -> Eigen::Index {
            using M = std::decay_t<decltype(mat)>;
            if constexpr (std::is_same_v<M, DenseMatrix>) {
                Eigen::Index count = 0;
                for (Eigen::Index c = 0; c < mat.cols(); ++c) {
                    for (Eigen::Index r = 0; r < mat.rows(); ++r) {
                        if (mat(r, c) != std::complex<double>(0.0, 0.0)) {
                            ++count;
                        }
                    }
                }
                return count;
            } else {
                return Eigen::Index(mat.nonZeros());
            }
        },
        _data);
}

std::complex<double> QTensorCpp::coeff(int row, int col) const {
    /*
    Return the entry at (row, col), regardless of the active storage backend.

    Args:
        row (int): The row index of the entry.
        col (int): The column index of the entry.

    Returns:
        std::complex<double>: The value of the entry at (row, col).
    */
    return std::visit(
        [&](const auto& mat) -> std::complex<double> {
            using M = std::decay_t<decltype(mat)>;
            if constexpr (std::is_same_v<M, DenseMatrix>) {
                return mat(row, col);
            } else {
                return mat.coeff(row, col);
            }
        },
        _data);
}

SparseMatrix QTensorCpp::to_row_sparse() const {
    /*
    Return the data as a row-major sparse matrix, converting from the active backend if necessary.

    Returns:
        SparseMatrix: The data as a row-major sparse matrix.
    */
    return std::visit(
        [](const auto& mat) -> SparseMatrix {
            using M = std::decay_t<decltype(mat)>;
            if constexpr (std::is_same_v<M, DenseMatrix>) {
                return mat.sparseView();
            } else if constexpr (std::is_same_v<M, SparseMatrix>) {
                return mat;
            } else {
                SparseMatrix row_major;
                row_major = mat;
                return row_major;
            }
        },
        _data);
}

DenseMatrix QTensorCpp::to_dense() const {
    /*
    Return the data as a dense matrix, converting from the active backend if necessary.

    Returns:
        DenseMatrix: The data as a dense matrix.
    */
    return std::visit(
        [](const auto& mat) -> DenseMatrix {
            using M = std::decay_t<decltype(mat)>;
            if constexpr (std::is_same_v<M, DenseMatrix>) {
                return mat;
            } else {
                return DenseMatrix(mat);
            }
        },
        _data);
}

std::string QTensorCpp::get_format_string() const {
    /*
    Return a human-readable name of the active storage backend (for introspection and tests).

    Returns:
        std::string: The name of the active storage backend.
    */
    switch (_format) {
        case StorageFormat::ColSparse:
            return "col_sparse";
        case StorageFormat::Dense:
            return "dense";
        case StorageFormat::RowSparse:
        default:
            return "row_sparse";
    }
}

QTensorCpp::QTensorCpp(int rows, int cols) {
    /*
    Construct an empty QTensor with the given number of rows and columns, which must be powers of 2.
    The storage backend is chosen automatically from the shape.

    Args:
        rows (int): The number of rows in the tensor, must be a power of 2.
        cols (int): The number of columns in the tensor, must be a power of 2.

    Raises:
        py::value_error: If rows or cols are not positive.
        py::value_error: If rows or cols are not powers of 2.
    */
    _validate_shape(Eigen::Index(rows), Eigen::Index(cols));
    _set_from_row_sparse(SparseMatrix(rows, cols));
}

QTensorCpp::QTensorCpp(const SparseMatrix& data) {
    /*
    Construct a QTensor from the given row-major Eigen::SparseMatrix, choosing the storage backend
    automatically from the data's shape and sparsity.

    Args:
        data (SparseMatrix): The row-major sparse matrix data for the tensor.
    */
    _validate_shape(data);
    _set_from_row_sparse(data);
}

QTensorCpp::QTensorCpp(const SparseMatrix& data, StorageFormat format) {
    /*
    Construct a QTensor from the given row-major Eigen::SparseMatrix, forcing a specific storage
    backend (or Auto to derive one). Use this to override the row-sparse default, e.g. to store a
    square operator densely or a square matrix column-major.

    Args:
        data (SparseMatrix): The row-major sparse matrix data for the tensor.
        format (StorageFormat): The storage backend to use, or Auto to derive one from the data.
    */
    _validate_shape(data);
    _set_from_row_sparse(data, format);
}

QTensorCpp::QTensorCpp(const SparseMatrix& data, bool no_checks) {
    /*
    Construct a QTensor from the given row-major Eigen::SparseMatrix, optionally skipping shape
    validation. The storage backend is chosen automatically.

    Args:
        data (SparseMatrix): The row-major sparse matrix data for the tensor.
        no_checks (bool): If true, skip shape validation.
    */
    if (!no_checks) {
        _validate_shape(data);
    }
    _set_from_row_sparse(data);
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
        const QTensorCpp& other = data.cast<QTensorCpp>();
        _data = other._data;
        _format = other._format;
        return;
    } else if (py::hasattr(data, "_qtensor_cpp")) {
        const QTensorCpp& other = data.attr("_qtensor_cpp").cast<QTensorCpp>();
        _data = other._data;
        _format = other._format;
        return;
    }
    SparseMatrix row_major;
    if (py::isinstance<py::list>(data)) {
        py::list data_list = data.cast<py::list>();
        int rows = int(data_list.size());
        if (rows == 0) {
            _set_from_row_sparse(SparseMatrix(0, 0));
            return;
        }
        py::object first_row = data_list[0];
        if (!py::isinstance<py::list>(first_row)) {
            throw py::value_error("Data object must be a list of lists");
        }
        int cols = int(first_row.cast<py::list>().size());
        row_major = SparseMatrix(rows, cols);
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
                    row_major.insert(i, j) = val;
                }
            }
        }
        row_major.makeCompressed();
    } else if (py::isinstance(data, csrmatrix) || py::isinstance(data, cscmatrix) || py::isinstance(data, coomatrix) || py::isinstance(data, sparray)) {
        row_major = from_spmatrix(data, default_atol);
    } else if (py::isinstance(data, numpy_array_type)) {
        row_major = from_numpy(data.cast<py::buffer>(), default_atol);
    } else {
        throw py::value_error("Data object must be a QTensor, a list of lists, a scipy sparse matrix, or a numpy array, got: " + std::string(py::str(data)));
    }
    _validate_shape(row_major);
    _set_from_row_sparse(row_major);
}

py::object QTensorCpp::as_scipy() const {
    /*
    Convert the internal Eigen::SparseMatrix representation of the QTensor to a scipy sparse matrix in Python.

    Returns:
        py::object: A scipy sparse matrix (csr, csc, coo, or sparse array) representing the same data as the QTensor.
    */
    return to_spmatrix(to_row_sparse());
}

py::object QTensorCpp::as_numpy() const {
    /*
    Convert the internal Eigen::SparseMatrix representation of the QTensor to a numpy array in Python.

    Returns:
        py::object: A numpy array representing the same data as the QTensor.
    */
    return to_numpy(to_row_sparse());
}

int QTensorCpp::get_nqubits() const {
    /*
    Get the number of qubits represented by the QTensor.

    Returns:
        int: The number of qubits represented by the QTensor.
    */
    int max_dim = int(std::max(rows(), cols()));
    return static_cast<int>(std::ceil(std::log2(max_dim)));
}

std::pair<int, int> QTensorCpp::get_shape() const {
    /*
    Get the shape of the QTensor as a pair of integers (rows, cols).

    Returns:
        std::pair<int, int>: A pair of integers representing the number of rows and columns in the QTensor.
    */
    return std::make_pair(int(rows()), int(cols()));
}

bool QTensorCpp::is_ket() const {
    /*
    Check if the QTensor represents a ket vector, which is defined as having a single column.

    Returns:
        bool: True if the QTensor is a ket vector, False otherwise.
    */
    return cols() == 1 && rows() > 1;
}

bool QTensorCpp::is_bra() const {
    /*
    Check if the QTensor represents a bra vector, which is defined as having a single row.

    Returns:
        bool: True if the QTensor is a bra vector, False otherwise.
    */
    return rows() == 1 && cols() > 1;
}

bool QTensorCpp::is_operator() const {
    /*
    Check if the QTensor represents an operator, which is defined as having more than one row and more than one column.

    Returns:
        bool: True if the QTensor is an operator, False otherwise.
    */
    return rows() == cols() && (rows() & (rows() - 1)) == 0 && rows() > 1;
}

bool QTensorCpp::is_scalar() const {
    /*
    Check if the QTensor represents a scalar, which is defined as having a single row and a single column.

    Returns:
        bool: True if the QTensor is a scalar, False otherwise.
    */
    return rows() == 1 && cols() == 1;
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
    if (rows() != cols()) {
        return false;
    }
    if (!_symmetric_diff_computed) {
        for_each_nonzero([&](int i, int j, std::complex<double> val) {
            std::complex<double> other_val = coeff(j, i);
            _max_symmetric_diff = std::max(_max_symmetric_diff, std::abs(val - other_val));
        });
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
    if (rows() != cols()) {
        return false;
    }
    if (!_max_adjoint_diff_computed) {
        for_each_nonzero([&](int i, int j, std::complex<double> val) {
            std::complex<double> conj_val = std::conj(val);
            std::complex<double> other_val = coeff(j, i);
            _max_adjoint_diff = std::max(_max_adjoint_diff, std::abs(conj_val - other_val));
        });
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
    if (rows() != cols()) {
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

        // Otherwise try an LDLT decomposition with a shift to check for positive semidefiniteness.
        // info() == Success alone is insufficient: LDLT succeeds on indefinite/negative-definite
        // matrices too (it only fails on zero pivots). We also require all diagonal entries of D
        // to be non-negative, which is the actual PSD condition.
        SparseMatrix row_sparse = to_row_sparse();
        SparseMatrix sparse_identity(row_sparse.rows(), row_sparse.cols());
        sparse_identity.setIdentity();
        Eigen::SimplicialLDLT<SparseMatrix> chol(row_sparse + atol * sparse_identity);
        _is_positive = (chol.info() == Eigen::Success) && (chol.vectorD().real().minCoeff() >= 0.0);
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
        _eigenvectors.push_back(to_row_sparse());
        return;
    }
    if (is_bra()) {
        _eigenvalues.push_back(1.0);
        _eigenvectors.push_back(SparseMatrix(to_row_sparse().transpose()));
        return;
    }
    DenseMatrix dense_data = to_dense();
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
    return QTensorCpp(SparseMatrix(to_row_sparse().conjugate()));
}

QTensorCpp QTensorCpp::transpose() const {
    /*
    Return a new QTensor that is the transpose of this QTensor.

    Returns:
        QTensorCpp: A new QTensor that is the transpose of this QTensor.
    */
    return QTensorCpp(SparseMatrix(to_row_sparse().transpose()));
}

QTensorCpp QTensorCpp::adjoint() const {
    /*
    Return a new QTensor that is the adjoint (conjugate transpose) of this QTensor.

    Returns:
        QTensorCpp: A new QTensor that is the adjoint of this QTensor.
    */
    return QTensorCpp(SparseMatrix(to_row_sparse().adjoint()));
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
    for_each_nonzero([&](int row, int col, std::complex<double> val) {
        if (row == col) {
            _trace += val;
        }
    });
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
    int total_dim = int(std::max(rows(), cols()));
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
    accum.reserve(nnz());
    for_each_nonzero([&](int row, int col, std::complex<double> val) {
        if (idx_trace[row] == idx_trace[col]) {
            accum[{idx_keep[row], idx_keep[col]}] += val;
        }
    });

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
        for_each_nonzero([&](int, int, std::complex<double> val) { sum += std::pow(std::abs(val), norm_order); });
        return std::pow(sum, 1.0 / norm_order);
    } else if (norm_type == "trace") {
        return trace().real();
    } else if (norm_type == "nuclear") {
        DenseMatrix dense_data = to_dense();
        Eigen::BDCSVD<DenseMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(dense_data);
        double sum_singular_values = svd.singularValues().array().abs().sum();
        return sum_singular_values;
    } else if (norm_type == "inf") {
        double max_abs_val = 0.0;
        for_each_nonzero([&](int, int, std::complex<double> val) { max_abs_val = std::max(max_abs_val, std::abs(val)); });
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
    return QTensorCpp(SparseMatrix(to_row_sparse() / nrm));
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
    Eigen::Index n = Eigen::Index(qubit_values.size());
    Eigen::Index dim = Eigen::Index(1) << n;
    Eigen::Index one_index = 0;
    for (size_t i = 0; i < qubit_values.size(); ++i) {
        if (qubit_values[qubit_values.size() - 1 - i] == 1) {
            one_index |= (Eigen::Index(1) << i);
        }
    }
    return _col_ket_from_entries(dim, {{one_index, 1.0}});
}

QTensorCpp QTensorCpp::_col_ket_from_entries(Eigen::Index dim, const std::vector<std::pair<Eigen::Index, std::complex<double>>>& entries) {
    /*
    Build a (dim x 1) ket stored directly column-major from the given (row index, value) entries.

    Args:
        dim (Eigen::Index): The length of the ket (number of rows).
        entries (std::vector<std::pair<Eigen::Index, std::complex<double>>>): The non-zero (row, value) entries.

    Returns:
        QTensorCpp: A column-major ket with the given entries.
    */
    SparseMatrixCol col_major(dim, 1);
    col_major.reserve(Eigen::Index(entries.size()));
    for (const auto& [row_index, value] : entries) {
        col_major.insert(row_index, 0) = value;
    }
    col_major.makeCompressed();
    QTensorCpp result;
    result._data = std::move(col_major);
    result._format = StorageFormat::ColSparse;
    return result;
}

QTensorCpp QTensorCpp::zero(int nqubits) {
    /*
    Construct a |0..0> ket vector QTensor for a given number of qubits.

    Args:
        nqubits (int): The number of qubits, which determines the size of the zero statevector (2^nqubits x 1).

    Returns:
        QTensorCpp: A QTensor representing the |0..0> ket vector for the specified number of qubits.
    */
    if (nqubits < 0) {
        throw py::value_error("Number of qubits must be non-negative");
    }
    Eigen::Index dim = Eigen::Index(1) << nqubits;
    return _col_ket_from_entries(dim, {{0, 1.0}});
}

QTensorCpp QTensorCpp::one(int nqubits) {
    /*
    Construct a |1..1> ket vector QTensor for a given number of qubits.

    Args:
        nqubits (int): The number of qubits, which determines the size of the one statevector (2^nqubits x 1).

    Returns:
        QTensorCpp: A QTensor representing the |1..1> ket vector for the specified number of qubits.
    */
    if (nqubits < 0) {
        throw py::value_error("Number of qubits must be non-negative");
    }
    Eigen::Index dim = Eigen::Index(1) << nqubits;
    return _col_ket_from_entries(dim, {{dim - 1, 1.0}});
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
    int one_index = 0;
    for (size_t i = 0; i < qubit_values.size(); ++i) {
        if (qubit_values[qubit_values.size() - 1 - i] == 1) {
            one_index |= (1 << i);
        }
    }
    SparseMatrix row_major(1, dim);
    row_major.reserve(1);
    row_major.insert(0, one_index) = 1.0;
    row_major.makeCompressed();
    return QTensorCpp(row_major);
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

QTensorCpp QTensorCpp::tensor_product_python(const py::list& others) {
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

QTensorCpp QTensorCpp::tensor_product(const std::vector<QTensorCpp>& others) {
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
            SparseMatrix scalar_matrix(rows(), cols());
            for (int i = 0; i < int(std::min(rows(), cols())); ++i) {
                scalar_matrix.insert(i, i) = scalar;
            }
            scalar_matrix.makeCompressed();
            return QTensorCpp(SparseMatrix(to_row_sparse() + scalar_matrix));
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
    return QTensorCpp(SparseMatrix(to_row_sparse() + other.to_row_sparse()));
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
    return QTensorCpp(SparseMatrix(to_row_sparse() - other.to_row_sparse()));
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
    return QTensorCpp(SparseMatrix(to_row_sparse() * scalar));
}

QTensorCpp QTensorCpp::mul(const QTensorCpp& other) const {
    /*
    Multiply this QTensor element-wise by another QTensor, returning a new QTensor as the result.

    Args:
        other (QTensorCpp): The other QTensor to multiply element-wise with this QTensor.

    Returns:
        QTensorCpp: A new QTensor that is the result of element-wise multiplying this QTensor by the other QTensor.
    */
    return QTensorCpp(SparseMatrix(to_row_sparse().cwiseProduct(other.to_row_sparse())));
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
    return QTensorCpp(SparseMatrix(to_row_sparse() * other.to_row_sparse()));
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
    if (rows() != other.rows() || cols() != other.cols()) {
        return false;
    }
    return other.to_row_sparse().isApprox(to_row_sparse());
}

std::string QTensorCpp::as_string() const {
    /*
    Return a string representation of the QTensor, including its shape, number of non-zero elements, and a dense representation of its data.

    Returns:
        std::string: A string representation of the QTensor.
    */
    DenseMatrix dense = to_dense();
    std::stringstream ss;
    ss << "QTensor(shape=" << rows() << "x" << cols() << ", nnz=" << nnz() << "):" << std::endl;
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
        SparseMatrix rs = to_row_sparse();
        QTensorCpp mat = QTensorCpp(SparseMatrix(rs * rs.adjoint()));
        return mat / mat.trace();
    } else if (is_bra()) {
        SparseMatrix rs = to_row_sparse();
        QTensorCpp mat = QTensorCpp(SparseMatrix(rs.adjoint() * rs));
        return mat / mat.trace();
        // If an operator, try to repair it
    } else if (is_operator()) {
        // Make self-adjoint by averaging with its adjoint
        SparseMatrix rs = to_row_sparse();
        QTensorCpp self_adjoint = (QTensorCpp(rs) + QTensorCpp(SparseMatrix(rs.adjoint()))) * 0.5;

        // Shift eigenvalues to make positive semidefinite if necessary
        if (!self_adjoint.is_positive_semidefinite(atol)) {
            if (self_adjoint._eigenvalues.empty()) {
                self_adjoint.compute_eigendecomposition();
            }
            std::complex<double> min_eval = *std::min_element(self_adjoint._eigenvalues.begin(), self_adjoint._eigenvalues.end(), [](const std::complex<double>& a, const std::complex<double>& b) { return a.real() < b.real(); });
            double shift = std::max(0.0, -min_eval.real() + atol);
            SparseMatrix shift_matrix(self_adjoint.rows(), self_adjoint.cols());
            for (int i = 0; i < int(std::min(self_adjoint.rows(), self_adjoint.cols())); ++i) {
                shift_matrix.insert(i, i) = shift;
            }
            shift_matrix.makeCompressed();
            self_adjoint = QTensorCpp(SparseMatrix(self_adjoint.to_row_sparse() + shift_matrix));
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
    Eigen::MatrixXcd dense = to_dense();
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
    Eigen::MatrixXcd dense = to_dense();
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
    if (nnz() == 0) {
        return QTensorCpp(to_row_sparse());
    }
    Eigen::MatrixXcd dense = to_dense();
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
    Eigen::MatrixXcd dense = to_dense();
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
    if (_eigenvalues.empty() && rows() > 0 && cols() > 0) {
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
    if (_eigenvectors.empty() && rows() > 0 && cols() > 0) {
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
    } else if (py::hasattr(other, "_elements")) {
        int nqubits = int(std::log2(rows()));
        py::list hamiltonians_py;
        hamiltonians_py.append(other);
        std::vector<MatrixFreeHamiltonian> hamiltonians = parse_hamiltonians_matrix_free(nqubits, hamiltonians_py);
        return expectation_value(hamiltonians[0]);
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return expectation_value(other.attr("_qtensor_cpp").cast<QTensorCpp>(), nshots);
    } else {
        throw py::value_error("Other object must be a QTensor");
    }
}

std::complex<double> QTensorCpp::expectation_value(const MatrixFreeHamiltonian& other) const {
    /*
    Compute the expectation value of a matrix-free Hamiltonian with respect to this QTensor.

    Args:
        other (MatrixFreeHamiltonian): The matrix-free Hamiltonian for which to compute the expectation value with respect to this QTensor.

    Returns:
        std::complex<double>: The expectation value of the matrix-free Hamiltonian with respect to this QTensor.
    */

    std::complex<double> expectation = 0.0;

    // For kets we need to do <psi|H|psi> by applying H to the left and then taking the inner product with |psi>
    if (cols() == 1) {
        DenseMatrix psi_dense = to_dense();
        DenseMatrix H_psi_dense;
        other.apply(psi_dense, MatrixFreeApplicationType::Left, H_psi_dense);
#ifndef _WIN32
#if defined(_OPENMP)
#pragma omp parallel for reduction(complex_double_reduction : expectation) schedule(static)
#endif
#endif
        for (int i = 0; i < H_psi_dense.rows(); ++i) {
            expectation += std::conj(psi_dense(i, 0)) * H_psi_dense(i, 0);
        }
        // for bras we need to do <psi|H by applying H to the right and then taking the inner product with |psi>
    } else if (rows() == 1) {
        DenseMatrix psi_dense = to_dense().adjoint();
        DenseMatrix H_psi_dense;
        other.apply(psi_dense, MatrixFreeApplicationType::Right, H_psi_dense);
#ifndef _WIN32
#if defined(_OPENMP)
#pragma omp parallel for reduction(complex_double_reduction : expectation) schedule(static)
#endif
#endif
        for (int i = 0; i < H_psi_dense.cols(); ++i) {
            expectation += H_psi_dense(0, i) * std::conj(psi_dense(0, i));
        }
    } else {
        // need to do trace(H * rho) for a general operator and density matrix
        DenseMatrix rho_dense = to_dense();
        DenseMatrix H_rho_dense;
        other.apply(rho_dense, MatrixFreeApplicationType::Left, H_rho_dense);
#ifndef _WIN32
#if defined(_OPENMP)
#pragma omp parallel for reduction(complex_double_reduction : expectation) schedule(static)
#endif
#endif
        for (int i = 0; i < H_rho_dense.rows(); ++i) {
            expectation += std::real(H_rho_dense(i, i));
        }
    }

    return expectation;
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
            // return (adjoint() * other * (*this)).trace();
            std::complex<double> expectation = 0.0;
            other.for_each_nonzero([&](int row, int col, std::complex<double> val) { expectation += val * coeff(row, 0) * std::conj(coeff(col, 0)); });
            return expectation;

        } else if (is_bra()) {
            std::complex<double> expectation = 0.0;
            other.for_each_nonzero([&](int row, int col, std::complex<double> val) { expectation += val * coeff(0, col) * std::conj(coeff(0, row)); });
            return expectation;
        } else {
            // return (other * (*this)).trace();
            std::complex<double> expectation = 0.0;
            other.for_each_nonzero([&](int row, int col, std::complex<double> val) { expectation += val * coeff(col, row); });
            return expectation;
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
                    overlap += val * coeff(row, col);
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
        std::vector<double> probs(rows(), 0.0);
        for_each_nonzero([&](int row, int, std::complex<double> val) { probs[row] += std::norm(val); });
        return probs;
    } else if (is_bra()) {
        std::vector<double> probs(cols(), 0.0);
        for_each_nonzero([&](int, int col, std::complex<double> val) { probs[col] += std::norm(val); });
        return probs;
    } else if (is_operator()) {
        std::vector<double> probs(rows(), 0.0);
        for_each_nonzero([&](int row, int col, std::complex<double> val) {
            if (row == col) {
                probs[row] += val.real();
            }
        });
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
        product.for_each_nonzero([&](int row, int col, std::complex<double> val) {
            std::complex<double> identity_val = (row == col) ? 1.0 : 0.0;
            _max_unitary_diff = std::max(_max_unitary_diff, std::abs(val - identity_val));
        });
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
    for_each_nonzero([&](int row, int col, std::complex<double> val) { result += std::conj(val) * other.coeff(row, col); });
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
    return to_dense();
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
            Eigen::MatrixXcd dense = to_dense();
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
    Eigen::MatrixXcd dense = to_dense();
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
    Eigen::Index dim = Eigen::Index(1) << nqubits;
    const std::complex<double> amplitude = 1.0 / std::sqrt(2.0);
    return _col_ket_from_entries(dim, {{0, amplitude}, {dim - 1, amplitude}});
}

QTensorCpp QTensorCpp::uniform(int nqubits) {
    /*
    Create a uniform superposition state for the specified number of qubits, returning a new QTensor as the result.

    Args:
        nqubits (int): The number of qubits in the uniform superposition state.

    Returns:
        QTensorCpp: A new QTensor that is the uniform superposition state for the specified number of qubits.
    */
    Eigen::Index dim = Eigen::Index(1) << nqubits;
    const std::complex<double> amplitude = 1.0 / std::sqrt(static_cast<double>(dim));
    // A uniform superposition is fully populated, so store it as a dense vector
    DenseMatrix vec(dim, 1);
    std::complex<double>* data = vec.data();
#ifndef _WIN32
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
#endif
    for (Eigen::Index i = 0; i < dim; ++i) {
        data[i] = amplitude;
    }
    QTensorCpp result;
    result._data = std::move(vec);
    result._format = StorageFormat::Dense;
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
    return QTensorCpp(SparseMatrix(to_row_sparse() / scalar));
}

// GCOV_EXCL_BR_STOP
