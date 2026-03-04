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
#include "../libs/pybind.h"
#include "../libs/numpy.h"

#include <sstream>

QTensorCpp::QTensorCpp(const py::object& data) {
    if (py::isinstance<QTensorCpp>(data)) {
        _data = data.cast<QTensorCpp>().get_data();
    } else if (py::hasattr(data, "_qtensor_cpp")) {
        _data = data.attr("_qtensor_cpp").cast<QTensorCpp>().get_data();
    } else if (py::isinstance<py::list>(data)) {
        py::list data_list = data.cast<py::list>();
        int rows = data_list.size();
        if (rows == 0) {
            _data = SparseMatrix(0, 0);
            return;
        }
        py::object first_row = data_list[0];
        if (!py::isinstance<py::list>(first_row)) {
            throw std::invalid_argument("Data object must be a list of lists");
        }
        int cols = first_row.cast<py::list>().size();
        _data = SparseMatrix(rows, cols);
        for (int i=0; i<rows; ++i) {
            py::object row_obj = data_list[i];
            if (!py::isinstance<py::list>(row_obj)) {
                throw std::invalid_argument("Data object must be a list of lists");
            }
            py::list row = row_obj.cast<py::list>();
            if (int(row.size()) != cols) {
                throw std::invalid_argument("All rows must have the same number of columns");
            }
            for (int j=0; j<cols; ++j) {
                std::complex<double> val = row[j].cast<std::complex<double>>();
                if (std::abs(val) > 1e-12) {
                    _data.insert(i, j) = val;
                }
            }
        }
        _data.makeCompressed();
    } else if (py::isinstance(data, csrmatrix)) {
        _data = from_spmatrix(data, 1e-12);
    } else if (py::isinstance(data, numpy_array)) {
        _data = from_numpy(data.cast<py::buffer>(), 1e-12);
    } else {
        throw std::invalid_argument("Data object must be a QTensor");
    }
}

py::object QTensorCpp::get_data_as_scipy() const {
    return to_spmatrix(_data);
}

py::object QTensorCpp::get_data_as_numpy() const {
    return to_numpy(_data);
}

int QTensorCpp::get_nqubits() const {
    int max_dim = std::max(_data.rows(), _data.cols());
    return static_cast<int>(std::ceil(std::log2(max_dim)));
}

std::pair<int, int> QTensorCpp::get_shape() const {
    return std::make_pair(_data.rows(), _data.cols());
}

bool QTensorCpp::is_ket() const {
    return _data.cols() == 1;
}

bool QTensorCpp::is_bra() const {
    return _data.rows() == 1;
}

bool QTensorCpp::is_operator() const {
    return _data.rows() == _data.cols() && (_data.rows() & (_data.rows() - 1)) == 0 && _data.rows() > 0;
}

bool QTensorCpp::is_scalar() const {
    return _data.rows() == 1 && _data.cols() == 1;
}


bool QTensorCpp::is_self_adjoint(double atol) const {
    if (_data.rows() != _data.cols()) {
        return false;
    }
    for (int k=0; k<_data.outerSize(); ++k) {
        for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
            int i = it.row();
            int j = it.col();
            std::complex<double> val = it.value();
            std::complex<double> conj_val = std::conj(val);
            std::complex<double> other_val = _data.coeff(j, i);
            if (std::abs(conj_val - other_val) > atol) {
                return false;
            }
        }
    }
    return true;
}

bool QTensorCpp::is_positive_semidefinite(double atol) const {
    if (_data.rows() != _data.cols()) {
        return false;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(_data);
    auto& eigenvalues = es.eigenvalues();
    for (const auto& ev : eigenvalues) {
        if (ev < -atol) {
            return false;
        }
    }
    return true;
}

bool QTensorCpp::is_density_matrix(double atol) const {
    if (!is_operator()) {
        return false;
    }
    if (!is_self_adjoint(atol)) {
        return false;
    }
    if (!is_positive_semidefinite(atol)) {
        return false;
    }
    double tr = trace().real();
    return std::abs(tr - 1.0) < atol;
}

QTensorCpp QTensorCpp::conjugate() const {
    return QTensorCpp(_data.conjugate());
}

QTensorCpp QTensorCpp::transpose() const {
    return QTensorCpp(_data.transpose());
}

QTensorCpp QTensorCpp::adjoint() const {
    return QTensorCpp(_data.adjoint());
}

std::complex<double> QTensorCpp::trace() const {
    std::complex<double> trace = 0.0;
    for (int k=0; k<_data.outerSize(); ++k) {
        for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
            if (it.row() == it.col()) {
                trace += it.value();
            }
        }
    }
    return trace;
}

QTensorCpp QTensorCpp::partial_trace(const std::vector<int>& keep) const {

    // Total number of qubits: matrix is 2^n x 2^n
    int total_dim = _data.rows();
    int n = static_cast<int>(std::log2(total_dim));

    // Determine which qubits to trace out
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

    // Dimension of the reduced (kept) subsystem
    int keep_dim = 1 << keep.size();

    // Dimension of the traced-out subsystem
    int trace_dim = 1 << trace_out.size();

    // Build a mapping from a full n-qubit basis index to
    // (kept_index, traced_index) and back, respecting qubit ordering.
    //
    // For a full basis state |b_{n-1}...b_1 b_0>, qubit q contributes
    // bit b_q at position q. We re-index the kept and traced qubits
    // preserving their relative order.
    //
    // kept_index   : index formed by bits of kept qubits (in their original order)
    // traced_index : index formed by bits of traced qubits (in their original order)

    // Precompute: for each full index, its (kept_index, traced_index)
    std::vector<int> full_to_kept(total_dim), full_to_traced(total_dim);
    for (int full = 0; full < total_dim; ++full) {
        int ki = 0, ti = 0;
        int kbit = 0, tbit = 0;
        for (int q = 0; q < n; ++q) {
            int b = (full >> q) & 1;
            if (kept[q])
                ki |= (b << kbit++);
            else
                ti |= (b << tbit++);
        }
        full_to_kept[full]   = ki;
        full_to_traced[full] = ti;
    }

    // Inverse map: (kept_index, traced_index) -> full index
    // full_index[ki][ti] = full
    std::vector<std::vector<int>> full_idx(keep_dim, std::vector<int>(trace_dim, 0));
    for (int full = 0; full < total_dim; ++full) {
        full_idx[full_to_kept[full]][full_to_traced[full]] = full;
    }

    // Compute the reduced density matrix:
    // rho_reduced[ki_row][ki_col] = sum_{ti} rho[full_idx[ki_row][ti]][full_idx[ki_col][ti]]
    SparseMatrix result(keep_dim, keep_dim);
    std::vector<Eigen::Triplet<std::complex<double>>> triplets;

    // Iterate over non-zero entries of the sparse matrix grouped by traced index
    // Strategy: iterate over all (ki_row, ki_col) pairs, summing over ti.
    // For efficiency with sparse matrices, iterate over stored non-zeros.

    // Build a lookup: full_row -> list of (full_col, value) from the sparse matrix
    // Then accumulate into result.
    std::unordered_map<int, std::vector<std::pair<int, std::complex<double>>>> row_entries;
    for (int k = 0; k < _data.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(_data, k); it; ++it) {
            int row = it.row(), col = it.col();
            int ki_row = full_to_kept[row];
            int ki_col = full_to_kept[col];
            int ti_row = full_to_traced[row];
            int ti_col = full_to_traced[col];

            // Only contribute when both row and col share the same traced-out index
            if (ti_row == ti_col) {
                triplets.emplace_back(ki_row, ki_col, it.value());
            }
        }
    }

    result.setFromTriplets(triplets.begin(), triplets.end());
    return QTensorCpp(result);

}

double QTensorCpp::norm(const std::string& norm_type) const {
    if (norm_type == "frobenius" || norm_type == "l2") {
        return _data.norm();
    } else if (norm_type == "l1") {
        double sum = 0.0;
        for (int k=0; k<_data.outerSize(); ++k) {
            for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
                sum += std::abs(it.value());
            }
        }
        return sum;
    } else if (norm_type == "trace") {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(_data);
        double trace_norm = 0.0;
        for (int i=0; i<es.eigenvalues().size(); ++i) {
            trace_norm += std::abs(es.eigenvalues()[i]);
        }
        return trace_norm;
    } else if (norm_type == "inf") {
        double max_row_sum = 0.0;
        for (int i=0; i<_data.rows(); ++i) {
            double row_sum = 0.0;
            for (int k=0; k<_data.outerSize(); ++k) {
                for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
                    if (it.row() == i) {
                        row_sum += std::abs(it.value());
                    }
                }
            }
            max_row_sum = std::max(max_row_sum, row_sum);
        }
        return max_row_sum;
    } else {
        throw std::invalid_argument("Unsupported norm type: " + norm_type);
    }
}

QTensorCpp QTensorCpp::normalized(const std::string& norm_type) const {
    double nrm = norm(norm_type);
    if (nrm == 0.0) {
        throw std::runtime_error("Cannot normalize a tensor with zero norm");
    }
    return QTensorCpp(_data / nrm);
}

int bitstring_to_index(const std::string& bitstring) {
    int index = 0;
    for (size_t i = 0; i < bitstring.size(); ++i) {
        if (bitstring[bitstring.size() - 1 - i] == '1') {
            index |= (1 << i);
        }
    }
    return index;
}

QTensorCpp QTensorCpp::ket(const std::string& bitstring) const {
    int dim = 1 << bitstring.size();
    QTensorCpp result(dim, 1);
    int one_index = bitstring_to_index(bitstring);
    result._data.insert(one_index, 0) = 1.0;
    result._data.makeCompressed();
    return result;
}

QTensorCpp QTensorCpp::bra(const std::string& bitstring) const {
    int dim = 1 << bitstring.size();
    QTensorCpp result(1, dim);
    int one_index = bitstring_to_index(bitstring);
    result._data.insert(0, one_index) = 1.0;
    result._data.makeCompressed();
    return result;
}

QTensorCpp QTensorCpp::tensor_product_python(const py::list& others) const {
    std::vector<QTensorCpp> other_tensors;
    for (const auto& other : others) {
        if (py::isinstance<QTensorCpp>(other)) {
            other_tensors.push_back(other.cast<QTensorCpp>());
        } else if (py::hasattr(other, "_qtensor_cpp")) {
            other_tensors.push_back(other.attr("_qtensor_cpp").cast<QTensorCpp>());
        } else {
            throw std::invalid_argument("All elements in the list must be a QTensor");
        }
    }
    return tensor_product(other_tensors);
}

QTensorCpp QTensorCpp::tensor_product(const std::vector<QTensorCpp>& others) const {
    Triplets triplets {{0, 0, 1.0}};
    int total_rows = 1;
    int total_cols = 1;
    for (const auto& other : others) {
        const SparseMatrix& B = other.get_data();
        total_rows *= B.rows();
        total_cols *= B.cols();
        Triplets new_triplets(triplets.size() * B.nonZeros());
        int index = 0;
        for (const auto& tA : triplets) {
            for (int k=0; k<B.outerSize(); ++k) {
                for (typename SparseMatrix::InnerIterator it(B, k); it; ++it) {
                    int row = tA.row() * B.rows() + it.row();
                    int col = tA.col() * B.cols() + it.col();
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
    if (py::isinstance<QTensorCpp>(other)) {
        return add(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return add(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else {
        throw std::invalid_argument("Other object must be a QTensor");
    }
}

QTensorCpp QTensorCpp::add(const QTensorCpp& other) const {
    return QTensorCpp(_data + other.get_data());
}

QTensorCpp QTensorCpp::sub_python(const py::object& other) const {
    if (py::isinstance<QTensorCpp>(other)) {
        return sub(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return sub(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else {
        throw std::invalid_argument("Other object must be a QTensor");
    }
}

QTensorCpp QTensorCpp::sub(const QTensorCpp& other) const {
    return QTensorCpp(_data - other.get_data());
}

QTensorCpp QTensorCpp::mul_python(const py::object& other) const {
    if (py::isinstance<QTensorCpp>(other)) {
        return mul(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return mul(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else {
        throw std::invalid_argument("Other object must be a QTensor");
    }
}

QTensorCpp QTensorCpp::mul(const QTensorCpp& other) const {
    return QTensorCpp(_data.cwiseProduct(other.get_data()));
}

QTensorCpp QTensorCpp::matmul_python(const py::object& other) const {
    if (py::isinstance<QTensorCpp>(other)) {
        return matmul(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return matmul(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else {
        throw std::invalid_argument("Other object must be a QTensor");
    }
}

QTensorCpp QTensorCpp::matmul(const QTensorCpp& other) const {
    return QTensorCpp(_data * other.get_data());
}

bool QTensorCpp::equals_python(const py::object& other) const {
    if (py::isinstance<QTensorCpp>(other)) {
        return equals(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return equals(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else {
        throw std::invalid_argument("Other object must be a QTensor");
    }
}

bool QTensorCpp::equals(const QTensorCpp& other) const {
    return other.get_data().isApprox(_data);
}

std::string QTensorCpp::as_string() const {
    DenseMatrix dense = _data;
    std::stringstream ss;
    ss << dense;
    return ss.str();
}


QTensorCpp QTensorCpp::identity(int dim) const {
    SparseMatrix id(dim, dim);
    for (int i=0; i<dim; ++i) {
        id.insert(i, i) = 1.0;
    }
    id.makeCompressed();
    return QTensorCpp(id);
}

QTensorCpp QTensorCpp::to_density_matrix() const {
    if (is_ket()) {
        return QTensorCpp(_data * _data.adjoint());
    } else if (is_bra()) {
        return QTensorCpp(_data.adjoint() * _data);
    } else {
        throw std::runtime_error("Only kets or bras can be converted to density matrix");
    }
}

QTensorCpp QTensorCpp::exponential() const {
    Eigen::MatrixXcd dense = _data;
    Eigen::MatrixXcd exp_dense = dense.exp();
    SparseMatrix exp_sparse = exp_dense.sparseView();
    return QTensorCpp(exp_sparse);
}