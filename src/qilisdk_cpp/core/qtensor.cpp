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
#include <random>

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
            throw py::value_error("Data object must be a list of lists");
        }
        int cols = first_row.cast<py::list>().size();
        _data = SparseMatrix(rows, cols);
        for (int i=0; i<rows; ++i) {
            py::object row_obj = data_list[i];
            if (!py::isinstance<py::list>(row_obj)) {
                throw py::value_error("Data object must be a list of lists");
            }
            py::list row = row_obj.cast<py::list>();
            if (int(row.size()) != cols) {
                throw py::value_error("All rows must have the same number of columns");
            }
            for (int j=0; j<cols; ++j) {
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
    if (_data.rows() == 0 || _data.cols() == 0) {
        throw py::value_error("A QTensor should be initialized with a non-empty 2D array; got shape (" + std::to_string(_data.rows()) + ", " + std::to_string(_data.cols()) + ")");
    }
    if (((_data.rows() & (_data.rows() - 1)) != 0) || ((_data.cols() & (_data.cols() - 1)) != 0)) {
        throw py::value_error("A QTensor should have dimensions that are powers of 2; got shape (" + std::to_string(_data.rows()) + ", " + std::to_string(_data.cols()) + ")");
    }
}

py::object QTensorCpp::as_scipy() const {
    return to_spmatrix(_data);
}

py::object QTensorCpp::as_numpy() const {
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


bool QTensorCpp::is_self_adjoint(double atol) {
    if (_data.rows() != _data.cols()) {
        return false;
    }
    if (!_max_adjoint_diff_computed) {
        for (int k=0; k<_data.outerSize(); ++k) {
            for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
                int i = it.row();
                int j = it.col();
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
    if (_data.rows() != _data.cols()) {
        return false;
    }
    if (!_is_positive_computed || atol != _atol_used_for_positive) {
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
    DenseMatrix dense_data = DenseMatrix(_data);
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
    if (!is_operator()) {
        return false;
    } else if (!is_self_adjoint(atol)) {
        return false;
    } else if (std::abs(trace().real() - 1.0) > atol) {
        return false;
    } else if (!is_positive_semidefinite(atol)) {
        return false;
    }
    return true;
}

bool QTensorCpp::is_pure(double atol) {
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
    if (std::abs(_trace_squared - tr*tr) < atol) {
        return true;
    }
    return false;
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

std::complex<double> QTensorCpp::trace() {
    if (_trace_computed) {
        return _trace;
    }
    _trace = 0.0;
    for (int k=0; k<_data.outerSize(); ++k) {
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
        idx_keep[i]  = extract_bits(i, keep_vec);
        idx_trace[i] = extract_bits(i, trace_out);
    }

    // Accumulate into a map keyed by (row, col) of the reduced matrix.
    using Index = std::pair<int, int>;
    struct PairHash {
        size_t operator()(const Index& p) const {
            return std::hash<long long>()((long long)p.first << 32 | p.second);
        }
    };
    std::unordered_map<Index, std::complex<double>, PairHash> accum;
    accum.reserve(_data.nonZeros());
    for (int k = 0; k < _data.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(_data, k); it; ++it) {
            int row = it.row();
            int col = it.col();
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
        if (!is_operator()) {
            return norm("frobenius");
        }
        if (_eigenvalues.empty()) {
            compute_eigendecomposition();
        }
        double sum = 0.0;
        for (const auto& eval : _eigenvalues) {
            sum += std::abs(eval);
        }
        return sum;
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
        throw py::value_error("Unsupported norm type: " + norm_type);
    }
}

QTensorCpp QTensorCpp::normalized(const std::string& norm_type) {
    double nrm = norm(norm_type);
    if (std::abs(nrm) < default_atol) {
        throw py::value_error("Cannot normalize a tensor with zero norm");
    }
    return QTensorCpp(_data / nrm);
}

QTensorCpp QTensorCpp::ket(const std::vector<int>& qubit_values) {
    if (qubit_values.empty()) {
        throw py::value_error("Ket state cannot be empty");
    }
    for (int bit : qubit_values) {
        if (bit != 0 && bit != 1) {
            throw py::value_error("Ket state must be a list of 0s and 1s");
        }
    }
    int n = qubit_values.size();
    int dim = 1 << n;
    QTensorCpp result(dim, 1);
    int one_index = 0;
    for (size_t i = 0; i < qubit_values.size(); ++i) {
        if (qubit_values[qubit_values.size() - 1 - i] == 1) {
            one_index |= (1 << i);
        }
    }
    result._data.insert(one_index, 0) = 1.0;
    result._data.makeCompressed();
    return result;
}

QTensorCpp QTensorCpp::ket_python(const py::object& state) {
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
    if (qubit_values.empty()) {
        throw py::value_error("Bra state cannot be empty");
    }
    for (int bit : qubit_values) {
        if (bit != 0 && bit != 1) {
            throw py::value_error("Bra state must be a list of 0s and 1s");
        }
    }
    int n = qubit_values.size();
    int dim = 1 << n;
    QTensorCpp result(1, dim);
    int one_index = 0;
    for (size_t i = 0; i < qubit_values.size(); ++i) {
        if (qubit_values[qubit_values.size() - 1 - i] == 1) {
            one_index |= (1 << i);
        }
    }
    result._data.insert(0, one_index) = 1.0;
    result._data.makeCompressed();
    return result;
}

QTensorCpp QTensorCpp::bra_python(const py::object& state) {
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
    if (others.empty()) {
        throw py::value_error("The tensor product requires at least one other tensor");
    }
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
    } else if (py::isinstance<py::int_>(other) || py::isinstance<py::float_>(other) || py::isinstance(other, py_complex)) {
        if (is_scalar()) {
            std::complex<double> scalar = other.cast<std::complex<double>>();
            SparseMatrix scalar_matrix(_data.rows(), _data.cols());
            for (int i=0; i<std::min(_data.rows(), _data.cols()); ++i) {
                scalar_matrix.insert(i, i) = scalar;
            }
            scalar_matrix.makeCompressed();
            return QTensorCpp(_data + scalar_matrix);
        } else if (std::abs(other.cast<std::complex<double>>()) < default_atol) {
            return *this;
        } else {
            throw py::type_error("unsupported operand type(s) for addition: 'QTensor' and '" + std::string(py::str(other.get_type())) + "'. Addition of a scalar is only supported for 1x1 QTensors.");
        }
    } else {
        throw py::type_error("Addition is only supported with another QTensors or a scalar");
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
        throw py::type_error("Subtraction is only supported between QTensors");
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
    } else if (py::isinstance<py::int_>(other) || py::isinstance<py::float_>(other) || py::isinstance(other, py_complex)) {
        std::complex<double> scalar = other.cast<std::complex<double>>();
        return mul(scalar);
    } else {
        throw py::type_error("Multiplication is only supported between QTensors");
    }
}

QTensorCpp QTensorCpp::mul(std::complex<double> scalar) const {
    return QTensorCpp(_data * scalar);
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
        throw py::type_error("Matrix multiplication is only supported between QTensors");
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
        return false;
    }
}

bool QTensorCpp::equals(const QTensorCpp& other) const {
    if (_data.rows() != other.get_data().rows() || _data.cols() != other.get_data().cols()) {
        return false;
    }
    return other.get_data().isApprox(_data);
}

std::string QTensorCpp::as_string() const {
    DenseMatrix dense = _data;
    std::stringstream ss;
    ss << "QTensor(shape=" << _data.rows() << "x" << _data.cols() << ", nnz=" << _data.nonZeros() << "):" << std::endl;
    ss << dense;
    return ss.str();
}


QTensorCpp QTensorCpp::identity(int dim) {
    SparseMatrix id(dim, dim);
    for (int i=0; i<dim; ++i) {
        id.insert(i, i) = 1.0;
    }
    id.makeCompressed();
    return QTensorCpp(id);
}

QTensorCpp QTensorCpp::as_density_matrix(double atol, double max_relative_correction) {
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
            std::complex<double> min_eval = *std::min_element(_eigenvalues.begin(), _eigenvalues.end(), [](const std::complex<double>& a, const std::complex<double>& b) {
                return a.real() < b.real();
            });
            double shift = std::max(0.0, -min_eval.real() + atol);
            SparseMatrix shift_matrix(self_adjoint.get_data().rows(), self_adjoint.get_data().cols());
            for (int i=0; i<std::min(self_adjoint.get_data().rows(), self_adjoint.get_data().cols()); ++i) {
                shift_matrix.insert(i, i) = shift;
            }
            shift_matrix.makeCompressed();
            self_adjoint = QTensorCpp(self_adjoint.get_data() + shift_matrix);
        }

        // Normalize to have trace 1
        QTensorCpp return_tensor = self_adjoint / self_adjoint.trace();

        // match=r"Repairing the density matrix required a large correction \(relative Frobenius correction=5.000e-01\)",
        double correction = (return_tensor - self_adjoint).norm("frobenius") / self_adjoint.norm("frobenius");
        if (correction > max_relative_correction) {
            throw py::value_error("Repairing the density matrix required a large correction (relative Frobenius correction=" + std::to_string(correction) + "). This likely indicates that the original tensor is not close to a valid density matrix, and the result may not be meaningful.");
        }

        return return_tensor;

    } else {
        throw py::value_error("Only kets, bras or operators can be converted to a density matrix");
    }
}

QTensorCpp QTensorCpp::exp() const {
    // If we have already computed the eigendecomposition, we can compute the exponential more efficiently
    if (!_eigenvalues.empty() && !_eigenvectors.empty()) {
        SparseMatrix result(_data.rows(), _data.cols());
        std::vector<Eigen::Triplet<std::complex<double>>> triplets;
        for (size_t i = 0; i < _eigenvalues.size(); ++i) {
            std::complex<double> eval_exp = std::exp(_eigenvalues[i]);
            const SparseMatrix& evec = _eigenvectors[i];
            for (int k=0; k<evec.outerSize(); ++k) {
                for (typename SparseMatrix::InnerIterator it(evec, k); it; ++it) {
                    int row = it.row();
                    int col = it.col();
                    std::complex<double> val = eval_exp * it.value();
                    triplets.emplace_back(row, col, val);
                }
            }
        }
        result.setFromTriplets(triplets.begin(), triplets.end());
        return QTensorCpp(result);
    }
    Eigen::MatrixXcd dense = _data;
    Eigen::MatrixXcd exp_dense = dense.exp();
    SparseMatrix exp_sparse = exp_dense.sparseView();
    return QTensorCpp(exp_sparse);
}

QTensorCpp QTensorCpp::log() const {
    // If we have already computed the eigendecomposition, we can compute the logarithm more efficiently
    if (!_eigenvalues.empty() && !_eigenvectors.empty()) {
        SparseMatrix result(_data.rows(), _data.cols());
        std::vector<Eigen::Triplet<std::complex<double>>> triplets;
        for (size_t i = 0; i < _eigenvalues.size(); ++i) {
            std::complex<double> eval_log = std::log(_eigenvalues[i]);
            const SparseMatrix& evec = _eigenvectors[i];
            for (int k=0; k<evec.outerSize(); ++k) {
                for (typename SparseMatrix::InnerIterator it(evec, k); it; ++it) {
                    int row = it.row();
                    int col = it.col();
                    std::complex<double> val = eval_log * it.value();
                    triplets.emplace_back(row, col, val);
                }
            }
        }
        result.setFromTriplets(triplets.begin(), triplets.end());
        return QTensorCpp(result);
    }
    Eigen::MatrixXcd dense = _data;
    Eigen::MatrixXcd log_dense = dense.log();
    SparseMatrix log_sparse = log_dense.sparseView();
    return QTensorCpp(log_sparse);
}

QTensorCpp QTensorCpp::sqrt() const {
    // If we have already computed the eigendecomposition, we can compute the square root more efficiently
    if (!_eigenvalues.empty() && !_eigenvectors.empty()) {
        SparseMatrix result(_data.rows(), _data.cols());
        std::vector<Eigen::Triplet<std::complex<double>>> triplets;
        for (size_t i = 0; i < _eigenvalues.size(); ++i) {
            std::complex<double> eval_sqrt = std::sqrt(_eigenvalues[i]);
            const SparseMatrix& evec = _eigenvectors[i];
            for (int k=0; k<evec.outerSize(); ++k) {
                for (typename SparseMatrix::InnerIterator it(evec, k); it; ++it) {
                    int row = it.row();
                    int col = it.col();
                    std::complex<double> val = eval_sqrt * it.value();
                    triplets.emplace_back(row, col, val);
                }
            }
        }
        result.setFromTriplets(triplets.begin(), triplets.end());
        return QTensorCpp(result);
    }
    Eigen::MatrixXcd dense = _data;
    Eigen::MatrixXcd sqrt_dense = dense.sqrt();
    SparseMatrix sqrt_sparse = sqrt_dense.sparseView();
    return QTensorCpp(sqrt_sparse);
}

QTensorCpp QTensorCpp::pow(int n) const {
    // If we have already computed the eigendecomposition, we can compute the power more efficiently
    if (!_eigenvalues.empty() && !_eigenvectors.empty()) {
        SparseMatrix result(_data.rows(), _data.cols());
        std::vector<Eigen::Triplet<std::complex<double>>> triplets;
        for (size_t i = 0; i < _eigenvalues.size(); ++i) {
            std::complex<double> eval_pow = std::pow(_eigenvalues[i], n);
            const SparseMatrix& evec = _eigenvectors[i];
            for (int k=0; k<evec.outerSize(); ++k) {
                for (typename SparseMatrix::InnerIterator it(evec, k); it; ++it) {
                    int row = it.row();
                    int col = it.col();
                    std::complex<double> val = eval_pow * it.value();
                    triplets.emplace_back(row, col, val);
                }
            }
        }
        result.setFromTriplets(triplets.begin(), triplets.end());
        return QTensorCpp(result);
    }
    Eigen::MatrixXcd dense = _data;
    Eigen::MatrixXcd pow_dense = dense.pow(n);
    SparseMatrix pow_sparse = pow_dense.sparseView();
    return QTensorCpp(pow_sparse);
}

std::vector<std::complex<double>> QTensorCpp::get_eigenvalues() const { 
    if (_eigenvalues.empty() && _data.rows() > 0 && _data.cols() > 0) {
        throw py::value_error("Eigenvalues have not been computed yet. Call compute_eigendecomposition() first.");
    }
    return _eigenvalues; 
}

py::object QTensorCpp::get_eigenvalues_python() const {
    py::list evals;
    for (const auto& eval : get_eigenvalues()) {
        evals.append(eval);
    }
    return evals;
}

std::vector<SparseMatrix> QTensorCpp::get_eigenvectors() const { 
    if (_eigenvectors.empty() && _data.rows() > 0 && _data.cols() > 0) {
        throw py::value_error("Eigenvectors have not been computed yet. Call compute_eigendecomposition() first.");
    }
    return _eigenvectors; 
}

py::object QTensorCpp::get_eigenvectors_python() const {
    py::list evecs;
    for (const auto& evec : get_eigenvectors()) {
        evecs.append(to_spmatrix(evec));
    }
    return evecs;
}

void QTensorCpp::clear_cache() {
    _is_self_adjoint_computed = false;
    _is_positive_computed = false;
    _trace_computed = false;
    _trace_squared_computed = false;
    _max_adjoint_diff_computed = false;
    _max_unitary_diff_computed = false;
    _trace_squared_computed = false;
    _rank_computed = false;
    _eigenvalues.clear();
    _eigenvectors.clear();
}

std::complex<double> QTensorCpp::expectation_value_python(const py::object& other, int nshots) const {
    if (py::isinstance<QTensorCpp>(other)) {
        return expectation_value(other.cast<QTensorCpp>(), nshots);
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return expectation_value(other.attr("_qtensor_cpp").cast<QTensorCpp>(), nshots);
    } else {
        throw py::value_error("Other object must be a QTensor");
    }
}

std::complex<double> QTensorCpp::expectation_value(const QTensorCpp& other, int nshots) const {
    
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
            for (int k=0; k<evec.outerSize(); ++k) {
                for (typename SparseMatrix::InnerIterator it(evec, k); it; ++it) {
                    int row = it.row();
                    int col = it.col();
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

QTensorCpp QTensorCpp::random_sparse(int rows, int cols, double density) {
    SparseMatrix random_matrix(rows, cols);
    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
    int total_elements = rows * cols;
    int nonzeros = static_cast<int>(total_elements * density);
    for (int i = 0; i < nonzeros; ++i) {
        int row = rand() % rows;
        int col = rand() % cols;
        std::complex<double> val(static_cast<double>(rand()) / RAND_MAX, static_cast<double>(rand()) / RAND_MAX);
        triplets.emplace_back(row, col, val);
    }
    random_matrix.setFromTriplets(triplets.begin(), triplets.end());
    return QTensorCpp(random_matrix);
}

QTensorCpp QTensorCpp::random(int rows, int cols) {
    DenseMatrix random_matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            random_matrix(i, j) = std::complex<double>(static_cast<double>(rand()) / RAND_MAX, static_cast<double>(rand()) / RAND_MAX);
        }
    }
    return QTensorCpp(random_matrix.sparseView());
}

QTensorCpp QTensorCpp::commutator_python(const py::object& other) const {
    if (py::isinstance<QTensorCpp>(other)) {
        return commutator(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return commutator(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else {
        throw py::value_error("Other object must be a QTensor");
    }
}

QTensorCpp QTensorCpp::commutator(const QTensorCpp& other) const {
    return (*this * other) - (other * (*this));
}

QTensorCpp QTensorCpp::anticommutator_python(const py::object& other) const {
    if (py::isinstance<QTensorCpp>(other)) {
        return anticommutator(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return anticommutator(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else {
        throw py::value_error("Other object must be a QTensor");
    }
}

QTensorCpp QTensorCpp::anticommutator(const QTensorCpp& other) const {
    return (*this * other) + (other * (*this));
}

std::vector<double> QTensorCpp::probabilities() const {
    if (is_ket()) {
        std::vector<double> probs(_data.rows(), 0.0);
        for (int k=0; k<_data.outerSize(); ++k) {
            for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
                int row = it.row();
                std::complex<double> val = it.value();
                probs[row] += std::norm(val);
            }
        }
        return probs;
    } else if (is_bra()) {
        std::vector<double> probs(_data.cols(), 0.0);
        for (int k=0; k<_data.outerSize(); ++k) {
            for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
                int col = it.col();
                std::complex<double> val = it.value();
                probs[col] += std::norm(val);
            }
        }
        return probs;
    } else if (is_operator()) {
        std::vector<double> probs(_data.rows(), 0.0);
        for (int k=0; k<_data.outerSize(); ++k) {
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
    if (!is_operator()) {
        return false;
    }
    if (!_max_unitary_diff_computed) {
        QTensorCpp product = (*this * adjoint());
        _max_unitary_diff = 0.0;
        for (int k=0; k<product.get_data().outerSize(); ++k) {
            for (typename SparseMatrix::InnerIterator it(product.get_data(), k); it; ++it) {
                int row = it.row();
                int col = it.col();
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
    if (!_trace_squared_computed) {
        for (int k=0; k<_data.outerSize(); ++k) {
            for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
                _trace_squared += std::norm(it.value());
            }
        }
        _trace_squared_computed = true;
    }
    return _trace_squared;
}

std::complex<double> QTensorCpp::dot(const QTensorCpp& other) const {
    std::complex<double> result = 0.0;
    for (int k=0; k<_data.outerSize(); ++k) {
        for (typename SparseMatrix::InnerIterator it(_data, k); it; ++it) {
            int row = it.row();
            int col = it.col();
            std::complex<double> val = it.value();
            std::complex<double> other_val = other.get_data().coeff(row, col);
            result += std::conj(val) * other_val;
        }
    }
    return result;
}

std::complex<double> QTensorCpp::dot_python(const py::object& other) const {
    if (py::isinstance<QTensorCpp>(other)) {
        return dot(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return dot(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else {
        throw py::value_error("Other object must be a QTensor");
    }
}

double QTensorCpp::fidelity(const QTensorCpp& other) const {
    if (is_ket() && other.is_ket()) {
        return std::pow(std::abs(dot(other)), 2);
    } else if (is_bra() && other.is_bra()) {
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

double QTensorCpp::fidelity_python(const py::object& other) const {
    if (py::isinstance<QTensorCpp>(other)) {
        return fidelity(other.cast<QTensorCpp>());
    } else if (py::hasattr(other, "_qtensor_cpp")) {
        return fidelity(other.attr("_qtensor_cpp").cast<QTensorCpp>());
    } else {
        throw py::value_error("Other object must be a QTensor");
    }
}

double QTensorCpp::entropy_von_neumann() {
    compute_eigendecomposition();
    if (!is_density_matrix()) {
        throw py::value_error("Von Neumann entropy can only be computed for density matrices");
    }
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
    compute_eigendecomposition();
    if (!is_density_matrix()) {
        throw py::value_error("Renyi entropy can only be computed for density matrices");
    }
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
    return DenseMatrix(_data);
}

int QTensorCpp::rank() {
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
            _rank = lu.rank();
        }
        _rank_computed = true;
    }
    return _rank;
}

QTensorCpp QTensorCpp::inverse() const {
    Eigen::MatrixXcd dense = _data;
    Eigen::MatrixXcd inv_dense = dense.inverse();
    SparseMatrix inv_sparse = inv_dense.sparseView();
    return QTensorCpp(inv_sparse);
}

QTensorCpp QTensorCpp::ghz(int nqubits) {
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
    std::set<int> qubits_set;
    for (const auto& item : qubits) {
        qubits_set.insert(item.cast<int>());
    }
    return reset_qubits(qubits_set);
}

QTensorCpp QTensorCpp::reset_qubits(const std::set<int>& qubits) {

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
        for (int k=0; k<rest_data.outerSize(); ++k) {
            for (typename SparseMatrix::InnerIterator it(rest_data, k); it; ++it) {
                int row = it.row();
                int col = it.col();
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
    if (std::abs(scalar) <= default_atol) {
        throw py::value_error("Cannot divide by zero");
    }
    return QTensorCpp(_data / scalar);
}