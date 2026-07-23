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
#pragma once

#include <set>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include "../backends/qilisim/representations/matrix_free_hamiltonian.h"
#include "../libs/eigen.h"
#include "../libs/pybind.h"

// GCOV_EXCL_BR_START

const double default_atol = 1e-12;

// A dense storage backend is chosen when the density (nnz / (rows * cols)) exceeds this fraction.
const double dense_storage_threshold = 1.0 / 3.0;

// The storage backend used internally by a QTensor
enum class StorageFormat { Auto, RowSparse, ColSparse, Dense };

// The main QiliSim C++ class
#pragma GCC visibility push(default)
class QTensorCpp {
   private:
    std::variant<SparseMatrix, SparseMatrixCol, DenseMatrix> _data;
    StorageFormat _format = StorageFormat::RowSparse;
    static StorageFormat _choose_format(Eigen::Index rows, Eigen::Index cols, Eigen::Index nnz, StorageFormat requested);
    void _set_from_row_sparse(const SparseMatrix& data, StorageFormat requested = StorageFormat::Auto);
    static QTensorCpp _col_ket_from_entries(Eigen::Index dim, const std::vector<std::pair<Eigen::Index, Complex>>& entries);

    // Cache various things to faster reaccess
    std::vector<Complex> _eigenvalues;
    std::vector<SparseMatrix> _eigenvectors;
    bool _is_positive_computed = false;
    bool _is_positive = false;
    double _atol_used_for_positive = 0.0;
    Complex _trace = 0.0;
    bool _trace_computed = false;
    bool _trace_squared_computed = false;
    double _trace_squared = 0.0;
    bool _max_adjoint_diff_computed = false;
    double _max_adjoint_diff = 0.0;
    bool _max_unitary_diff_computed = false;
    double _max_unitary_diff = 0.0;
    bool _rank_computed = false;
    int _rank = 0;
    bool _symmetric_diff_computed = false;
    double _max_symmetric_diff = 0.0;

   public:
    // Constructors and basic accessors
    QTensorCpp() {}
    QTensorCpp(const SparseMatrix& data);
    QTensorCpp(const SparseMatrix& data, StorageFormat format);
    QTensorCpp(const SparseMatrix& data, bool no_checks);
    QTensorCpp(const py::object& data);
    QTensorCpp(int rows, int cols);
    SparseMatrix get_data() const { return to_row_sparse(); }

    // Backend-agnostic accessors
    StorageFormat get_format() const { return _format; }
    std::string get_format_string() const;
    Eigen::Index rows() const;
    Eigen::Index cols() const;
    Eigen::Index nnz() const;
    SparseMatrix to_row_sparse() const;
    DenseMatrix to_dense() const;

    template <typename F>
    void for_each_nonzero(F&& f) const {
        std::visit(
            [&](const auto& mat) {
                using M = std::decay_t<decltype(mat)>;
                if constexpr (std::is_same_v<M, DenseMatrix>) {
                    for (Eigen::Index c = 0; c < mat.cols(); ++c) {
                        for (Eigen::Index r = 0; r < mat.rows(); ++r) {
                            const Complex value = mat(r, c);
                            if (value != Complex(0.0, 0.0)) {
                                f(int(r), int(c), value);
                            }
                        }
                    }
                } else {
                    for (Eigen::Index k = 0; k < mat.outerSize(); ++k) {
                        for (typename M::InnerIterator it(mat, k); it; ++it) {
                            f(int(it.row()), int(it.col()), it.value());
                        }
                    }
                }
            },
            _data);
    }

    py::object as_scipy() const;
    py::object as_numpy() const;
    int get_nqubits() const;
    std::pair<int, int> get_shape() const;
    void clear_cache();
    std::string as_string() const;
    DenseMatrix as_dense() const;
    Complex coeff(int row, int col) const;

    // Matrix arithmetic
    double norm(const std::string& norm_type);
    void compute_eigendecomposition();
    bool equals_python(const py::object& other) const;
    bool equals(const QTensorCpp& other) const;
    Complex dot(const QTensorCpp& other) const;
    Complex dot_python(const py::object& other) const;
    QTensorCpp normalized(const std::string& norm_type);
    QTensorCpp inverse();
    QTensorCpp pow(double n);
    QTensorCpp sqrt();
    QTensorCpp log();
    QTensorCpp exp();
    int rank();
    std::vector<Complex> get_eigenvalues() const;
    py::object get_eigenvalues_python() const;
    std::vector<SparseMatrix> get_eigenvectors() const;
    py::object get_eigenvectors_python() const;
    QTensorCpp conjugate() const;
    QTensorCpp transpose() const;
    QTensorCpp adjoint() const;
    Complex trace();
    QTensorCpp add_python(const py::object& other) const;
    QTensorCpp add(const QTensorCpp& other) const;
    QTensorCpp sub_python(const py::object& other) const;
    QTensorCpp sub(const QTensorCpp& other) const;
    QTensorCpp mul_python(const py::object& other) const;
    QTensorCpp mul(const QTensorCpp& other) const;
    QTensorCpp matmul_python(const py::object& other) const;
    QTensorCpp matmul(const QTensorCpp& other) const;
    QTensorCpp div(Complex scalar) const;
    QTensorCpp mul(Complex scalar) const;

    // Cached checks
    bool is_ket() const;
    bool is_bra() const;
    bool is_operator() const;
    bool is_scalar() const;
    bool is_pure(double atol = default_atol);
    bool is_density_matrix(double atol = default_atol);
    bool is_symmetric(double atol = default_atol);
    bool is_self_adjoint(double atol = default_atol);
    bool is_positive_semidefinite(double atol = default_atol);
    bool is_unitary(double atol = default_atol);
    bool is_hermitian(double atol = default_atol) { return is_self_adjoint(atol); }

    // Specifically quantum things
    QTensorCpp as_density_matrix(double atol = default_atol, double max_relative_correction = 0.1);
    double entropy_von_neumann();
    double entropy_renyi(double alpha);
    double magic(double alpha = 2.0);
    double fidelity(const QTensorCpp& other);
    double fidelity_python(const py::object& other);
    double purity();
    std::vector<double> probabilities() const;
    py::list probabilities_python() const;
    Complex expectation_value(const QTensorCpp& other, int nshots = 0) const;
    Complex expectation_value(const MatrixFreeHamiltonian& other) const;
    Complex expectation_value_python(const py::object& other, int nshots = 0) const;
    QTensorCpp partial_trace_python(const py::object& keep) const;
    QTensorCpp partial_trace(const std::set<int>& keep) const;
    QTensorCpp commutator(const QTensorCpp& other) const;
    QTensorCpp commutator_python(const py::object& other) const;
    QTensorCpp anticommutator(const QTensorCpp& other) const;
    QTensorCpp anticommutator_python(const py::object& other) const;
    QTensorCpp reset_qubits_python(const py::object& qubits);
    QTensorCpp reset_qubits(const std::set<int>& qubits);

    // Static initializers for common states
    static QTensorCpp identity(int nqubits);
    static QTensorCpp zero(int nqubits);
    static QTensorCpp one(int nqubits);
    static QTensorCpp ket_python(const py::object& state);
    static QTensorCpp ket(const std::vector<int>& qubit_values);
    static QTensorCpp bra_python(const py::object& state);
    static QTensorCpp bra(const std::vector<int>& qubit_values);
    static QTensorCpp ghz(int nqubits);
    static QTensorCpp uniform(int nqubits);
    static QTensorCpp tensor_product_python(const py::list& others);
    static QTensorCpp tensor_product(const std::vector<QTensorCpp>& others);

    // C++ specific overloads
    QTensorCpp operator+(const QTensorCpp& other) const { return add(other); }
    QTensorCpp operator-(const QTensorCpp& other) const { return sub(other); }
    QTensorCpp operator*(const QTensorCpp& other) const { return matmul(other); }
    friend std::ostream& operator<<(std::ostream& os, const QTensorCpp& qt) {
        os << qt.as_string();
        return os;
    }
    bool operator==(const QTensorCpp& other) const { return equals(other); }
    bool operator!=(const QTensorCpp& other) const { return !equals(other); }
    QTensorCpp operator/(Complex scalar) const { return div(scalar); }
    QTensorCpp operator/(double scalar) const { return div(Complex(scalar, 0.0)); }
    Complex operator[](const std::pair<int, int>& index) const { return coeff(index.first, index.second); }
    QTensorCpp operator*(Complex scalar) const { return mul(scalar); }
    QTensorCpp operator*(double scalar) const { return mul(Complex(scalar, 0.0)); }
};
#pragma GCC visibility pop

// GCOV_EXCL_BR_STOP
