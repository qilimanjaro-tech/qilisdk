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
#include "../libs/eigen.h"
#include "../libs/pybind.h"

const double default_atol = 1e-12;

// The main QiliSim C++ class
class QTensorCpp {
   private:
    // The main data of the class, an Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>
    SparseMatrix _data;

    // Cache various things to faster reaccess
    std::vector<std::complex<double>> _eigenvalues;
    std::vector<SparseMatrix> _eigenvectors;
    bool _is_positive_computed = false;
    bool _is_positive = false;
    double _atol_used_for_positive = 0.0;
    std::complex<double> _trace = 0.0;
    bool _trace_computed = false;
    bool _is_self_adjoint_computed = false;
    bool _is_self_adjoint = false;
    bool _trace_squared_computed = false;
    double _trace_squared = 0.0;
    bool _max_adjoint_diff_computed = false;
    double _max_adjoint_diff = 0.0;
    bool _max_unitary_diff_computed = false;
    double _max_unitary_diff = 0.0;
    bool _rank_computed = false;
    int _rank = 0;

   public:
    // Constructors and basic accessors
    QTensorCpp() {}
    QTensorCpp(const SparseMatrix& data);
    QTensorCpp(const py::object& data);
    QTensorCpp(int rows, int cols);
    const SparseMatrix& get_data() const { return _data; }
    py::object as_scipy() const;
    py::object as_numpy() const;
    int get_nqubits() const;
    std::pair<int, int> get_shape() const;
    void clear_cache();
    std::string as_string() const;
    DenseMatrix as_dense() const;
    std::complex<double> coeff(int row, int col) const { return _data.coeff(row, col); }

    // Matrix arithmetic
    double norm(const std::string& norm_type);
    void compute_eigendecomposition();
    bool equals_python(const py::object& other) const;
    bool equals(const QTensorCpp& other) const;
    std::complex<double> dot(const QTensorCpp& other) const;
    std::complex<double> dot_python(const py::object& other) const;
    QTensorCpp normalized(const std::string& norm_type);
    QTensorCpp inverse() const;
    QTensorCpp pow(int n) const;
    QTensorCpp sqrt() const;
    QTensorCpp log() const;
    QTensorCpp exp() const;
    int rank();
    std::vector<std::complex<double>> get_eigenvalues() const;
    py::object get_eigenvalues_python() const;
    std::vector<SparseMatrix> get_eigenvectors() const;
    py::object get_eigenvectors_python() const;
    QTensorCpp conjugate() const;
    QTensorCpp transpose() const;
    QTensorCpp adjoint() const;
    std::complex<double> trace();
    QTensorCpp tensor_product_python(const py::list& others) const;
    QTensorCpp tensor_product(const std::vector<QTensorCpp>& others) const;
    QTensorCpp add_python(const py::object& other) const;
    QTensorCpp add(const QTensorCpp& other) const;
    QTensorCpp sub_python(const py::object& other) const;
    QTensorCpp sub(const QTensorCpp& other) const;
    QTensorCpp mul_python(const py::object& other) const;
    QTensorCpp mul(const QTensorCpp& other) const;
    QTensorCpp matmul_python(const py::object& other) const;
    QTensorCpp matmul(const QTensorCpp& other) const;
    QTensorCpp div(std::complex<double> scalar) const;
    QTensorCpp mul(std::complex<double> scalar) const;

    // Cached checks
    bool is_ket() const;
    bool is_bra() const;
    bool is_operator() const;
    bool is_scalar() const;
    bool is_pure(double atol = default_atol);
    bool is_density_matrix(double atol = default_atol);
    bool is_self_adjoint(double atol = default_atol);
    bool is_positive_semidefinite(double atol = default_atol);
    bool is_unitary(double atol = default_atol);
    bool is_hermitian(double atol = default_atol) { return is_self_adjoint(atol); }

    // Specifically quantum things
    QTensorCpp as_density_matrix(double atol = default_atol, double max_relative_correction = 0.1);
    double entropy_von_neumann();
    double entropy_renyi(double alpha);
    double fidelity(const QTensorCpp& other) const;
    double fidelity_python(const py::object& other) const;
    double purity();
    std::vector<double> probabilities() const;
    py::list probabilities_python() const;
    std::complex<double> expectation_value(const QTensorCpp& other, int nshots = 0) const;
    std::complex<double> expectation_value_python(const py::object& other, int nshots = 0) const;
    QTensorCpp partial_trace_python(const py::object& keep) const;
    QTensorCpp partial_trace(const std::set<int>& keep) const;
    QTensorCpp commutator(const QTensorCpp& other) const;
    QTensorCpp commutator_python(const py::object& other) const;
    QTensorCpp anticommutator(const QTensorCpp& other) const;
    QTensorCpp anticommutator_python(const py::object& other) const;
    QTensorCpp reset_qubits_python(const py::object& qubits);
    QTensorCpp reset_qubits(const std::set<int>& qubits);

    // Static initializers for common states
    static QTensorCpp identity(int dim);
    static QTensorCpp zero(int rows, int cols);
    static QTensorCpp ket_python(const py::object& state);
    static QTensorCpp ket(const std::vector<int>& qubit_values);
    static QTensorCpp bra_python(const py::object& state);
    static QTensorCpp bra(const std::vector<int>& qubit_values);
    static QTensorCpp ghz(int nqubits);

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
    QTensorCpp operator/(std::complex<double> scalar) const { return div(scalar); }
    QTensorCpp operator/(double scalar) const { return div(std::complex<double>(scalar, 0.0)); }
    std::complex<double> operator[](const std::pair<int, int>& index) const { return coeff(index.first, index.second); }
    QTensorCpp operator*(std::complex<double> scalar) const { return mul(scalar); }
    QTensorCpp operator*(double scalar) const { return mul(std::complex<double>(scalar, 0.0)); }
};
