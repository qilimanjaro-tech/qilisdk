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

#include <complex>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "../../../libs/eigen.h"
#include "../digital/gate.h"
#include "../representations/matrix_free_hamiltonian.h"

// GCOV_EXCL_BR_START

class Tensor {
   protected:
    std::vector<int> shape;
    std::vector<Complex> data;
    int64_t flat_index(const std::vector<int>& index) const;

   public:
    
    // Constructors
    Tensor() = default;
    explicit Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape, const std::vector<Complex>& values);

    // Various ways to access the shape and data
    int rank() const { return static_cast<int>(shape.size()); }
    int64_t size() const { return static_cast<int64_t>(data.size()); }
    int extent(int leg) const { return shape[leg]; }
    const std::vector<int>& get_shape() const { return shape; }
    const std::vector<Complex>& raw() const { return data; }
    std::vector<Complex>& raw() { return data; }
    Complex operator()(const std::vector<int>& index) const { return data[flat_index(index)]; }
    Complex& operator()(const std::vector<int>& index) { return data[flat_index(index)]; }

    // Leg manipulation
    Tensor permute(const std::vector<int>& perm) const;
    void reshape(const std::vector<int>& new_shape);
    Tensor fuse(const std::vector<std::vector<int>>& groups) const;

    // Matrix views
    Eigen::Map<const DenseMatrix> matrix_view(int nrow_legs) const;
    Eigen::Map<DenseMatrix> matrix_view(int nrow_legs);
    DenseMatrix as_matrix(const std::vector<int>& row_legs) const;
    static Tensor from_matrix(const DenseMatrix& m, const std::vector<int>& shape);

    // Contracts two tensors along the given legs
    Tensor contract(const Tensor& other, const std::vector<int>& legs_a, const std::vector<int>& legs_b) const;

    // Truncated SVD
    void split(const std::vector<int>& left_legs, int max_bond_dimension, Real cutoff, Tensor& left, RealVector& singular_values, Tensor& right, Real* truncation_error = nullptr) const;

    // Reductions
    Tensor conjugate() const;
    Complex trace_all_with(const Tensor& other) const;
    Real norm() const;
    void scale(Complex factor);
    void set_zero();
    bool has_nan() const;

};

class MPSTensor : public Tensor {
   public:
    static constexpr int PHYSICAL_DIMENSION = 2;
    MPSTensor() : Tensor({1, PHYSICAL_DIMENSION, 1}) {}
    MPSTensor(int left, int right);
    MPSTensor(int left, int right, const std::vector<Complex>& values);
    explicit MPSTensor(const Tensor& t);
    int left() const { return shape[0]; }
    int right() const { return shape[2]; }
    Complex operator()(int l, int p, int r) const { return data[l + shape[0] * (p + PHYSICAL_DIMENSION * r)]; }
    Complex& operator()(int l, int p, int r) { return data[l + shape[0] * (p + PHYSICAL_DIMENSION * r)]; }
    Eigen::Map<const DenseMatrix> right_fused() const { return matrix_view(1); }
    Eigen::Map<const DenseMatrix> left_fused() const { return matrix_view(2); }
    Eigen::Map<const DenseMatrix, 0, Eigen::OuterStride<>> physical_slice(int p) const;
};

class MPSState {
   private:
    int nqubits = 0;
    std::vector<MPSTensor> sites;
    int centre = 0;
    int max_bond_dimension = 64;
    Real truncation_cutoff = 1e-10;
    Real total_truncation_error = 0.0;
    mutable std::mt19937_64 rng{std::random_device{}()};
    Real split_two_site(int q, const Tensor& theta, int keep_centre_on);

   public:
    
    // Constructors
    explicit MPSState(int nqubits);
    MPSState(int nqubits, const std::string& b);

    // Getters and setters
    void set_seed(uint64_t seed) const { rng.seed(seed); }
    void set_max_bond_dimension(int d) { max_bond_dimension = d; }
    void set_truncation_cutoff(Real c) { truncation_cutoff = c; }
    int get_nqubits() const { return nqubits; }
    const MPSTensor& get_site_tensor(int q) const { return sites[q]; }
    int get_bond_dimension(int bond) const;
    int get_max_bond_dimension_used() const;
    Real get_truncation_error() const { return total_truncation_error; }

    // Evolving the state
    Real apply_gate(const Gate& gate);
    Real apply_one_site(const DenseMatrix& u, int q);
    Real apply_two_site(const DenseMatrix& u, int q);
    void normalize();
    void move_centre(int q);

    // Outputs
    Real norm() const;
    Complex amplitude(const std::string& b) const;
    Complex overlap(const MPSState& other) const;
    Complex expectation_value(const DenseMatrix& observable, const std::vector<int>& qubits) const;
    Real expectation_value(const MatrixFreeHamiltonian& H) const;
    std::string sample() const;
    std::string sample(std::mt19937_64& engine) const;
    std::map<std::string, int> sample(int nshots) const;
    DenseMatrix as_dense() const;

};

std::ostream& operator<<(std::ostream& os, const MPSState& state);

// GCOV_EXCL_BR_STOP
