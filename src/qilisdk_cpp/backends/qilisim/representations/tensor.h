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
#include <vector>

#include "../../../libs/eigen.h"

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
    static Tensor from_matrix(Eigen::Ref<const DenseMatrix> m, const std::vector<int>& shape);

    // Contracts two tensors along the given legs
    Tensor contract(const Tensor& other, const std::vector<int>& legs_a, const std::vector<int>& legs_b) const;

    // Contracts every leg of two identically shaped tensors, giving a scalar
    Complex contract_all(const Tensor& other) const;

    // Truncated SVD
    void split(const std::vector<int>& left_legs, int max_bond_dimension, Real cutoff, Tensor& left, RealVector& singular_values, Tensor& right, Real* truncation_error = nullptr) const;

    // Reductions
    Tensor conjugate() const;
    Real norm() const;
    void scale(Complex factor);
    void set_zero();
    bool has_nan() const;
};

// GCOV_EXCL_BR_STOP
