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
#pragma once

#include <vector>
#include <map>
#include <complex>
#include "../libs/eigen.h"
#include "../digital/gate.h"

typedef std::vector<std::pair<double, double>> ProductState;
typedef std::pair<ProductState, ProductState> StateElement;

class MatrixFreeState {
    private:
        // std::map<StateElement, std::complex<double>> state;
        // std::vector<std::pair<StateElement, std::complex<double>>> state;

        // state might look something like (1+2i)*z1*z1,3 |0+-1><0-+1|
        // basis chars:
        //  - |+> = (|0> + |1>) / sqrt(2)
        //  - |-> = (|0> - |1>) / sqrt(2)
        //  - |0>
        //  - |1>
        //  - |+s1> = (|00> + |11>) / sqrt(2)
        //  - |+d1> = (|01> + |10>) / sqrt(2)
        // coefficient chars:
        //  - z1 = -1 if q[1], else 1
        //  - z1,2 = -1 if q[1] and q[2], else 1
        //  - i1 = i if q[1], else 1
        //  - i1,2 = i if q[1] and q[2], else 1
        typedef std::pair<std::vector<std::tuple<char, int, int>>, std::complex<double>> StateCoefficient;
        typedef std::vector<std::tuple<char, int>> StateBasis;
        std::vector<std::tuple<StateCoefficient, StateBasis, StateBasis>> state;

    public:
        
        MatrixFreeState() {}
        MatrixFreeState(int n_qubits, bool density=false);
        MatrixFreeState(const MatrixFreeState& other) : state(other.state) {}
        MatrixFreeState(const SparseMatrix& initial_state);
        MatrixFreeState& operator=(const MatrixFreeState& other);

        bool is_ket() const;
        bool is_bra() const;
        bool is_density_matrix() const;
        bool is_pure(double atol=1e-12) const;
        bool empty() const;
        int n_qubits() const;
        size_t size() const;

        void to_density_matrix();
        void normalize();
        void prune(double atol);

        void add_state(const std::pair<StateElement, std::complex<double>>& new_element) {
            state.push_back(new_element);
        }

        auto begin() { return state.begin(); }
        auto end() { return state.end(); }
        auto begin() const { return state.begin(); }
        auto end() const { return state.end(); }

        std::map<std::string, int> sample(int n_samples, int seed) const;

        friend std::ostream& operator<<(std::ostream& os, const MatrixFreeState& mfs);

        MatrixFreeState operator+(const MatrixFreeState& other) const;
        MatrixFreeState operator-(const MatrixFreeState& other) const;
        MatrixFreeState& operator+=(const MatrixFreeState& other);
        MatrixFreeState& operator-=(const MatrixFreeState& other);

};

class MatrixFreeOperator {
    private:
        SparseMatrix matrix;
        std::vector<int> target_qubits;
    public:
        MatrixFreeOperator(const SparseMatrix& mat, const std::vector<int>& targets = {}) : matrix(mat), target_qubits(targets) {}
        MatrixFreeOperator(const Gate& gate);
        MatrixFreeState apply(const MatrixFreeState& input_state, bool only_multiply=false) const;
        MatrixFreeState operator*(const MatrixFreeState& input_state) const;
        friend std::ostream& operator<<(std::ostream& os, const MatrixFreeOperator& mfo);
};