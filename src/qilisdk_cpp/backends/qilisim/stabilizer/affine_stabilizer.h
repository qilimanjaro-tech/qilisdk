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
#include <set>
#include <complex>
#include "../libs/eigen.h"
#include "../digital/gate.h"

typedef std::set<int> IndexSet;
typedef std::pair<std::complex<double>, std::set<std::pair<char, IndexSet>>> StateCoefficient;
typedef std::vector<std::pair<char, IndexSet>> StateBasis;
typedef std::vector<std::tuple<StateCoefficient, StateBasis, StateBasis>> State;

class AffineStabilizerState {
    private:
        // state might look something like (1+2i)*z1*z1,3 |0+-1s2><0-+1s1|
        // basis chars:
        //  - |+> = (|0> + |1>) / sqrt(2)
        //  - |0>
        //  - |1>
        //  - |+s1> = (|00> + |11>) / sqrt(2)
        //  - |+d1> = (|01> + |10>) / sqrt(2)
        // coefficient chars:
        //  - z1 = -1 if q[1], else 1
        //  - z1,2 = -1 if q[1] and q[2], else 1
        //  - i1 = i if q[1], else 1
        //  - i1,2 = i if q[1] and q[2], else 1
        State state;

    public:
        
        AffineStabilizerState() {}
        AffineStabilizerState(int n_qubits, bool density=false);
        AffineStabilizerState(const AffineStabilizerState& other) : state(other.state) {}
        AffineStabilizerState(const SparseMatrix& initial_state);
        AffineStabilizerState& operator=(const AffineStabilizerState& other);

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

        auto begin() { return state.begin(); }
        auto end() { return state.end(); }
        auto begin() const { return state.begin(); }
        auto end() const { return state.end(); }

        std::map<std::string, int> sample(int n_samples, int seed) const;

        friend std::ostream& operator<<(std::ostream& os, const AffineStabilizerState& mfs);

        AffineStabilizerState operator+(const AffineStabilizerState& other) const;
        AffineStabilizerState operator-(const AffineStabilizerState& other) const;
        AffineStabilizerState& operator+=(const AffineStabilizerState& other);
        AffineStabilizerState& operator-=(const AffineStabilizerState& other);

};

class AffineStabilizerOperator {
    private:
        std::string name;
        int target_qubit;
        int control_qubit;
    public:
        AffineStabilizerOperator(const Gate& gate);
        void apply(AffineStabilizerState& output_state) const;
        AffineStabilizerState operator*(const AffineStabilizerState& input_state) const;
        friend std::ostream& operator<<(std::ostream& os, const AffineStabilizerOperator& mfo);
};