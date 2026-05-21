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

#include "matrix_free_operator.h"
#include <bitset>

// GCOV_EXCL_BR_START

typedef std::bitset<256> Bitset;

// moving now to use a bitmask for x and z rather than storing operators and strings
class PauliString {

public:

    // neither means i, x means x, z means z, both means y
    Bitset x_mask;
    Bitset z_mask;
    int nqubits;

    struct HashFunction {
        std::size_t operator()(const PauliString& ps) const {
            std::size_t hash = 0;
            for (size_t i = 0; i < ps.nqubits; ++i) {
                hash ^= std::hash<bool>()(ps.x_mask[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                hash ^= std::hash<bool>()(ps.z_mask[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };

    PauliString() {throw std::runtime_error("Default constructor for PauliString is not allowed. Please specify the number of qubits.");}
    PauliString(int num_qubits) : x_mask(), z_mask(), nqubits(num_qubits) {}
    PauliString(int num_qubits, char pauli, int target_qubit) : x_mask(), z_mask(), nqubits(num_qubits) {
        if (pauli == 'X') {
            x_mask.set(target_qubit);
        } else if (pauli == 'Z') {
            z_mask.set(target_qubit);
        } else if (pauli == 'Y') {
            x_mask.set(target_qubit);
            z_mask.set(target_qubit);
        }
    }

    size_t size() const {
        return (x_mask | z_mask).count();
    }

    bool operator==(const PauliString& other) const {
        return x_mask == other.x_mask && z_mask == other.z_mask;
    }

    friend std::ostream& operator<<(std::ostream& os, const PauliString& ps) {
        bool printed_something = false;
        for (size_t i = 0; i < ps.x_mask.size(); ++i) {
            if (ps.x_mask[i] && ps.z_mask[i]) {
                os << "Y(" << i << ")";
                printed_something = true;
            } else if (ps.x_mask[i]) {
                os << "X(" << i << ")";
                printed_something = true;
            } else if (ps.z_mask[i]) {
                os << "Z(" << i << ")";
                printed_something = true;
            }
        }
        if (!printed_something) {
            os << "I";
        }
        return os;
    }
                
};

class MatrixFreeHamiltonian {
   private:
    int nqubits = 0;
    std::unordered_map<PauliString, std::complex<double>, PauliString::HashFunction> operators;
    mutable DenseMatrix m_temp_state;
    mutable DenseMatrix m_new_state;

   public:
    MatrixFreeHamiltonian(int nqubits) : nqubits(nqubits) {}
    MatrixFreeHamiltonian(int nqubits, double val) : nqubits(nqubits) { operators[PauliString(nqubits)] = std::complex<double>(val, 0.0); }
    MatrixFreeHamiltonian(int nqubits, const PauliString& op) : nqubits(nqubits) { operators[op] = 1.0; }
    MatrixFreeHamiltonian(int nqubits, const MatrixFreeOperator& op) : nqubits(nqubits) {
        PauliString ps(nqubits);
        for (int target : op.get_target_qubits()) {
            if (op.get_name() == "X") {
                ps.x_mask.flip(target);
            } else if (op.get_name() == "Z") {
                ps.z_mask.flip(target);
            } else if (op.get_name() == "Y") {
                ps.x_mask.flip(target);
                ps.z_mask.flip(target);
            }
        }
        operators[ps] = 1.0;
    }
    MatrixFreeHamiltonian(int nqubits, const PauliString& op, const std::complex<double>& coeff) : nqubits(nqubits) { operators[op] = coeff; }
    MatrixFreeHamiltonian(int nqubits, const std::unordered_map<PauliString, std::complex<double>, PauliString::HashFunction>& ops) : nqubits(nqubits), operators(ops) {}
    void apply(const DenseMatrix& input_state, MatrixFreeApplicationType application_type, DenseMatrix& output_state) const;
    double expectation_value(const DenseMatrix& state) const;
    double expectation_value(const MatrixFreeHamiltonian& other) const;
    MatrixFreeHamiltonian operator*(const std::complex<double>& scalar) const;
    MatrixFreeHamiltonian operator*(const double& scalar) const;
    friend MatrixFreeHamiltonian operator*(const std::complex<double>& scalar, const MatrixFreeHamiltonian& hamiltonian);
    MatrixFreeHamiltonian operator*(const MatrixFreeHamiltonian& other) const;
    MatrixFreeHamiltonian operator+(const MatrixFreeHamiltonian& other) const;
    MatrixFreeHamiltonian operator-(const MatrixFreeHamiltonian& other) const;
    MatrixFreeHamiltonian& operator*=(const std::complex<double>& scalar);
    MatrixFreeHamiltonian& operator+=(const MatrixFreeHamiltonian& other);
    bool operator==(const MatrixFreeHamiltonian& other) const;
    void add(const std::complex<double>& coeff, const PauliString& op);
    void add(const std::complex<double>& coeff, const std::vector<MatrixFreeOperator>& ops);
    friend std::ostream& operator<<(std::ostream& os, const MatrixFreeHamiltonian& hamiltonian);
    std::unordered_map<PauliString, std::complex<double>, PauliString::HashFunction> get_operators() const { return operators; }
    void prune(double threshold, int max_terms);
    int get_nqubits() const { return nqubits; }
    MatrixFreeHamiltonian conjugate() const;
    size_t size() const { return operators.size(); }
};


// GCOV_EXCL_BR_STOP