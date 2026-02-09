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

#include "matrix_free.h"
#include <random>
#include <iostream> // TODO(luke) remove

// TODO(luke) docstrings

void element_to_computational(const StateElement& product_state, const std::vector<int>& target_qubits, DenseMatrix& vec) {
    // density matrix
    if (product_state.first.size() > 0 && product_state.second.size() > 0) {
        DenseMatrix bra_vec = DenseMatrix::Ones(1, 1);
        DenseMatrix ket_vec = DenseMatrix::Ones(1, 1);
        for (int i : target_qubits) {
            double theta_ket = product_state.first[i].first;
            double phi_ket = product_state.first[i].second;
            double theta_bra = product_state.second[i].first;
            double phi_bra = product_state.second[i].second;
            DenseMatrix single_qubit_ket(2, 1);
            DenseMatrix single_qubit_bra(2, 1);
            single_qubit_ket(0, 0) = std::cos(theta_ket / 2);
            single_qubit_ket(1, 0) = std::sin(theta_ket / 2) * std::exp(std::complex<double>(0, phi_ket));
            single_qubit_bra(0, 0) = std::cos(theta_bra / 2);
            single_qubit_bra(1, 0) = std::sin(theta_bra / 2) * std::exp(std::complex<double>(0, phi_bra));
            ket_vec = kroneckerProduct(ket_vec, single_qubit_ket).eval();
            bra_vec = kroneckerProduct(bra_vec, single_qubit_bra).eval();
        }
        vec = ket_vec * bra_vec.adjoint();
    // ket
    } else if (product_state.first.size() > 0) {
        DenseMatrix single_qubit_vec;
        vec = DenseMatrix::Ones(1, 1);
        for (int i : target_qubits) {
            double theta = product_state.first[i].first;
            double phi = product_state.first[i].second;
            single_qubit_vec = DenseMatrix(2, 1);
            single_qubit_vec(0, 0) = std::cos(theta / 2);
            single_qubit_vec(1, 0) = std::sin(theta / 2) * std::exp(std::complex<double>(0, phi));
            vec = kroneckerProduct(vec, single_qubit_vec).eval();
        }   
    // bra
    } else if (product_state.second.size() > 0) {
        DenseMatrix single_qubit_vec;
        vec = DenseMatrix::Ones(1, 1);
        for (int i : target_qubits) {
            double theta = product_state.second[i].first;
            double phi = product_state.second[i].second;
            single_qubit_vec = DenseMatrix(1, 2);
            single_qubit_vec(0, 0) = std::cos(theta / 2);
            single_qubit_vec(0, 1) = std::sin(theta / 2) * std::exp(std::complex<double>(0, phi));
            vec = kroneckerProduct(vec, single_qubit_vec).eval();
        }
    }
}

void computational_to_elements(const DenseMatrix& vec, const std::vector<int>& target_qubits, const StateElement& original, MatrixFreeState& output_state) {
    // density matrix
    if (vec.rows() == vec.cols()) {
        // if it's a single-qubit state, can do theta and phi directly
        if (target_qubits.size() == 1) {
            StateElement element = original;
            double theta_ket = std::atan2(std::abs(vec(1, 0)), std::abs(vec(0, 0))) * 2;
            double phi_ket = std::arg(vec(1, 0)) - std::arg(vec(0, 0));
            double theta_bra = std::atan2(std::abs(vec(0, 1)), std::abs(vec(0, 0))) * 2;
            double phi_bra = std::arg(vec(0, 1)) - std::arg(vec(0, 0));
            element.first[target_qubits[0]] = std::make_pair(theta_ket, phi_ket);
            element.second[target_qubits[0]] = std::make_pair(theta_bra, phi_bra);
            std::complex<double> coeff = 1.0 / sqrt(std::norm(vec(0, 0)) + std::norm(vec(1, 1)));
            // output_state[element] += coeff;
            output_state.add_state(std::make_pair(element, coeff));
        // otherwise, need to iterate through the vector and form the product state from the computational
        } else {
            for (int i = 0; i < vec.rows(); ++i) {
                for (int j = 0; j < vec.cols(); ++j) {
                    StateElement element = original;
                    std::complex<double> coeff = vec(i, j);
                    if (std::norm(coeff) > 1e-12) {
                        for (size_t q = 0; q < target_qubits.size(); ++q) {
                            int bit_i = (i >> (target_qubits.size() - 1 - q)) & 1;
                            int bit_j = (j >> (target_qubits.size() - 1 - q)) & 1;
                            double theta_i = bit_i == 0 ? 0.0 : M_PI;
                            double phi_i = 0.0;
                            double theta_j = bit_j == 0 ? 0.0 : M_PI;
                            double phi_j = 0.0;
                            int qubit = target_qubits[q];
                            element.first[qubit] = std::make_pair(theta_i, phi_i);
                            element.second[qubit] = std::make_pair(theta_j, phi_j);
                        }
                    }
                    output_state.add_state(std::make_pair(element, coeff));
                }
            }
        }
    // ket
    } else if (vec.cols() == 1) {
        // if it's a single-qubit state, can do theta and phi directly
        if (target_qubits.size() == 1) {
            StateElement element = original;
            double theta = std::atan2(std::abs(vec(1, 0)), std::abs(vec(0, 0))) * 2;
            double phi = std::arg(vec(1, 0)) - std::arg(vec(0, 0));
            element.first[target_qubits[0]] = std::make_pair(theta, phi);
            std::complex<double> coeff = 1.0 / sqrt(std::norm(vec(0, 0)) + std::norm(vec(1, 0)));
            output_state.add_state(std::make_pair(element, coeff));
        // otherwise, need to iterate through the vector and form the product state from the computational
        } else {
            for (int i = 0; i < vec.rows(); ++i) {
                std::complex<double> coeff = vec(i, 0);
                if (std::norm(coeff) > 1e-12) {
                    StateElement element = original;
                    for (size_t q = 0; q < target_qubits.size(); ++q) {
                        int bit = (i >> (target_qubits.size() - 1 - q)) & 1;
                        double theta = bit == 0 ? 0.0 : M_PI;
                        double phi = 0.0;
                        int qubit = target_qubits[q];
                        element.first[qubit] = std::make_pair(theta, phi);
                    }
                    output_state.add_state(std::make_pair(element, coeff));
                }
            }
        }
    // bra
    } else if (vec.rows() == 1) {
        // if it's a single-qubit state, can do theta and phi directly
        if (target_qubits.size() == 1) {
            StateElement element = original;
            double theta = std::atan2(std::abs(vec(0, 1)), std::abs(vec(0, 0))) * 2;
            double phi = std::arg(vec(0, 1)) - std::arg(vec(0, 0));
            element.second[target_qubits[0]] = std::make_pair(theta, phi);
            std::complex<double> coeff = 1.0 / sqrt(std::norm(vec(0, 0))     + std::norm(vec(0, 1)));
            output_state.add_state(std::make_pair(element, coeff));
        // otherwise, need to iterate through the vector and form the product state from the computational
        } else { 
            for (int j = 0; j < vec.cols(); ++j) {
                std::complex<double> coeff = vec(0, j);
                if (std::norm(coeff) > 1e-12) {
                    StateElement element = original;
                    for (size_t q = 0; q < target_qubits.size(); ++q) {
                        int bit = (j >> (target_qubits.size() - 1 - q)) & 1;
                        double theta = bit == 0 ? 0.0 : M_PI;
                        double phi = 0.0;
                        int qubit = target_qubits[q];
                        element.second[qubit] = std::make_pair(theta, phi);
                    }
                    output_state.add_state(std::make_pair(element, coeff));
                }
            }
        }
    }
}

MatrixFreeState::MatrixFreeState(int n_qubits, bool density) {
    StateElement zero_state;
    for (int i = 0; i < n_qubits; ++i) {
        if (density) {
            zero_state.second.emplace_back(0.0, 0.0);
        }
        zero_state.first.emplace_back(0.0, 0.0);
    }
    // state[zero_state] = 1.0;
    state.push_back(std::make_pair(zero_state, 1.0));
}

MatrixFreeState::MatrixFreeState(const SparseMatrix& initial_state) {
    MatrixFreeState new_state;
    int n_qubits = std::ceil(std::log2(initial_state.rows()));
    bool is_density = initial_state.rows() == initial_state.cols();
    StateElement zero_state_element = MatrixFreeState(n_qubits, is_density).state.begin()->first;
    std::vector<int> all_qubits(n_qubits);
    for (int i = 0; i < n_qubits; ++i) {
        all_qubits[i] = i;
    }
    computational_to_elements(initial_state, all_qubits, zero_state_element, *this);
}

MatrixFreeState& MatrixFreeState::operator=(const MatrixFreeState& other) {
    if (this != &other) {
        state = other.state;
    }
    return *this;
}

bool MatrixFreeState::is_ket() const {
    if (state.empty()) {
        return false;
    }
    return state.begin()->first.second.empty();
}

bool MatrixFreeState::is_bra() const {
    if (state.empty()) {
        return false;
    }
    return state.begin()->first.first.empty();
}

bool MatrixFreeState::is_density_matrix() const {
    if (state.empty()) {
        return false;
    }
    return !state.begin()->first.first.empty() && !state.begin()->first.second.empty();
}

void MatrixFreeState::to_density_matrix() {
    if (is_density_matrix()) {
        return;
    }
    MatrixFreeState new_state;
    if (is_ket()) {
        for (const auto& pair : state) {
            const auto& ket = pair.first.first;
            std::complex<double> amplitude = pair.second;
            StateElement element = std::make_pair(ket, ket);
            new_state.state.push_back(std::make_pair(element, amplitude * std::conj(amplitude)));
        }
    } else if (is_bra()) {
        for (const auto& pair : state) {
            const auto& bra = pair.first.second;
            std::complex<double> amplitude = pair.second;
            StateElement element = std::make_pair(bra, bra);
            new_state.state.push_back(std::make_pair(element, amplitude * std::conj(amplitude)));
        }
    }
    state = new_state.state;
}

bool MatrixFreeState::is_pure(double atol) const {
    if (!is_density_matrix()) {
        return true;
    }
    double trace_squared = 0.0;
    for (const auto& pair : state) {
        if (pair.first.first == pair.first.second) {
            trace_squared += std::norm(pair.second);
        }
    }
    return std::abs(trace_squared - 1.0) < atol;
}

bool MatrixFreeState::empty() const {
    return state.empty();
}

void MatrixFreeState::normalize() {
    if (is_density_matrix()) {
        double trace = 0.0;
        for (const auto& pair : state) {
            if (pair.first.first == pair.first.second) {
                trace += pair.second.real();
            }
        }
        for (auto& pair : state) {
            pair.second /= trace;
        }
    } else {
        double norm = 0.0;
        for (const auto& pair : state) {
            norm += std::norm(pair.second);
        }
        norm = std::sqrt(norm);
        for (auto& pair : state) {
            pair.second /= norm;
        }
    }
}

int MatrixFreeState::n_qubits() const {
    if (state.empty()) {
        return 0;
    }
    return std::max(state.begin()->first.first.size(), state.begin()->first.second.size());
}

void MatrixFreeState::prune(double atol) {
    for (auto it = state.begin(); it != state.end(); ) {
        if (std::norm(it->second) < atol * atol) {
            it = state.erase(it);
        } else {
            ++it;
        }
    }
}

// std::complex<double>& MatrixFreeState::operator[](const StateElement& key) {
//     for (auto& pair : state) {
//         if (pair.first == key) {
//             return pair.second;
//         }
//     }
//     state.push_back(std::make_pair(key, std::complex<double>(0.0, 0.0)));
//     return state.back().second;
// }
// const std::complex<double>& MatrixFreeState::operator[](const StateElement& key) const {
//     for (const auto& pair : state) {
//         if (pair.first == key) {
//             return pair.second;
//         }
//     }
//     static std::complex<double> zero(0.0, 0.0);
//     return zero;
// }

std::map<std::string, int> MatrixFreeState::sample(int n_samples, int seed) const {
    // TODO (luke)
    std::mt19937 gen(seed);
    std::string zero_string = "";
    for (int i = 0; i < n_qubits(); ++i) {
        zero_string += "0";
    }
    std::map<std::string, int> counts;
    counts[zero_string] = n_samples;
    return counts;
}

std::ostream& operator<<(std::ostream& os, const StateElement& element) {
    if (!element.first.empty()) {
        os << "|";
        for (const auto& amp : element.first) {
            os << "(" << amp.first << "," << amp.second << ")";
            if (&amp != &element.first.back()) {
                os << "x";
            }
        }
        os << ">";
    }
    if (!element.second.empty()) {
        os << "<";
        for (const auto& amp : element.second) {
            os << "(" << amp.first << "," << amp.second << ")";
            if (&amp != &element.second.back()) {
                os << "x";
            }
        }
        os << "|";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<int>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

std::ostream& operator<<(std::ostream& os, const MatrixFreeState& mfs) {
    for (const auto& pair : mfs.state) {
        os << pair.first << " : " << pair.second;
        if (pair != *mfs.state.rbegin()) {
            os << "\n";
        }
    }
    return os;
}

MatrixFreeState MatrixFreeState::operator+(const MatrixFreeState& other) const {
    MatrixFreeState result = *this;
    for (const auto& pair : other.state) {
        // result.state[pair.first] += pair.second;
        result.state.push_back(std::make_pair(pair.first, pair.second));
    }
    return result;
}
MatrixFreeState MatrixFreeState::operator-(const MatrixFreeState& other) const {
    MatrixFreeState result = *this;
    for (const auto& pair : other.state) {
        // result.state[pair.first] -= pair.second;
        result.state.push_back(std::make_pair(pair.first, -pair.second));
    }
    return result;
}
MatrixFreeState& MatrixFreeState::operator+=(const MatrixFreeState& other) {
    for (const auto& pair : other.state) {
        // state[pair.first] += pair.second;
        state.push_back(std::make_pair(pair.first, pair.second));
    }
    return *this;
}
MatrixFreeState& MatrixFreeState::operator-=(const MatrixFreeState& other) {
    for (const auto& pair : other.state) {
        // state[pair.first] -= pair.second;
        state.push_back(std::make_pair(pair.first, -pair.second));
    }
    return *this;
}

size_t MatrixFreeState::size() const {
    return state.size();
}

 



MatrixFreeState MatrixFreeOperator::apply(const MatrixFreeState& input_state, bool only_multiply) const {
    MatrixFreeState output_state;
    DenseMatrix vec;
    DenseMatrix output_vec;
    for (auto& pair : input_state) {

        // Convert product state of target qubits to computational basis
        element_to_computational(pair.first, target_qubits, vec);

        // std::cout << "---------------------------------" << std::endl;
        // std::cout << "From:\n" << pair.first << std::endl;
        // std::cout << "Targets:\n" << target_qubits << std::endl;
        // std::cout << "To vec:\n" << vec << std::endl;
        // std::cout << "Operator:\n" << DenseMatrix(matrix) << std::endl;

        // Apply matrix
        if (input_state.is_density_matrix() && !only_multiply) {
            output_vec = matrix * vec * matrix.adjoint();
        } else {
            output_vec = matrix * vec;
        }
        output_vec *= pair.second;

        // Convert back to product state and add to output_state
        computational_to_elements(output_vec, target_qubits, pair.first, output_state);

        // std::cout << "To vec:\n" << output_vec << std::endl;
        // std::cout << "To state:\n" << output_state << std::endl;
        // std::cout << "---------------------------------" << std::endl;

    }
    return output_state;
}

MatrixFreeState MatrixFreeOperator::operator*(const MatrixFreeState& input_state) const {
    return apply(input_state, true);
}

MatrixFreeOperator::MatrixFreeOperator(const Gate& gate) {
    matrix = gate.get_base_matrix();
    target_qubits = {};
    for (int q : gate.get_control_qubits()) {
        target_qubits.push_back(q);
    }
    for (int q : gate.get_target_qubits()) {
        target_qubits.push_back(q);
    }
    // Expand the matrix if it's not already the right size
    int matrix_qubits = std::ceil(std::log2(matrix.rows()));
    while (matrix_qubits < int(target_qubits.size())) {
        DenseMatrix controlled = DenseMatrix::Zero(2 * matrix.rows(), 2 * matrix.cols());
        controlled.topLeftCorner(matrix.rows(), matrix.cols()) = DenseMatrix::Identity(matrix.rows(), matrix.cols());
        controlled.bottomRightCorner(matrix.rows(), matrix.cols()) = matrix;
        matrix = controlled.sparseView();
        matrix_qubits += 1;
    }
}
