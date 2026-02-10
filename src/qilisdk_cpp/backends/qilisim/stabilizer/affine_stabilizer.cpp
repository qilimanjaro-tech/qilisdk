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

#include "affine_stabilizer.h"
#include <random>
#include <iostream> // TODO(luke) remove

// TODO(luke) docstrings

AffineStabilizerState::AffineStabilizerState(int n_qubits, bool density) {
    StateCoefficient one_coeff = std::make_pair(std::complex<double>(1.0, 0.0), StateCoefficient::second_type());
    StateBasis zero_basis;
    for (int i = 0; i < n_qubits; ++i) {
        zero_basis.push_back(std::make_pair('0', std::set<int>()));
    }
    if (density) {
        state.push_back(std::make_tuple(one_coeff, zero_basis, zero_basis));
    } else {
        state.push_back(std::make_tuple(one_coeff, zero_basis, StateBasis()));
    }
}

AffineStabilizerState::AffineStabilizerState(const SparseMatrix& initial_state) {
    AffineStabilizerState new_state;
    int n_qubits = std::ceil(std::log2(initial_state.rows()));
    bool is_density = initial_state.rows() == initial_state.cols();
    for (int k = 0; k < initial_state.outerSize(); ++k) {
        for (typename SparseMatrix::InnerIterator it(initial_state, k); it; ++it) {
            int row = it.row();
            int col = it.col();
            std::complex<double> value = it.value();
            StateCoefficient coeff = std::make_pair(value, StateCoefficient::second_type());
            StateBasis ket_basis;
            StateBasis bra_basis;
            for (int i = 0; i < n_qubits; ++i) {
                char ket_char = ((row >> i) & 1) ? '1' : '0';
                char bra_char = ((col >> i) & 1) ? '1' : '0';
                ket_basis.push_back(std::make_pair(ket_char, std::set<int>()));
                bra_basis.push_back(std::make_pair(bra_char, std::set<int>()));
            }
            if (is_density) {
                new_state.state.push_back(std::make_tuple(coeff, ket_basis, bra_basis));
            } else {
                new_state.state.push_back(std::make_tuple(coeff, ket_basis, StateBasis()));
            }
        }
    }
    state = new_state.state;
}

AffineStabilizerState& AffineStabilizerState::operator=(const AffineStabilizerState& other) {
    if (this != &other) {
        state = other.state;
    }
    return *this;
}

bool AffineStabilizerState::is_ket() const {
    if (state.empty()) {
        return false;
    }
    return std::get<2>(*state.begin()).empty();
}

bool AffineStabilizerState::is_bra() const {
    if (state.empty()) {
        return false;
    }
    return std::get<1>(*state.begin()).empty();
}

bool AffineStabilizerState::is_density_matrix() const {
    if (state.empty()) {
        return false;
    }
    return !std::get<1>(*state.begin()).empty() && !std::get<2>(*state.begin()).empty();
}

void AffineStabilizerState::to_density_matrix() {
    if (is_density_matrix()) {
        return;
    }
    AffineStabilizerState new_state;
    if (is_ket()) {
        for (const auto& tuple : state) {
            const auto& coeff = std::get<0>(tuple);
            const auto& ket = std::get<1>(tuple);
            StateBasis bra;
            for (const auto& [char_, indices] : ket) {
                bra.push_back(std::make_pair(char_, indices));
            }
            new_state.state.push_back(std::make_tuple(coeff, ket, bra));
        }
    } else if (is_bra()) {
        for (const auto& tuple : state) {
            const auto& coeff = std::get<0>(tuple);
            const auto& bra = std::get<2>(tuple);
            StateBasis ket;
            for (const auto& [char_, indices] : bra) {
                ket.push_back(std::make_pair(char_, indices));
            }
            new_state.state.push_back(std::make_tuple(coeff, ket, bra));
        }
    }
    state = new_state.state;
}

bool AffineStabilizerState::is_pure(double atol) const {
    if (!is_density_matrix()) {
        return true;
    }
    double trace_squared = 0.0;
    for (const auto& tuple : state) {
        const auto& coeff = std::get<0>(tuple);
        const auto& ket = std::get<1>(tuple);
        const auto& bra = std::get<2>(tuple);
        if (ket == bra) {
            trace_squared += std::norm(coeff.first);
        }
    }
    return std::abs(trace_squared - 1.0) < atol;
}

bool AffineStabilizerState::empty() const {
    return state.empty();
}

void AffineStabilizerState::normalize() {
    if (is_density_matrix()) {
        double trace = 0.0;
        for (const auto& tuple : state) {
            const auto& coeff = std::get<0>(tuple);
            const auto& ket = std::get<1>(tuple);
            const auto& bra = std::get<2>(tuple);
            if (ket == bra) {
                trace += std::norm(coeff.first);
            }
        }
        if (trace > 0.0) {
            for (auto& tuple : state) {
                std::get<0>(tuple).first /= trace;
            }
        }
    } else {
        double norm = 0.0;
        for (const auto& tuple : state) {
            const auto& coeff = std::get<0>(tuple);
            norm += std::norm(coeff.first);
        }
        norm = std::sqrt(norm);
        if (norm > 0.0) {
            for (auto& tuple : state) {
                std::get<0>(tuple).first /= norm;
            }
        }
    }
}

int AffineStabilizerState::n_qubits() const {
    if (state.empty()) {
        return 0;
    }
    return std::max(std::get<1>(*state.begin()).size(), std::get<2>(*state.begin()).size());
}

void AffineStabilizerState::prune(double atol) {
    State new_state;
    for (const auto& tuple : state) {
        const auto& coeff = std::get<0>(tuple);
        if (std::abs(coeff.first) > atol) {
            new_state.push_back(tuple);
        }
    }
    state = new_state;
}

std::map<std::string, int> AffineStabilizerState::sample(int n_samples, int seed) const {
    
    // Set up the random generator
    std::mt19937 gen(seed);

    // For each sample needed
    std::map<std::string, int> counts;
    for (int i = 0; i < n_samples; ++i) {

        // Pick an element from the state with probability proportional to the square of its coefficient
        std::discrete_distribution<> dist(state.size(), 0.0, 1.0, [&](size_t index) {
            const auto& coeff = std::get<0>(state[index]);
            return std::norm(coeff.first);
        });
        size_t index = dist(gen);
        const auto& tuple = state[index];
        auto ket = std::get<1>(tuple);
        
        // Keep evaluating the ket until all chars are 0 or 1, randomly breaking ties for +/-
        bool things_to_do = true;
        while (things_to_do) {
            things_to_do = false;
            for (auto& [char_, indices] : ket) {
                if (char_ == '+' || char_ == '-') {
                    if (std::bernoulli_distribution(0.5)(gen)) {
                        char_ = '1';
                    } else {
                        char_ = '0';
                    }
                } else if (char_ == 's' || char_ == 'd') {
                    for (int index : indices) {
                        if (index != -1) {
                            if (ket[index].first != '0' && ket[index].first != '1') {
                                things_to_do = true;
                                break;
                            }
                        }
                    }
                    if (things_to_do) {
                        continue;
                    }
                    if (char_ == 'd') {
                        char_ = '1';
                    } else {
                        char_ = '0';
                    }
                    for (int index : indices) {
                        if (index != -1) {
                            if (ket[index].first == '1') {
                                char_ = (char_ == '1') ? '0' : '1';
                            }
                        }
                    }
                }
            }
        }
        
        // Convert the final ket to a bitstring and increment the count
        std::string bitstring;
        for (const auto& [char_, indices] : ket) {
            bitstring += char_;
        }
        counts[bitstring]++;

    }

    return counts;

}

std::ostream& operator<<(std::ostream& os, const StateBasis& basis) {
    for (size_t i = 0; i < basis.size(); ++i) {
        const auto& [char_, indices] = basis[i];
        os << char_;
        int count = 0;
        for (int index : indices) {
            if (index != -1) {
                os << index;
                if (count < static_cast<int>(indices.size()) - 1) {
                    os << "&";
                }
                count++;
            }
        }
        if (i != basis.size() - 1) {
            os << ", ";
        }
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const StateCoefficient& coeff) {
    os << "(" << coeff.first.real() << " + " << coeff.first.imag() << "i)";
    for (const auto& [char_, indices] : coeff.second) {
        os << " * " << char_;
        int count = 0;
        for (int index : indices) {
            if (index != -1) {
                os << index;
                if (count < static_cast<int>(indices.size()) - 1) {
                    os << "&";
                }
                count++;
            }
        }
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

std::ostream& operator<<(std::ostream& os, const AffineStabilizerState& state) {
    std::cout << "State with " << state.n_qubits() << " qubits and " << state.size() << " terms:" << std::endl;
    for (const auto& tuple : state.state) {
        const auto& coeff = std::get<0>(tuple);
        const auto& ket = std::get<1>(tuple);
        const auto& bra = std::get<2>(tuple);
        os << coeff;
        if (!ket.empty()) {
            os << " |" << ket << ">";
        }
        if (!bra.empty()) {
            os << " <" << bra << "|";
        }
    }
    return os;
}

AffineStabilizerState AffineStabilizerState::operator+(const AffineStabilizerState& other) const {
    AffineStabilizerState result = *this;
    for (const auto& pair : other.state) {
        result.state.push_back(pair);
    }
    return result;
}
AffineStabilizerState AffineStabilizerState::operator-(const AffineStabilizerState& other) const {
    AffineStabilizerState result = *this;
    for (const auto& pair : other.state) {
        auto negated_pair = pair;
        std::get<0>(negated_pair).first *= -1.0;
        result.state.push_back(negated_pair);
    }
    return result;
}
AffineStabilizerState& AffineStabilizerState::operator+=(const AffineStabilizerState& other) {
    for (const auto& pair : other.state) {
        state.push_back(pair);
    }
    return *this;
}
AffineStabilizerState& AffineStabilizerState::operator-=(const AffineStabilizerState& other) {
    for (const auto& pair : other.state) {
        auto negated_pair = pair;
        std::get<0>(negated_pair).first *= -1.0;
        state.push_back(negated_pair);
    }
    return *this;
}

size_t AffineStabilizerState::size() const {
    return state.size();
}

 
void handle_insert_basis(std::set<int>& vec, int value, char& target_char) {
    auto it = vec.find(value);
    if (it != vec.end()) {
        vec.erase(it);
    } else {
        vec.insert(value);
    }
    if (vec.empty()) {
        if (target_char == 's') {
            target_char = '0';
        } else if (target_char == 'd') {
            target_char = '1';
        }
    }
}

std::complex<double> handle_insert_coeff(std::set<std::pair<char, std::set<int>>>& vec, const std::pair<char, std::set<int>>& value) {
    std::complex<double> result(1.0, 0.0);
    auto it = vec.find(value);
    if (it != vec.end()) {
        vec.erase(it);
        if (value.first == 'i') {
            result *= handle_insert_coeff(vec, std::make_pair('i', value.second)); // i1*i1 = z1
        }
    } else {
        vec.insert(value);
    }
    return result;
}

AffineStabilizerState AffineStabilizerOperator::apply(const AffineStabilizerState& input_state, bool only_multiply) const {
    AffineStabilizerState output_state = input_state;
    for (auto& tuple : output_state) {
        
        // Get things as reference from the state element
        auto& coeff = std::get<0>(tuple);
        auto& ket = std::get<1>(tuple);
        auto& target_char = ket[target_qubit].first;
        auto& target_indices = ket[target_qubit].second;

        // All qubits this that depends on (e.g. for kickback)
        std::set<int> all_affected_qubits = {target_qubit};
        for (const auto& [char_, indices] : ket) {
            for (int index : indices) {
                if (index != -1) {
                    all_affected_qubits.insert(index);
                }
            }
        }

        // X
        if (name == "X") {
            // X |0> = |1>
            if (target_char == '0') {
                target_char = '1';
            // X |1> = |0>
            } else if (target_char == '1') {
                target_char = '0';
            // X |si&j> = |di&j>
            } else if (target_char == 's') {
                target_char = 'd';
            // X |di&j> = |si&j>
            } else if (target_char == 'd') {
                target_char = 's';
            // X |+ s0> = |+ d0>
            } else if (target_char == '+') {
                // propogate bitflip TODO
            }
        // Y
        } else if (name == "Y") {
            coeff.first *= std::complex<double>(0.0, 1.0);
            // Y |0> = i|1>
            if (target_char == '0') {
                target_char = '1';
            // Y |1> = -i|0>
            } else if (target_char == '1') {
                coeff.first *= -1.0;
                target_char = '0';
            // Y |si&j> = z0 i|di&j>
            // Y |di&j> = z0 i|si&j>
            } else if (target_char == 's' || target_char == 'd') {
                target_char = (target_char == 's') ? 'd' : 's';
                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', std::set<int>{target_qubit}));
            // Y |+> = z0 i|+>
            } else if (target_char == '+') {
                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', std::set<int>{target_qubit}));
                // propogate bitflip TODO
            }
        // Z
        } else if (name == "Z") {
            // Z |1> = -|1>
            if (target_char == '1') {
                coeff.first *= -1.0;
            // Z |+> = z0 |+>
            // Z |si&j> = z0 |si&j>
            // Z |di&j> = z0 |di&j>
            } else if (target_char == '+' || target_char == 's' || target_char == 'd') {
                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', std::set<int>{target_qubit}));
            }
        // H
        } else if (name == "H") {
            // H |0> = |+>
            if (target_char == '0') {
                target_char = '+';
            // H |1> = z0 |+>
            } else if (target_char == '1') {
                target_char = '+';
                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', std::set<int>{target_qubit}));
            // H |+> = |0>
            // H z0 |+> = |1>
            // H z0&1 |+ +> = |s1 +>
            // H z0 |+ s0> = coeff |+ +>
            } else if (target_char == '+') {

                // Check if anything references this plus
                bool has_reference = false;
                for (size_t i = 0; i < ket.size(); ++i) {
                    auto& char_ = std::get<0>(ket[i]);
                    auto& indices = std::get<1>(ket[i]);
                    if (indices.find(target_qubit) != indices.end()) {
                        char_ = '+';
                        indices.clear();
                        has_reference = true;
                        break;
                    }
                }

                // If it's not referenced anywhere, destroy it
                if (!has_reference) {
                    target_char = '0';
                    target_indices.clear();
                }

            } else if (target_char == 's' || target_char == 'd') {
                for (int index : target_indices) {
                    coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', std::set<int>{index}));
                }
                target_char = '+';
                target_indices.clear();
            }
        // CNOT
        } else if (name == "CNOT") {
            auto& control_char = std::get<0>(ket[control_qubit]);
            auto& control_indices = std::get<1>(ket[control_qubit]);
            if (control_char == '1') {
                // CNOT |1 0> = |1 1>
                if (target_char == '0') {
                    target_char = '1';
                // CNOT |1 1> = |1 0>
                } else if (target_char == '1') {
                    target_char = '0';
                // CNOT |1 si&j> = |1 di&j>
                } else if (target_char == 's') {
                    target_char = 'd';
                // CNOT |1 di&j> = |1 si&j>
                } else if (target_char == 'd') {
                    target_char = 's';
                }
            } else if (control_char == '+') {
                // CNOT |+ 0> = |+ s0>
                if (target_char == '0') {
                    target_char = 's';
                    target_indices = std::set<int>{control_qubit};
                // CNOT |+ 1> = |+ d0>
                } else if (target_char == '1') {
                    target_char = 'd';
                    target_indices = std::set<int>{control_qubit};
                // CNOT |+ s2 +> = |+ s0&2 +>
                } else if (target_char == 's' || target_char == 'd') {
                    handle_insert_basis(target_indices, control_qubit, target_char);
                }
            } else if (control_char == 's' || control_char == 'd') {
                // CNOT |s2 0 +> = |s2 s0 +>
                if (target_char == '0') {
                    target_char = 's';
                    target_indices = control_indices;
                // CNOT |s2 1 +> = |s2 d0 +>
                } else if (target_char == '1') {
                    target_char = 'd';
                    target_indices = control_indices;
                // CNOT |s1 +> = |+ 0>
                // CNOT |s1&2 + +> = |+ s2 +>
                } else if (target_char == '+') {
                    for (int index : control_indices) {
                        if (index == target_qubit) {
                            
                            // This is getting destroyed
                            std::set<int> new_indices = control_indices;
                            new_indices.erase(index);
                            if (new_indices.empty()) {
                                target_char = (control_char == 's') ? '0' : '1';
                                target_indices.clear();
                            } else {
                                target_char = 's';
                                target_indices = new_indices;
                            }

                            // Which means the control is now a plus
                            control_char = '+';
                            control_indices.clear();

                            // Anything that referenced the target should now reference the control
                            for (auto& [char_, indices] : ket) {
                                if (indices.find(target_qubit) != indices.end()) {
                                    indices.insert(control_qubit);
                                }
                            }

                            break;
                            
                        }
                    }
                // CNOT |s2 s3 + +> = |s2 s0&3 + +>
                } else if (target_char == 's' || target_char == 'd') {
                    handle_insert_basis(target_indices, control_qubit, target_char);
                }
            }
        // S
        } else if (name == "S") {
            if (target_char == '1') {
                coeff.first *= std::complex<double>(0.0, 1.0);
            } else if (target_char == '+' || target_char == '-' || target_char == 's' || target_char == 'd') {
                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('i', std::set<int>{target_qubit}));
            }
        // T TODO
        } else if (name == "T") {
        }

        // Check to make sure we're still valid
        for (size_t i = 0; i < ket.size(); ++i) {
            auto& char_ = std::get<0>(ket[i]);
            auto& indices = std::get<1>(ket[i]);
            
            // Ensure all s and d chars have at least one index, otherwise convert to 0 or 1
            if ((char_ == 's' || char_ == 'd') && indices.empty()) {
                char_ = (char_ == 's') ? '0' : '1';
            }

            // Ensure all s and d have indices pointing to chars that are not 0 or 1
            if (char_ == 's' || char_ == 'd') {
                char new_char = char_;
                std::set<int> new_indices;
                for (int index : indices) {
                    if (index != -1) {
                        auto& dependent_char = std::get<0>(ket[index]);
                        if (dependent_char == '1') {
                            new_char = (new_char == 's') ? 'd' : 's';
                        } else {
                            new_indices.insert(index);
                        }
                    }
                }
                char_ = new_char;
                indices = new_indices;
            }

        }

    }
    return output_state;
}

AffineStabilizerState AffineStabilizerOperator::operator*(const AffineStabilizerState& input_state) const {
    return apply(input_state, true);
}

AffineStabilizerOperator::AffineStabilizerOperator(const Gate& gate) {
    if (gate.get_control_qubits().size() > 1) {
        throw std::invalid_argument("AffineStabilizerOperator only supports gates with 1 or fewer total control qubits.");
    }
    if (gate.get_target_qubits().size() != 1) {
        throw std::invalid_argument("AffineStabilizerOperator requires a gate with exactly 1 target qubit.");
    }
    target_qubit = gate.get_target_qubits()[0];
    control_qubit = gate.get_control_qubits().empty() ? -1 : gate.get_control_qubits()[0];
    name = gate.get_name();
}
