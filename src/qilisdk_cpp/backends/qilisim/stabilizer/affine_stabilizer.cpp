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

std::ostream& operator<<(std::ostream& os, const AffineStabilizerState& mfs) {
    std::cout << "State with " << mfs.n_qubits() << " qubits and " << mfs.size() << " terms:" << std::endl;
    for (const auto& tuple : mfs.state) {
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

        // TODO(luke)

        // X
        if (name == "X") {
            if (target_char == '0') {
                target_char = '1';
            } else if (target_char == '1') {
                target_char = '0';
            } else if (target_char == '-') {
                coeff.first *= -1.0;
            } else if (target_char == 's') {
                target_char = 'd';
            } else if (target_char == 'd') {
                target_char = 's';
            }
        // Y
        } else if (name == "Y") {
            coeff.first *= std::complex<double>(0.0, 1.0);
            if (target_char == '0') {
                target_char = '1';
            } else if (target_char == '1') {
                target_char = '0';
            } else if (target_char == '+') {
                target_char = '-';
            } else if (target_char == '-') {
                target_char = '+';
            } else if (target_char == 's') {
                target_char = 'd';
            } else if (target_char == 'd') {
                target_char = 's';
            }
        // Z
        } else if (name == "Z") {
            if (target_char == '1') {
                coeff.first *= -1.0;
            } else if (target_char == '+') {                
                target_char = '-';
            } else if (target_char == '-') {
                target_char = '+';
            } else if (target_char == 's') {
                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', std::set<int>{target_qubit}));
            } else if (target_char == 'd') {
                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', std::set<int>{target_qubit}));
            }
        // H (this one has kickback)
        } else if (name == "H") {
            if (target_char == '0') {
                target_char = '+';
            } else if (target_char == '1') {
                target_char = '-';
            } else if (target_char == '+') {
                target_char = '0';
            } else if (target_char == '-') {
                target_char = '1';
            } else if (target_char == 's') {
                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', all_affected_qubits));
            } else if (target_char == 'd') {
                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', all_affected_qubits));
            }
        // CNOT
        } else if (name == "CNOT") {
            auto& control_char = std::get<0>(ket[control_qubit]);
            auto& control_indices = std::get<1>(ket[control_qubit]);
            if (control_char == '1') {
                if (target_char == '0') {
                    target_char = '1';
                } else if (target_char == '1') {
                    target_char = '0';
                } else if (target_char == '-') {
                    coeff.first *= -1.0;
                } else if (target_char == 's') {
                    target_char = 'd';
                } else if (target_char == 'd') {
                    target_char = 's';
                }
            } else if (control_char == '+') {
                if (target_char == '0') {
                    target_char = 's';
                    target_indices = std::set<int>{control_qubit};
                } else if (target_char == '1') {
                    target_char = 'd';
                    target_indices = std::set<int>{control_qubit};
                } else if (target_char == '-') {
                    control_char = '-';
                } else if (target_char == 's') {
                    handle_insert_basis(target_indices, control_qubit, target_char);
                } else if (target_char == 'd') {
                    handle_insert_basis(target_indices, control_qubit, target_char);
                }
            } else if (control_char == '-') {
                if (target_char == '0') {
                    target_char = 's';
                    target_indices = std::set<int>{control_qubit};
                } else if (target_char == '1') {
                    target_char = 'd';
                    target_indices = std::set<int>{control_qubit};
                } else if (target_char == '-') {
                    control_char = '+';
                } else if (target_char == 's') {
                    handle_insert_basis(target_indices, control_qubit, target_char);
                } else if (target_char == 'd') {
                    handle_insert_basis(target_indices, control_qubit, target_char);
                }
            } else if (control_char == 's' || control_char == 'd') {
                if (target_char == '0') {
                    target_char = control_char;
                    target_indices = control_indices;
                } else if (target_char == '1') {
                    target_char = (control_char == 's') ? 'd' : 's';
                    target_indices = control_indices;
                } else if (target_char == '+') {
                    for (int index : control_indices) {
                        if (index == target_qubit) {
                            target_char = (control_char == 's') ? '0' : '1';
                            target_indices.clear();
                            control_char = '+';
                            control_indices.clear();
                            break;
                        }
                    }
                } else if (target_char == '-') {
                    coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', all_affected_qubits));
                } else if (target_char == 's' || target_char == 'd') {
                    for (int index : control_indices) {
                        handle_insert_basis(target_indices, index, target_char);
                    }
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
