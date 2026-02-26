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

#ifdef VERBOSE
#include <iostream>
#endif

const StateCoefficient one_coeff = std::make_pair(std::complex<double>(1.0, 0.0), StateCoefficient::second_type());

std::string basis_to_bitstring(const StateBasis& basis) {
    /*
    Convert a computational basis StateBasis to a bitstring.

    Args:
        basis (const StateBasis&): The StateBasis to convert.

    Returns:
        std::string: The resulting bitstring.

    Raises:
        std::runtime_error: If the StateBasis is not a computational basis state.
    */
    std::string bitstring;
    for (const auto& [char_, indices] : basis) {
        bitstring += char_;
    }
    return bitstring;
}

AffineStabilizerState::AffineStabilizerState(int n_qubits, bool density) {
    /* 
    Create a new AffineStabilizerState in the |0...0> state.

    Args:
        n_qubits (int): The number of qubits in the state.
        density (bool): Whether to create the state as a density matrix (default: false).

    Returns:
        AffineStabilizerState: The initialized state.
    */
    StateBasis zero_basis;
    for (int i = 0; i < n_qubits; ++i) {
        zero_basis.push_back(std::make_pair('0', IndexSet()));
    }
    if (density) {
        add_element(one_coeff, zero_basis, one_coeff, zero_basis);
    } else {
        add_ket(one_coeff, zero_basis);
    }
    #ifdef VERBOSE
    std::cout << "Initialized state:" << std::endl;
    std::cout << *this << std::endl;
    std::cout << "Current basis indices:" << std::endl;
    for (const auto& [bitstring, index] : basis_indices) {
        std::cout << bitstring << ": " << index << std::endl;
    }
    #endif
}

AffineStabilizerState::AffineStabilizerState(const SparseMatrix& initial_state) {
    /*
    Create a new AffineStabilizerState from a sparse matrix (statevector or density matrix).

    Args:
        initial_state (SparseMatrix): The initial state as a sparse matrix.

    Returns:
        AffineStabilizerState: The initialized state.
    */
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
                ket_basis.push_back(std::make_pair(ket_char, IndexSet()));
                bra_basis.push_back(std::make_pair(bra_char, IndexSet()));
            }
            if (is_density) {
                add_element(coeff, ket_basis, one_coeff, bra_basis);
            } else {
                add_ket(coeff, ket_basis);
            }
        }
    }
    state = new_state.state;
}

AffineStabilizerState& AffineStabilizerState::operator=(const AffineStabilizerState& other) {
    /*
    Assignment operator for AffineStabilizerState.

    Args:
        other (AffineStabilizerState): The state to copy from.

    Returns:        
        AffineStabilizerState&: A reference to this state after assignment.
    */
    if (this != &other) {
        state = other.state;
        basis_indices = other.basis_indices;
    }
    return *this;
}

bool AffineStabilizerState::is_ket() const {
    /*
    Check if the state is a ket.

    Returns:
        bool: True if the state is a ket, False otherwise.
    */
    if (state.empty()) {
        return false;
    }
    return std::get<3>(*state.begin()).empty();
}

bool AffineStabilizerState::is_bra() const {
    /*
    Check if the state is a bra.

    Returns:
        bool: True if the state is a bra, False otherwise.
    */
    if (state.empty()) {
        return false;
    }
    return std::get<1>(*state.begin()).empty();
}

bool AffineStabilizerState::is_density_matrix() const {
    /*
    Check if the state is a density matrix.

    Returns:
        bool: True if the state is a density matrix, False otherwise.
    */
    if (state.empty()) {
        return false;
    }
    return !std::get<1>(*state.begin()).empty() && !std::get<3>(*state.begin()).empty();
}

AffineStabilizerState AffineStabilizerState::as_expanded() const {
    /*
    Expand each term in the stabilizer state, turning + into 0 and 1 etc.
    */
    AffineStabilizerState expanded_state = *this;
    AffineStabilizerState new_state;
    
    // While we still have plusses, keep expanding
    bool has_plus = true;
    while (has_plus) {
        has_plus = false;
        for (const auto& tuple : expanded_state.state) {
            const auto& ket_coeff = std::get<0>(tuple);
            const auto& ket = std::get<1>(tuple);
            const auto& bra_coeff = std::get<2>(tuple);
            const auto& bra = std::get<3>(tuple);
            for (size_t i = 0; i < ket.size(); ++i) {
                if (ket[i].first == '+') {
                    has_plus = true;
                    StateBasis ket_zero = ket;
                    StateBasis ket_one = ket;
                    ket_zero[i].first = '0';
                    ket_one[i].first = '1';
                    StateCoefficient new_coeff = std::make_pair(ket_coeff.first / std::sqrt(2.0), ket_coeff.second);
                    new_state.add_element(new_coeff, ket_zero, bra_coeff, bra);
                    new_state.add_element(new_coeff, ket_one, bra_coeff, bra);
                    break;
                }
            }
            if (!has_plus) {
                for (size_t i = 0; i < bra.size(); ++i) {
                    if (bra[i].first == '+') {
                        has_plus = true;
                        StateBasis bra_zero = bra;
                        StateBasis bra_one = bra;
                        bra_zero[i].first = '0';
                        bra_one[i].first = '1';
                        StateCoefficient new_coeff = std::make_pair(bra_coeff.first / std::sqrt(2.0), bra_coeff.second);
                        new_state.add_element(new_coeff, ket, new_coeff, bra_zero);
                        new_state.add_element(new_coeff, ket, new_coeff, bra_one);
                        break;
                    }
                }
            }
        }
        if (has_plus) {
            expanded_state.state = new_state.state;
            new_state.state.clear();
        }
    }

    // Go through and evaluate all of the s and d's
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& tuple : expanded_state.state) {
            auto& ket = std::get<1>(tuple);
            auto& bra = std::get<3>(tuple);
            for (size_t i = 0; i < ket.size(); ++i) {
                if (ket[i].first == 's' || ket[i].first == 'd') {
                    changed = true;
                    char new_char = (ket[i].first == 's') ? '0' : '1';
                    for (int index : ket[i].second) {
                        if (ket[index].first == '1') {
                            new_char = (new_char == '0') ? '1' : '0';
                        } else if (ket[index].first != '0') {
                            new_char = ket[i].first;
                            break;
                        }
                    }
                    ket[i].first = new_char;
                    if (new_char != 's' && new_char != 'd') {
                        ket[i].second.clear();
                    }
                }
            }
            for (size_t i = 0; i < bra.size(); ++i) {
                if (bra[i].first == 's' || bra[i].first == 'd') {
                    changed = true;
                    char new_char = (bra[i].first == 's') ? '0' : '1';
                    for (int index : bra[i].second) {
                        if (bra[index].first == '1') {
                            new_char = (new_char == '0') ? '1' : '0';
                        } else if (bra[index].first != '0') {
                            new_char = bra[i].first;
                            break;
                        }
                    }
                    bra[i].first = new_char;
                    if (new_char != 's' && new_char != 'd') {
                        bra[i].second.clear();
                    }
                }
            }
        }
    }
    
    // Evaluate all of the phases in the coeff
    for (auto& tuple : expanded_state.state) {
        auto& ket_coeff = std::get<0>(tuple);
        auto& ket = std::get<1>(tuple);
        auto& bra_coeff = std::get<2>(tuple);
        auto& bra = std::get<3>(tuple);
        for (const auto& [char_, indices] : ket_coeff.second) {
            if (char_ == 'z') {
                bool all_one = true;
                for (int index : indices) {
                    if (ket[index].first != '1') {
                        all_one = false;
                        break;
                    }
                }
                if (all_one) {
                    ket_coeff.first *= -1.0;
                }
            } else if (char_ == 'i') {
                bool all_one = true;
                for (int index : indices) {
                    if (ket[index].first != '1') {
                        all_one = false;
                        break;
                    }
                }
                if (all_one) {
                    ket_coeff.first *= std::complex<double>(0.0, 1.0);
                }
            }
        }
        for (const auto& [char_, indices] : bra_coeff.second) {
            if (char_ == 'z') {
                bool all_one = true;
                for (int index : indices) {
                    if (bra[index].first != '1') {
                        all_one = false;
                        break;
                    }
                }
                if (all_one) {
                    bra_coeff.first *= -1.0;
                }
            } else if (char_ == 'i') {
                bool all_one = true;
                for (int index : indices) {
                    if (bra[index].first != '1') {
                        all_one = false;
                        break;
                    }
                }
                if (all_one) {
                    bra_coeff.first *= std::complex<double>(0.0, 1.0);
                }
            }
        }
        ket_coeff.second.clear();
        bra_coeff.second.clear();
    }
    
    return expanded_state;

}

DenseMatrix AffineStabilizerState::as_dense() const {
    /*
    Convert the state to a dense matrix representation.
    
    Returns:
        DenseMatrix: The dense matrix representation of the state.

    Raises:
        std::runtime_error: If the state is neither a ket nor a density matrix.
    */
    int n_qubits = this->n_qubits();
    int dim = 1 << n_qubits;
    AffineStabilizerState expanded_state = as_expanded();
    if (expanded_state.is_ket()) {
        DenseMatrix result = DenseMatrix::Zero(dim, 1);
        for (const auto& tuple : expanded_state.state) {
            const auto& ket_coeff = std::get<0>(tuple);
            const auto& ket = std::get<1>(tuple);
            int index = 0;
            for (size_t i = 0; i < ket.size(); ++i) {
                if (ket[i].first == '1') {
                    index |= (1 << (n_qubits - 1 - i));
                }
            }
            result(index, 0) += ket_coeff.first;
        }
        return result;
    } else if (expanded_state.is_bra()) {
        DenseMatrix result = DenseMatrix::Zero(1, dim);
        for (const auto& tuple : expanded_state.state) {
            const auto& bra_coeff = std::get<2>(tuple);
            const auto& bra = std::get<3>(tuple);
            int index = 0;
            for (size_t i = 0; i < bra.size(); ++i) {
                if (bra[i].first == '1') {
                    index |= (1 << (n_qubits - 1 - i));
                }
            }
            result(0, index) += bra_coeff.first;
        }
        return result;
    } else {
        DenseMatrix result = DenseMatrix::Zero(dim, dim);
        for (const auto& tuple : expanded_state.state) {
            const auto& ket_coeff = std::get<0>(tuple);
            const auto& ket = std::get<1>(tuple);
            const auto& bra_coeff = std::get<2>(tuple);
            const auto& bra = std::get<3>(tuple);
            int ket_index = 0;
            int bra_index = 0;
            for (size_t i = 0; i < ket.size(); ++i) {
                if (ket[i].first == '1') {
                    ket_index |= (1 << (n_qubits - 1 - i));
                }
                if (bra[i].first == '1') {
                    bra_index |= (1 << (n_qubits - 1 - i));
                }
            }
            result(ket_index, bra_index) += ket_coeff.first * bra_coeff.first;
        }
        return result;
    }

}

void AffineStabilizerState::to_density_matrix() {
    /*
    Convert the state to a density matrix if it is currently a ket or bra.
    */
    if (is_density_matrix()) {
        return;
    }
    AffineStabilizerState new_state;
    if (is_ket()) {
        for (const auto& tuple : state) {
            const auto& ket_coeff = std::get<0>(tuple);
            const auto& ket = std::get<1>(tuple);
            StateBasis bra;
            for (const auto& [char_, indices] : ket) {
                bra.push_back(std::make_pair(char_, indices));
            }
            new_state.add_element(ket_coeff, ket, one_coeff, bra);
        }
    } else if (is_bra()) {
        for (const auto& tuple : state) {
            const auto& bra_coeff = std::get<2>(tuple);
            const auto& bra = std::get<3>(tuple);
            StateBasis ket;
            for (const auto& [char_, indices] : bra) {
                ket.push_back(std::make_pair(char_, indices));
            }
            new_state.add_element(one_coeff, ket, bra_coeff, bra);
        }
    }
    state = new_state.state;
}

bool AffineStabilizerState::is_pure(double atol) const {
    /*
    Check if the state is pure (only applies to density matrices).

    Args:
        atol (double): The absolute tolerance for checking purity (default: 1e-12).
    
    Returns:
        bool: True if the state is pure, False otherwise.
    */
    if (!is_density_matrix()) {
        return true;
    }
    double trace_squared = 0.0;
    for (const auto& tuple : state) {
        const auto& ket_coeff = std::get<0>(tuple);
        const auto& ket = std::get<1>(tuple);
        const auto& bra_coeff = std::get<2>(tuple);
        const auto& bra = std::get<3>(tuple);
        if (ket == bra) {
            trace_squared += std::norm(ket_coeff.first * bra_coeff.first);
        }
    }
    return std::abs(trace_squared - 1.0) < atol;
}

bool AffineStabilizerState::empty() const {
    /*
    Check if the state is empty (has no components).

    Returns:
        bool: True if the state is empty, False otherwise.
    */
    return state.empty();
}

void AffineStabilizerState::normalize() {
    /*
    Normalize the state so that it has trace 1 if it's a density matrix, or norm 1 if it's a ket.
    */
    if (is_density_matrix()) {
        double trace = 0.0;
        for (const auto& tuple : state) {
            const auto& ket_coeff = std::get<0>(tuple);
            const auto& ket = std::get<1>(tuple);
            const auto& bra_coeff = std::get<2>(tuple);
            const auto& bra = std::get<3>(tuple);
            if (ket == bra) {
                trace += std::norm(ket_coeff.first * bra_coeff.first);
            }
        }
        if (trace > 0.0) {
            for (auto& tuple : state) {
                std::get<0>(tuple).first /= trace;
            }
        }
    } else if (is_ket()) {
        double norm = 0.0;
        for (const auto& tuple : state) {
            const auto& ket_coeff = std::get<0>(tuple);
            norm += std::norm(ket_coeff.first);
        }
        norm = std::sqrt(norm);
        if (norm > 0.0) {
            for (auto& tuple : state) {
                auto& ket_coeff = std::get<0>(tuple);
                ket_coeff.first /= norm;
            }
        }
    } else if (is_bra()) {
        double norm = 0.0;
        for (const auto& tuple : state) {
            const auto& bra_coeff = std::get<2>(tuple);
            norm += std::norm(bra_coeff.first);
        }
        norm = std::sqrt(norm);
        if (norm > 0.0) {
            for (auto& tuple : state) {
                auto& bra_coeff = std::get<2>(tuple);
                bra_coeff.first /= norm;
            }
        }
    }
}

int AffineStabilizerState::n_qubits() const {
    /*
    Get the number of qubits in the state.

    Returns:
        int: The number of qubits in the state.
    */
    if (state.empty()) {
        return 0;
    }
    return std::max(std::get<1>(*state.begin()).size(), std::get<3>(*state.begin()).size());
}

void AffineStabilizerState::prune(double atol) {
    /*
    Prune components of the state that have coefficients with magnitude below the given absolute tolerance.

    Args:
        atol (double): The absolute tolerance for pruning components (default: 1e-12).
    */
    State new_state;
    std::unordered_map<std::string, int> new_basis_indices;
    for (const auto& tuple : state) {
        const auto& ket_coeff = std::get<0>(tuple);
        const auto& bra_coeff = std::get<2>(tuple);
        #ifdef VERBOSE
        std::cout << std::abs(ket_coeff.first) << " " << std::abs(bra_coeff.first) << std::endl;
        #endif
        if (std::abs(ket_coeff.first * bra_coeff.first) > atol) {
            new_state.push_back(tuple);
            std::string bitstring = basis_to_bitstring(std::get<1>(tuple)) + "|" + basis_to_bitstring(std::get<3>(tuple));
            new_basis_indices[bitstring] = new_state.size() - 1;
        }
    }
    #ifdef VERBOSE
    std::cout << "Pruned state from:" << std::endl << *this << std::endl;
    #endif  
    state = new_state;
    basis_indices = new_basis_indices;
    #ifdef VERBOSE
    std::cout << "To:" << std::endl << *this << std::endl;
    #endif  
}

std::map<std::string, int> AffineStabilizerState::sample(int n_samples, int seed) const {
    /*
    Sample from the state by randomly picking components with probability proportional to the 
    square of their coefficients, and then randomly resolving any superposition characters in the resulting ket.

    In the case of a sum of stabilizer states, we need to do two rounds of sampling, first to
    get some evaulated states and their probabilities (which might be zero), and then to sample 
    from those probabilities to get the final counts.

    Args:
        n_samples (int): The number of samples to draw from the state.
        seed (int): The seed for the random number generator.

    Returns:
        std::map<std::string, int>: A map from bitstrings to their corresponding counts in the samples.
    */
    
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
                        if (ket[index].first != '0' && ket[index].first != '1') {
                            things_to_do = true;
                            break;
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
                        if (ket[index].first == '1') {
                            char_ = (char_ == '1') ? '0' : '1';
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
    /*
    Print a StateBasis in a human-readable format, showing the characters and their corresponding indices.
    e.g. std::cout << basis << std::endl;

    Args:
        os (std::ostream&): The output stream to write to.
        basis (const StateBasis&): The StateBasis to print.

    Returns:
        std::ostream&: The output stream after writing the StateBasis.
    */
    for (size_t i = 0; i < basis.size(); ++i) {
        const auto& [char_, indices] = basis[i];
        os << char_;
        int count = 0;
        for (int index : indices) {
            os << index;
            if (count < static_cast<int>(indices.size()) - 1) {
                os << "x";
            }
            count++;
        }
        if (i != basis.size() - 1) {
            os << " ";
        }
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const StateCoefficient& coeff) {
    /*
    Print a StateCoefficient in a human-readable format, showing the complex coefficient and its corresponding characters and indices.
    e.g. std::cout << coeff << std::endl;

    Args:
        os (std::ostream&): The output stream to write to.
        coeff (const StateCoefficient&): The StateCoefficient to print.

    Returns:
        std::ostream&: The output stream after writing the StateCoefficient.
    */
    os << "(" << coeff.first.real() << " + " << coeff.first.imag() << "i)";
    for (const auto& [char_, indices] : coeff.second) {
        os << " " << char_;
        int count = 0;
        for (int index : indices) {
            os << index;
            if (count < static_cast<int>(indices.size()) - 1) {
                os << "&";
            }
            count++;
        }
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<int>& vec) {
    /*
    Print a vector of integers in a human-readable format, showing the elements in square brackets.
    e.g. std::cout << vec << std::endl;

    Args:
        os (std::ostream&): The output stream to write to.
        vec (const std::vector<int>&): The vector of integers to print.

    Returns:
        std::ostream&: The output stream after writing the vector.
    */
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

std::ostream& operator<<(std::ostream& os, const IndexSet& vec) {
    /*
    Print an IndexSet in a human-readable format, showing the elements in curly braces.
    e.g. std::cout << index_set << std::endl;

    Args:
        os (std::ostream&): The output stream to write to.
        vec (const IndexSet&): The IndexSet to print.

    Returns:
        std::ostream&: The output stream after writing the IndexSet.
    */
    os << "{";
    size_t count = 0;
    for (int value : vec) {
        os << value;
        if (count < vec.size() - 1) {
            os << ", ";
        }
        count++;
    }
    os << "}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const AffineStabilizerState& state) {
    /*
    Print an AffineStabilizerState in a human-readable format, showing the coefficients and their corresponding kets and bras.
    e.g. std::cout << state << std::endl;

    Args:
        os (std::ostream&): The output stream to write to.
        state (const AffineStabilizerState&): The AffineStabilizerState to print.

    Returns:
        std::ostream&: The output stream after writing the AffineStabilizerState.
    */
    for (const auto& tuple : state.state) {
        const auto& ket_coeff = std::get<0>(tuple);
        const auto& ket = std::get<1>(tuple);
        const auto& bra_coeff = std::get<2>(tuple);
        const auto& bra = std::get<3>(tuple);
        if (!ket.empty()) {
            os << ket_coeff;
            os << " |" << ket << ">";
        }
        if (!bra.empty()) {
            os << bra_coeff;
            os << " <" << bra << "|";
        }
        if (&tuple != &state.state.back()) {
            os << std::endl;
        }
    }
    return os;
}

AffineStabilizerState AffineStabilizerState::operator+(const AffineStabilizerState& other) const {
    /*
    Add two AffineStabilizerStates together by concatenating their state vectors.

    Args:
        other (const AffineStabilizerState&): The state to add to this state.

    Returns:
        AffineStabilizerState: The resulting state after addition.
    */
    AffineStabilizerState result = *this;
    result += other;
    return result;
}

AffineStabilizerState AffineStabilizerState::operator-(const AffineStabilizerState& other) const {
    /*
    Subtract one AffineStabilizerState from another by negating the coefficients 
    of the second state and concatenating their state vectors.

    Args:
        other (const AffineStabilizerState&): The state to subtract from this state.

    Returns:
        AffineStabilizerState: The resulting state after subtraction.
    */
    AffineStabilizerState result = *this;
    result -= other;
    return result;
}

bool is_computational_basis_state(const StateBasis& basis) {
    /*
    Check if a StateBasis is a computational basis state (i.e. all characters are '0' or '1').

    Args:
        basis (const StateBasis&): The StateBasis to check.

    Returns:
        bool: True if the StateBasis is a computational basis state, False otherwise.
    */
    for (const auto& [char_, indices] : basis) {
        if (char_ != '0' && char_ != '1') {
            return false;
        }
    }
    return true;
}

AffineStabilizerState& AffineStabilizerState::operator+=(const AffineStabilizerState& other) {
    /*
    Add another AffineStabilizerState to this state in-place by concatenating their state vectors.

    Args:
        other (const AffineStabilizerState&): The state to add to this state.

    Returns:
        AffineStabilizerState&: A reference to this state after addition.
    */
    for (const auto& pair : other.state) {
        if (is_computational_basis_state(std::get<1>(pair)) && is_computational_basis_state(std::get<3>(pair))) {
            std::string bitstring = basis_to_bitstring(std::get<1>(pair)) + "|" + basis_to_bitstring(std::get<3>(pair));
            #ifdef VERBOSE
            std::cout << "Adding pair with bitstring " << bitstring << " and coeff " << std::get<0>(pair) << " to state." << std::endl;
            std::cout << "Current state:" << std::endl << *this << std::endl;
            std::cout << "Current basis_indices:" << std::endl;
            for (const auto& [key, value] : basis_indices) {
                std::cout << key << ": " << value << std::endl;
            }
            #endif
            if (basis_indices.find(bitstring) != basis_indices.end()) {
                #ifdef VERBOSE
                std::cout << "Found existing bitstring " << bitstring << " in state, updating coefficients." << std::endl;
                #endif
                size_t index = basis_indices[bitstring];
                std::get<0>(state[index]).first += std::get<0>(pair).first;
                std::get<2>(state[index]).first += std::get<2>(pair).first;
            } else {
                #ifdef VERBOSE
                std::cout << "Bitstring " << bitstring << " not found in state, adding new pair." << std::endl;
                #endif  
                basis_indices[bitstring] = state.size();
                state.push_back(pair);
            }
        } else {
            state.push_back(pair);
        }
    }
    return *this;
}

AffineStabilizerState& AffineStabilizerState::operator-=(const AffineStabilizerState& other) {
    /*
    Subtract another AffineStabilizerState from this state in-place by negating the 
    coefficients of the second state and concatenating their state vectors.

    Args:
        other (const AffineStabilizerState&): The state to subtract from this state.

    Returns:
        AffineStabilizerState&: A reference to this state after subtraction.
    */
    for (const auto& pair : other.state) {
        if (is_computational_basis_state(std::get<1>(pair)) && is_computational_basis_state(std::get<3>(pair))) {
            std::string bitstring = basis_to_bitstring(std::get<1>(pair)) + "|" + basis_to_bitstring(std::get<3>(pair));
            if (basis_indices.find(bitstring) != basis_indices.end()) {
                size_t index = basis_indices[bitstring];
                std::get<0>(state[index]).first -= std::get<0>(pair).first;
                std::get<2>(state[index]).first -= std::get<2>(pair).first;
            } else {
                basis_indices[bitstring] = state.size();
                auto negated_pair = pair;
                std::get<0>(negated_pair).first *= -1.0;
                std::get<2>(negated_pair).first *= -1.0;
                state.push_back(negated_pair);
            }
        } else {
            auto negated_pair = pair;
            std::get<0>(negated_pair).first *= -1.0;
            state.push_back(negated_pair);
        }
    }
    return *this;
}

size_t AffineStabilizerState::size() const {
    /*
    Get the number of components in the state.

    Returns:
        size_t: The number of components in the state.
    */
    return state.size();
}

void handle_insert_basis(IndexSet& vec, int value, char& target_char) {
    /*
    Handle inserting or removing an index from an IndexSet, and update the target character accordingly.
    Rules:
     - If the index to insert is not unique, remove both (i.e. inserting 0 to s0&1 -> s1).
     - If we no longer have any indices, convert s->0 and d->1, and if we gain any indices, convert 0->s and 1->d.

    Args:
        vec (IndexSet&): The IndexSet to modify.
        value (int): The index to insert or remove from the set.
        target_char (char&): The character to update based on the presence of indices in the set.
    */
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
    } else {
        if (target_char == '0') {
            target_char = 's';
        } else if (target_char == '1') {
            target_char = 'd';
        }
    }
}

std::complex<double> handle_insert_coeff(std::set<std::pair<char, IndexSet>>& vec, const std::pair<char, IndexSet>& value) {
    /*
    Handle inserting a coefficient to the coefficient set and return any 
    global coefficient that should be applied based on the rules.
    Rules:
     - If the coeff to insert is not unique, remove both 
       (e.g. inserting z0&1 and we already have z0&1, removed both, since -1*-1 = 1).
     - If we have a collision with an 'i', remove both and replace with a 'z' with the same indices
       (e.g. inserting i1 to i1 -> remove both and add z1, since i1*i1 = z1).

    Args:
        vec (std::set<std::pair<char, IndexSet>>&): The set of characters and index sets to modify.
        value (const std::pair<char, IndexSet>&): The character and index set to insert or remove from the set.

    Returns:
        std::complex<double>: The coefficient to apply based on the insertion of the coefficient.
    */
    std::complex<double> result(1.0, 0.0);
    auto it = vec.find(value);
    if (it != vec.end()) {
        vec.erase(it);
        if (value.first == 'i') {
            result *= handle_insert_coeff(vec, std::make_pair('z', value.second)); // i1*i1 = z1
        }
    } else {
        vec.insert(value);
    }
    return result;
}

void AffineStabilizerState::add_ket(const StateCoefficient& ket_coeff, const StateBasis& ket) {
    /*
    Add an element to the state with the given coefficient and ket, and an empty bra.

    Args:
        ket_coeff (const StateCoefficient&): The coefficient for the ket.
        ket (const StateBasis&): The basis for the ket.
    */
    add_element(ket_coeff, ket, one_coeff, StateBasis());
}

void AffineStabilizerState::add_bra(const StateCoefficient& bra_coeff, const StateBasis& bra) {
    /*
    Add an element to the state with the given coefficient and bra, and an empty ket.

    Args:
        bra_coeff (const StateCoefficient&): The coefficient for the bra.
        bra (const StateBasis&): The basis for the bra.
    */
    add_element(one_coeff, StateBasis(), bra_coeff, bra);
}

void AffineStabilizerState::add_element(const StateCoefficient& ket_coeff, const StateBasis& ket, const StateCoefficient& bra_coeff, const StateBasis& bra) {
    /*
    Add an element to the state with the given coefficient, ket, and bra.

    Args:
        ket_coeff (const StateCoefficient&): The coefficient for the ket.
        ket (const StateBasis&): The basis for the ket.
        bra_coeff (const StateCoefficient&): The coefficient for the bra.
        bra (const StateBasis&): The basis for the bra.
    */
    if (is_computational_basis_state(ket) && is_computational_basis_state(bra)) {
        std::string bitstring = basis_to_bitstring(ket) + "|" + basis_to_bitstring(bra);
        basis_indices[bitstring] = state.size();
    }
    state.push_back(std::make_tuple(ket_coeff, ket, bra_coeff, bra));
}

void AffineStabilizerOperator::apply(AffineStabilizerState& output_state) const {
    /*
    Apply the operator to an AffineStabilizerState by iterating through each component 
    of the state and applying the transformation rules for the given operator name.
    The specifics of this are quite complex, have a look at the list of rules in the implementation for details.

    Args:
        output_state (AffineStabilizerState&): The state to apply the operator to.

    Raises:
        std::runtime_error: If the operator name is unknown.
    */
    // keep track of new things to add
    AffineStabilizerState to_add;
    for (auto& tuple : output_state) {
        
        // Get things as reference from the state element
        auto& coeff = std::get<0>(tuple);
        auto& ket = std::get<1>(tuple);
        auto& target_char = ket[target_qubit].first;
        auto& target_indices = ket[target_qubit].second;

        // X
        if (name == "X") {
            // X |0> = |1>
            // X |1> = |0>
            if (target_char == '0' || target_char == '1') {
                target_char = (target_char == '0') ? '1' : '0';
            // X |sixj> = |dixj>
            // X |dixj> = |sixj>
            } else if (target_char == 's' || target_char == 'd') {
                target_char = (target_char == 's') ? 'd' : 's';
                // add a global -1 for any linear phase that references this
                for (const auto& [char_, indices] : coeff.second) {
                    if (indices.size() == 1 && indices.find(target_qubit) != indices.end() && char_ == 'z') {
                        coeff.first *= -1.0;
                    }
                }
                // add a z(other) for any quadratic phase that references this
                for (const auto& [char_, indices] : coeff.second) {
                    if (indices.size() == 2 && indices.find(target_qubit) != indices.end() && char_ == 'z') {
                        IndexSet other_indices = indices;
                        other_indices.erase(target_qubit);
                        coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', other_indices));
                    }
                }
            // X |+ s0> = |+ d0>
            // X z0 |+> = - z0 |+>
            // X z0&1 |+ s0> = z0&z1 z1 |+ s0>
            } else if (target_char == '+') {
                // find anything in the basis that references this and swap it
                for (size_t i = 0; i < ket.size(); ++i) {
                    auto& char_ = std::get<0>(ket[i]);
                    auto& indices = std::get<1>(ket[i]);
                    if (indices.find(target_qubit) != indices.end()) {
                        char_ = (char_ == 's') ? 'd' : 's';
                    }
                }
                // add a global -1 for any linear phase that references this
                for (const auto& [char_, indices] : coeff.second) {
                    if (indices.size() == 1 && indices.find(target_qubit) != indices.end() && char_ == 'z') {
                        coeff.first *= -1.0;
                    }
                }
                // add a z(other) for any quadratic phase that references this
                for (const auto& [char_, indices] : coeff.second) {
                    if (indices.size() == 2 && indices.find(target_qubit) != indices.end() && char_ == 'z') {
                        IndexSet other_indices = indices;
                        other_indices.erase(target_qubit);
                        coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', other_indices));
                    }
                }
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
            // Y |sixj> = z0 i|dixj>
            // Y |dixj> = z0 i|sixj>
            } else if (target_char == 's' || target_char == 'd') {
                target_char = (target_char == 's') ? 'd' : 's';
                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', IndexSet{target_qubit}));
            // Y |+> = z0 i|+>
            // Y z0 |+> = i|+>
            // Y z0 |+ s0> = i|+ d0>
            // Y z0&1 |+ s0> = i|+ d0>
            } else if (target_char == '+') {
                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', IndexSet{target_qubit}));
                // find anything in the basis that references this and swap it
                for (size_t i = 0; i < ket.size(); ++i) {
                    auto& char_ = std::get<0>(ket[i]);
                    auto& indices = std::get<1>(ket[i]);
                    if (indices.find(target_qubit) != indices.end()) {
                        char_ = (char_ == 's') ? 'd' : 's';
                    }
                }
                // add a z(other) for any quadratic phase that references this
                for (const auto& [char_, indices] : coeff.second) {
                    if (indices.size() == 2 && indices.find(target_qubit) != indices.end() && char_ == 'z') {
                        IndexSet other_indices = indices;
                        other_indices.erase(target_qubit);
                        coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', other_indices));
                    }
                }
            }
        // Z
        } else if (name == "Z") {
            // Z |1> = -|1>
            if (target_char == '1') {
                coeff.first *= -1.0;
            // Z |+> = z0 |+>
            // Z |sixj> = z0 |sixj>
            // Z |dixj> = z0 |dixj>
            } else if (target_char == '+' || target_char == 's' || target_char == 'd') {
                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', IndexSet{target_qubit}));
            }
        // H
        } else if (name == "H") {
            // H |0> = |+>
            if (target_char == '0') {
                target_char = '+';
            // H |1> = z0 |+>
            } else if (target_char == '1') {
                target_char = '+';
                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', IndexSet{target_qubit}));
            // H |+> = |0>
            // H z0 |+> = |1>
            // H z0 z0&1 |+ +> = |d1 +>
            // H z0&1 |+ +> = |s1 +>
            // H z0&1 z0&2 |+ + +> = | s1&2 + +>
            // H |+ s0> = z0&1 |+ +>
            // H |+ d0> = z0 z0&1 |+ +> 
            // H z0 |+ s0> = z0&1 z1 |+ +>
            // H |+ s0&2 +> = z0&1 z0&2 |+ + +>
            // H z0 |+ s0&2 +> = z1 z2 (z0&1 z0&2) |+ + +>
            // H z2 |+ s0&2 +> = z2 (z0&1 z0&2) |+ + +>
            // H z0&2 |+ s0&2 +> = z2 z1&2 (z0&1 z0&2) |+ + +>
            // H z0&2 |+ s0 +> = z0&2 z1&2 |+ + +>
            // H z0&2 |+ d0 +> = z0 z2 z0&1 z1&2 |+ + +>
            // H z0 z0&2 |+ s0 +> = z1 z0&1 z1&2 |+ + +> TODO
            // H z0 z0&2 |+ d0 +> = - z0 z1 z2 z0&1 z1&2 |+ + +> TODO
            } else if (target_char == '+') {

                // Check if we have a linear z phase that references this
                bool has_linear_phase = false;
                for (const auto& [char_, indices] : coeff.second) {
                    if (char_ == 'z' && indices.size() == 1 && indices.find(target_qubit) != indices.end()) {
                        has_linear_phase = true;
                        break;
                    }
                }

                // Check if we have any quadratic z phase that references this
                std::vector<int> quadratic_phase_other_qubits;
                for (const auto& [char_, indices] : coeff.second) {
                    if (char_ == 'z' && indices.size() == 2 && indices.find(target_qubit) != indices.end()) {
                        IndexSet other_indices = indices;
                        other_indices.erase(target_qubit);
                        quadratic_phase_other_qubits.push_back(*other_indices.begin());
                    }
                }

                // Keep track of the phases to add
                std::vector<std::pair<char, IndexSet>> phases_to_add;

                // Check if anything references this plus
                bool has_reference = false;
                for (size_t i = 0; i < ket.size(); ++i) {
                    auto& char_ = std::get<0>(ket[i]);
                    auto& indices = std::get<1>(ket[i]);
                    
                    // If so, it is now a plus too, and we add a phase
                    if (indices.find(target_qubit) != indices.end()) {
                        phases_to_add.push_back(std::make_pair('z', IndexSet{target_qubit, int(i)}));
                        if (has_linear_phase) {
                            phases_to_add.push_back(std::make_pair('z', IndexSet{int(i)}));
                            for (int other_qubit : quadratic_phase_other_qubits) {
                                phases_to_add.push_back(std::make_pair('z', IndexSet{other_qubit, int(i)}));
                            }
                        }
                        for (int other_index : quadratic_phase_other_qubits) {
                            if (char_ == 'd') {
                                phases_to_add.push_back(std::make_pair('z', IndexSet{other_index}));
                            }
                            phases_to_add.push_back(std::make_pair('z', IndexSet{other_index, int(i)}));
                        }
                        for (int index : indices) {
                            if (index != target_qubit) {
                                phases_to_add.push_back(std::make_pair('z', IndexSet{target_qubit, index}));
                                if (has_linear_phase) {
                                    phases_to_add.push_back(std::make_pair('z', IndexSet{index}));
                                }
                            }
                        }
                        // if it's a d, add an extra phase
                        if (char_ == 'd') {
                            phases_to_add.push_back(std::make_pair('z', IndexSet{int(target_qubit)}));
                            if (has_linear_phase) {
                                coeff.first *= -1.0;
                            }
                        }
                        char_ = '+';
                        indices.clear();
                        has_reference = true;
                        break;
                    }

                }

                // If it's not referenced anywhere, destroy it
                if (!has_reference) {
                    if (quadratic_phase_other_qubits.empty()) {
                        target_char = (has_linear_phase) ? '1' : '0';
                        target_indices.clear();
                    } else {
                        target_char = (has_linear_phase) ? 'd' : 's';
                        target_indices = IndexSet(quadratic_phase_other_qubits.begin(), quadratic_phase_other_qubits.end());
                    }
                }

                // Remove any phases that reference this
                std::vector<std::pair<char, IndexSet>> to_remove;
                for (const auto& [char_, indices] : coeff.second) {
                    if (char_ == 'z' && indices.find(target_qubit) != indices.end()) {
                        to_remove.push_back(std::make_pair(char_, indices));
                    }
                }
                for (const auto& pair : to_remove) {
                    coeff.first *= handle_insert_coeff(coeff.second, pair);
                }

                // Add any new phases 
                for (const auto& pair : phases_to_add) {
                    coeff.first *= handle_insert_coeff(coeff.second, pair);
                }

            // H |s1 +> = z0&1 |+ +>
            // H |d1 +> = z0 z0&1 |+ +> 
            // H z0 |s1 +> = z1 z0&1 |+ +> 
            // H z0 |d1 +> = - z0 z1 z0&1 |+ +>
            // H z1 |s1 +> = z1 z0&1 |+ +> 
            // H z1 |d1 +> = z0 z1 z0&1 |+ +>
            // H z0&1 |s1 +> = z1 z0&1 |+ +> 
            // H z0&1 |d1 +> = z0 z0&1 |+ +>
            // H z0 z1 |s1 +> = |+ +> 
            // H z0 z1 |d1 +> = z0 z0&1 |+ +>
            // H |s1&2 + +> = (z0&1 z0&2) |+ + +>
            // H |d1&2 + +> = z0 (z0&1 z0&2) |+ + +>
            // H z2 |s1&2 + +> = z2 (z0&1 z0&2) |+ + +>
            // H z2 |d1&2 + +> = z0 z2 (z0&1 z0&2) |+ + +>
            // H z0 |s1&2 + +> = z1 z2 (z0&1 z0&2) |+ + +> 
            // H z0 |d1&2 + +> = z0 z1 z2 z0&1 z0&2 |+ + +>
            // H z1&2 |s1&2 + +> = z1&2 (z0&1 z0&2) |+ + +> 
            // H z1&2 |d1&2 + +> = z0 z1&2 (z0&1 z0&2) |+ + +>
            // H z0&2 |s1&2 + +> = z2 z1&2 (z0&1 z0&2) |+ + +>
            // H z0&2 |d1&2 + +> = z0 z1&2 (z0&1 z0&2) |+ + +>
            // H z0&1 |s1&2 + +> = z1 z1&2 (z0&1 z0&2) |+ + +>
            // H z0&1 |d1&2 + +> = z0 z1&2 (z0&1 z0&2) |+ + +>
            // H z0&1 z0&2 |s1&2 + +> = z1 z2 z0&1 z0&2 |+ + +>
            // H z0&1 z0&2 |d1&2 + +> = z0 z0&1 z0&2 |+ + +>
            } else if (target_char == 's' || target_char == 'd') {

                // If acting on sixj, add z0&i z0&j
                // If acting on dixj, add z0 z0&i z0&j
                // If acting on sixj and there is a linear phase on 0, add zi and zj
                // If acting on dixj and there is a linear phase on 0, multiply by -1 and add zi and zj
                // If acting on sixj and there is a quadratic phase on k, add zi&k if i!=k and add zj&k if j==k, otherwise add zj
                // If acting on dixj and there is a quadratic phase on k, add zi&k if i!=k and add zj&k if j==k

                // Check for linear phase
                bool has_linear_phase = false;
                for (const auto& [char_, indices] : coeff.second) {
                    if (char_ == 'z' && indices.size() == 1 && indices.find(target_qubit) != indices.end()) {
                        has_linear_phase = true;
                        break;
                    }
                }

                // Check for quadratic phases
                std::vector<int> quadratic_phase_other_qubits;
                for (const auto& [char_, indices] : coeff.second) {
                    if (char_ == 'z' && indices.size() == 2 && indices.find(target_qubit) != indices.end()) {
                        IndexSet other_indices = indices;
                        other_indices.erase(target_qubit);
                        quadratic_phase_other_qubits.push_back(*other_indices.begin());
                    }
                }

                // Remove any phases that reference this
                std::vector<std::pair<char, IndexSet>> to_remove;
                for (const auto& [char_, indices] : coeff.second) {
                    if (char_ == 'z' && indices.find(target_qubit) != indices.end()) {
                        to_remove.push_back(std::make_pair(char_, indices));
                    }
                }
                for (const auto& pair : to_remove) {
                    coeff.first *= handle_insert_coeff(coeff.second, pair);
                }

                // Add the phase
                if (target_char == 's') {
                    for (int index : target_indices) {
                        coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', IndexSet{target_qubit, index}));
                    }
                    if (has_linear_phase) {
                        for (int index : target_indices) {
                            coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', IndexSet{index}));
                        }
                    }
                    for (int other_qubit : quadratic_phase_other_qubits) {
                        for (int index : target_indices) {
                            if (index != other_qubit) {
                                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', IndexSet{other_qubit, index}));
                            } else {
                                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', IndexSet{other_qubit}));
                            }
                        }
                    }
                } else if (target_char == 'd') {
                    coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', IndexSet{target_qubit}));
                    for (int index : target_indices) {
                        coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', IndexSet{target_qubit, index}));
                    }
                    if (has_linear_phase) {
                        coeff.first *= -1.0;
                        for (int index : target_indices) {
                            coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', IndexSet{index}));
                        }
                    }
                    for (int other_qubit : quadratic_phase_other_qubits) {
                        for (int index : target_indices) {
                            if (index != other_qubit) {
                                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', IndexSet{other_qubit, index}));
                            }
                        }
                    }
                }
                
                // It will always then be a free variable
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
                // CNOT |1 sixj> = |1 di&j>
                } else if (target_char == 's') {
                    target_char = 'd';
                // CNOT |1 dixj> = |1 sixj>
                } else if (target_char == 'd') {
                    target_char = 's';
                // CNOT |1 +> = |1 +>
                // CNOT |1 + s1> = |1 + d1>
                } else if (target_char == '+') {
                    // find anything in the basis that references this and swap it
                    for (size_t i = 0; i < ket.size(); ++i) {
                        auto& char_ = std::get<0>(ket[i]);
                        auto& indices = std::get<1>(ket[i]);
                        if (indices.find(target_qubit) != indices.end()) {
                            char_ = (char_ == 's') ? 'd' : 's';
                        }
                    }
                    // add a global -1 for any linear phase that references this
                    for (const auto& [char_, indices] : coeff.second) {
                        if (indices.size() == 1 && indices.find(target_qubit) != indices.end() && char_ == 'z') {
                            coeff.first *= -1.0;
                        }
                    }
                    // add a z(other) for any quadratic phase that references this
                    for (const auto& [char_, indices] : coeff.second) {
                        if (indices.size() == 2 && indices.find(target_qubit) != indices.end() && char_ == 'z') {
                            IndexSet other_indices = indices;
                            other_indices.erase(target_qubit);
                            coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', other_indices));
                        }
                    }
                }
            } else if (control_char == '+') {
                // CNOT |+ 0> = |+ s0>
                if (target_char == '0') {
                    target_char = 's';
                    target_indices = IndexSet{control_qubit};
                // CNOT |+ 1> = |+ d0>
                } else if (target_char == '1') {
                    target_char = 'd';
                    target_indices = IndexSet{control_qubit};
                // CNOT |+ s2 +> = |+ s0&2 +>
                } else if (target_char == 's' || target_char == 'd') {
                    handle_insert_basis(target_indices, control_qubit, target_char);
                // CNOT |+ +> = |+ +>
                // CNOT z1 |+ +> = z1 z0 z0&1 |+ +> TODO
                // CNOT z0 |+ +> = z1 |+ +> TODO
                // CNOT |+ + s1> = |+ + s0&1>
                } else if (target_char == '+') {
                    // find anything in the basis that references this and add the control to it
                    for (size_t i = 0; i < ket.size(); ++i) {
                        auto& char_ = std::get<0>(ket[i]);
                        auto& indices = std::get<1>(ket[i]);
                        if ((char_ == 's' || char_ == 'd') && indices.find(target_qubit) != indices.end()) {
                            handle_insert_basis(indices, control_qubit, char_);
                        }
                    }
                }
            } else if (control_char == 's' || control_char == 'd') {
                // CNOT |s2 0 +> = |s2 s2 +>
                // CNOT |d2 0 +> = |d2 d2 +>
                if (target_char == '0') {
                    target_char = control_char;
                    target_indices = control_indices;
                // CNOT |s2 1 +> = |s2 d2 +>
                // CNOT |d2 1 +> = |d2 s2 +>
                } else if (target_char == '1') {
                    target_char = (control_char == 's') ? 'd' : 's';
                    target_indices = control_indices;
                // CNOT |s1 +> = |+ 0>
                // CNOT |s1 + s1> = |+ 0 s0>
                // CNOT |s1&2 + +> = |+ s2 +>
                // CNOT |d1 +> = |+ 0>
                // CNOT z1 |s1 +> = z0 |s1 0>
                } else if (target_char == '+') {
                    bool control_was_s = (control_char == 's');
                    bool self_reference = control_indices.find(target_qubit) != control_indices.end();
                    if (self_reference) {
                        for (int index : control_indices) {
                            if (index == target_qubit) {

                                // This is getting destroyed
                                IndexSet new_indices = control_indices;
                                new_indices.erase(index);
                                if (new_indices.empty()) {
                                    target_char = (control_was_s) ? '0' : '1';
                                    target_indices.clear();
                                } else {
                                    target_char = (control_was_s) ? 's' : 'd';
                                    target_indices = new_indices;
                                }

                                // Which means the control is now a plus
                                control_char = '+';
                                control_indices.clear();

                                // Anything that referenced the target should now reference the control
                                for (auto& [char_, indices] : ket) {
                                    if ((char_ == 's' || char_ == 'd') && indices.find(target_qubit) != indices.end()) {
                                        handle_insert_basis(indices, control_qubit, char_);
                                    }
                                }
                                for (const auto& [char_, indices] : coeff.second) {
                                    if (indices.find(target_qubit) != indices.end()) {
                                        handle_insert_coeff(coeff.second, std::make_pair(char_, IndexSet{control_qubit}));
                                    }
                                }

                                break;
                                
                            }
                        }
                    } else {

                        // Anything that references this should have the control qubits added
                        for (auto& [char_, indices] : ket) {
                            if ((char_ == 's' || char_ == 'd') && indices.find(target_qubit) != indices.end()) {
                                if (!control_was_s) {
                                    char_ = (char_ == 's') ? 'd' : 's';
                                }
                                for (int index : control_indices) {
                                    handle_insert_basis(indices, index, char_);
                                }
                            }
                        }

                    }
                // CNOT |s2 s2 +> = |s2 0 +>
                // CNOT |s2 d2 +> = |s2 1 +>
                // CNOT |d2 s2 +> = |d2 1 +>
                // CNOT |d2 d2 +> = |d2 0 +>
                // CNOT |s2 s3 + +> = |s2 s2&3 + +>
                // CNOT |s2 d3 + +> = |s2 d2&3 + +>
                // CNOT |d2 s3 + +> = |d2 d2&3 + +>
                // CNOT |d2 d3 + +> = |d2 s2&3 + +>
                } else if (target_char == 's' || target_char == 'd') {
                    if (control_char == 'd') {
                        target_char = (target_char == 's') ? 'd' : 's';
                    }
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
                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('i', IndexSet{target_qubit}));
            }
        } else if (name == "CZ") {
            auto& control_char = std::get<0>(ket[control_qubit]);
            // CZ |1 1> = -|1 1>
            if (target_char == '1' && control_char == '1') {
                coeff.first *= -1.0;
            // CZ |+ +> = z0&1 |+ +>
            } else if (target_char != '0' && control_char != '0') {
                coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', IndexSet{target_qubit, control_qubit}));
            }
        // T TODO
        } else if (name == "T") {
        // Toffoli TODO
        } else if (name == "CCX") {
        // Generic gate TODO
        } else {
            throw std::runtime_error("Unknown operator name: " + name);
        }

        // Check to make sure we're still valid
        bool consistent = false;
        while (!consistent) {
            consistent = true;
            for (size_t i = 0; i < ket.size(); ++i) {
                auto& char_ = std::get<0>(ket[i]);
                auto& indices = std::get<1>(ket[i]);
                
                // Ensure all s and d chars have at least one index, otherwise convert to 0 or 1
                if ((char_ == 's' || char_ == 'd') && indices.empty()) {
                    char_ = (char_ == 's') ? '0' : '1';
                    consistent = false;
                }

                // Ensure all other character have no indices
                if (char_ != 's' && char_ != 'd' && !indices.empty()) {
                    indices.clear(); 
                    consistent = false;
                }

                // Ensure all s and d have indices pointing to chars that are not 0 or 1
                if (char_ == 's' || char_ == 'd') {
                    char new_char = char_;
                    IndexSet new_indices;
                    for (int index : indices) {
                        auto& dependent_char = std::get<0>(ket[index]);
                        auto& dependent_indices = std::get<1>(ket[index]);
                        if (dependent_char == '1') {
                            new_char = (new_char == 's') ? 'd' : 's';
                        } else if (dependent_char == 's' || dependent_char == 'd') {
                            if (dependent_char == 'd') {
                                new_char = (new_char == 's') ? 'd' : 's';
                            }
                            for (int dependent_index : dependent_indices) {
                                handle_insert_basis(new_indices, dependent_index, new_char);
                            }
                        } else if (dependent_char != '0') {
                            handle_insert_basis(new_indices, index, new_char);
                        }
                    }
                    if (new_char != char_ || new_indices != indices) {
                        char_ = new_char;
                        indices = new_indices;
                        consistent = false;
                    }
                }
            }
        }

        // Make sure any phases that can evaluate, do
        // e.g. z0 |1> should become -|1>
        std::vector<std::pair<char, IndexSet>> to_remove;
        for (const auto& [char_, indices] : coeff.second) {
            if (char_ == 'z' || char_ == 'i') {
                bool applies = true;
                for (int index : indices) {
                    auto& char_ = std::get<0>(ket[index]);
                    if (char_ != '1') {
                        applies = false;
                        break;
                    }
                }
                if (applies) {
                    if (char_ == 'z') {
                        coeff.first *= -1.0;
                    } else {
                        coeff.first *= std::complex<double>(0.0, 1.0);
                    }
                    to_remove.push_back(std::make_pair(char_, indices));
                }
            }
        }
        for (const auto& pair : to_remove) {
            coeff.first *= handle_insert_coeff(coeff.second, pair);
        }

    }

}

void AffineStabilizerOperator::apply(DenseMatrix& output_state) const {
    /*
    Apply the operator to a dense state.

    Args:
        output_state (DenseMatrix&): The dense state to apply the operator to.
    */

    // If we have an X on qubit i, we swap the amplitudes of all basis states where qubit i is 0 with those where qubit i is 1
    int num_qubits = static_cast<int>(std::log2(output_state.size()));
    long long mask = 1LL << (num_qubits - 1 - target_qubit);
    if (name == "X") {
        size_t N = output_state.size();
        size_t stride = mask;
        size_t step = stride << 1;
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t base = 0; base < N; base += step) {
            for (size_t offset = 0; offset < stride; ++offset) {
                size_t i = base + offset;
                size_t j = i + stride;
                std::swap(output_state(i), output_state(j));
            }
        }

    // If we have a Y on qubit i, we swap the amplitudes of all basis states where qubit i is 0 with those where qubit i is 1, and multiply the amplitude of all basis states where qubit i is 1 by i or -i depending on whether it was originally 0 or 1
    } else if (name == "Y") {
        size_t N = output_state.size();
        size_t stride = mask;
        size_t step = stride << 1;
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t base = 0; base < N; base += step) {
            for (size_t offset = 0; offset < stride; ++offset) {
                size_t i = base + offset;
                size_t j = i + stride;
                std::swap(output_state(i), output_state(j));
                output_state(i) *= std::complex<double>(0.0, -1.0);
                output_state(j) *= std::complex<double>(0.0, 1.0);
            }
        }

    // If we have a Z on qubit i, we multiply the amplitude of all basis states where qubit i is 1 by -1
    } else if (name == "Z") {
        size_t N = output_state.size();
        size_t stride = mask;
        size_t step = stride << 1;
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t base = 0; base < N; base += step) {
            for (size_t offset = 0; offset < stride; ++offset) {
                size_t i = base + offset;
                output_state(i + stride) *= -1.0;
            }
        }

    // If we have a H on qubit i, we apply the transformation |0> -> (|0> + |1>)/sqrt(2), |1> -> (|0> - |1>)/sqrt(2) to all basis states
    } else if (name == "H") {
        size_t N = output_state.size();
        size_t stride = mask;
        size_t step = stride << 1;
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t base = 0; base < N; base += step) {
            for (size_t offset = 0; offset < stride; ++offset) {
                size_t i = base + offset;
                size_t j = i + stride;
                std::complex<double> temp_i = output_state(i);
                std::complex<double> temp_j = output_state(j);
                output_state(i) = (temp_i + temp_j) / std::sqrt(2.0);
                output_state(j) = (temp_i - temp_j) / std::sqrt(2.0);
            }
        }

    // If we have a S on qubit i, we multiply the amplitude of all basis states where qubit i is 1 by i
    } else if (name == "S") {
        size_t N = output_state.size();
        size_t stride = mask;
        size_t step = stride << 1;
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t base = 0; base < N; base += step) {
            for (size_t offset = 0; offset < stride; ++offset) {
                size_t i = base + offset;
                output_state(i + stride) *= std::complex<double>(0.0, 1.0);
            }
        }

    // If we have a T on qubit i, we multiply the amplitude of all basis states where qubit i is 1 by exp(i*pi/4)
    } else if (name == "T") {
        std::complex<double> phase = std::exp(std::complex<double>(0.0, M_PI / 4.0));
        size_t N = output_state.size();
        size_t stride = mask;
        size_t step = stride << 1;
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t base = 0; base < N; base += step) {
            for (size_t offset = 0; offset < stride; ++offset) {
                size_t i = base + offset;
                output_state(i + stride) *= phase;
            }
        }

    // If we have a CNOT with control qubit j and target qubit i, we swap the amplitudes of all basis states where qubit j is 1 and qubit i is 0 with those where qubit j is 1 and qubit i is 1
    } else if (name == "CNOT") {
        size_t N = output_state.size();
        size_t control_mask = 1LL << (num_qubits - 1 - control_qubit);
        size_t stride = mask;
        size_t step = stride << 1;
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t base = 0; base < N; base += step) {
            for (size_t offset = 0; offset < stride; ++offset) {
                size_t i = base + offset;
                if (i & control_mask) {
                    size_t j = i ^ mask;
                    if (i < j) {
                        std::swap(output_state(i), output_state(j));
                    }
                }
            }
        }

    // If we have a CZ with control qubit j and target qubit i, we multiply the amplitude of all basis states where qubit j is 1 and qubit i is 1 by -1
    } else if (name == "CZ") {
        size_t N = output_state.size();
        size_t control_mask = 1LL << (num_qubits - 1 - control_qubit);
        size_t stride = mask;
        size_t step = stride << 1;
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t base = 0; base < N; base += step) {
            for (size_t offset = 0; offset < stride; ++offset) {
                size_t i = base + offset;
                if (i & control_mask) {
                    size_t j = i ^ mask;
                    if (i < j) {
                        output_state(j) *= -1.0;
                    }
                }
            }
        }

    // If we have a 2x2 base matrix, we apply it by treating the target qubit as the least significant bit and iterating through pairs of basis states
    } else if (base_matrix.rows() == 2 && base_matrix.cols() == 2) {
        size_t N = output_state.size();
        size_t stride = mask;
        size_t step = stride << 1;
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t base = 0; base < N; base += step) {
            for (size_t offset = 0; offset < stride; ++offset) {
                size_t i = base + offset;
                size_t j = i + stride;
                std::complex<double> temp_i = output_state(i);
                std::complex<double> temp_j = output_state(j);
                output_state(i) = base_matrix(0, 0) * temp_i + base_matrix(0, 1) * temp_j;
                output_state(j) = base_matrix(1, 0) * temp_i + base_matrix(1, 1) * temp_j;
            }
        }
    } else {
        throw std::runtime_error("Unknown operator name: " + name);
    }
}

AffineStabilizerOperator::AffineStabilizerOperator(const Gate& gate) {
    /*
    Construct an AffineStabilizerOperator from a given gate by extracting the target and control qubits, and the name of the gate.

    Args:
        gate (const Gate&): The gate to construct the operator from.

    Returns:
        AffineStabilizerOperator: The resulting operator after construction.

    Raises:
        std::invalid_argument: If the gate has more than 1 control qubits or does not have exactly 1 target qubit.
    */
    if (gate.get_control_qubits().size() > 1) {
        throw std::invalid_argument("AffineStabilizerOperator only supports gates with 1 or fewer total control qubits.");
    }
    if (gate.get_target_qubits().size() != 1) {
        throw std::invalid_argument("AffineStabilizerOperator requires a gate with exactly 1 target qubit.");
    }
    target_qubit = gate.get_target_qubits()[0];
    control_qubit = gate.get_control_qubits().empty() ? -1 : gate.get_control_qubits()[0];
    base_matrix = gate.get_base_matrix();
    name = gate.get_name();
}
