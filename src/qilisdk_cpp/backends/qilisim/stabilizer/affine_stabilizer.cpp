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

AffineStabilizerState::AffineStabilizerState(int n_qubits, bool density) {
    /* 
    Create a new AffineStabilizerState in the |0...0> state.

    Args:
        n_qubits (int): The number of qubits in the state.
        density (bool): Whether to create the state as a density matrix (default: false).

    Returns:
        AffineStabilizerState: The initialized state.
    */
    StateCoefficient one_coeff = std::make_pair(std::complex<double>(1.0, 0.0), StateCoefficient::second_type());
    StateBasis zero_basis;
    for (int i = 0; i < n_qubits; ++i) {
        zero_basis.push_back(std::make_pair('0', IndexSet()));
    }
    if (density) {
        state.push_back(std::make_tuple(one_coeff, zero_basis, zero_basis));
    } else {
        state.push_back(std::make_tuple(one_coeff, zero_basis, StateBasis()));
    }
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
                new_state.state.push_back(std::make_tuple(coeff, ket_basis, bra_basis));
            } else {
                new_state.state.push_back(std::make_tuple(coeff, ket_basis, StateBasis()));
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
    return std::get<2>(*state.begin()).empty();
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
    return !std::get<1>(*state.begin()).empty() && !std::get<2>(*state.begin()).empty();
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
    /*
    Get the number of qubits in the state.

    Returns:
        int: The number of qubits in the state.
    */
    if (state.empty()) {
        return 0;
    }
    return std::max(std::get<1>(*state.begin()).size(), std::get<2>(*state.begin()).size());
}

void AffineStabilizerState::prune(double atol) {
    /*
    Prune components of the state that have coefficients with magnitude below the given absolute tolerance.

    Args:
        atol (double): The absolute tolerance for pruning components (default: 1e-12).
    */
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
                os << "&";
            }
            count++;
        }
        if (i != basis.size() - 1) {
            os << ", ";
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
        os << " * " << char_;
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
    /*
    Add two AffineStabilizerStates together by concatenating their state vectors.

    Args:
        other (const AffineStabilizerState&): The state to add to this state.

    Returns:
        AffineStabilizerState: The resulting state after addition.
    */
    AffineStabilizerState result = *this;
    for (const auto& pair : other.state) {
        result.state.push_back(pair);
    }
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
    for (const auto& pair : other.state) {
        auto negated_pair = pair;
        std::get<0>(negated_pair).first *= -1.0;
        result.state.push_back(negated_pair);
    }
    return result;
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
        state.push_back(pair);
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
        auto negated_pair = pair;
        std::get<0>(negated_pair).first *= -1.0;
        state.push_back(negated_pair);
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
            // X |si&j> = |di&j>
            // X |di&j> = |si&j>
            } else if (target_char == 's' || target_char == 'd') {
                target_char = (target_char == 's') ? 'd' : 's';
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
            // Y |si&j> = z0 i|di&j>
            // Y |di&j> = z0 i|si&j>
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
            // Z |si&j> = z0 |si&j>
            // Z |di&j> = z0 |di&j>
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
            // H z0&1 |+ +> = |s1 +>
            // H |+ s0> = z0&1 |+ +>
            // H |+ s0&2 +> = z0&1 z0&2 |+ + +>
            // H z0 |+ s0&2 +> = z1 z2 z0&1 z0&2 |+ + +>
            // H z2 |+ s0&2 +> = z2 z0&1 z0&2 |+ + +>
            // H z0&2 |+ s0&2 +> = z2 z0&1 z0&2 z1&2 |+ + +>
            // H z0 |+ s0> = z1 z0&1 |+ +>
            // H z1 |+ s0> = z1 z0&1 |+ +>
            // H z0&1 |+ s0> = z1 z0&1 |+ +> 
            } else if (target_char == '+') {
                // TODO

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

                    bool should_be_zero = true;

                    // Check for linear phases
                    for (const auto& [char_, indices] : coeff.second) {
                        if (indices.size() == 1 && indices.find(target_qubit) != indices.end() && char_ == 'z') {
                            should_be_zero = !should_be_zero;
                        }
                    }

                    // Check for quadratic phases

                    target_char = '0';
                    target_indices.clear();

                }

            // H |s1 +> = z1 |+ +>
            } else if (target_char == 's' || target_char == 'd') {
                for (int index : target_indices) {
                    coeff.first *= handle_insert_coeff(coeff.second, std::make_pair('z', IndexSet{index}));
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
                                        // if (!control_was_s) {
                                        //     char_ = (char_ == 's') ? 'd' : 's';
                                        // }
                                        // indices.erase(target_qubit);
                                        handle_insert_basis(indices, control_qubit, char_);
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
        // T TODO
        } else if (name == "T") {
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
    name = gate.get_name();
}
