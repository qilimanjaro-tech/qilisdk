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

#include "stabilizer_state.h"

std::ostream& operator<<(std::ostream& os, const StabilizerStateSum& sss) {
    /*
    Print a StabilizerStateSum in a human-readable format. This is mostly for debugging purposes.

    Args:
        os (std::ostream&): The output stream to print to.
        sss (StabilizerStateSum&): The StabilizerStateSum to print.

    Returns:
        std::ostream&: The output stream after printing.
    */
    os << "StabilizerStateSum with " << sss.get_states().size() << " terms:\n";
    for (size_t i = 0; i < sss.get_states().size(); ++i) {
        const int n = sss.get_states()[i].get_nqubits();
        auto truncate = [n](const std::bitset<MAX_ROWS_STABILIZER>& bs) {
            auto s = bs.to_string().substr(MAX_ROWS_STABILIZER - n);
            std::reverse(s.begin(), s.end());
            return s;
        };
        os << "Coefficient: " << sss.get_coefficients()[i] << "\n";
        os << "X stabilizers:\n";
        for (int j = 0; j < n; ++j) {
            os << "For qubit " << j << ": " << truncate(sss.get_states()[i].get_x_bits()[j]) << "\n";
        }
        os << "Z stabilizers:\n";
        for (int j = 0; j < n; ++j) {
            os << "For qubit " << j << ": " << truncate(sss.get_states()[i].get_z_bits()[j]) << "\n";
        }
        os << "Phases: " << truncate(sss.get_states()[i].get_phases()) << "\n";
    }
    return os;
}

// GF(2) Gaussian elimination to find the ±1 phase of Z_target in a stabilizer tableau.
// Takes a column-major tableau (x[qubit][row], z[qubit][row], ph[row]) and returns
// false = +1 eigenvalue, true = -1 eigenvalue.
static bool z_phase_by_ge(
        const std::vector<std::bitset<MAX_ROWS_STABILIZER>>& x,
        const std::vector<std::bitset<MAX_ROWS_STABILIZER>>& z,
        const std::bitset<MAX_ROWS_STABILIZER>& ph,
        int n, int target) {
    struct Row { std::vector<bool> xb, zb; bool ph; };
    std::vector<Row> rows(n);
    for (int r = 0; r < n; ++r) {
        rows[r].xb.resize(n); rows[r].zb.resize(n);
        for (int q = 0; q < n; ++q) { rows[r].xb[q] = (bool)x[q][r]; rows[r].zb[q] = (bool)z[q][r]; }
        rows[r].ph = (bool)ph[r];
    }
    auto rs = [&](Row& h, const Row& g) {
        int contrib = 0;
        for (int q = 0; q < n; ++q) {
            if (h.xb[q] && g.zb[q]) contrib ^= 1;
            if (h.zb[q] && g.xb[q]) contrib ^= 1;
        }
        h.ph = h.ph ^ g.ph ^ (bool)contrib;
        for (int q = 0; q < n; ++q) { h.xb[q] = h.xb[q] ^ g.xb[q]; h.zb[q] = h.zb[q] ^ g.zb[q]; }
    };
    Row tgt; tgt.xb.assign(n, false); tgt.zb.assign(n, false); tgt.zb[target] = true; tgt.ph = false;
    int pivot_row = 0;
    for (int bit = 0; bit < 2 * n && pivot_row < n; ++bit) {
        int q = bit < n ? bit : bit - n;
        bool is_x = bit < n;
        int pivot = -1;
        for (int r = pivot_row; r < n; ++r)
            if (is_x ? rows[r].xb[q] : rows[r].zb[q]) { pivot = r; break; }
        if (pivot < 0) continue;
        std::swap(rows[pivot], rows[pivot_row]);
        for (int r = 0; r < n; ++r)
            if (r != pivot_row && (is_x ? rows[r].xb[q] : rows[r].zb[q]))
                rs(rows[r], rows[pivot_row]);
        if (is_x ? tgt.xb[q] : tgt.zb[q]) rs(tgt, rows[pivot_row]);
        ++pivot_row;
    }
    return tgt.ph;
}

std::string StabilizerState::sample() const {
    auto x = x_bits;
    auto z = z_bits;
    auto ph = phases;

    std::string result(nqubits, '0');

    for (int k = 0; k < nqubits; ++k) {
        int p = -1;
        for (int i = 0; i < nqubits; ++i) {
            if (x[k][i]) { p = i; break; }
        }

        if (p >= 0) {
            // Random outcome: rowsum every anticommuting generator into g_p, then collapse.
            for (int i = 0; i < nqubits; ++i) {
                if (i != p && x[k][i]) {
                    int contrib = 0;
                    for (int q = 0; q < nqubits; ++q) {
                        if (x[q][i] && z[q][p]) contrib ^= 1;
                        if (z[q][i] && x[q][p]) contrib ^= 1;
                    }
                    if ((bool)ph[p] ^ (bool)contrib) ph.flip(i);
                    for (int q = 0; q < nqubits; ++q) {
                        if (x[q][p]) x[q].flip(i);
                        if (z[q][p]) z[q].flip(i);
                    }
                }
            }
            int b = rand() & 1;
            for (int q = 0; q < nqubits; ++q) { x[q].reset(p); z[q].reset(p); }
            z[k].set(p);
            if (b) ph.set(p); else ph.reset(p);
            result[k] = '0' + b;

        } else {
            // Deterministic: GF(2) elimination on the current working tableau.
            result[k] = z_phase_by_ge(x, z, ph, nqubits, k) ? '1' : '0';
        }
    }

    return result;
}

std::map<std::string, int> StabilizerStateSum::sample(int nshots) const {
    /*
    Sample from the StabilizerStateSum. This is done by first picking a StabilizerState 
    according to the probabilities given by the coefficients, then sampling from that state.

    Args:
        nshots (int): The number of samples to draw.

    Returns:
        std::map<std::string, int>: A map from bitstrings to counts.
    */

    // Repeat for each shot:
    std::map<std::string, int> sample_counts;
    for (int shot = 0; shot < nshots; ++shot) {

        // First, pick an element of the sum according to the probabilities given by the coefficients
        // int index = 0;
        double r = ((double) rand() / (RAND_MAX));
        double cumulative_prob = 0.0;
        size_t index = 0;
        for (size_t i = 0; i < coefficients.size(); ++i) {
            cumulative_prob += std::norm(coefficients[i]);
            if (r < cumulative_prob) {
                index = i;
                break;
            }
        }

        // Sample from that
        const StabilizerState& state = states[index];
        std::string sample_str = state.sample();
        sample_counts[sample_str]++;

    }

    // Return the counts
    return sample_counts;

}

void StabilizerState::apply_gate(const Gate& gate) {
    /*
    Apply a gate to the StabilizerState. This uses the standard tableau update rules for Clifford gates. Non-Clifford gates are not supported here.

    Args:
        gate (Gate&): The gate to apply.
    */

    // Get info about the gate
    const std::string name = gate.get_name();
    const auto& controls = gate.get_control_qubits();
    const auto& targets = gate.get_target_qubits();

    if (name == "SWAP") {
        int i = targets[0];
        int j = targets[1];
        std::swap(x_bits[i], x_bits[j]);
        std::swap(z_bits[i], z_bits[j]);
    } else if (controls.empty()) {
        int i = targets[0];
        if (name == "H") {
            phases ^= (x_bits[i] & z_bits[i]);
            std::swap(x_bits[i], z_bits[i]);
        } else if (name == "S") {
            phases ^= (x_bits[i] & z_bits[i]);
            z_bits[i] ^= x_bits[i];
        } else if (name == "X") {
            phases ^= z_bits[i];
        } else if (name == "Y") {
            phases ^= (x_bits[i] ^ z_bits[i]);
        } else if (name == "Z") {
            phases ^= x_bits[i];
        }
    } else {
        // Two-qubit controlled gates: CNOT is stored as X with control, CZ as Z with control
        int i = controls[0], j = targets[0];
        if (name == "X") {
            // CNOT: control=i, target=j — phase computed before modifying x[j], z[i]
            phases ^= (x_bits[i] & z_bits[j] & ~(x_bits[j] ^ z_bits[i]));
            x_bits[j] ^= x_bits[i];
            z_bits[i] ^= z_bits[j];
        } else if (name == "Z") {
            // CZ: qubits i, j — phase computed before modifying z[i], z[j]
            phases ^= (x_bits[i] & x_bits[j] & (z_bits[i] ^ z_bits[j]));
            z_bits[j] ^= x_bits[i];
            z_bits[i] ^= x_bits[j];
        }
    }
}

// Returns index of any row r where x_bits[q][r] == 1, or -1 if none.
int StabilizerState::find_x_pivot(int q) const {
    for (int r = 0; r < nqubits; ++r) {
        if (x_bits[q][r]) return r;
    }
    return -1;
}

// Computes the phase contribution when multiplying two Pauli operators on a single qubit.
// Returns an integer in {0, 1, 2, 3} representing multiples of i (i^n).
// x1,z1: Pauli on qubit from row i; x2,z2: Pauli on qubit from row h (the one being updated).
static int pauli_phase(bool x1, bool z1, bool x2, bool z2) {
    // g(x1,z1,x2,z2) from AG Table 1
    if (!x1 && !z1) return 0; // I * anything = no phase
    if ( x1 &&  z1) return z2 - x2; // Y case: z2 - x2 in {-1, 0, 1} -> mod 4
    if ( x1 && !z1) return z2 * (2*x2 - 1); // X case
    if (!x1 &&  z1) return x2 * (1 - 2*z2); // Z case
    return 0;
}

// Row h <- row_h * row_i, with correct phase update.
// Implements AG rowsum(h, i): used during measurement collapse.
void StabilizerState::rowsum(int h, int i) {
    // Accumulate total phase exponent (in units of i) across all qubits
    int exp = 0;
    // Add phase contributions from the two input rows
    exp += 2 * (int)phases[i]; // existing phase of row i: +1 -> 0, -1 -> 2 (in units of i^2)
    exp += 2 * (int)phases[h]; // existing phase of row h
    for (int q = 0; q < nqubits; ++q) {
        exp += pauli_phase(x_bits[q][i], z_bits[q][i], x_bits[q][h], z_bits[q][h]);
    }
    // exp mod 4: result is 0 -> phase +1 (false), 2 -> phase -1 (true)
    exp = ((exp % 4) + 4) % 4;
    // assert(exp == 0 || exp == 2); // must be real for a valid stabilizer product
    phases[h] = (exp == 2);

    // Update Pauli content: XOR the bit columns
    for (int q = 0; q < nqubits; ++q) {
        if (x_bits[q][i]) x_bits[q].flip(h);
        if (z_bits[q][i]) z_bits[q].flip(h);
    }
}

// Project onto Z_q = (-1)^outcome. Assumes x_bits[q] is not all-zero (random outcome).
// Mutates this state in place.
void StabilizerState::project_z(int q, bool outcome) {
    int p = find_x_pivot(q);
    // assert(p != -1);

    // Rowsum every other generator that anticommutes with Z_q into row p first,
    // so that after we overwrite row p, all others commute with the new stabilizer.
    for (int r = 0; r < nqubits; ++r) {
        if (r != p && x_bits[q][r]) {
            rowsum(r, p); // row r <- row_r * row_p, with phase update
        }
    }

    // Overwrite row p with Z_q (phase = outcome)
    for (int i = 0; i < nqubits; ++i) {
        x_bits[i][p] = false;
        z_bits[i][p] = (i == q);
    }
    phases[p] = outcome;
}

// Call only when find_x_pivot(q) == -1 (outcome is deterministic).
// Returns the Z_q eigenvalue: false = +1, true = -1.
bool StabilizerState::z_eigenvalue(int q) const {
    return z_phase_by_ge(x_bits, z_bits, phases, nqubits, q);
}

void StabilizerStateSum::apply_gate(const Gate& gate) {
    /*
    Apply a gate to the StabilizerStateSum. Clifford gates get applied to each state in the sum.  Non-Clifford gates expand the sum.

    Args:
        gate (Gate&): The gate to apply.
    */

    const std::string name = gate.get_name();

    if ((name == "H" || name == "S" || name == "X" || name == "Y" ||
         name == "Z" || name == "SWAP") && gate.get_control_qubits().size() <= 1) {
        for (auto& state : states) {
            state.apply_gate(gate);
        }

    } else if (name == "T" && gate.get_control_qubits().empty()) {
        int q = gate.get_target_qubits()[0];

        // T = diag(1, e^{iπ/4}), so:
        //   T|ψ⟩ = P₀|ψ⟩  +  e^{iπ/4} · P₁|ψ⟩
        // where P₀, P₁ project onto Z_q = ±1 eigenstates.
        // P₁|ψ⟩ is also written as S · P₁|ψ⟩ / (up to overall phase) — but
        // here we do it directly: project, then absorb the e^{iπ/4} into coefficients.
        static const std::complex<double> t_phase =
            std::exp(std::complex<double>(0.0, M_PI / 4.0));

        std::vector<StabilizerState>        new_states;
        std::vector<std::complex<double>>   new_coeffs;
        new_states.reserve(states.size() * 2);
        new_coeffs.reserve(states.size() * 2);

        for (int k = 0; k < static_cast<int>(states.size()); ++k) {
            const StabilizerState& s = states[k];
            const std::complex<double>  c = coefficients[k];

            bool is_random = (s.find_x_pivot(q) != -1);

            if (!is_random) {
                // Deterministic outcome: only one branch survives.
                // Read the Z_q eigenvalue from the stabilizer tableau.
                bool outcome = s.z_eigenvalue(q); // see helper below
                if (!outcome) {
                    // Z_q = +1: T acts as identity, coefficient unchanged
                    new_states.push_back(s);
                    new_coeffs.push_back(c);
                } else {
                    // Z_q = -1: T acts as e^{iπ/4}
                    new_states.push_back(s);
                    new_coeffs.push_back(c * t_phase);
                }
            } else {
                // Random outcome: both branches exist.
                // Branch 0: Z_q = +1, coefficient unchanged
                // Branch 1: Z_q = -1, coefficient gets e^{iπ/4}
                // Both branches get 1/√2 because P_{0,1}|φ⟩ has norm 1/√2 for a random qubit.
                static const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
                {
                    StabilizerState s0 = s;
                    s0.project_z(q, false);
                    new_states.push_back(std::move(s0));
                    new_coeffs.push_back(c * inv_sqrt2);
                }
                {
                    StabilizerState s1 = s;
                    s1.project_z(q, true);
                    new_states.push_back(std::move(s1));
                    new_coeffs.push_back(c * t_phase * inv_sqrt2);
                }
            }
        }

        states       = std::move(new_states);
        coefficients = std::move(new_coeffs);

    } else {
        throw std::runtime_error("Gate " + name + " not supported for StabilizerStateSum yet");
    }
}