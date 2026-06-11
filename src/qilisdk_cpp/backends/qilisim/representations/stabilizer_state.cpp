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

StabilizerState::StabilizerState(int nqubits) : nqubits(nqubits) {
    /*
    Initialize a StabilizerState to the |0⟩^n state. This means that the Z stabilizers are Z_i for each qubit i, and there are no X stabilizers.

    Args:
        nqubits (int): The number of qubits in the state.

    Raises:
        std::invalid_argument: If nqubits is too large to fit in the tableau.
    */
    x_bits.resize(nqubits);
    z_bits.resize(nqubits);
    for (int i = 0; i < nqubits; ++i) {
        z_bits[i].set(i);
    }
    if (2 * nqubits + 1 > MAX_ROWS_STABILIZER) {
        throw std::invalid_argument("Number of qubits should be at most " + std::to_string(MAX_ROWS_STABILIZER / 2));
    }
}

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

static bool z_phase_by_ge(const std::vector<std::bitset<MAX_ROWS_STABILIZER>>& x, const std::vector<std::bitset<MAX_ROWS_STABILIZER>>& z, const std::bitset<MAX_ROWS_STABILIZER>& ph, int n, int target) {
    /*
    Gaussian elimination to find the Z_target eigenvalue in a stabilizer tableau.
    This is used during sampling when we have a deterministic outcome for a qubit.

    Args:
        x (std::vector<std::bitset<MAX_ROWS_STABILIZER>>&): The X part of the stabilizer tableau.
        z (std::vector<std::bitset<MAX_ROWS_STABILIZER>>&): The Z part of the stabilizer tableau.
        ph (std::bitset<MAX_ROWS_STABILIZER>&): The phases of the stabilizers.
        n (int): The number of qubits in the state.
        target (int): The index of the qubit to find the Z eigenvalue for.

    Returns:
        bool: The Z_target eigenvalue: false = +1, true = -1.
    */

    // Convert the tableau into a more convenient format for Gaussian elimination
    struct Row {
        std::vector<bool> xb;
        std::vector<bool> zb;
        bool ph;
    };
    std::vector<Row> rows(n);
    for (int r = 0; r < n; ++r) {
        rows[r].xb.resize(n);
        rows[r].zb.resize(n);
        for (int q = 0; q < n; ++q) {
            rows[r].xb[q] = bool(x[q][r]);
            rows[r].zb[q] = bool(z[q][r]);
        }
        rows[r].ph = bool(ph[r]);
    }

    // Helper to do row sum: row h <- row_h * row_g
    auto rowsum_local = [&](Row& h, const Row& g) {
        int contrib = 0;
        for (int q = 0; q < n; ++q) {
            if (h.xb[q] && g.zb[q]) {
                contrib ^= 1;
            }
            if (h.zb[q] && g.xb[q]) {
                contrib ^= 1;
            }
        }
        h.ph = h.ph ^ g.ph ^ bool(contrib);
        for (int q = 0; q < n; ++q) {
            h.xb[q] = h.xb[q] ^ g.xb[q];
            h.zb[q] = h.zb[q] ^ g.zb[q];
        }
    };

    // Set up the target row for elimination: we want to find a combination of the stabilizers that
    // give us Z_target, then read off the phase of that combination
    Row tgt;
    tgt.xb.assign(n, false);
    tgt.zb.assign(n, false);
    tgt.zb[target] = true;
    tgt.ph = false;

    // Do Gaussian elimination
    int pivot_row = 0;
    for (int bit = 0; bit < 2 * n && pivot_row < n; ++bit) {
        int q = bit < n ? bit : bit - n;
        bool is_x = bit < n;
        int pivot = -1;
        for (int r = pivot_row; r < n; ++r) {
            if (is_x ? rows[r].xb[q] : rows[r].zb[q]) {
                pivot = r;
                break;
            }
        }
        if (pivot < 0) {
            continue;
        }
        std::swap(rows[pivot], rows[pivot_row]);
        for (int r = 0; r < n; ++r) {
            if (r != pivot_row && (is_x ? rows[r].xb[q] : rows[r].zb[q])) {
                rowsum_local(rows[r], rows[pivot_row]);
            }
        }
        if (is_x ? tgt.xb[q] : tgt.zb[q])
            rowsum_local(tgt, rows[pivot_row]);
        ++pivot_row;
    }

    // Return the phase of the resulting row, which gives the Z_target eigenvalue
    return tgt.ph;
}

std::string StabilizerState::sample() const {
    /*
    Sample from the StabilizerState.
    This is done by iterating through each qubit and determining whether the outcome is random or deterministic.

    For random outcomes, we perform the necessary row operations to collapse the state and record the outcome.
    For deterministic outcomes, we use Gaussian elimination to find the Z eigenvalue.

    Returns:
        std::string: A bitstring representing the measurement outcome for each qubit.
    */

    // Create copies of the tableau data structures that we can modify during sampling
    auto x = x_bits;
    auto z = z_bits;
    auto ph = phases;

    // The bitstring to return
    std::string result(nqubits, '0');

    // For each qubit, determine if the outcome is random or deterministic
    for (int k = 0; k < nqubits; ++k) {
        int p = -1;
        for (int i = 0; i < nqubits; ++i) {
            if (x[k][i]) {
                p = i;
                break;
            }
        }

        // We have a random outcome if we have an X stabilizer that anticommutes with Z_k
        if (p >= 0) {
            // To collapse the state, we first need to make sure that the X stabilizer is the only one that anticommutes with Z_k.
            // We do this by rowsumming it into any other stabilizer that anticommutes with Z_k.
            for (int i = 0; i < nqubits; ++i) {
                if (i != p && x[k][i]) {
                    int contrib = 0;
                    for (int q = 0; q < nqubits; ++q) {
                        if (x[q][i] && z[q][p])
                            contrib ^= 1;
                        if (z[q][i] && x[q][p])
                            contrib ^= 1;
                    }
                    if ((bool)ph[p] ^ (bool)contrib)
                        ph.flip(i);
                    for (int q = 0; q < nqubits; ++q) {
                        if (x[q][p])
                            x[q].flip(i);
                        if (z[q][p])
                            z[q].flip(i);
                    }
                }
            }

            // Flip a coin to decide the outcome of the measurement for this qubit
            int b = rand() & 1;

            // Finally, we overwrite the pivot row to set Z_k = (-1)^b, which collapses the state. The phase of that row is set to b as well.
            for (int q = 0; q < nqubits; ++q) {
                x[q].reset(p);
                z[q].reset(p);
            }
            z[k].set(p);
            if (b) {
                ph.set(p);
            } else {
                ph.reset(p);
            }
            result[k] = b ? '1' : '0';

            // Otherwise the outcome is deterministic: we can read off the Z eigenvalue using Gaussian elimination on the tableau
        } else {
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
        double r = ((double)rand() / (RAND_MAX));
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

    // SWAP is easy, we just swap the corresponding rows in the tableau
    if (name == "SWAP") {
        int i = targets[0];
        int j = targets[1];
        std::swap(x_bits[i], x_bits[j]);
        std::swap(z_bits[i], z_bits[j]);

        // For gates without controls
    } else if (controls.empty()) {
        int i = targets[0];

        // H means X <-> Z and phase flips if we have both X and Z
        if (name == "H") {
            phases ^= (x_bits[i] & z_bits[i]);
            std::swap(x_bits[i], z_bits[i]);

            // S means X -> Y -> -X and Z -> Z, so phase flips if we have both X and Z (Y case) or if we have X but not Z (X case)
        } else if (name == "S") {
            phases ^= (x_bits[i] & z_bits[i]);
            z_bits[i] ^= x_bits[i];

            // X means Z -> -Z
        } else if (name == "X") {
            phases ^= z_bits[i];

            // Y means X -> -X, Z -> -Z, so phase flips if we have either but not both
        } else if (name == "Y") {
            phases ^= (x_bits[i] ^ z_bits[i]);

            // Z means X -> -X
        } else if (name == "Z") {
            phases ^= x_bits[i];
        }

        // Gates with a single control
    } else if (controls.size() == 1) {
        int i = controls[0], j = targets[0];

        // CNOT means X_j -> X_i X_j, Z_i -> Z_i Z_j, so phase flips if we have X on one and Z on the other (but not both)
        if (name == "X") {
            phases ^= (x_bits[i] & z_bits[j] & ~(x_bits[j] ^ z_bits[i]));
            x_bits[j] ^= x_bits[i];
            z_bits[i] ^= z_bits[j];

            // CZ means X_i -> X_i Z_j, X_j -> Z_i X_j, so phase flips if we have X on one and Z on the other (but not both)
        } else if (name == "Z") {
            phases ^= (x_bits[i] & x_bits[j] & (z_bits[i] ^ z_bits[j]));
            z_bits[j] ^= x_bits[i];
            z_bits[i] ^= x_bits[j];
        }

        // Gates with two controls (Toffoli family): called by StabilizerStateSum after branching on controls[0]=1,
        // so the effective operation is CNOT(controls[1], target)
    } else if (controls.size() == 2) {
        int i = controls[1], j = targets[0];

        if (name == "X") {
            phases ^= (x_bits[i] & z_bits[j] & ~(x_bits[j] ^ z_bits[i]));
            x_bits[j] ^= x_bits[i];
            z_bits[i] ^= z_bits[j];
        }
    }
}

int StabilizerState::find_x_pivot(int q) const {
    /*
    Find the pivot row for X_q, which is the unique row that has an X on qubit q (if it exists).

    Args:
        q (int): The index of the qubit to find the pivot for.

    Returns:
        int: The index of the pivot row, or -1 if there is no X stabilizer that anticommutes with Z_q (i.e. the outcome is deterministic).
    */
    for (int r = 0; r < nqubits; ++r) {
        if (x_bits[q][r]) {
            return r;
        }
    }
    return -1;
}

static int pauli_phase(bool x1, bool z1, bool x2, bool z2) {
    /*
    Computes the phase contribution when multiplying two Pauli operators on a single qubit.
    This is used during the rowsum operation to update the phase correctly.

    Args:
        x1 (bool): Whether the first operator has an X on this qubit.
        z1 (bool): Whether the first operator has a Z on this qubit.
        x2 (bool): Whether the second operator has an X on this qubit.
        z2 (bool): Whether the second operator has a Z on this qubit.

    Returns:
        int: An integer in {0, 1, 2, 3} representing multiples of i (i^n).
             0 means no phase, 2 means a -1 phase, and 1 or 3 would mean ±i (which should not happen for valid stabilizer products).
    */
    if (!x1 && !z1)
        return 0;  // I * anything = no phase
    if (x1 && z1)
        return z2 - x2;  // Y case: z2 - x2 in {-1, 0, 1} -> mod 4
    if (x1 && !z1)
        return z2 * (2 * x2 - 1);  // X case
    if (!x1 && z1)
        return x2 * (1 - 2 * z2);  // Z case
    return 0;
}

void StabilizerState::rowsum(int h, int i) {
    /*
    Perform the row sum operation on the tableau, which corresponds to
    multiplying the stabilizer in row h by the stabilizer in row i. This mutates the tableau in place.

    Args:
        h (int): The index of the row to be updated (row_h <- row_h * row_i).
        i (int): The index of the row to multiply by.
    */

    // Accumulate total phase exponent (in units of i) across all qubits
    int exp = 0;

    // Add phase contributions from the two input rows
    exp += 2 * (int)phases[i];  // existing phase of row i: +1 -> 0, -1 -> 2 (in units of i^2)
    exp += 2 * (int)phases[h];  // existing phase of row h
    for (int q = 0; q < nqubits; ++q) {
        exp += pauli_phase(x_bits[q][i], z_bits[q][i], x_bits[q][h], z_bits[q][h]);
    }

    // exp mod 4: result is 0 -> phase +1 (false), 2 -> phase -1 (true)
    exp = ((exp % 4) + 4) % 4;
    phases[h] = (exp == 2);

    // Update Pauli content: XOR the bit columns
    for (int q = 0; q < nqubits; ++q) {
        if (x_bits[q][i])
            x_bits[q].flip(h);
        if (z_bits[q][i])
            z_bits[q].flip(h);
    }
}

void StabilizerState::project_z(int q, bool outcome) {
    /*
    Project the state onto the Z_q = (-1)^outcome eigenspace.
    This is used during sampling when we have a random outcome for a qubit, and we need to collapse the state accordingly.

    Args:
        q (int): The index of the qubit to project.
        outcome (bool): The measurement outcome for Z_q: false = +1, true = -1.
    */
    int p = find_x_pivot(q);

    // Rowsum every other generator that anticommutes with Z_q into row p first,
    // so that after we overwrite row p, all others commute with the new stabilizer.
    for (int r = 0; r < nqubits; ++r) {
        if (r != p && x_bits[q][r]) {
            rowsum(r, p);  // row r <- row_r * row_p, with phase update
        }
    }

    // Overwrite row p with Z_q (phase = outcome)
    for (int i = 0; i < nqubits; ++i) {
        x_bits[i][p] = false;
        z_bits[i][p] = (i == q);
    }
    phases[p] = outcome;
}

bool StabilizerState::z_eigenvalue(int q) const {
    /*
    Find the Z_q eigenvalue using Gaussian elimination on the tableau.
    This is only valid to call when we have already determined that the outcome is
    deterministic (i.e. there are no X stabilizers that anticommute with Z_q).

    Args:
        q (int): The index of the qubit to find the Z eigenvalue for.

    Returns:
        bool: The Z_q eigenvalue: false = +1, true = -1.
    */
    return z_phase_by_ge(x_bits, z_bits, phases, nqubits, q);
}

void StabilizerStateSum::apply_gate(const Gate& gate) {
    /*
    Apply a gate to the StabilizerStateSum.
    Clifford gates get applied to each state in the sum.
    Non-Clifford gates expand the sum by branching on qubit eigenstates.

    Args:
        gate (Gate&): The gate to apply.
    */
    const std::string name = gate.get_name();
    const auto& controls = gate.get_control_qubits();
    const auto& targets = gate.get_target_qubits();

    // If it's a clifford gate with at most one control, we can just apply it to each state in the sum
    if ((name == "H" || name == "S" || name == "X" || name == "Y" || name == "Z" || name == "SWAP") && controls.size() <= 1) {
        for (auto& state : states) {
            state.apply_gate(gate);
        }
        return;
    }

    // For non-Clifford gates we branch on the eigenvalue of a qubit
    int branch_qubit = -1;
    std::function<void(StabilizerState&, std::complex<double>&)> on_zero;
    std::function<void(StabilizerState&, std::complex<double>&)> on_one;

    // T gates: branch on the target qubit. On the 0 branch we do nothing, on the 1 branch we apply a phase of e^{iπ/4}
    if (name == "T" && controls.empty()) {
        static const std::complex<double> t_phase = std::exp(std::complex<double>(0.0, M_PI / 4.0));
        branch_qubit = targets[0];
        on_zero = [](StabilizerState&, std::complex<double>&) {};
        on_one = [](StabilizerState&, std::complex<double>& c) { c *= t_phase; };

        // Toffoli gates: branch on the first control. On the 0 branch we do nothing, on the 1 branch we apply a CNOT from the second control to the target
    } else if (name == "X" && controls.size() == 2) {
        branch_qubit = controls[0];
        on_zero = [](StabilizerState&, std::complex<double>&) {};
        on_one = [&gate](StabilizerState& s, std::complex<double>&) { s.apply_gate(gate); };

        // Otherwise not supported
    } else {
        throw std::runtime_error("Gate " + name + " not supported for StabilizerStateSum yet");
    }

    // Branch: for each current (state, coeff), split on the eigenvalue of branch_qubit
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    std::vector<StabilizerState> new_states;
    std::vector<std::complex<double>> new_coeffs;
    new_states.reserve(states.size() * 2);
    new_coeffs.reserve(states.size() * 2);

    // Need to branch each state
    for (int k = 0; k < static_cast<int>(states.size()); ++k) {
        // Check if the outcome of the branch_qubit is deterministic or not
        bool is_random = (states[k].find_x_pivot(branch_qubit) != -1);

        // Deterministic outcome: one branch, choose the right callback
        if (!is_random) {
            bool outcome = states[k].z_eigenvalue(branch_qubit);
            StabilizerState new_s = states[k];
            std::complex<double> new_c = coefficients[k];
            if (!outcome) {
                on_zero(new_s, new_c);
            } else {
                on_one(new_s, new_c);
            }
            new_states.push_back(std::move(new_s));
            new_coeffs.push_back(new_c);

            // Otherwise if it's a random outcome we make two branches, each weighted by 1/√2
        } else {
            // Branch on zero
            StabilizerState s0 = states[k];
            s0.project_z(branch_qubit, false);
            std::complex<double> c0 = coefficients[k] * inv_sqrt2;
            on_zero(s0, c0);
            new_states.push_back(std::move(s0));
            new_coeffs.push_back(c0);

            // Branch on one
            StabilizerState s1 = states[k];
            s1.project_z(branch_qubit, true);
            std::complex<double> c1 = coefficients[k] * inv_sqrt2;
            on_one(s1, c1);
            new_states.push_back(std::move(s1));
            new_coeffs.push_back(c1);
        }
    }

    // Copy the new states and coeffs back over
    states = std::move(new_states);
    coefficients = std::move(new_coeffs);
}