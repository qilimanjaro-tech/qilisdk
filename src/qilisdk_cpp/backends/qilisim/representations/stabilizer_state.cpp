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

std::string StabilizerState::sample() const {
    auto x = x_bits;
    auto z = z_bits;
    auto ph = phases;

    std::string result(nqubits, '0');

    for (int k = 0; k < nqubits; ++k) {
        // Find first generator anticommuting with Z_k (has X on qubit k).
        // Column-major: x[k][i] is a contiguous scan — cache-friendly.
        int p = -1;
        for (int i = 0; i < nqubits; ++i) {
            if (x[k][i]) { p = i; break; }
        }

        if (p >= 0) {
            // Random outcome: rowsum every other anticommuting generator into g_p.
            // g_i <- g_i * g_p eliminates the X_k component from g_i.
            for (int i = 0; i < nqubits; ++i) {
                if (i != p && x[k][i]) {
                    // Phase contribution: (-1)^(z_i · x_p) from commuting Z^z_i past X^x_p.
                    int contrib = 0;
                    for (int q = 0; q < nqubits; ++q)
                        if (z[q][i] && x[q][p]) contrib ^= 1;
                    // new_phase[i] = phase[i] ^ phase[p] ^ contrib
                    if ((bool)ph[p] ^ (bool)contrib) ph.flip(i);
                    // Update x and z: bit-i of each row (strided write — unavoidable in column-major).
                    for (int q = 0; q < nqubits; ++q) {
                        if (x[q][p]) x[q].flip(i);
                        if (z[q][p]) z[q].flip(i);
                    }
                }
            }

            // Replace g_p with (-1)^b * Z_k and record the coin-flip outcome.
            int b = rand() & 1;
            for (int q = 0; q < nqubits; ++q) { x[q].reset(p); z[q].reset(p); }
            z[k].set(p);
            if (b) ph.set(p); else ph.reset(p);
            result[k] = '0' + b;

        } else {
            // Deterministic: Gaussian elimination to find the phase of Z_k.
            // Build row-major copies so we can do standard GF(2) elimination.
            struct Row { std::vector<bool> xb, zb; bool ph; };
            std::vector<Row> rows(nqubits);
            for (int i = 0; i < nqubits; ++i) {
                rows[i].xb.resize(nqubits);
                rows[i].zb.resize(nqubits);
                for (int q = 0; q < nqubits; ++q) {
                    rows[i].xb[q] = (bool)x[q][i];
                    rows[i].zb[q] = (bool)z[q][i];
                }
                rows[i].ph = (bool)ph[i];
            }

            // Target starts as Z_k (phase 0).  After elimination it becomes identity;
            // the accumulated phase is the measurement outcome.
            Row target;
            target.xb.assign(nqubits, false);
            target.zb.assign(nqubits, false);
            target.zb[k] = true;
            target.ph = false;

            // rowsum: h <- h * g  (same phase formula as in the random branch)
            auto rs = [&](Row& h, const Row& g) {
                int contrib = 0;
                for (int q = 0; q < nqubits; ++q)
                    if (h.zb[q] && g.xb[q]) contrib ^= 1;
                h.ph = h.ph ^ g.ph ^ (bool)contrib;
                for (int q = 0; q < nqubits; ++q) {
                    h.xb[q] = h.xb[q] ^ (bool)g.xb[q];
                    h.zb[q] = h.zb[q] ^ (bool)g.zb[q];
                }
            };

            // Reduced row-echelon form over GF(2), x-bits first then z-bits.
            int pivot_row = 0;
            for (int bit = 0; bit < 2 * nqubits && pivot_row < nqubits; ++bit) {
                int q = bit < nqubits ? bit : bit - nqubits;
                bool is_x = bit < nqubits;

                int pivot = -1;
                for (int i = pivot_row; i < nqubits; ++i)
                    if (is_x ? rows[i].xb[q] : rows[i].zb[q]) { pivot = i; break; }
                if (pivot < 0) continue;

                std::swap(rows[pivot], rows[pivot_row]);

                for (int i = 0; i < nqubits; ++i)
                    if (i != pivot_row && (is_x ? rows[i].xb[q] : rows[i].zb[q]))
                        rs(rows[i], rows[pivot_row]);

                if (is_x ? target.xb[q] : target.zb[q])
                    rs(target, rows[pivot_row]);

                ++pivot_row;
            }

            // target.ph now equals the phase of Z_k in the stabilizer group.
            result[k] = target.ph ? '1' : '0';
        }
    }

    return result;
}

std::map<std::string, int> StabilizerStateSum::sample(int nshots) const {

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
        } else if (name == "Sdg") {
            phases ^= (x_bits[i] & ~z_bits[i]);
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
            phases ^= (x_bits[i] & x_bits[j] & ~(z_bits[i] ^ z_bits[j]));
            z_bits[j] ^= x_bits[i];
            z_bits[i] ^= x_bits[j];
        }
    }
}

void StabilizerStateSum::apply_gate(const Gate& gate) {
    const std::string name = gate.get_name();
    if (name == "H" || name == "S" || name == "Sdg" || name == "X" || name == "Y" || name == "Z" || name == "SWAP") {
        for (auto& state : states) {
            state.apply_gate(gate);
        }
    } else {
        throw std::runtime_error("Gate " + name + " not supported for StabilizerStateSum yet");
    }
}