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
#include "../../../libs/pybind.h"

#include <iostream>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>

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
        os << "Term " << i << ":\n";
        const int n = sss.get_states()[i].get_nqubits();
        auto truncate = [n](const std::bitset<MAX_ROWS_STABILIZER>& bs) {
            auto s = bs.to_string().substr(MAX_ROWS_STABILIZER - n);
            std::reverse(s.begin(), s.end());
            return s;
        };
        os << "  Coefficient: " << sss.get_coefficients()[i] << " (norm " << std::norm(sss.get_coefficients()[i]) << ")\n";
        os << "  X stabilizers:\n";
        for (int j = 0; j < n; ++j) {
            os << "    For qubit " << j << ": " << truncate(sss.get_states()[i].get_x_bits()[j]) << "\n";
        }
        os << "  Z stabilizers:\n";
        for (int j = 0; j < n; ++j) {
            os << "    For qubit " << j << ": " << truncate(sss.get_states()[i].get_z_bits()[j]) << "\n";
        }
        os << "  Phases: " << truncate(sss.get_states()[i].get_phases()) << "\n";
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
        int exp = 2 * (int)g.ph + 2 * (int)h.ph;
        for (int q = 0; q < n; ++q) {
            bool x1 = g.xb[q], z1 = g.zb[q], x2 = h.xb[q], z2 = h.zb[q];
            if (!x1 && !z1) {
                continue;
            } else if (x1 && z1) {
                exp += (int)z2 - (int)x2;
            } else if (x1 && !z1) {
                exp += (int)z2 * (2 * (int)x2 - 1);
            } else {
                exp += (int)x2 * (1 - 2 * (int)z2);
            }
        }
        exp = ((exp % 4) + 4) % 4;
        h.ph = (exp == 2);
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
            // Rowsum every other generator that anticommutes with Z_k into row p to isolate the pivot
            for (int i = 0; i < nqubits; ++i) {
                if (i != p && x[k][i]) {
                    int exp = 2 * (int)(bool)ph[p] + 2 * (int)(bool)ph[i];
                    for (int q = 0; q < nqubits; ++q) {
                        bool x1 = (bool)x[q][p], z1 = (bool)z[q][p];
                        bool x2 = (bool)x[q][i], z2 = (bool)z[q][i];
                        if (!x1 && !z1)
                            continue;
                        else if (x1 && z1)
                            exp += (int)z2 - (int)x2;
                        else if (x1 && !z1)
                            exp += (int)z2 * (2 * (int)x2 - 1);
                        else
                            exp += (int)x2 * (1 - 2 * (int)z2);
                    }
                    exp = ((exp % 4) + 4) % 4;
                    if (exp == 2) {
                        ph.set(i);
                    } else {
                        ph.reset(i);
                    }
                    for (int q = 0; q < nqubits; ++q) {
                        if (x[q][p]) {
                            x[q].flip(i);
                        }
                        if (z[q][p]) {
                            z[q].flip(i);
                        }
                    }
                }
            }

            // Flip a coin to decide the outcome of the measurement for this qubit
            int b = std::uniform_int_distribution<int>(0, 1)(rng);

            // Overwrite the pivot row to set Z_k = (-1)^b, collapsing the state
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

            // Otherwise the outcome is deterministic: read off the Z eigenvalue via Gaussian elimination
        } else {
            result[k] = z_phase_by_ge(x, z, ph, nqubits, k) ? '1' : '0';
        }
    }

    return result;
}

std::complex<double> StabilizerState::amplitude(const std::string& b) const {
    /*
    Compute the amplitude <b|s> for a given computational basis bitstring b.

    Args:
        b (std::string): The bitstring to compute the amplitude for.

    Returns:
        std::complex<double>: The complex amplitude <b|s>.
    */

    // Pull the n stabilizer generators into a row-major scratch tableau we can reduce in place
    const int n = nqubits;
    std::vector<std::vector<uint8_t>> X(n, std::vector<uint8_t>(n, 0)), Z(n, std::vector<uint8_t>(n, 0));
    std::vector<int> p4(n, 0);
    for (int r = 0; r < n; ++r) {
        for (int q = 0; q < n; ++q) {
            X[r][q] = x_bits[q][r];
            Z[r][q] = z_bits[q][r];
        }
        p4[r] = phases[r] ? 2 : 0;
    }

    // Helper to compute the phase when multiplying two Paulis together, given their X and Z bits
    auto mul_phase = [](int x1, int z1, int x2, int z2) {
        int m = x1 * z1 + x2 * z2 + 2 * z1 * x2 - ((x1 ^ x2) * (z1 ^ z2));
        return ((m % 4) + 4) % 4;
    };

    // Helper to do row sum: row h <- row_h * row_i, where * is the Pauli product. This updates the tableau in place and tracks the phase
    auto rowmul = [&](int dst, int src) {
        int acc = p4[dst] + p4[src];
        for (int q = 0; q < n; ++q) {
            acc += mul_phase(X[dst][q], Z[dst][q], X[src][q], Z[src][q]);
            X[dst][q] ^= X[src][q];
            Z[dst][q] ^= Z[src][q];
        }
        p4[dst] = ((acc % 4) + 4) % 4;
    };

    // Gaussian-eliminate the X block into reduced row echelon form
    std::vector<int> pivot_qubit;
    int rank = 0;
    for (int q = 0; q < n && rank < n; ++q) {
        int sel = -1;
        for (int r = rank; r < n; ++r) {
            if (X[r][q]) {
                sel = r;
                break;
            }
        }
        if (sel < 0)
            continue;
        std::swap(X[sel], X[rank]);
        std::swap(Z[sel], Z[rank]);
        std::swap(p4[sel], p4[rank]);
        for (int r = 0; r < n; ++r) {
            if (r != rank && X[r][q])
                rowmul(r, rank);
        }
        pivot_qubit.push_back(q);
        ++rank;
    }
    const int r = rank;

    // Rows [r, n) are now pure Z. Reduce their Z block to RREF so each fixes one (non-X-pivot) qubit
    std::vector<uint8_t> is_xpivot(n, 0);
    for (int qa : pivot_qubit) {
        is_xpivot[qa] = 1;
    }
    std::vector<int> zpivot_qubit(n - r, -1);
    int zrank = 0;
    for (int q = 0; q < n && zrank < n - r; ++q) {
        // Pure-Z rows carry no Z on X-pivot qubits (they commute)
        if (is_xpivot[q]) {
            continue;
        }
        int sel = -1;
        for (int row = r + zrank; row < n; ++row) {
            if (Z[row][q]) {
                sel = row;
                break;
            }
        }
        if (sel < 0) {
            continue;
        }
        std::swap(X[sel], X[r + zrank]);
        std::swap(Z[sel], Z[r + zrank]);
        std::swap(p4[sel], p4[r + zrank]);
        for (int row = r; row < n; ++row) {
            if (row != r + zrank && Z[row][q]) {
                rowmul(row, r + zrank);
            }
        }
        zpivot_qubit[zrank] = q;
        ++zrank;
    }

    // Build a concrete reference basis state in the support: X-pivot qubits are free (set to 0),
    // and each pure-Z row fixes its Z-pivot qubit to that row's sign
    std::vector<uint8_t> ref(n, 0);
    for (int row = r; row < n; ++row) {
        int d = zpivot_qubit[row - r];
        if (d < 0) {
            continue;
        }
        ref[d] = (p4[row] == 2) ? 1 : 0;
    }

    // Determine which X-pivot generators map the reference toward b (free bits = b on pivot qubits),
    // then check b is actually reachable (i.e. lies in the support)
    std::vector<int> subset;
    for (int a = 0; a < r; ++a) {
        if ((b[pivot_qubit[a]] == '1') != (ref[pivot_qubit[a]] != 0)) {
            subset.push_back(a);
        }
    }
    std::vector<uint8_t> target = ref;
    for (int a : subset) {
        for (int q = 0; q < n; ++q) {
            target[q] ^= X[a][q];
        }
    }
    for (int q = 0; q < n; ++q) {
        if ((b[q] == '1') != (target[q] != 0)) {
            return {0.0, 0.0};
        }
    }

    // Multiply the selected X-pivot generators together into a single Pauli, tracking the phase
    std::vector<uint8_t> Xp(n, 0), Zp(n, 0);
    int pw = 0;
    for (int a : subset) {
        for (int q = 0; q < n; ++q) {
            pw += mul_phase(Xp[q], Zp[q], X[a][q], Z[a][q]);
            Xp[q] ^= X[a][q];
            Zp[q] ^= Z[a][q];
        }
        pw += p4[a];
    }
    pw = ((pw % 4) + 4) % 4;

    // <b| (selected product) |ref> = i^pw * prod_q <b_q| tau(Xp,Zp) |ref_q>
    std::complex<double> amp(1.0, 0.0);
    for (int q = 0; q < n; ++q) {
        int x = Xp[q];
        int zz = Zp[q];
        int rq = ref[q];
        if (x == 0) {
            if (zz && rq) {
                amp = -amp;  // Z|1> = -|1>
            }
        } else if (zz) {
            amp *= (rq == 0) ? std::complex<double>(0.0, 1.0)    // <1|Y|0> = i
                             : std::complex<double>(0.0, -1.0);  // <0|Y|1> = -i
        }
    }
    static const std::complex<double> ipow[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    amp *= ipow[pw];

    // Apply the 2^(-r/2) magnitude of the equal superposition over the support.
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    for (int a = 0; a < r; ++a)
        amp *= inv_sqrt2;

    return amp;
}

std::complex<double> StabilizerStateSum::amplitude(const std::string& b) const {
    /*
    Compute the full amplitude <b|ψ> = sum_i c_i <b|s_i> for a given bitstring b.

    Args:
        b (std::string): The bitstring to compute the amplitude for.

    Returns:
        std::complex<double>: The total complex amplitude for b across all terms.
    */
    std::complex<double> total = 0.0;
    for (size_t i = 0; i < states.size(); ++i) {
        total += coefficients[i] * states[i].amplitude(b);
    }
    return total;
}

std::map<std::string, int> StabilizerStateSum::sample(int nshots) const {
    /*
    Sample from the StabilizerStateSum. This is done by first picking a StabilizerState
    according to the probabilities given by the coefficients, then sampling from that state.

    Importance sampling: pick states proportionally to |c_i| so that we naturally focus
    on bitstrings with the highest probability. For each sampled bitstring we compute the
    exact amplitude A(b) = sum_i c_i <b|s_i> using amplitude(), so pre_samples[b] is always
    exact (no Monte Carlo noise in the amplitude itself). We just need enough shots to discover
    all bitstrings with significant probability.

    Args:
        nshots (int): The number of samples to draw.

    Returns:
        std::map<std::string, int>: A map from bitstrings to counts.
    */

    // We add an extra factor to try make sure we find all interesting bitstrings
    // Could be made a seperate parameter, but the user can increase this by increasing the nshots anyway
    const size_t M = states.size();
    const int extra_sample_factor = 10;

    // Build a cumulative weight vector for efficient importance sampling
    std::vector<double> cum_weights(M);
    double total_weight = 0.0;
    for (size_t i = 0; i < M; ++i) {
        total_weight += std::abs(coefficients[i]);
        cum_weights[i] = total_weight;
    }

    // Try to find all interesting bitstrings
    std::uniform_real_distribution<double> weight_dist(0.0, total_weight);
    std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
    std::map<std::string, std::complex<double>> pre_samples;
    for (int shot = 0; shot < nshots * extra_sample_factor; ++shot) {
        // Pick a state with probability proportional to |c_i|
        double r = weight_dist(rng);
        size_t index = static_cast<size_t>(std::lower_bound(cum_weights.begin(), cum_weights.end(), r) - cum_weights.begin());
        if (index >= M)
            index = M - 1;

        // Sample a bitstring from that state
        const std::string sample_str = states[index].sample();

        // Compute the exact amplitude
        if (!pre_samples.count(sample_str)) {
            pre_samples[sample_str] = amplitude(sample_str);
        }
    }

    // Convert to probabilities
    std::map<std::string, double> probabilities;
    double total = 0.0;
    for (const auto& [sample_str, amp] : pre_samples) {
        probabilities[sample_str] = std::norm(amp);
        total += probabilities[sample_str];
    }

    // If the total doesn't cover enough, raise a warning
    const float coverage_threshold = 0.95;
    if (total < coverage_threshold) {
        warning("Sampled probability mass is only " + std::to_string(total) + " (try increasing nshots to find more of the support)");
    }

    // Normalize probabilities
    for (auto& [sample_str, prob] : probabilities) {
        prob /= total;
    }

    // Build cumulative distribution
    std::vector<std::pair<double, std::string>> cdf;
    cdf.reserve(probabilities.size());
    double so_far = 0.0;
    for (const auto& [sample_str, prob] : probabilities) {
        so_far += prob;
        cdf.emplace_back(so_far, sample_str);
    }

    // Sample from this distribution
    std::map<std::string, int> sample_counts;
    for (int shot = 0; shot < nshots; ++shot) {
        double r = unit_dist(rng);
        auto it = std::lower_bound(cdf.begin(), cdf.end(), std::make_pair(r, std::string{}), [](const auto& a, const auto& b) { return a.first < b.first; });
        const std::string& s = (it != cdf.end()) ? it->second : cdf.back().second;
        sample_counts[s]++;
    }

    // Return the counts
    return sample_counts;
}

std::complex<double> StabilizerState::apply_gate(const Gate& gate, bool track_global_phase) {
    /*
    Apply a gate to the StabilizerState. This uses the standard tableau update rules for Clifford gates. Non-Clifford gates are not supported here.

    Args:
        gate (Gate&): The gate to apply.
        track_global_phase (bool): Whether to compute the global phase factor the tableau update drops.

    Returns:
        std::complex<double>: The global phase factor introduced by the gate on this particular state.
            A stabilizer tableau only represents a state up to a global phase, so the update rules
            silently drop the phase a gate imparts (e.g. S|1>=i|1>, Z|1>=-|1>, but also Pauli/Hadamard
            gates acting on a qubit that is in superposition, e.g. X|->=-|->, X|Y+>=i|Y->). Within a
            single stabilizer state that phase is unobservable, but as a term of a StabilizerStateSum
            it is a relative phase between terms and must be folded back into the coefficient. The
            caller must multiply its coefficient by this returned value. When track_global_phase is
            false (a lone state, where the phase cannot matter) we skip the computation and return 1.
    */

    // Get info about the gate
    const std::string name = gate.get_name();
    const auto& controls = gate.get_control_qubits();
    const auto& targets = gate.get_target_qubits();

    // Snapshot the pre-gate state so we can recover the global phase the tableau update drops.
    std::unique_ptr<StabilizerState> before;
    if (track_global_phase) {
        before = std::make_unique<StabilizerState>(*this);
    }

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
    } else {
        throw std::invalid_argument("Unsupported gate with name " + name + " and controls " + std::to_string(controls.size()));
    }

    // Recover the global phase the tableau update dropped (only when it can matter).
    if (!track_global_phase) {
        return 1.0;
    }
    return before->dropped_global_phase(*this, gate);
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

std::string StabilizerState::representative() const {
    /*
    Return a computational basis bitstring that is guaranteed to have nonzero amplitude in this
    state. This mirrors sample() but always chooses outcome 0 for every random (X-pivot) qubit,
    so the result is deterministic and matches the |0>-at-every-random-qubit reference that
    amplitude() assigns a positive real amplitude to.

    Returns:
        std::string: A bitstring in the support of this state.
    */
    auto x = x_bits;
    auto z = z_bits;
    auto ph = phases;
    std::string b(nqubits, '0');
    for (int k = 0; k < nqubits; ++k) {
        int p = -1;
        for (int i = 0; i < nqubits; ++i) {
            if (x[k][i]) {
                p = i;
                break;
            }
        }
        if (p >= 0) {
            // Isolate the pivot, exactly as sample() does, then collapse to the |0> outcome.
            for (int i = 0; i < nqubits; ++i) {
                if (i != p && x[k][i]) {
                    int exp = 2 * (int)(bool)ph[p] + 2 * (int)(bool)ph[i];
                    for (int q = 0; q < nqubits; ++q) {
                        bool x1 = (bool)x[q][p], z1 = (bool)z[q][p];
                        bool x2 = (bool)x[q][i], z2 = (bool)z[q][i];
                        if (!x1 && !z1)
                            continue;
                        else if (x1 && z1)
                            exp += (int)z2 - (int)x2;
                        else if (x1 && !z1)
                            exp += (int)z2 * (2 * (int)x2 - 1);
                        else
                            exp += (int)x2 * (1 - 2 * (int)z2);
                    }
                    exp = ((exp % 4) + 4) % 4;
                    if (exp == 2)
                        ph.set(i);
                    else
                        ph.reset(i);
                    for (int q = 0; q < nqubits; ++q) {
                        if (x[q][p])
                            x[q].flip(i);
                        if (z[q][p])
                            z[q].flip(i);
                    }
                }
            }
            for (int q = 0; q < nqubits; ++q) {
                x[q].reset(p);
                z[q].reset(p);
            }
            z[k].set(p);
            ph.reset(p);
            b[k] = '0';
        } else {
            b[k] = z_phase_by_ge(x, z, ph, nqubits, k) ? '1' : '0';
        }
    }
    return b;
}

// The single-qubit Clifford matrix element <r|U|c> keyed by the gate's name. We key off the name
// (not gate.get_base_matrix()) so this stays consistent with the tableau update, which is itself
// name-driven, and so it is correct even for the internally-constructed flip gate the arbitrary
// single-qubit path reuses under the name "X".
static std::complex<double> clifford_elem(const std::string& name, int r, int c) {
    /*
    Compute the matrix element <r|U|c> for a single-qubit Clifford gate U, where r and c are 0 or 1.

    Args:
        name (std::string): The name of the Clifford gate ("H", "X", "Y", "Z", "S").
        r (int): The row index (0 or 1).
        c (int): The column index (0 or 1).

    Returns:
        std::complex<double>: The complex matrix element <r|U|c>.
    */
    using cd = std::complex<double>;
    const double s = 1.0 / std::sqrt(2.0);
    if (name == "H") {
        return cd((r == 1 && c == 1) ? -s : s, 0.0);
    } else if (name == "X") {
        return (r != c) ? cd(1.0, 0.0) : cd(0.0, 0.0);
    } else if (name == "Y") {
        if (r == 1 && c == 0)
            return cd(0.0, 1.0);
        if (r == 0 && c == 1)
            return cd(0.0, -1.0);
        return cd(0.0, 0.0);
    } else if (name == "Z") {
        return (r == c) ? cd(r == 0 ? 1.0 : -1.0, 0.0) : cd(0.0, 0.0);
    } else if (name == "S") {
        if (r == 0 && c == 0)
            return cd(1.0, 0.0);
        if (r == 1 && c == 1)
            return cd(0.0, 1.0);
        return cd(0.0, 0.0);
    }

    // Fall back to the identity
    return (r == c) ? cd(1.0, 0.0) : cd(0.0, 0.0);
}

std::complex<double> StabilizerState::matrix_element(const Gate& gate, const std::string& b) const {
    /*
    Compute <b| G |this>, the exact amplitude of basis state b in G|this>, using this state's
    amplitudes. G is restricted to the Clifford gates the tableau understands. For permutation /
    diagonal multi-qubit gates this is a single amplitude lookup; for a single-qubit gate it is the
    2-term contraction sum_a <b_t|U|a> * amplitude(b with qubit t = a).

    Args:
        gate (Gate&): The gate G.
        b (std::string): The output basis state bitstring.

    Returns:
        std::complex<double>: The amplitude <b|G|this>.
    */
    const std::string name = gate.get_name();
    const auto& controls = gate.get_control_qubits();
    const auto& targets = gate.get_target_qubits();

    // SWAP is easy, we just swap the corresponding bits in b and look up the amplitude
    if (name == "SWAP") {
        std::string bp = b;
        std::swap(bp[targets[0]], bp[targets[1]]);
        return amplitude(bp);
    }

    // For single-qubit gates, we need to sum over the two possible input states of the target qubit
    if (controls.empty() && targets.size() == 1) {
        int t = targets[0];
        int bt = (b[t] == '1') ? 1 : 0;
        std::complex<double> sum = 0.0;
        for (int a = 0; a < 2; ++a) {
            std::complex<double> u = clifford_elem(name, bt, a);
            if (std::abs(u) < 1e-15) {
                continue;
            }
            std::string bp = b;
            bp[t] = a ? '1' : '0';
            sum += u * amplitude(bp);
        }
        return sum;
    }

    // For gates with one control
    if (controls.size() == 1) {
        int c = controls[0];
        int t = targets[0];

        // CNOT: <b|CNOT|psi> = psi(CNOT b) since CNOT is a self-inverse permutation
        if (name == "X") {
            std::string bp = b;
            if (b[c] == '1') {
                bp[t] = (b[t] == '1') ? '0' : '1';
            }
            return amplitude(bp);

            // CZ is diagonal with eigenvalue (-1)^(b_c * b_t)
        } else if (name == "Z") {
            std::complex<double> v = amplitude(b);
            if (b[c] == '1' && b[t] == '1') {
                v = -v;
            }
            return v;
        }
    }

    // Toffoli flips the target iff both controls are 1
    if (controls.size() == 2 && name == "X") {
        int c1 = controls[0], c2 = controls[1], t = targets[0];
        std::string bp = b;
        if (b[c1] == '1' && b[c2] == '1') {
            bp[t] = (b[t] == '1') ? '0' : '1';
        }
        return amplitude(bp);
    }

    // Fall back to the identity
    return amplitude(b);
}

std::complex<double> StabilizerState::dropped_global_phase(const StabilizerState& after, const Gate& gate) const {
    /*
    Recover the global phase the tableau update for `gate` dropped. *this is the pre-gate state and
    `after` is the post-gate tableau. We pick a reference basis state b in the support of `after`,
    then compare the true amplitude <b|G|this> against the amplitude amplitude()'s fixed global-phase
    convention assigns to `after`. Their ratio is the unit-modulus phase the coefficient must absorb.

    Args:
        after (StabilizerState&): The post-gate state.
        gate (Gate&): The gate that was applied.

    Returns:
        std::complex<double>: The dropped global phase (unit modulus), or 1 if it cannot be determined.
    */
    const std::string b = after.representative();
    const std::complex<double> conv = after.amplitude(b);
    if (std::abs(conv) < 1e-12) {
        return {1.0, 0.0};
    }
    const std::complex<double> truth = matrix_element(gate, b);
    std::complex<double> g = truth / conv;
    const double m = std::abs(g);
    return (m > 1e-12) ? g / m : std::complex<double>(1.0, 0.0);
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
        // The dropped global phase is a relative phase only when there is more than one term
        const bool track = states.size() > 1;
        for (size_t k = 0; k < states.size(); ++k) {
            coefficients[k] *= states[k].apply_gate(gate, track);
        }
        return;
    }

    // Each callback receives the projected StabilizerState and its coefficient, and returns a
    // StabilizerStateSum whose terms are all added into the new sum
    using BranchFn = std::function<StabilizerStateSum(const StabilizerState&, std::complex<double>)>;
    int branch_qubit = -1;
    BranchFn on_zero, on_one;

    // T gates: branch on the target qubit. On the 0 branch we do nothing, on the 1 branch we apply a phase of e^{iπ/4}
    if (name == "T" && controls.empty()) {
        static const std::complex<double> t_phase = std::exp(std::complex<double>(0.0, M_PI / 4.0));
        branch_qubit = targets[0];
        on_zero = [](const StabilizerState& s, std::complex<double> c) { return StabilizerStateSum(s.get_nqubits(), {s}, {c}); };
        on_one = [](const StabilizerState& s, std::complex<double> c) { return StabilizerStateSum(s.get_nqubits(), {s}, {c * t_phase}); };

        // Toffoli gates: branch on the first control. On the 0 branch we do nothing, on the 1 branch we apply a CNOT from the second control to the target
    } else if (name == "X" && controls.size() == 2) {
        branch_qubit = controls[0];
        on_zero = [](const StabilizerState& s, std::complex<double> c) { return StabilizerStateSum(s.get_nqubits(), {s}, {c}); };
        on_one = [&gate](const StabilizerState& s, std::complex<double> c) {
            StabilizerState s1 = s;
            auto phase = s1.apply_gate(gate);
            return StabilizerStateSum(s.get_nqubits(), {s1}, {c * phase});
        };

        // Arbitrary single-qubit gate: the two columns of the 2x2 matrix give the output states for |0⟩ and |1⟩ inputs
        // U|0⟩ = u00|0⟩ + u10|1⟩  and  U|1⟩ = u01|0⟩ + u11|1⟩
    } else if (controls.empty() && targets.size() == 1) {
        branch_qubit = targets[0];
        SparseMatrix base = gate.get_base_matrix();
        std::complex<double> u00 = base.coeff(0, 0), u01 = base.coeff(0, 1);
        std::complex<double> u10 = base.coeff(1, 0), u11 = base.coeff(1, 1);
        Gate x_gate("X", base, {}, {branch_qubit}, {});
        const double tol = 1e-12;
        on_zero = [u00, u10, tol, x_gate](const StabilizerState& s, std::complex<double> c) {
            std::vector<StabilizerState> ss;
            std::vector<std::complex<double>> cs;
            if (std::abs(u00) > tol) {
                ss.push_back(s);
                cs.push_back(c * u00);
            }
            if (std::abs(u10) > tol) {
                StabilizerState s1 = s;
                std::complex<double> flip_phase = s1.apply_gate(x_gate);
                ss.push_back(std::move(s1));
                cs.push_back(c * u10 * flip_phase);
            }
            return StabilizerStateSum(s.get_nqubits(), ss, cs);
        };
        on_one = [u01, u11, tol, x_gate](const StabilizerState& s, std::complex<double> c) {
            std::vector<StabilizerState> ss;
            std::vector<std::complex<double>> cs;
            if (std::abs(u01) > tol) {
                StabilizerState s0 = s;
                std::complex<double> flip_phase = s0.apply_gate(x_gate);
                ss.push_back(std::move(s0));
                cs.push_back(c * u01 * flip_phase);
            }
            if (std::abs(u11) > tol) {
                ss.push_back(s);
                cs.push_back(c * u11);
            }
            return StabilizerStateSum(s.get_nqubits(), ss, cs);
        };

        // Otherwise not supported
    } else {
        throw std::runtime_error("Gate " + name + " not supported for StabilizerStateSum yet");
    }

    // Branch: for each current (state, coeff), split on the Z eigenvalue of branch_qubit
    std::vector<StabilizerState> new_states;
    std::vector<std::complex<double>> new_coeffs;
    new_states.reserve(states.size() * 2);
    new_coeffs.reserve(states.size() * 2);

    // Add the terms from a StabilizerStateSum into the new sum we're building after branching
    auto append_sss = [&](StabilizerStateSum sss) {
        for (auto& s : sss.get_states()) {
            new_states.push_back(std::move(s));
        }
        for (auto& c : sss.get_coefficients()) {
            new_coeffs.push_back(c);
        }
    };

    // Fo each state
    for (int k = 0; k < static_cast<int>(states.size()); ++k) {
        // Check if it's deterministic
        bool is_random = (states[k].find_x_pivot(branch_qubit) != -1);

        // Deterministic outcome: call the matching callback
        if (!is_random) {
            bool outcome = states[k].z_eigenvalue(branch_qubit);
            append_sss(outcome ? on_one(states[k], coefficients[k]) : on_zero(states[k], coefficients[k]));

            // Random outcome: project onto each Z eigenstate and call both callbacks
        } else {
            StabilizerState s0 = states[k];
            s0.project_z(branch_qubit, false);
            StabilizerState s1 = states[k];
            s1.project_z(branch_qubit, true);
            const std::string r0 = s0.representative();
            const std::string r1 = s1.representative();
            const std::complex<double> d0 = s0.amplitude(r0);
            const std::complex<double> d1 = s1.amplitude(r1);
            const std::complex<double> a0 = (std::abs(d0) > 1e-12) ? states[k].amplitude(r0) / d0 : std::complex<double>(0.0, 0.0);
            const std::complex<double> a1 = (std::abs(d1) > 1e-12) ? states[k].amplitude(r1) / d1 : std::complex<double>(0.0, 0.0);
            append_sss(on_zero(s0, coefficients[k] * a0));
            append_sss(on_one(s1, coefficients[k] * a1));
        }
    }

    // Copy the new states and coeffs back over
    states = std::move(new_states);
    coefficients = std::move(new_coeffs);

    // Simplify the state as much as possible
    combine_duplicates();
    truncate();
    normalize();
}

void StabilizerStateSum::truncate() {
    /*
    Truncate the StabilizerStateSum to keep only the top max_terms terms by coefficient
    magnitude. This is a heuristic to prevent exponential blowup of the number of terms
    after applying many non-Clifford gates.

    Also, check that there are not equal steady states, if so, combine their coefficients.
    */

    // Don't truncate if max_terms <= 0
    if (max_terms <= 0) {
        return;
    }

    // End early if we don't have more than max_terms terms
    if (states.size() <= max_terms) {
        return;
    }

    // Sort by coefficient magnitude and keep only the top max_terms terms
    std::vector<size_t> indices(states.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + max_terms, indices.end(), [this](size_t a, size_t b) { return std::norm(coefficients[a]) > std::norm(coefficients[b]); });
    indices.resize(max_terms);
    std::vector<StabilizerState> new_states;
    std::vector<std::complex<double>> new_coeffs;
    new_states.reserve(max_terms);
    new_coeffs.reserve(max_terms);
    for (size_t idx : indices) {
        new_states.push_back(std::move(states[idx]));
        new_coeffs.push_back(coefficients[idx]);
    }

    // Copy the new states and coeffs back over
    states = std::move(new_states);
    coefficients = std::move(new_coeffs);
}

void StabilizerStateSum::combine_duplicates() {
    /*
    Combine duplicate StabilizerStates in the sum by adding their coefficients.
    */

    // Prepare the arrays to fill
    std::vector<StabilizerState> new_states;
    std::vector<std::complex<double>> new_coeffs;
    new_states.reserve(states.size());
    new_coeffs.reserve(coefficients.size());

    // For each state, add it and hash it
    std::unordered_map<std::string, size_t> state_to_index;
    for (size_t i = 0; i < states.size(); ++i) {
        const StabilizerState& s = states[i];
        std::string key;
        for (const auto& xb : s.get_x_bits()) {
            key += xb.to_string();
        }
        for (const auto& zb : s.get_z_bits()) {
            key += zb.to_string();
        }
        key += s.get_phases().to_string();
        auto it = state_to_index.find(key);
        if (it != state_to_index.end()) {
            new_coeffs[it->second] += coefficients[i];
        } else {
            state_to_index[key] = new_states.size();
            new_states.push_back(s);
            new_coeffs.push_back(coefficients[i]);
        }
    }

    // Move the new states and coeffs back over
    states = std::move(new_states);
    coefficients = std::move(new_coeffs);
}

void StabilizerStateSum::normalize() {
    /*
    Normalize the StabilizerStateSum so that the sum of the squared magnitudes of the coefficients is 1.
    */
    double norm = 0.0;
    for (const auto& c : coefficients) {
        norm += std::norm(c);
    }
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (auto& c : coefficients) {
            c /= norm;
        }
    }
}