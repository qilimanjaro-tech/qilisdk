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

#if defined(_OPENMP)
#include <omp.h>
#endif

// GCOV_EXCL_BR_START

StabilizerState::StabilizerState(int nqubits) : nqubits(nqubits) {
    /*
    Initialize a StabilizerState to the |0⟩^n state. This means that the Z stabilizers are Z_i for each qubit i, and there are no X stabilizers.

    Args:
        nqubits (int): The number of qubits in the state.

    Raises:
        std::invalid_argument: If nqubits is too large to fit in the tableau.
    */
    if (nqubits > MAX_ROWS_STABILIZER) {
        throw std::invalid_argument("Number of qubits should be at most " + std::to_string(MAX_ROWS_STABILIZER));
    }
    x_bits.resize(nqubits);
    z_bits.resize(nqubits);
    for (int i = 0; i < nqubits; ++i) {
        z_bits[i].set(i);
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

namespace {

// Reduces a stabilizer tableau to row-echelon form once so that the Z eigenvalue of any
// target qubit can be read off cheaply via query(). Callers that need many deterministic
// eigenvalues from the same tableau (sample(), representative()) build the solver once and
// query each target, turning the old per-qubit O(n^3) Gaussian elimination into a single
// O(n^3) reduction plus O(n^2) per query.
//
// Rows store the X/Z support over qubits as bitsets (n <= MAX_ROWS_STABILIZER/2), so the
// vector XORs that dominate elimination become single word-parallel bitset operations.
struct ZPhaseSolver {
    int n = 0;
    std::vector<std::bitset<MAX_ROWS_STABILIZER>> rx;  // X support per row, indexed by qubit
    std::vector<std::bitset<MAX_ROWS_STABILIZER>> rz;  // Z support per row, indexed by qubit
    std::vector<bool> rph;                             // phase bit per row
    std::vector<std::pair<bool, int>> pivots;          // (is_x, qubit) for each accepted pivot, in order

    // Symplectic phase exponent contributed by multiplying generator g (gx, gz) into h (hx, hz)
    static int phase_exp(const std::bitset<MAX_ROWS_STABILIZER>& gx, const std::bitset<MAX_ROWS_STABILIZER>& gz, const std::bitset<MAX_ROWS_STABILIZER>& hx, const std::bitset<MAX_ROWS_STABILIZER>& hz, int /*n*/) {
        const std::bitset<MAX_ROWS_STABILIZER> Yg = gx & gz;   // g has Y here: contributes z2 - x2
        const std::bitset<MAX_ROWS_STABILIZER> Xg = gx & ~gz;  // g has X here: contributes z2 * (2*x2 - 1)
        const std::bitset<MAX_ROWS_STABILIZER> Zg = ~gx & gz;  // g has Z here: contributes x2 * (1 - 2*z2)
        const std::bitset<MAX_ROWS_STABILIZER> plus = (Yg & hz & ~hx) | (Xg & hz & hx) | (Zg & hx & ~hz);
        const std::bitset<MAX_ROWS_STABILIZER> minus = (Yg & hx & ~hz) | (Xg & hz & ~hx) | (Zg & hx & hz);
        return static_cast<int>(plus.count()) - static_cast<int>(minus.count());
    }

    // row h <- row_h * row_g (Pauli product), updating its phase.
    void rowsum(int h, int g) {
        int exp = 2 * (int)rph[g] + 2 * (int)rph[h] + phase_exp(rx[g], rz[g], rx[h], rz[h], n);
        exp = ((exp % 4) + 4) % 4;
        rph[h] = (exp == 2);
        rx[h] ^= rx[g];
        rz[h] ^= rz[g];
    }

    // Transpose the tableau into rows and reduce to row-echelon form, recording the pivots.
    void build(const std::vector<std::bitset<MAX_ROWS_STABILIZER>>& x, const std::vector<std::bitset<MAX_ROWS_STABILIZER>>& z, const std::bitset<MAX_ROWS_STABILIZER>& ph, int n_) {
        n = n_;
        rx.assign(n, {});
        rz.assign(n, {});
        rph.assign(n, false);
        for (int r = 0; r < n; ++r) {
            for (int q = 0; q < n; ++q) {
                if (x[q][r])
                    rx[r].set(q);
                if (z[q][r])
                    rz[r].set(q);
            }
            rph[r] = bool(ph[r]);
        }

        pivots.clear();
        int pivot_row = 0;
        for (int bit = 0; bit < 2 * n && pivot_row < n; ++bit) {
            int q = bit < n ? bit : bit - n;
            bool is_x = bit < n;
            int pivot = -1;
            for (int r = pivot_row; r < n; ++r) {
                if (is_x ? rx[r][q] : rz[r][q]) {
                    pivot = r;
                    break;
                }
            }
            if (pivot < 0) {
                continue;
            }
            std::swap(rx[pivot], rx[pivot_row]);
            std::swap(rz[pivot], rz[pivot_row]);
            bool tmp = rph[pivot];
            rph[pivot] = rph[pivot_row];
            rph[pivot_row] = tmp;
            for (int r = 0; r < n; ++r) {
                if (r != pivot_row && (is_x ? rx[r][q] : rz[r][q])) {
                    rowsum(r, pivot_row);
                }
            }
            pivots.emplace_back(is_x, q);
            ++pivot_row;
        }
    }

    // Read off the Z_target eigenvalue: false = +1, true = -1.
    bool query(int target) const {
        std::bitset<MAX_ROWS_STABILIZER> tx, tz;
        tz.set(target);
        bool tph = false;
        // The rows are fully reduced (each pivot column is isolated), so a single ordered pass
        // clears every pivot column of the target without reintroducing earlier ones.
        for (size_t pr = 0; pr < pivots.size(); ++pr) {
            bool is_x = pivots[pr].first;
            int q = pivots[pr].second;
            if (is_x ? tx[q] : tz[q]) {
                int exp = 2 * (int)rph[pr] + 2 * (int)tph + phase_exp(rx[pr], rz[pr], tx, tz, n);
                exp = ((exp % 4) + 4) % 4;
                tph = (exp == 2);
                tx ^= rx[pr];
                tz ^= rz[pr];
            }
        }
        return tph;
    }

    // Expectation <P> of a bare Pauli P given by (tx, tz) on this pure stabilizer state:
    //   +1 if P lies in the stabilizer group, -1 if -P does, and 0 otherwise.
    // We reduce P against the same isolated pivots query() uses. If every X/Z component cancels then
    // P equals (-1)^phase times the product of the corresponding generators, so P (or -P) is in the
    // group and the accumulated phase is the sign. Any leftover support means P anticommutes with
    // some generator, so it averages to zero.
    int expectation_pauli(std::bitset<MAX_ROWS_STABILIZER> tx, std::bitset<MAX_ROWS_STABILIZER> tz) const {
        bool tph = false;
        for (size_t pr = 0; pr < pivots.size(); ++pr) {
            bool is_x = pivots[pr].first;
            int q = pivots[pr].second;
            if (is_x ? tx[q] : tz[q]) {
                int exp = 2 * (int)rph[pr] + 2 * (int)tph + phase_exp(rx[pr], rz[pr], tx, tz, n);
                exp = ((exp % 4) + 4) % 4;
                tph = (exp == 2);
                tx ^= rx[pr];
                tz ^= rz[pr];
            }
        }
        if (tx.any() || tz.any()) {
            return 0;
        }
        return tph ? -1 : 1;
    }
};

}  // namespace

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

    ZPhaseSolver solver;
    solver.build(x, z, ph, n);
    return solver.query(target);
}

std::string StabilizerState::sample() const {
    return sample(rng);
}

std::string StabilizerState::sample(std::mt19937_64& engine) const {
    /*
    Sample from the StabilizerState.
    This is done by iterating through each qubit and determining whether the outcome is random or deterministic.

    For random outcomes, we perform the necessary row operations to collapse the state and record the outcome.
    For deterministic outcomes, we use Gaussian elimination to find the Z eigenvalue.

    Args:
        engine (std::mt19937_64&): The random engine used for the coin flips on random outcomes.

    Returns:
        std::string: A bitstring representing the measurement outcome for each qubit.
    */

    // Create copies of the tableau data structures that we can modify during sampling
    auto x = x_bits;
    auto z = z_bits;
    auto ph = phases;

    // The bitstring to return
    std::string result(nqubits, '0');

    // A solver shared across all deterministic qubits. The tableau only changes in the random
    // (X-pivot) branch, so we reduce it once and reuse the reduction until it is invalidated.
    ZPhaseSolver solver;
    bool dirty = true;

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
            int b = std::uniform_int_distribution<int>(0, 1)(engine);

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
            dirty = true;  // the tableau changed, so any cached reduction is stale

            // Otherwise the outcome is deterministic: read off the Z eigenvalue via Gaussian elimination
        } else {
            if (dirty) {
                solver.build(x, z, ph, nqubits);
                dirty = false;
            }
            result[k] = solver.query(k) ? '1' : '0';
        }
    }

    return result;
}

namespace {

// Reduces a stabilizer tableau once so the amplitude <b|s> of any basis state b can be read off
// via query(b). The reduction (X-block then Z-block to RREF, plus the reference support state)
// depends only on the state, so callers that need several amplitudes of the same state
// (matrix_element's single-qubit contraction) build the reducer once and query each b.
struct AmplitudeReducer {
    int n = 0;
    int r = 0;  // X-pivot rank == log2 of the support size
    std::vector<std::vector<uint8_t>> X, Z;
    std::vector<int> p4;
    std::vector<int> pivot_qubit;
    std::vector<uint8_t> ref;  // reference support state (X-pivot qubits set to 0)

    // Phase (mod 4) of multiplying two Paulis together, given their X and Z bits.
    static int mul_phase(int x1, int z1, int x2, int z2) {
        int m = x1 * z1 + x2 * z2 + 2 * z1 * x2 - ((x1 ^ x2) * (z1 ^ z2));
        return ((m % 4) + 4) % 4;
    }

    void build(const std::vector<std::bitset<MAX_ROWS_STABILIZER>>& x_bits, const std::vector<std::bitset<MAX_ROWS_STABILIZER>>& z_bits, const std::bitset<MAX_ROWS_STABILIZER>& phases, int n_) {
        /*
        Build a reduced tableau and reference support state from the stabilizer generators.

        Args:
            x_bits (std::vector<std::bitset<MAX_ROWS_STABILIZER>>&): The X part of the stabilizer tableau.
            z_bits (std::vector<std::bitset<MAX_ROWS_STABILIZER>>&): The Z part of the stabilizer tableau.
            phases (std::bitset<MAX_ROWS_STABILIZER>&): The phases of the stabilizers.
            n_ (int): The number of qubits in the state.
        */
        n = n_;
        // Pull the n stabilizer generators into a row-major scratch tableau we can reduce in place
        X.assign(n, std::vector<uint8_t>(n, 0));
        Z.assign(n, std::vector<uint8_t>(n, 0));
        p4.assign(n, 0);
        for (int row = 0; row < n; ++row) {
            for (int q = 0; q < n; ++q) {
                X[row][q] = x_bits[q][row];
                Z[row][q] = z_bits[q][row];
            }
            p4[row] = phases[row] ? 2 : 0;
        }

        // Row sum: row dst <- row_dst * row_src (Pauli product), updating the tableau and phase.
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
        pivot_qubit.clear();
        int rank = 0;
        for (int q = 0; q < n && rank < n; ++q) {
            int sel = -1;
            for (int row = rank; row < n; ++row) {
                if (X[row][q]) {
                    sel = row;
                    break;
                }
            }
            if (sel < 0)
                continue;
            std::swap(X[sel], X[rank]);
            std::swap(Z[sel], Z[rank]);
            std::swap(p4[sel], p4[rank]);
            for (int row = 0; row < n; ++row) {
                if (row != rank && X[row][q])
                    rowmul(row, rank);
            }
            pivot_qubit.push_back(q);
            ++rank;
        }
        r = rank;

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
                continue;  // GCOVR_EXCL_LINE
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
        ref.assign(n, 0);
        for (int row = r; row < n; ++row) {
            int d = zpivot_qubit[row - r];
            // Unreachable for a valid tableau: every pure-Z row obtains a Z-pivot above, so no entry of
            // zpivot_qubit is left at its -1 sentinel. Defensive only.
            if (d < 0) {
                continue;  // GCOVR_EXCL_LINE
            }
            ref[d] = (p4[row] == 2) ? 1 : 0;
        }
    }

    std::complex<double> query(const std::string& b) const {
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
};

}  // namespace

std::complex<double> StabilizerState::amplitude(const std::string& b) const {
    /*
    Compute the amplitude <b|s> for a given computational basis bitstring b.

    Args:
        b (std::string): The bitstring to compute the amplitude for.

    Returns:
        std::complex<double>: The complex amplitude <b|s>.
    */
    AmplitudeReducer reducer;
    reducer.build(x_bits, z_bits, phases, nqubits);
    return reducer.query(b);
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

    // Build a cumulative weight vector for efficient importance sampling
    std::vector<double> cum_weights(M);
    double total_weight = 0.0;
    for (size_t i = 0; i < M; ++i) {
        total_weight += std::abs(coefficients[i]);
        cum_weights[i] = total_weight;
    }

    // Try to find all interesting bitstrings
    const int total_shots = nshots;
    const uint64_t base_seed = rng();
    std::set<std::string> unique_samples;
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
#else
        const int tid = 0;
        const int nthreads = 1;
#endif
        std::mt19937_64 engine(base_seed + static_cast<uint64_t>(tid) + 1);
        std::uniform_real_distribution<double> weight_dist(0.0, total_weight);
        std::set<std::string> local_samples;
        const int lo = (tid * total_shots) / nthreads;
        const int hi = ((tid + 1) * total_shots) / nthreads;
        for (int shot = lo; shot < hi; ++shot) {
            // Pick a state with probability proportional to |c_i|
            double r = weight_dist(engine);
            size_t index = static_cast<size_t>(std::lower_bound(cum_weights.begin(), cum_weights.end(), r) - cum_weights.begin());
            if (index >= M) {
                index = M - 1;
            }

            // Sample a bitstring from that state using this thread's engine
            local_samples.insert(states[index].sample(engine));
        }

#if defined(_OPENMP)
#pragma omp critical
#endif
        unique_samples.insert(local_samples.begin(), local_samples.end());
    }

    // Compute the exact amplitude of each distinct bitstring. amplitude() is a const, race-free
    // O(M * n^3) reduction, so distinct bitstrings are computed in parallel.
    const std::vector<std::string> sample_list(unique_samples.begin(), unique_samples.end());
    const int nsamples = static_cast<int>(sample_list.size());
    std::vector<std::complex<double>> sample_amps(nsamples);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < nsamples; ++i) {
        sample_amps[i] = amplitude(sample_list[i]);
    }

    std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
    std::map<std::string, std::complex<double>> pre_samples;
    for (int i = 0; i < nsamples; ++i) {
        pre_samples[sample_list[i]] = sample_amps[i];
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
    // Only the Z case (!x1 && z1) remains
    return x2 * (1 - 2 * z2);
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

std::string StabilizerState::representative(int* support_rank) const {
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

    // Count the random (X-pivot) qubits; this is the log2 of the support size, which fixes the
    // magnitude (2^(-rank/2)) of every amplitude in the support.
    int rank = 0;

    // Reuse one tableau reduction across all deterministic qubits, rebuilding only after the
    // tableau is mutated by the X-pivot branch (see sample() for the same pattern).
    ZPhaseSolver solver;
    bool dirty = true;
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
            for (int q = 0; q < nqubits; ++q) {
                x[q].reset(p);
                z[q].reset(p);
            }
            z[k].set(p);
            ph.reset(p);
            b[k] = '0';
            ++rank;
            dirty = true;
        } else {
            if (dirty) {
                solver.build(x, z, ph, nqubits);
                dirty = false;
            }
            b[k] = solver.query(k) ? '1' : '0';
        }
    }
    if (support_rank != nullptr) {
        *support_rank = rank;
    }
    return b;
}

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

    // For single-qubit gates, we need to sum over the two possible input states of the target qubit.
    // Both contributions are amplitudes of this same state, so reduce its tableau once and reuse it.
    if (controls.empty() && targets.size() == 1) {
        int t = targets[0];
        int bt = (b[t] == '1') ? 1 : 0;
        AmplitudeReducer reducer;
        bool built = false;
        std::complex<double> sum = 0.0;
        for (int a = 0; a < 2; ++a) {
            std::complex<double> u = clifford_elem(name, bt, a);
            if (std::abs(u) < 1e-15) {
                continue;
            }
            if (!built) {
                reducer.build(x_bits, z_bits, phases, nqubits);
                built = true;
            }
            std::string bp = b;
            bp[t] = a ? '1' : '0';
            sum += u * reducer.query(bp);
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
    // The representative is, by construction, the basis state amplitude() assigns a positive real
    // amplitude to, so conv = after.amplitude(b) is exactly 2^(-rank/2) (real, positive). We get
    // rank straight out of representative() and avoid a second full tableau reduction of `after`.
    int rank = 0;
    const std::string b = after.representative(&rank);
    double conv = 1.0;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    for (int i = 0; i < rank; ++i) {
        conv *= inv_sqrt2;
    }
    if (conv < 1e-12) {
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
        // Each term is an independent StabilizerState (its own tableau and rng), and apply_gate's
        // dropped_global_phase work is O(n^3), so the terms parallelize cleanly with no sharing.
        const int nterms = static_cast<int>(states.size());
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int k = 0; k < nterms; ++k) {
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
        // M_PI is not a standard C++ macro (undefined under MSVC without _USE_MATH_DEFINES), so use an explicit constant.
        constexpr double pi = 3.14159265358979323846;
        static const std::complex<double> t_phase = std::exp(std::complex<double>(0.0, pi / 4.0));
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

    // Each input term branches independently and the per-term work (projection, representative,
    // amplitude) is heavy, so compute the resulting terms into per-term buffers in parallel and
    // then concatenate them in input order. Keeping the merge ordered preserves a deterministic
    // term ordering (important for reproducibility and combine_duplicates).
    const int nterms = static_cast<int>(states.size());
    std::vector<std::vector<StabilizerState>> part_states(nterms);
    std::vector<std::vector<std::complex<double>>> part_coeffs(nterms);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int k = 0; k < nterms; ++k) {
        // Add the terms from a branch callback's StabilizerStateSum into this term's buffer
        auto add_sss = [&](StabilizerStateSum sss) {
            for (auto& s : sss.get_states()) {
                part_states[k].push_back(std::move(s));
            }
            for (auto& c : sss.get_coefficients()) {
                part_coeffs[k].push_back(c);
            }
        };

        // Check if it's deterministic
        bool is_random = (states[k].find_x_pivot(branch_qubit) != -1);

        // Deterministic outcome: call the matching callback
        if (!is_random) {
            bool outcome = states[k].z_eigenvalue(branch_qubit);
            add_sss(outcome ? on_one(states[k], coefficients[k]) : on_zero(states[k], coefficients[k]));

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
            add_sss(on_zero(s0, coefficients[k] * a0));
            add_sss(on_one(s1, coefficients[k] * a1));
        }
    }

    // Concatenate the per-term buffers in input order
    for (int k = 0; k < nterms; ++k) {
        for (auto& s : part_states[k]) {
            new_states.push_back(std::move(s));
        }
        for (auto& c : part_coeffs[k]) {
            new_coeffs.push_back(c);
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
    if (int(states.size()) <= max_terms) {
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

    // Hash and compare states directly from their raw bitsets
    auto hash_state = [](const StabilizerState& s) {
        std::hash<std::bitset<MAX_ROWS_STABILIZER>> bh;
        size_t h = 1469598103934665603ULL;  // FNV-1a offset basis
        auto mix = [&](size_t v) { h = (h ^ v) * 1099511628211ULL; };
        for (const auto& xb : s.get_x_bits())
            mix(bh(xb));
        for (const auto& zb : s.get_z_bits())
            mix(bh(zb));
        mix(bh(s.get_phases()));
        return h;
    };
    auto states_equal = [](const StabilizerState& a, const StabilizerState& b) { return a.get_x_bits() == b.get_x_bits() && a.get_z_bits() == b.get_z_bits() && a.get_phases() == b.get_phases(); };
    auto hash = [&](size_t i) { return hash_state(states[i]); };
    auto eq = [&](size_t a, size_t b) { return states_equal(states[a], states[b]); };

    // For each state, add it and hash it
    std::unordered_map<size_t, size_t, decltype(hash), decltype(eq)> state_to_index(states.size(), hash, eq);
    for (size_t i = 0; i < states.size(); ++i) {
        auto it = state_to_index.find(i);
        if (it != state_to_index.end()) {
            new_coeffs[it->second] += coefficients[i];
        } else {
            state_to_index[i] = new_states.size();
            new_states.push_back(states[i]);
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

DenseMatrix StabilizerState::as_dense() const {
    /*
    Convert the StabilizerState to a dense vector representation.
    This is done by computing the amplitude of each computational basis state.

    Returns:
        DenseMatrix: The dense vector representation of the StabilizerState.
    */
    int dim = 1 << nqubits;
    DenseMatrix result(dim, 1);
    for (int i = 0; i < dim; ++i) {
        std::string b = std::bitset<MAX_ROWS_STABILIZER>(i).to_string().substr(MAX_ROWS_STABILIZER - nqubits);
        result(i, 0) = amplitude(b);
    }
    return result;
}

DenseMatrix StabilizerStateSum::as_dense() const {
    /*
    Convert the StabilizerStateSum to a dense matrix representation.
    Each StabilizerState is converted to its dense vector, and the sum is computed.

    Returns:
        DenseMatrix: The dense matrix representation of the StabilizerStateSum.
    */
    int dim = 1 << nqubits;
    DenseMatrix result(dim, 1);
    result.setZero();
    for (size_t k = 0; k < states.size(); ++k) {
        DenseMatrix vec = states[k].as_dense();
        result += coefficients[k] * vec;
    }
    return result;
}

double StabilizerState::expectation_value(const MatrixFreeHamiltonian& h) const {
    /*
    Compute the expectation value of a Hamiltonian with respect to the StabilizerState.
    This is done by summing the expectation values of each term in the Hamiltonian.

    Args:
        h (MatrixFreeHamiltonian&): The Hamiltonian to compute the expectation value for.

    Returns:
        double: The expectation value <psi|H|psi> where |psi> is the StabilizerState.
    */

    // Reduce the tableau once; each Pauli term is then a cheap O(n^2/word) membership/sign query.
    ZPhaseSolver solver;
    solver.build(x_bits, z_bits, phases, nqubits);

    // PauliString masks are bounded by MAX_QUBITS_PAULI, so never read past that when copying them.
    const int nq = std::min(nqubits, MAX_QUBITS_PAULI);

    // The operator map isn't random-access, so snapshot the terms to fan the per-term queries out.
    const auto& operators = h.get_operators();
    std::vector<const std::pair<const PauliString, std::complex<double>>*> terms;
    terms.reserve(operators.size());
    for (const auto& term : operators) {
        terms.push_back(&term);
    }

    double total = 0.0;
    const int nterms = static_cast<int>(terms.size());
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : total) schedule(static)
#endif
    for (int t = 0; t < nterms; ++t) {
        const PauliString& pauli = terms[t]->first;
        std::bitset<MAX_ROWS_STABILIZER> tx, tz;
        for (int q = 0; q < nq; ++q) {
            if (pauli.x_mask[q]) {
                tx.set(q);
            }
            if (pauli.z_mask[q]) {
                tz.set(q);
            }
        }
        // <P> is real (+1/-1/0); a Hermitian Hamiltonian has real Pauli coefficients.
        int ev = solver.expectation_pauli(tx, tz);
        if (ev != 0) {
            total += terms[t]->second.real() * ev;
        }
    }
    return total;
}

namespace {

// The support of a stabilizer state is an affine subspace of {0,1}^n: a base point plus the GF(2)
// span of the X-supports of its X-pivot generators. Enumerating it lets us evaluate overlaps
// <a|P|b> between two (generally non-orthogonal) stabilizer states by direct summation.
struct Support {
    std::string base;
    std::vector<std::bitset<MAX_ROWS_STABILIZER>> dirs;  // difference-space basis (generator X-supports)
};

Support get_support(const StabilizerState& s) {
    /*
    Compute the support of a stabilizer state as an affine subspace of {0,1}^n.
    The support is represented by a base point (a computational basis state with nonzero amplitude)
    and a set of direction vectors (the X-supports of the X-pivot generators).

    Args:
        s (StabilizerState&): The stabilizer state to compute the support for.

    Returns:
        Support: The support of the stabilizer state.
    */
    Support sup;
    sup.base = s.representative();
    ZPhaseSolver solver;
    solver.build(s.get_x_bits(), s.get_z_bits(), s.get_phases(), s.get_nqubits());
    for (size_t pr = 0; pr < solver.pivots.size(); ++pr) {
        if (solver.pivots[pr].first) {
            sup.dirs.push_back(solver.rx[pr]);
        }
    }
    return sup;
}

std::complex<double> pauli_overlap(const StabilizerState& a, const StabilizerState& b, const PauliString& P, const Support& supA) {
    /*
    Compute the overlap <a|P|b> between two stabilizer states a and b, where P is a Pauli operator.
    The support of a is enumerated, and for each basis state x in the support,
    the amplitude <x|a> is computed, and the corresponding basis state y = x XOR px is found,
    where px is the X-support of P. The amplitude <y|b> is then computed, and the contribution
    to the overlap is accumulated.

    Args:
        a (StabilizerState&): The first stabilizer state.
        b (StabilizerState&): The second stabilizer state.
        P (PauliString&): The Pauli operator.
        supA (Support&): The support of the first stabilizer state.

    Returns:
        std::complex<double>: The overlap <a|P|b>.
    */
    const int n = a.get_nqubits();
    const int k = static_cast<int>(supA.dirs.size());
    const int MAX_SUPPORT_DIM = 24;
    if (k > MAX_SUPPORT_DIM) {
        throw std::runtime_error("StabilizerStateSum::expectation_value: an off-diagonal term has support dimension " + std::to_string(k) + " (> " + std::to_string(MAX_SUPPORT_DIM) + "); the cross-term overlap enumeration would be too expensive.");
    }
    const int nq = std::min(n, MAX_QUBITS_PAULI);
    std::complex<double> total = 0.0;
    const uint64_t lim = uint64_t(1) << k;
    for (uint64_t m = 0; m < lim; ++m) {
        // Build the support point x = base XOR (selected direction vectors).
        std::string x = supA.base;
        for (int d = 0; d < k; ++d) {
            if ((m >> d) & 1u) {
                const auto& dir = supA.dirs[d];
                for (int q = 0; q < n; ++q) {
                    if (dir[q]) {
                        x[q] = (x[q] == '1') ? '0' : '1';
                    }
                }
            }
        }
        const std::complex<double> amp_a = a.amplitude(x);
        // Unreachable: x is enumerated from a's own support (base XOR a's X-pivot directions), so a
        // always assigns it a nonzero amplitude. Defensive only.
        if (std::abs(amp_a) < 1e-15) {
            continue;  // GCOVR_EXCL_LINE
        }
        // y = x XOR px is the only basis state P connects x to with nonzero matrix element.
        std::string y = x;
        for (int q = 0; q < nq; ++q) {
            if (P.x_mask[q]) {
                y[q] = (y[q] == '1') ? '0' : '1';
            }
        }
        const std::complex<double> amp_b = b.amplitude(y);
        if (std::abs(amp_b) < 1e-15) {
            continue;
        }
        // <x|P|x^px>: Z gives (-1)^{x_q}, Y gives -i (x_q=0) or i (x_q=1), X and I give 1.
        std::complex<double> mu(1.0, 0.0);
        for (int q = 0; q < nq; ++q) {
            const bool xq = (x[q] == '1');
            if (P.z_mask[q] && !P.x_mask[q]) {
                if (xq) {
                    mu = -mu;
                }
            } else if (P.x_mask[q] && P.z_mask[q]) {
                mu *= xq ? std::complex<double>(0.0, 1.0) : std::complex<double>(0.0, -1.0);
            }
        }
        total += std::conj(amp_a) * mu * amp_b;
    }
    return total;
}

}  // namespace

double StabilizerStateSum::expectation_value(const MatrixFreeHamiltonian& h) const {
    /*
    Compute the expectation value of a Hamiltonian with respect to the StabilizerStateSum.

    |psi> = sum_i c_i |s_i> is a superposition of (generally non-orthogonal) stabilizer states, so
        <psi|H|psi> = sum_{i,j} conj(c_i) c_j <s_i|H|s_j>
    has both diagonal and off-diagonal terms. Diagonal terms use the fast tableau query; off-diagonal
    terms use the support-enumeration overlap. Because the terms need not be orthogonal, |psi> is
    generally unnormalised even when sum |c_i|^2 = 1, so we divide by <psi|psi>.

    Args:
        h (MatrixFreeHamiltonian&): The Hamiltonian to compute the expectation value for.

    Returns:
        double: The expectation value <psi|H|psi> / <psi|psi>.
    */
    const int M = static_cast<int>(states.size());
    if (M == 0) {
        return 0.0;
    }

    // Diagonal contributions: <s_i|H|s_i> (fast tableau path) and <s_i|s_i> = 1.
    double numer = 0.0;
    double denom = 0.0;
    for (int i = 0; i < M; ++i) {
        const double w = std::norm(coefficients[i]);
        numer += w * states[i].expectation_value(h);
        denom += w;
    }

    // Off-diagonal contributions. H is Hermitian so <s_i|H|s_j> = conj(<s_j|H|s_i>), and the {i,j}
    // pair contributes 2 Re(conj(c_i) c_j <s_i|H|s_j>); the overlaps <s_i|s_j> feed the denominator.
    if (M > 1) {
        std::vector<Support> sup(M);
        for (int i = 0; i < M; ++i) {
            sup[i] = get_support(states[i]);
        }
        const auto& ops = h.get_operators();
        const PauliString identity(nqubits);
        for (int i = 0; i < M; ++i) {
            for (int j = i + 1; j < M; ++j) {
                // Enumerate the smaller support; <s_i|P|s_j> = conj(<s_j|P|s_i>).
                const bool enum_i = sup[i].dirs.size() <= sup[j].dirs.size();
                auto overlap = [&](const PauliString& P) { return enum_i ? pauli_overlap(states[i], states[j], P, sup[i]) : std::conj(pauli_overlap(states[j], states[i], P, sup[j])); };
                std::complex<double> hij = 0.0;
                for (const auto& [P, w] : ops) {
                    hij += w * overlap(P);
                }
                const std::complex<double> sij = overlap(identity);
                const std::complex<double> cc = std::conj(coefficients[i]) * coefficients[j];
                numer += 2.0 * std::real(cc * hij);
                denom += 2.0 * std::real(cc * sij);
            }
        }
    }

    if (std::abs(denom) < 1e-300) {
        return 0.0;
    }
    return numer / denom;
}

// GCOV_EXCL_BR_STOP