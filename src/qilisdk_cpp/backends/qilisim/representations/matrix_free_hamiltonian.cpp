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

#include "matrix_free_hamiltonian.h"
#include <algorithm>
#include <unordered_map>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#if defined(__SSE3__) && defined(__FMA__)
#include <immintrin.h>
#define QILI_PACKED_COMPLEX 1
#endif
#include "../../../libs/pybind.h"
#include "../utils/matrix_utils.h"
#if defined(_OPENMP)
#include <omp.h>
#endif

// GCOV_EXCL_BR_START

namespace {
int popcount_u64(unsigned long long value) {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(value));
#else
    return __builtin_popcountll(value);
#endif
}

// A single complex value held for arithmetic. GCC will not pack std::complex
// accumulation into 128-bit SIMD on its own, so on x86 (SSE3 + FMA, i.e. the
// -march=x86-64-v3 baseline) we operate on __m128d = [real, imag] explicitly.
// Elsewhere it degrades to plain std::complex, which the compiler handles fine.
#if defined(QILI_PACKED_COMPLEX)
using cvec = __m128d;
inline cvec cv_zero() {
    return _mm_setzero_pd();
}
inline cvec cv_load(const Complex* p) {
    return _mm_loadu_pd(reinterpret_cast<const double*>(p));
}
inline cvec cv_add(cvec a, cvec b) {
    return _mm_add_pd(a, b);
}
// (ar + i ai)(br + i bi) = (ar*br - ai*bi) + i (ar*bi + ai*br)
inline cvec cv_cmul(cvec a, cvec b) {
    cvec ar = _mm_movedup_pd(a);              // [ar, ar]
    cvec ai = _mm_unpackhi_pd(a, a);          // [ai, ai]
    cvec b_swap = _mm_shuffle_pd(b, b, 0x1);  // [bi, br]
    cvec t = _mm_mul_pd(ai, b_swap);          // [ai*bi, ai*br]
    return _mm_fmaddsub_pd(ar, b, t);         // [ar*br - ai*bi, ar*bi + ai*br]
}
// acc += r * x, with r a real scalar
inline cvec cv_fma_real(cvec acc, double r, cvec x) {
    return _mm_fmadd_pd(_mm_set1_pd(r), x, acc);
}
// acc += phase * x, with phase complex
inline cvec cv_fma_cplx(cvec acc, const Complex& phase, cvec x) {
    return cv_add(acc, cv_cmul(cv_load(&phase), x));
}
inline Complex cv_store(cvec v) {
    Complex out;
    _mm_storeu_pd(reinterpret_cast<double*>(&out), v);
    return out;
}
#else
using cvec = Complex;
inline cvec cv_zero() {
    return Complex(0.0, 0.0);
}
inline cvec cv_load(const Complex* p) {
    return *p;
}
inline cvec cv_add(cvec a, cvec b) {
    return a + b;
}
inline cvec cv_cmul(cvec a, cvec b) {
    return a * b;
}
inline cvec cv_fma_real(cvec acc, double r, cvec x) {
    return acc + r * x;
}
inline cvec cv_fma_cplx(cvec acc, const Complex& phase, cvec x) {
    return acc + phase * x;
}
inline Complex cv_store(cvec v) {
    return v;
}
#endif
}  // namespace

MatrixFreeHamiltonian::MatrixFreeHamiltonian(int nqubits, const MatrixFreeOperator& op, Complex coeff) : nqubits(nqubits) {
    PauliString ps(nqubits, {op});
    operators[ps] = coeff;
}

void MatrixFreeHamiltonian::apply(const DenseMatrix& input_state, MatrixFreeApplicationType application_type, DenseMatrix& output_state) const {
    /*
    Applies the matrix-free Hamiltonian to the given input state and writes the
    result to a separate output state.

    Args:
        input_state: The state to which the Hamiltonian will be applied.
        application_type: The type of application (Left, Right, or LeftAndRight).
        output_state: The state where the result will be stored.
    */
    struct Term {
        Complex base_phase;      // coefficient * (-i)^n_Y, precomputed
        Complex base_phase_neg;  // -coefficient * (-i)^n_Y, precomputed
        long flip_mask;          // XOR of all X and Y qubit masks
        long sign_mask;          // OR of all Y and Z qubit masks (popcount parity = sign flip)
    };

    // Precompute things
    std::vector<Term> terms;
    terms.reserve(operators.size());
    const int num_qubits = static_cast<int>(std::log2(input_state.rows()));
    for (const auto& [pauli, coefficient] : operators) {
        long flip_mask = 0;
        long sign_mask = 0;
        int n_y = 0;
        for (int i = 0; i < num_qubits; ++i) {
            long mask = 1LL << (num_qubits - 1 - i);
            if (pauli.x_mask[i] && !pauli.z_mask[i]) {  // X
                flip_mask ^= mask;
            } else if (!pauli.x_mask[i] && pauli.z_mask[i]) {  // Z
                sign_mask |= mask;
            } else if (pauli.x_mask[i] && pauli.z_mask[i]) {  // Y
                flip_mask ^= mask;
                sign_mask |= mask;
                ++n_y;
            }
        }
        static const Complex neg_i_powers[4] = {{1, 0}, {0, -1}, {-1, 0}, {0, 1}};
        Complex base_phase = coefficient * neg_i_powers[n_y & 3];
        Complex base_phase_neg = -base_phase;
        terms.push_back({base_phase, base_phase_neg, flip_mask, sign_mask});
    }

    // Move all diagonal terms to the front of the vector, preserving relative order
    auto diag_mid = std::stable_partition(terms.begin(), terms.end(), [](const Term& t) { return t.flip_mask == 0; });
    const long n_diag = static_cast<long>(diag_mid - terms.begin());

    // Seperate simple (i.e. no sign flip, real phase) off-diagonal terms from general off-diagonal terms
    auto simple_mid = std::stable_partition(diag_mid, terms.end(), [](const Term& t) { return t.sign_mask == 0 && t.base_phase.imag() == 0.0; });
    const long n_simple_off = static_cast<long>(simple_mid - diag_mid);

    // Make sure output_state has the right shape
    output_state.resizeLike(input_state);

    // Cache some pointers
    const Complex* in_ptr = input_state.data();
    Complex* out_ptr = output_state.data();
    const Term* t_begin = terms.data();
    const Term* t_end = t_begin + terms.size();
    const Term* diag_end = t_begin + n_diag;
    const Term* simple_end = diag_end + n_simple_off;

    // Per-index kernel shared by the statevector and (column-wise) density-matrix
    // paths: returns amplitude i of (H * vec) for a single contiguous vector `vec`.
    // Terms are pre-partitioned into diagonal / simple real off-diagonal / general,
    // so most terms avoid a full complex multiply.
    // The three loops below accumulate into complex (not split real/imag scalar)
    // values so that, with -fcx-limited-range + AVX2, each term becomes a packed
    // 128-bit load and FMA instead of pairs of scalar ops. Several independent
    // accumulators per loop break the reduction dependency chain (reassociation
    // is off without -ffast-math) and keep multiple gathers in flight.
    auto apply_index = [&](long i, const Complex* vec) -> Complex {
        const unsigned long long ii = static_cast<unsigned long long>(i);

        // Diagonal terms only contribute a per-index phase to vec[i]
        cvec d0 = cv_zero(), d1 = cv_zero();
        const Term* t = t_begin;
        for (; t + 2 <= diag_end; t += 2) {
            bool n0 = popcount_u64(ii & static_cast<unsigned long long>(t[0].sign_mask)) & 1;
            bool n1 = popcount_u64(ii & static_cast<unsigned long long>(t[1].sign_mask)) & 1;
            d0 = cv_add(d0, cv_load(n0 ? &t[0].base_phase_neg : &t[0].base_phase));
            d1 = cv_add(d1, cv_load(n1 ? &t[1].base_phase_neg : &t[1].base_phase));
        }
        for (; t != diag_end; ++t) {
            bool n0 = popcount_u64(ii & static_cast<unsigned long long>(t->sign_mask)) & 1;
            d0 = cv_add(d0, cv_load(n0 ? &t->base_phase_neg : &t->base_phase));
        }
        cvec acc = cv_cmul(cv_add(d0, d1), cv_load(&vec[i]));

        // Simple off-diagonal terms: real phase, no sign flips
        cvec s0 = cv_zero(), s1 = cv_zero(), s2 = cv_zero(), s3 = cv_zero();
        t = diag_end;
        for (; t + 4 <= simple_end; t += 4) {
            s0 = cv_fma_real(s0, t[0].base_phase.real(), cv_load(&vec[i ^ t[0].flip_mask]));
            s1 = cv_fma_real(s1, t[1].base_phase.real(), cv_load(&vec[i ^ t[1].flip_mask]));
            s2 = cv_fma_real(s2, t[2].base_phase.real(), cv_load(&vec[i ^ t[2].flip_mask]));
            s3 = cv_fma_real(s3, t[3].base_phase.real(), cv_load(&vec[i ^ t[3].flip_mask]));
        }
        for (; t != simple_end; ++t) {
            s0 = cv_fma_real(s0, t->base_phase.real(), cv_load(&vec[i ^ t->flip_mask]));
        }
        acc = cv_add(acc, cv_add(cv_add(s0, s1), cv_add(s2, s3)));

        // General off-diagonal terms: full complex multiply, possible sign flip
        cvec g0 = cv_zero(), g1 = cv_zero();
        t = simple_end;
        for (; t + 2 <= t_end; t += 2) {
            bool n0 = popcount_u64(ii & static_cast<unsigned long long>(t[0].sign_mask)) & 1;
            bool n1 = popcount_u64(ii & static_cast<unsigned long long>(t[1].sign_mask)) & 1;
            g0 = cv_fma_cplx(g0, n0 ? t[0].base_phase_neg : t[0].base_phase, cv_load(&vec[i ^ t[0].flip_mask]));
            g1 = cv_fma_cplx(g1, n1 ? t[1].base_phase_neg : t[1].base_phase, cv_load(&vec[i ^ t[1].flip_mask]));
        }
        for (; t != t_end; ++t) {
            bool n0 = popcount_u64(ii & static_cast<unsigned long long>(t->sign_mask)) & 1;
            g0 = cv_fma_cplx(g0, n0 ? t->base_phase_neg : t->base_phase, cv_load(&vec[i ^ t->flip_mask]));
        }
        acc = cv_add(acc, cv_add(g0, g1));

        return cv_store(acc);
    };

    // Statevector: a single column, out = H * psi
    if (input_state.cols() == 1) {
        const long total = output_state.size();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (long i = 0; i < total; ++i) {
            out_ptr[i] = apply_index(i, in_ptr);
        }
        return;
    }

    // Density matrix (or a batch of trajectory state vectors). Left multiplication
    // H * rho acts within each column: column c of the result is H applied to
    // column c of the input, so we reuse the contiguous per-column kernel above
    // instead of striding across rows. N is the column length (state-space
    // dimension); the matrix need not be square - Monte Carlo passes an
    // (dim x n_trajectories) buffer - so iterate over the real column count.
    const long N = output_state.rows();
    const long ncols = output_state.cols();
    if (application_type == MatrixFreeApplicationType::Left || application_type == MatrixFreeApplicationType::LeftAndRight) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (long c = 0; c < ncols; ++c) {
            const Complex* in_col = in_ptr + c * N;
            Complex* out_col = out_ptr + c * N;
            for (long i = 0; i < N; ++i) {
                out_col[i] = apply_index(i, in_col);
            }
        }
    }

    // Right multiplication rho * H^dagger permutes and scales whole columns, which
    // is already contiguous in column-major storage.
    if (application_type == MatrixFreeApplicationType::Right) {
        output_state.setZero();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (long j = 0; j < N; ++j) {
            for (const Term* t = t_begin; t != t_end; ++t) {
                long index = j ^ t->flip_mask;
                bool neg = popcount_u64(static_cast<unsigned long long>(j) & static_cast<unsigned long long>(t->sign_mask)) & 1;
                output_state.col(j) += std::conj(neg ? t->base_phase_neg : t->base_phase) * input_state.col(index);
            }
        }
    } else if (application_type == MatrixFreeApplicationType::LeftAndRight) {
        // The Left pass above already wrote H * rho into output_state; now apply
        // the right factor (H * rho) * H^dagger, again as contiguous column work.
        DenseMatrix hr_temp = output_state;
        output_state.setZero();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (long j = 0; j < N; ++j) {
            for (const Term* t = t_begin; t != t_end; ++t) {
                long index = j ^ t->flip_mask;
                bool neg = popcount_u64(static_cast<unsigned long long>(j) & static_cast<unsigned long long>(t->sign_mask)) & 1;
                output_state.col(j) += std::conj(neg ? t->base_phase_neg : t->base_phase) * hr_temp.col(index);
            }
        }
    }
}

double MatrixFreeHamiltonian::expectation_value(const DenseMatrix& state) const {
    /*
    Calculate the expectation value of the Hamiltonian with respect to a given state.

    Args:
        state: The state for which the expectation value will be calculated.

    Raises:
        std::invalid_argument: If any operator in the Hamiltonian acts on a qubit that is out of bounds for the given state.
    */
    int num_qubits_in_state = static_cast<int>(std::log2(state.rows()));
    int num_qubits_in_hamiltonian = get_nqubits();
    if (num_qubits_in_hamiltonian > num_qubits_in_state) {
        throw std::invalid_argument("Hamiltonian acts on more qubits than the state has.");
    }

    // Do <state|H|state> by applying H to state and then taking the dot product with state.
    m_temp_state.resizeLike(state);
    m_new_state.resizeLike(state);
    apply(state, MatrixFreeApplicationType::Left, m_temp_state);
    double exp_val = std::real(dot(state, m_temp_state));

    return exp_val;
}

double MatrixFreeHamiltonian::expectation_value(const MatrixFreeHamiltonian& other) const {
    /*
    Calculate the expectation value of this Hamiltonian with respect to another Hamiltonian.
    The assumption is that the state is this*|+>.

    Args:
        other: The Hamiltonian with respect to which the expectation value will be calculated.

    Returns:
        The expectation value of this Hamiltonian with respect to the other Hamiltonian.
    */

    // first we calculate H_this^dag H_other H_this:
    MatrixFreeHamiltonian temp = (*this).conjugate() * other * (*this);

    // Since <+|P|+> is 1 if P is identity or X and 0 otherwise, we just need to sum the coefficients
    Complex exp_val = 0.0;
    for (const auto& [pauli, coefficient] : temp.operators) {
        if (pauli.z_mask.none()) {
            exp_val += coefficient;
        }
    }
    return std::real(exp_val);
}

MatrixFreeHamiltonian& MatrixFreeHamiltonian::operator*=(const Complex& scalar) {
    /*
    Scale in-place by a complex scalar.

    Args:
        scalar: The complex scalar by which to scale the Hamiltonian.

    Returns:
        A reference to the scaled Hamiltonian.
    */
    for (auto& [pauli, coefficient] : operators) {
        coefficient *= scalar;
    }
    return *this;
}

MatrixFreeHamiltonian MatrixFreeHamiltonian::operator*(const Complex& scalar) const {
    /*
    Scale by a complex scalar and return a new Hamiltonian.

    Args:
        scalar: The complex scalar by which to scale the Hamiltonian.

    Returns:
        A new Hamiltonian that is the result of scaling this Hamiltonian by the given scalar.
    */
    MatrixFreeHamiltonian result = *this;
    result *= scalar;
    return result;
}

MatrixFreeHamiltonian MatrixFreeHamiltonian::operator*(const double& scalar) const {
    /*
    Scale by a real scalar and return a new Hamiltonian.

    Args:
        scalar: The real scalar by which to scale the Hamiltonian.

    Returns:
        A new Hamiltonian that is the result of scaling this Hamiltonian by the given scalar.
    */
    MatrixFreeHamiltonian result = *this;
    result *= scalar;
    return result;
}

MatrixFreeHamiltonian& MatrixFreeHamiltonian::operator+=(const MatrixFreeHamiltonian& other) {
    /*
    Add another Hamiltonian to this one in-place.

    Args:
        other: The Hamiltonian to be added to this one.

    Returns:
        A reference to the resulting Hamiltonian after addition.
    */
    for (const auto& [pauli, coefficient] : other.operators) {
        operators[pauli] += coefficient;
    }
    return *this;
}

std::ostream& operator<<(std::ostream& os, const MatrixFreeHamiltonian& hamiltonian) {
    /*
    Output stream operator for MatrixFreeHamiltonian.
    Used like std::cout << hamiltonian; to print the Hamiltonian in a human-readable format.

    Args:
        os: The output stream to which the Hamiltonian will be written.
        hamiltonian: The Hamiltonian to be written to the output stream.

    Returns:
        A reference to the output stream after writing the Hamiltonian.
    */
    int count = 0;
    for (const auto& [pauli, coefficient] : hamiltonian.operators) {
        os << coefficient << " * " << pauli;
        if (count < int(hamiltonian.operators.size()) - 1) {
            os << " + ";
        }
        count++;
    }
    return os;
}

void MatrixFreeHamiltonian::add(const Complex& coeff, const PauliString& op) {
    /*
    Add a term to the Hamiltonian with a given coefficient and operator.

    Args:
        coeff: The complex coefficient for the term being added.
        op: The PauliString that defines the term being added to the Hamiltonian.
    */
    operators[op] += coeff;
}

void MatrixFreeHamiltonian::add(const Complex& coeff, const std::vector<MatrixFreeOperator>& ops) {
    /*
    Add a term to the Hamiltonian with a given coefficient and a vector of operators.

    Args:
        coeff: The complex coefficient for the term being added.
        ops: The vector of MatrixFreeOperator that defines the term being added to the Hamiltonian.
    */
    PauliString ps(get_nqubits(), ops);  // start with identity
    operators[ps] += coeff;
}

void MatrixFreeHamiltonian::add(const Complex& coeff, const MatrixFreeOperator& op) {
    add(coeff, std::vector<MatrixFreeOperator>{op});
}

bool MatrixFreeHamiltonian::operator==(const MatrixFreeHamiltonian& other) const {
    /*
    Equality operator for MatrixFreeHamiltonian. Two Hamiltonians are considered equal if they have the same terms with the same coefficients.

    Args:
        other: The Hamiltonian to compare with this one.

    Returns:
        True if the Hamiltonians are equal, false otherwise.
    */
    return operators == other.operators;
}

MatrixFreeHamiltonian operator*(const Complex& scalar, const MatrixFreeHamiltonian& hamiltonian) {
    /*
    Scale a Hamiltonian by a complex scalar from the left.

    Args:
        scalar: The complex scalar by which to scale the Hamiltonian.
        hamiltonian: The Hamiltonian to be scaled.

    Returns:
        A new Hamiltonian that is the result of scaling the given Hamiltonian by the given scalar.
    */
    return hamiltonian * scalar;
}

std::pair<PauliString, Complex> _multiply_pauli_strings(const PauliString& a, const PauliString& b) {
    size_t n = a.nqubits;
    PauliString result(n);

    // Phase contribution lookup table indexed by [ax][az][bx][bz]
    // Derived from: I=00, X=10, Z=01, Y=11
    static const int phase_lut[2][2][2][2] = {// ax=0
                                              {
                                                  // az=0 (I)
                                                  {{0, 0},   // bx=0: I*I=+1(0), I*Z=+1(0)
                                                   {0, 0}},  // bx=1: I*X=+1(0), I*Y=+1(0)
                                                             // az=1 (Z)
                                                  {{0, 0},   // bx=0: Z*I=+1(0), Z*Z=+1(0)
                                                   {1, -1}}  // bx=1: Z*X=+i(+1), Z*Y=-i(-1)
                                              },
                                              // ax=1
                                              {
                                                  // az=0 (X)
                                                  {{0, -1},  // bx=0: X*I=+1(0), X*Z=-i(-1)
                                                   {0, 1}},  // bx=1: X*X=+1(0), X*Y=+i(+1)
                                                             // az=1 (Y)
                                                  {{0, 1},   // bx=0: Y*I=+1(0), Y*Z=+i(+1)
                                                   {-1, 0}}  // bx=1: Y*X=-i(-1), Y*Y=+1(0)
                                              }};

    result.x_mask = a.x_mask ^ b.x_mask;
    result.z_mask = a.z_mask ^ b.z_mask;

    int phase_exp = 0;
    for (size_t q = 0; q < n; ++q) {
        const int ax = a.x_mask[q], az = a.z_mask[q];
        const int bx = b.x_mask[q], bz = b.z_mask[q];
        phase_exp += phase_lut[ax][az][bx][bz];
    }

    phase_exp = ((phase_exp % 4) + 4) % 4;

    static const Complex phase_table[4] = {{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}};

    return {result, phase_table[phase_exp]};
}

MatrixFreeHamiltonian MatrixFreeHamiltonian::operator*(const MatrixFreeHamiltonian& other) const {
    /*
    Multiply two Hamiltonians together.

    Args:
        other: The Hamiltonian to multiply with this one.

    Returns:
        A new Hamiltonian that is the result of multiplying this Hamiltonian with the other Hamiltonian.
    */
    MatrixFreeHamiltonian result(get_nqubits());

#if defined(_OPENMP)
    // Convert to vector for indexed parallel access
    std::vector<std::pair<PauliString, Complex>> ops_vec(operators.begin(), operators.end());
    const int nops = static_cast<int>(ops_vec.size());
    const int nthreads = omp_get_max_threads();
    std::vector<std::unordered_map<PauliString, Complex, PauliString::HashFunction>> local(nthreads);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < nops; ++i) {
        const int tid = omp_get_thread_num();
        const PauliString& ps_a = ops_vec[i].first;
        const Complex coeff_a = ops_vec[i].second;
        for (const auto& [ps_b, coeff_b] : other.operators) {
            auto [ps_result, phase] = _multiply_pauli_strings(ps_a, ps_b);
            local[tid][ps_result] += coeff_a * coeff_b * phase;
        }
    }

    for (int t = 0; t < nthreads; ++t) {
        for (auto& [ps, coeff] : local[t]) {
            result.operators[ps] += coeff;
        }
    }
#else
    result.operators.reserve(operators.size() * other.operators.size());
    for (const auto& [ps_a, coeff_a] : operators) {
        for (const auto& [ps_b, coeff_b] : other.operators) {
            auto [ps_result, phase] = _multiply_pauli_strings(ps_a, ps_b);
            result.operators[ps_result] += coeff_a * coeff_b * phase;
        }
    }
#endif

    return result;
}

MatrixFreeHamiltonian MatrixFreeHamiltonian::operator+(const MatrixFreeHamiltonian& other) const {
    /*
    Add this Hamiltonian to another Hamiltonian.

    Args:
        other: The Hamiltonian to add to this one.

    Returns:
        A new Hamiltonian that is the result of adding this Hamiltonian to the other Hamiltonian.
    */
    MatrixFreeHamiltonian result = *this;
    result += other;
    return result;
}

MatrixFreeHamiltonian MatrixFreeHamiltonian::operator-(const MatrixFreeHamiltonian& other) const {
    /*
    Subtract another Hamiltonian from this Hamiltonian.

    Args:
        other: The Hamiltonian to subtract from this one.

    Returns:
        A new Hamiltonian that is the result of subtracting the other Hamiltonian from this Hamiltonian.
    */
    MatrixFreeHamiltonian result = *this;
    result += other * Complex(-1.0, 0.0);
    return result;
}

void MatrixFreeHamiltonian::prune(double threshold, int max_terms) {
    /*
    Prune the Hamiltonian by removing terms with coefficients below a certain threshold and limiting the total number of terms.

    Args:
        threshold: The minimum absolute value of coefficients for terms to be kept in the Hamiltonian.
        max_terms: The maximum number of terms to keep in the Hamiltonian.
    */

    // Create a vector of terms and sort by absolute value of coefficients
    std::vector<std::pair<PauliString, Complex>> term_vector(operators.begin(), operators.end());
    std::sort(term_vector.begin(), term_vector.end(), [](const auto& a, const auto& b) { return std::abs(a.second) > std::abs(b.second); });
    if (term_vector.size() > static_cast<size_t>(max_terms)) {
        term_vector.resize(max_terms);
    }

    // Rebuild the operators map from the pruned vector
    operators.clear();
    for (const auto& [ps, coeff] : term_vector) {
        if (std::abs(coeff) >= threshold) {
            operators[ps] = coeff;
        }
    }
}

MatrixFreeHamiltonian MatrixFreeHamiltonian::conjugate() const {
    /*
    Conjugate the Hamiltonian by taking the complex conjugate of all coefficients.

    Returns:
        A new Hamiltonian that is the complex conjugate of this Hamiltonian.
    */
    MatrixFreeHamiltonian result = *this;
    for (auto& [ps, coeff] : result.operators) {
        coeff = std::conj(coeff);
    }
    return result;
}

// GCOV_EXCL_BR_STOP