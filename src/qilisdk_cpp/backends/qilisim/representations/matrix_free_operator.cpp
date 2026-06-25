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

#include "matrix_free_operator.h"
#include <sstream>
#include "../../../libs/pybind.h"
#if defined(__BMI2__)
#include <immintrin.h>  // _pdep_u64: deposit anchor-counter bits into free positions
#endif

// GCOV_EXCL_BR_START

const Complex imag(0.0, 1.0);
const Complex imag_conj(0.0, -1.0);
const Real inv_sqrt_2 = 1.0 / std::sqrt(2.0);
constexpr double pi = 3.14159265358979323846;
const Complex t_phase = std::exp(Complex(0.0, pi / 4.0));
const Complex t_phase_conj = std::conj(t_phase);

void MatrixFreeOperator::apply(DenseMatrix& output_state, MatrixFreeApplicationType application_type) const {
    /*
    Apply the operator to a dense state.

    Args:
        output_state (DenseMatrix&): The dense state to apply the operator to.
        as_density_matrix (bool): Whether to treat the state as a density matrix.
    */

    // Precompute things that are used in all branches
    int num_qubits = static_cast<int>(std::log2(output_state.rows()));
    long long mask = 1LL << (num_qubits - 1 - target_qubits[0]);
    long N = output_state.rows();
    long stride = mask;
    // stride is always a single-bit power of two, so the per-iteration
    // k / stride and k % stride below reduce to cheap bit-masking:
    //   k % stride          == k & offset_mask
    //   (k / stride) * 2*stride == (k & block_mask) << 1
    const long offset_mask = stride - 1;
    const long block_mask = ~offset_mask;
    long half = N >> 1;
    long dim = 1LL << num_qubits;

    // If we have an X on qubit i, we swap the amplitudes of all basis states where qubit i is 0 with those where qubit i is 1
    if (name == "X" && control_qubits.empty()) {
        if (output_state.cols() == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long i = base + offset;
                long j = i + stride;
                std::swap(output_state(i), output_state(j));
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long r0 = base + offset;
                long r1 = r0 + stride;
                output_state.row(r0).swap(output_state.row(r1));
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long c0 = base + offset;
                long c1 = c0 + stride;
                output_state.col(c0).swap(output_state.col(c1));
            }
        } else if (application_type == MatrixFreeApplicationType::LeftAndRight) {
#if defined(_OPENMP)
#pragma omp parallel
#endif
            {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long k = 0; k < half; ++k) {
                    long offset = k & offset_mask;
                    long base = (k & block_mask) << 1;
                    long r0 = base + offset;
                    long r1 = r0 + stride;
                    output_state.row(r0).swap(output_state.row(r1));
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long k = 0; k < half; ++k) {
                    long offset = k & offset_mask;
                    long base = (k & block_mask) << 1;
                    long c0 = base + offset;
                    long c1 = c0 + stride;
                    output_state.col(c0).swap(output_state.col(c1));
                }
            }
        }

        // If we have a Y on qubit i, we swap the amplitudes of all basis states where qubit i is 0 with those where qubit i is 1, and multiply the amplitude of all basis states where qubit i is 1 by i or -i depending on whether it was originally 0 or 1
    } else if (name == "Y" && control_qubits.empty()) {
        if (output_state.cols() == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long i = base + offset;
                long j = i + stride;
                Complex temp = output_state(i);
                output_state(i) = output_state(j) * imag_conj;
                output_state(j) = temp * imag;
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long r0 = base + offset;
                long r1 = r0 + stride;
                output_state.row(r0).swap(output_state.row(r1));
                output_state.row(r0) *= imag_conj;
                output_state.row(r1) *= imag;
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long c0 = base + offset;
                long c1 = c0 + stride;
                output_state.col(c0).swap(output_state.col(c1));
                output_state.col(c0) *= imag;
                output_state.col(c1) *= imag_conj;
            }
        } else if (application_type == MatrixFreeApplicationType::LeftAndRight) {
#if defined(_OPENMP)
#pragma omp parallel
#endif
            {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long k = 0; k < half; ++k) {
                    long offset = k & offset_mask;
                    long base = (k & block_mask) << 1;
                    long r0 = base + offset;
                    long r1 = r0 + stride;
                    output_state.row(r0).swap(output_state.row(r1));
                    output_state.row(r0) *= imag_conj;
                    output_state.row(r1) *= imag;
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long k = 0; k < half; ++k) {
                    long offset = k & offset_mask;
                    long base = (k & block_mask) << 1;
                    long c0 = base + offset;
                    long c1 = c0 + stride;
                    output_state.col(c0).swap(output_state.col(c1));
                    output_state.col(c0) *= imag;
                    output_state.col(c1) *= imag_conj;
                }
            }
        }

        // If we have a Z on qubit i, we multiply the amplitude of all basis states where qubit i is 1 by -1
    } else if (name == "Z" && control_qubits.empty()) {
        if (output_state.cols() == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long i = base + offset;
                output_state(i + stride) *= -1.0;
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long r = 0; r < half; ++r) {
                long offset = r & offset_mask;
                long base = (r & block_mask) << 1;
                long i = base + offset;
                output_state.row(i + stride) *= -1.0;
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long c = 0; c < half; ++c) {
                long offset = c & offset_mask;
                long base = (c & block_mask) << 1;
                long i = base + offset;
                output_state.col(i + stride) *= -1.0;
            }
        } else if (application_type == MatrixFreeApplicationType::LeftAndRight) {
#if defined(_OPENMP)
#pragma omp parallel
#endif
            {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long r = 0; r < half; ++r) {
                    long offset = r & offset_mask;
                    long base = (r & block_mask) << 1;
                    long i = base + offset;
                    output_state.row(i + stride) *= -1.0;
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long c = 0; c < half; ++c) {
                    long offset = c & offset_mask;
                    long base = (c & block_mask) << 1;
                    long i = base + offset;
                    output_state.col(i + stride) *= -1.0;
                }
            }
        }

        // If we have a H on qubit i, we apply the transformation |0> -> (|0> + |1>)/sqrt(2), |1> -> (|0> - |1>)/sqrt(2) to all basis states
    } else if (name == "H" && control_qubits.empty()) {
        if (output_state.cols() == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long i = base + offset;
                long j = i + stride;
                Complex temp_i = output_state(i);
                Complex temp_j = output_state(j);
                output_state(i) = (temp_i + temp_j) * inv_sqrt_2;
                output_state(j) = (temp_i - temp_j) * inv_sqrt_2;
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long r0 = base + offset;
                long r1 = r0 + stride;
                DenseRowVector temp0 = output_state.row(r0);
                DenseRowVector temp1 = output_state.row(r1);
                output_state.row(r0) = (temp0 + temp1) * inv_sqrt_2;
                output_state.row(r1) = (temp0 - temp1) * inv_sqrt_2;
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long c0 = base + offset;
                long c1 = c0 + stride;
                DenseVector temp0 = output_state.col(c0);
                DenseVector temp1 = output_state.col(c1);
                output_state.col(c0) = (temp0 + temp1) * inv_sqrt_2;
                output_state.col(c1) = (temp0 - temp1) * inv_sqrt_2;
            }
        } else if (application_type == MatrixFreeApplicationType::LeftAndRight) {
#if defined(_OPENMP)
#pragma omp parallel
#endif
            {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long k = 0; k < half; ++k) {
                    long offset = k & offset_mask;
                    long base = (k & block_mask) << 1;
                    long r0 = base + offset;
                    long r1 = r0 + stride;
                    DenseRowVector temp0 = output_state.row(r0);
                    DenseRowVector temp1 = output_state.row(r1);
                    output_state.row(r0) = (temp0 + temp1) * inv_sqrt_2;
                    output_state.row(r1) = (temp0 - temp1) * inv_sqrt_2;
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long k = 0; k < half; ++k) {
                    long offset = k & offset_mask;
                    long base = (k & block_mask) << 1;
                    long c0 = base + offset;
                    long c1 = c0 + stride;
                    DenseVector temp0 = output_state.col(c0);
                    DenseVector temp1 = output_state.col(c1);
                    output_state.col(c0) = (temp0 + temp1) * inv_sqrt_2;
                    output_state.col(c1) = (temp0 - temp1) * inv_sqrt_2;
                }
            }
        }

        // If we have a S on qubit i, we multiply the amplitude of all basis states where qubit i is 1 by i
    } else if (name == "S" && control_qubits.empty()) {
        if (output_state.cols() == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long i = base + offset;
                output_state(i + stride) *= imag;
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long r = 0; r < half; ++r) {
                long offset = r & offset_mask;
                long base = (r & block_mask) << 1;
                long i = base + offset;
                output_state.row(i + stride) *= imag;
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long c = 0; c < half; ++c) {
                long offset = c & offset_mask;
                long base = (c & block_mask) << 1;
                long i = base + offset;
                output_state.col(i + stride) *= imag_conj;
            }
        } else if (application_type == MatrixFreeApplicationType::LeftAndRight) {
#if defined(_OPENMP)
#pragma omp parallel
#endif
            {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long r = 0; r < half; ++r) {
                    long offset = r & offset_mask;
                    long base = (r & block_mask) << 1;
                    long i = base + offset;
                    output_state.row(i + stride) *= imag;
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long c = 0; c < half; ++c) {
                    long offset = c & offset_mask;
                    long base = (c & block_mask) << 1;
                    long i = base + offset;
                    output_state.col(i + stride) *= imag_conj;
                }
            }
        }

        // If we have a T on qubit i, we multiply the amplitude of all basis states where qubit i is 1 by exp(i*pi/4)
    } else if (name == "T" && control_qubits.empty()) {
        if (output_state.cols() == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long i = base + offset;
                output_state(i + stride) *= t_phase;
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long r = 0; r < half; ++r) {
                long offset = r & offset_mask;
                long base = (r & block_mask) << 1;
                long i = base + offset;
                output_state.row(i + stride) *= t_phase;
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long c = 0; c < half; ++c) {
                long offset = c & offset_mask;
                long base = (c & block_mask) << 1;
                long i = base + offset;
                output_state.col(i + stride) *= t_phase_conj;
            }
        } else if (application_type == MatrixFreeApplicationType::LeftAndRight) {
#if defined(_OPENMP)
#pragma omp parallel
#endif
            {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long r = 0; r < half; ++r) {
                    long offset = r & offset_mask;
                    long base = (r & block_mask) << 1;
                    long i = base + offset;
                    output_state.row(i + stride) *= t_phase;
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long c = 0; c < half; ++c) {
                    long offset = c & offset_mask;
                    long base = (c & block_mask) << 1;
                    long i = base + offset;
                    output_state.col(i + stride) *= t_phase_conj;
                }
            }
        }

        // If we have a SWAP between qubits i and j, we swap the amplitudes of all basis states where qubit i is 0 and qubit j is 1 with those where qubit i is 1 and qubit j is 0
    } else if (name == "SWAP" && target_qubits.size() == 2 && control_qubits.empty()) {
        long other_mask = 1L << (num_qubits - 1 - target_qubits[1]);
        long mask0 = mask;
        long mask1 = other_mask;
        if (mask0 > mask1) {
            std::swap(mask0, mask1);
        }
        long half_dim = dim >> 2;

        if (output_state.cols() == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long k = 0; k < half_dim; ++k) {
                long i = (k & (mask0 - 1)) | ((k & ~(mask1 - 1)) << 2) | ((k & (mask1 - 1) & ~(mask0 - 1)) << 1) | mask1;
                long j = i ^ (mask0 | mask1);
                std::swap(output_state(i), output_state(j));
            }

        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long k = 0; k < half_dim; ++k) {
                long i = (k & (mask0 - 1)) | ((k & ~(mask1 - 1)) << 2) | ((k & (mask1 - 1) & ~(mask0 - 1)) << 1) | mask1;
                long j = i ^ (mask0 | mask1);
                output_state.row(i).swap(output_state.row(j));
            }

        } else if (application_type == MatrixFreeApplicationType::Right) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long k = 0; k < half_dim; ++k) {
                long i = (k & (mask0 - 1)) | ((k & ~(mask1 - 1)) << 2) | ((k & (mask1 - 1) & ~(mask0 - 1)) << 1) | mask1;
                long j = i ^ (mask0 | mask1);
                output_state.col(i).swap(output_state.col(j));
            }

        } else if (application_type == MatrixFreeApplicationType::LeftAndRight) {
#if defined(_OPENMP)
#pragma omp parallel
#endif
            {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long k = 0; k < half_dim; ++k) {
                    long i = (k & (mask0 - 1)) | ((k & ~(mask1 - 1)) << 2) | ((k & (mask1 - 1) & ~(mask0 - 1)) << 1) | mask1;
                    long j = i ^ (mask0 | mask1);
                    output_state.row(i).swap(output_state.row(j));
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long k = 0; k < half_dim; ++k) {
                    long i = (k & (mask0 - 1)) | ((k & ~(mask1 - 1)) << 2) | ((k & (mask1 - 1) & ~(mask0 - 1)) << 1) | mask1;
                    long j = i ^ (mask0 | mask1);
                    output_state.col(i).swap(output_state.col(j));
                }
            }
        }

        // If we have a CNOT with control qubit j and target qubit i, we swap the amplitudes of all basis states where qubit j is 1 and qubit i is 0 with those where qubit j is 1 and qubit i is 1
    } else if (name == "X" && control_qubits.size() == 1) {
        long control_mask = 1L << (num_qubits - 1 - control_qubits[0]);
        if (output_state.cols() == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long offset = k & offset_mask;
                long i = ((k & block_mask) << 1) + offset;
                if (i & control_mask) {
                    long j = i ^ mask;
                    if (i < j) {
                        std::swap(output_state(i), output_state(j));
                    }
                }
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long i = 0; i < long(dim); ++i) {
                if (i & control_mask) {
                    long j = i ^ mask;
                    if (i < j) {
                        output_state.row(i).swap(output_state.row(j));
                    }
                }
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long i = 0; i < dim; ++i) {
                if (i & control_mask) {
                    long j = i ^ mask;
                    if (i < j) {
                        output_state.col(i).swap(output_state.col(j));
                    }
                }
            }
        } else if (application_type == MatrixFreeApplicationType::LeftAndRight) {
#if defined(_OPENMP)
#pragma omp parallel
#endif
            {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long i = 0; i < dim; ++i) {
                    if (i & control_mask) {
                        long j = i ^ mask;
                        if (i < j) {
                            output_state.row(i).swap(output_state.row(j));
                        }
                    }
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long i = 0; i < long(dim); ++i) {
                    if (i & control_mask) {
                        long j = i ^ mask;
                        if (i < j) {
                            output_state.col(i).swap(output_state.col(j));
                        }
                    }
                }
            }
        }

        // If we have a Toffoli with control qubits j and k and target qubit i, we swap the amplitudes of all basis states where qubits j and k are 1 and qubit i is 0 with those where qubits j and k are 1 and qubit i is 1
    } else if (name == "X" && control_qubits.size() == 2) {
        long control_mask = 0;
        for (int control_qubit : control_qubits) {
            control_mask |= 1L << (num_qubits - 1 - control_qubit);
        }
        if (output_state.cols() == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long offset = k & offset_mask;
                long i = ((k & block_mask) << 1) + offset;
                if ((i & control_mask) == control_mask) {
                    long j = i ^ mask;
                    if (i < j) {
                        std::swap(output_state(i), output_state(j));
                    }
                }
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long i = 0; i < long(dim); ++i) {
                if ((i & control_mask) == control_mask) {
                    long j = i ^ mask;
                    if (i < j) {
                        output_state.row(i).swap(output_state.row(j));
                    }
                }
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long i = 0; i < dim; ++i) {
                if ((i & control_mask) == control_mask) {
                    long j = i ^ mask;
                    if (i < j) {
                        output_state.col(i).swap(output_state.col(j));
                    }
                }
            }
        } else if (application_type == MatrixFreeApplicationType::LeftAndRight) {
#if defined(_OPENMP)
#pragma omp parallel
#endif
            {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long i = 0; i < long(dim); ++i) {
                    if ((i & control_mask) == control_mask) {
                        long j = i ^ mask;
                        if (i < j) {
                            output_state.row(i).swap(output_state.row(j));
                        }
                    }
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long i = 0; i < dim; ++i) {
                    if ((i & control_mask) == control_mask) {
                        long j = i ^ mask;
                        if (i < j) {
                            output_state.col(i).swap(output_state.col(j));
                        }
                    }
                }
            }
        }

        // If we have a CZ with control qubit j and target qubit i, we multiply the amplitude of all basis states where qubit j is 1 and qubit i is 1 by -1
    } else if (name == "Z" && control_qubits.size() == 1) {
        long control_mask = 1L << (num_qubits - 1 - control_qubits[0]);
        if (output_state.cols() == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long k = 0; k < long(half); ++k) {
                long offset = k & offset_mask;
                long i = ((k & block_mask) << 1) + offset;
                if (i & control_mask) {
                    long j = i ^ mask;
                    if (i < j) {
                        output_state(j) *= -1.0;
                    }
                }
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long i = 0; i < long(dim); ++i) {
                if (i & control_mask) {
                    long j = i ^ mask;
                    if (i < j) {
                        output_state.row(j) *= -1.0;
                    }
                }
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long i = 0; i < long(dim); ++i) {
                if (i & control_mask) {
                    long j = i ^ mask;
                    if (i < j) {
                        output_state.col(j) *= -1.0;
                    }
                }
            }
        } else if (application_type == MatrixFreeApplicationType::LeftAndRight) {
#if defined(_OPENMP)
#pragma omp parallel
#endif
            {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long i = 0; i < long(dim); ++i) {
                    if (i & control_mask) {
                        long j = i ^ mask;
                        if (i < j) {
                            output_state.row(j) *= -1.0;
                        }
                    }
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long i = 0; i < long(dim); ++i) {
                    if (i & control_mask) {
                        long j = i ^ mask;
                        if (i < j) {
                            output_state.col(j) *= -1.0;
                        }
                    }
                }
            }
        }

        // If we have a 2x2 base matrix and a control qubit
    } else if (base_matrix.rows() == 2 && base_matrix.cols() == 2 && control_qubits.size() == 1) {
        long control_mask = 1L << (num_qubits - 1 - control_qubits[0]);
        if (output_state.cols() == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long k = 0; k < long(half); ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long i = base + offset;
                if (i & control_mask) {
                    long j = i ^ mask;
                    Complex temp_i = output_state(i);
                    Complex temp_j = output_state(j);
                    output_state(i) = base_matrix(0, 0) * temp_i + base_matrix(0, 1) * temp_j;
                    output_state(j) = base_matrix(1, 0) * temp_i + base_matrix(1, 1) * temp_j;
                }
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < long(half); ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long r0 = base + offset;
                long r1 = r0 + stride;
                if (r0 & control_mask) {
                    DenseRowVector temp0 = output_state.row(r0);
                    DenseRowVector temp1 = output_state.row(r1);
                    output_state.row(r0) = base_matrix(0, 0) * temp0 + base_matrix(0, 1) * temp1;
                    output_state.row(r1) = base_matrix(1, 0) * temp0 + base_matrix(1, 1) * temp1;
                }
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
            DenseMatrix base_matrix_conj = base_matrix.conjugate().transpose();
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < long(half); ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long c0 = base + offset;
                long c1 = c0 + stride;
                if (c0 & control_mask) {
                    DenseVector temp0 = output_state.col(c0);
                    DenseVector temp1 = output_state.col(c1);
                    output_state.col(c0) = base_matrix_conj(0, 0) * temp0 + base_matrix_conj(1, 0) * temp1;
                    output_state.col(c1) = base_matrix_conj(0, 1) * temp0 + base_matrix_conj(1, 1) * temp1;
                }
            }
        } else if (application_type == MatrixFreeApplicationType::LeftAndRight) {
            DenseMatrix base_matrix_conj = base_matrix.conjugate().transpose();
#if defined(_OPENMP)
#pragma omp parallel
#endif
            {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long k = 0; k < long(half); ++k) {
                    long offset = k & offset_mask;
                    long base = (k & block_mask) << 1;
                    long r0 = base + offset;
                    long r1 = r0 + stride;
                    if (r0 & control_mask) {
                        DenseRowVector temp0 = output_state.row(r0);
                        DenseRowVector temp1 = output_state.row(r1);
                        output_state.row(r0) = base_matrix(0, 0) * temp0 + base_matrix(0, 1) * temp1;
                        output_state.row(r1) = base_matrix(1, 0) * temp0 + base_matrix(1, 1) * temp1;
                    }
                }

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long k = 0; k < long(half); ++k) {
                    long offset = k & offset_mask;
                    long base = (k & block_mask) << 1;
                    long c0 = base + offset;
                    long c1 = c0 + stride;
                    if (c0 & control_mask) {
                        DenseVector temp0 = output_state.col(c0);
                        DenseVector temp1 = output_state.col(c1);
                        output_state.col(c0) = base_matrix_conj(0, 0) * temp0 + base_matrix_conj(1, 0) * temp1;
                        output_state.col(c1) = base_matrix_conj(0, 1) * temp0 + base_matrix_conj(1, 1) * temp1;
                    }
                }
            }
        }

        // If we have a 2x2 base matrix, we apply it by treating the target qubit as the least significant bit and iterating through pairs of basis states
    } else if (base_matrix.rows() == 2 && base_matrix.cols() == 2) {
        if (output_state.cols() == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long k = 0; k < long(half); ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long i = base + offset;
                long j = i + stride;
                Complex temp_i = output_state(i);
                Complex temp_j = output_state(j);
                output_state(i) = base_matrix(0, 0) * temp_i + base_matrix(0, 1) * temp_j;
                output_state(j) = base_matrix(1, 0) * temp_i + base_matrix(1, 1) * temp_j;
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < long(half); ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long r0 = base + offset;
                long r1 = r0 + stride;
                DenseRowVector temp0 = output_state.row(r0);
                DenseRowVector temp1 = output_state.row(r1);
                output_state.row(r0) = base_matrix(0, 0) * temp0 + base_matrix(0, 1) * temp1;
                output_state.row(r1) = base_matrix(1, 0) * temp0 + base_matrix(1, 1) * temp1;
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
            DenseMatrix base_matrix_conj = base_matrix.conjugate().transpose();
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < long(half); ++k) {
                long offset = k & offset_mask;
                long base = (k & block_mask) << 1;
                long c0 = base + offset;
                long c1 = c0 + stride;
                DenseVector temp0 = output_state.col(c0);
                DenseVector temp1 = output_state.col(c1);
                output_state.col(c0) = base_matrix_conj(0, 0) * temp0 + base_matrix_conj(1, 0) * temp1;
                output_state.col(c1) = base_matrix_conj(0, 1) * temp0 + base_matrix_conj(1, 1) * temp1;
            }
        } else if (application_type == MatrixFreeApplicationType::LeftAndRight) {
            DenseMatrix base_matrix_conj = base_matrix.conjugate().transpose();
#if defined(_OPENMP)
#pragma omp parallel
#endif
            {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long k = 0; k < long(half); ++k) {
                    long offset = k & offset_mask;
                    long base = (k & block_mask) << 1;
                    long r0 = base + offset;
                    long r1 = r0 + stride;
                    DenseRowVector temp0 = output_state.row(r0);
                    DenseRowVector temp1 = output_state.row(r1);
                    output_state.row(r0) = base_matrix(0, 0) * temp0 + base_matrix(0, 1) * temp1;
                    output_state.row(r1) = base_matrix(1, 0) * temp0 + base_matrix(1, 1) * temp1;
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long k = 0; k < long(half); ++k) {
                    long offset = k & offset_mask;
                    long base = (k & block_mask) << 1;
                    long c0 = base + offset;
                    long c1 = c0 + stride;
                    DenseVector temp0 = output_state.col(c0);
                    DenseVector temp1 = output_state.col(c1);
                    output_state.col(c0) = base_matrix_conj(0, 0) * temp0 + base_matrix_conj(1, 0) * temp1;
                    output_state.col(c1) = base_matrix_conj(0, 1) * temp0 + base_matrix_conj(1, 1) * temp1;
                }
            }
        }

        // A general gate acting on k target qubits (e.g. a fused block). We
        // iterate over the 2^(num_qubits - k) "anchors" (basis states with all
        // target bits zero); for each, gather the 2^k amplitudes that differ only
        // in the target bits, multiply by the gate matrix, and scatter back. This
        // touches the statevector once per fused block instead of once per gate.
        //
        // The matrix is applied in sparse (CSR) form so the per-anchor cost scales
        // with the number of nonzero entries rather than (2^k)^2. Fused blocks made
        // of permutations/diagonals (CNOT, Pauli, Z, S, T, CZ, ...) stay cheap,
        // while genuinely dense blocks (e.g. stacked rotations) still work.
    } else if (control_qubits.empty() && base_matrix.rows() == base_matrix.cols() && base_matrix.rows() == (1LL << target_qubits.size())) {
        int k = static_cast<int>(target_qubits.size());
        long dim_k = 1L << k;

        // Full-state bit position of each (sorted) target qubit. In the dense
        // matrix, local qubit li corresponds to bit (k - 1 - li), matching the
        // big-endian convention used when the fused matrix was built.
        std::vector<long> target_pos(k);
        for (int li = 0; li < k; ++li) {
            target_pos[li] = num_qubits - 1 - target_qubits[li];
        }

        // Precompute, for each matrix index m, the offset of its target bits
        // within a full-state index.
        std::vector<long> offsets(dim_k, 0);
        for (long m = 0; m < dim_k; ++m) {
            long off = 0;
            for (int li = 0; li < k; ++li) {
                if ((m >> (k - 1 - li)) & 1L) {
                    off |= 1L << target_pos[li];
                }
            }
            offsets[m] = off;
        }

        // Sorted (ascending) target bit positions, used to turn an anchor
        // counter into a full-state index by inserting a zero bit at each
        // target position (k cheap steps instead of scanning every free bit).
        std::vector<long> sorted_target_pos = target_pos;
        std::sort(sorted_target_pos.begin(), sorted_target_pos.end());
        long num_anchors = 1L << (num_qubits - k);

        // Mask of "free" (non-target) bit positions within the num_qubits-bit
        // index. Depositing an anchor counter's bits into these positions is
        // exactly the zero-bit-insertion above, but as a single pdep instruction
        // (BMI2) instead of a k-iteration loop. The loop remains as a fallback for
        // targets without BMI2 (e.g. aarch64).
        [[maybe_unused]] long free_mask = (num_qubits >= 64) ? ~0L : ((1L << num_qubits) - 1);
        for (int li = 0; li < k; ++li) {
            free_mask &= ~(1L << sorted_target_pos[li]);
        }

        // Build a compact CSR view of the gate matrix, skipping structural zeros
        // (built once, not per anchor; (2^k)^2 work, negligible vs the state pass).
        // The values are kept as split real/imag arrays so the per-nonzero multiply
        // in the hot loop is plain scalar FMA. Using std::complex here instead would
        // make libstdc++ emit its C99 Annex G NaN-correction branch (a compare + jp
        // per element) inside the innermost loop, which dominates the profile and
        // blocks vectorization.
        std::vector<long> row_start(dim_k + 1, 0);
        std::vector<int> col_idx;
        std::vector<Real> val_re;
        std::vector<Real> val_im;
        col_idx.reserve(dim_k * dim_k);
        val_re.reserve(dim_k * dim_k);
        val_im.reserve(dim_k * dim_k);
        for (long r = 0; r < dim_k; ++r) {
            row_start[r] = static_cast<long>(col_idx.size());
            for (long col = 0; col < dim_k; ++col) {
                Complex v = base_matrix(r, col);
                if (v != Complex(0.0, 0.0)) {
                    col_idx.push_back(static_cast<int>(col));
                    val_re.push_back(v.real());
                    val_im.push_back(v.imag());
                }
            }
        }
        row_start[dim_k] = static_cast<long>(col_idx.size());

        if (output_state.cols() == 1) {
            // Process anchors in batches of B. The gate matrix is identical for
            // every anchor, so the CSR coefficients (vr/vi) are broadcast scalars
            // while the per-anchor amplitudes form contiguous B-wide arrays. The
            // accumulators then become B independent lanes instead of one serial
            // real/imag reduction, so the innermost loop vectorizes (SIMD across
            // anchors) instead of running one scalar matrix element at a time.
            constexpr long B = 8;
#if defined(_OPENMP)
#pragma omp parallel
#endif
            {
                // Layout: [m][b] with b contiguous, so the b-loop is unit-stride.
                std::vector<Real> in_re(dim_k * B);
                std::vector<Real> in_im(dim_k * B);
                std::vector<Real> out_re(dim_k * B);
                std::vector<Real> out_im(dim_k * B);
                std::vector<long> anchors(B);
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long c0 = 0; c0 < num_anchors; c0 += B) {
                    const long bw = std::min<long>(B, num_anchors - c0);

                    // Full-state base index for each anchor in the batch: insert a
                    // zero bit at each (ascending) target position so the remaining
                    // bits of the counter land in the free positions in order.
                    for (long b = 0; b < bw; ++b) {
#if defined(__BMI2__)
                        anchors[b] = static_cast<long>(
                            _pdep_u64(static_cast<unsigned long>(c0 + b), static_cast<unsigned long>(free_mask)));
#else
                        long anchor = c0 + b;
                        for (int s = 0; s < k; ++s) {
                            long p = sorted_target_pos[s];
                            long lower = anchor & ((1L << p) - 1);
                            long upper = anchor & ~((1L << p) - 1);
                            anchor = lower | (upper << 1);
                        }
                        anchors[b] = anchor;
#endif
                    }

                    // Gather the 2^k amplitudes for every anchor in the batch.
                    for (long m = 0; m < dim_k; ++m) {
                        const long off = offsets[m];
                        Real* ir = &in_re[m * B];
                        Real* ii = &in_im[m * B];
                        for (long b = 0; b < bw; ++b) {
                            const Complex amp = output_state(anchors[b] + off);
                            ir[b] = amp.real();
                            ii[b] = amp.imag();
                        }
                    }

                    // Sparse matrix * (dim_k x B) block. Accumulators are local
                    // fixed-size arrays (kept in registers, stored once at the end)
                    // and the gathered inputs are marked __restrict, so the compiler
                    // can prove no aliasing between the in/out buffers and emit a
                    // packed b-loop. Without both, it falls back to a scalar
                    // load+FMA+store per element (no SIMD). B is a power of two so
                    // the b-loop maps cleanly onto whatever vector width the generic
                    // target provides (SSE2 is guaranteed on all x86-64).
                    for (long r = 0; r < dim_k; ++r) {
                        Real sr[B];
                        Real si[B];
                        for (long b = 0; b < B; ++b) {
                            sr[b] = 0.0;
                            si[b] = 0.0;
                        }
                        for (long j = row_start[r]; j < row_start[r + 1]; ++j) {
                            const int col = col_idx[j];
                            const Real vr = val_re[j];
                            const Real vi = val_im[j];
                            const Real* __restrict ir = &in_re[col * B];
                            const Real* __restrict ii = &in_im[col * B];
                            // Request true SIMD over the B independent lanes: the
                            // compiler packs ir[0..B)/ii[0..B) into vector loads and
                            // emits one packed FMA-pair per nonzero, instead of B
                            // scalar FMAs. A plain unroll pragma here forces a scalar
                            // unroll that the vectorizer then fails to re-pack.
#if defined(_OPENMP)
#pragma omp simd
#endif
                            for (long b = 0; b < B; ++b) {
                                sr[b] += vr * ir[b] - vi * ii[b];
                                si[b] += vr * ii[b] + vi * ir[b];
                            }
                        }
                        Real* __restrict sro = &out_re[r * B];
                        Real* __restrict sio = &out_im[r * B];
                        for (long b = 0; b < B; ++b) {
                            sro[b] = sr[b];
                            sio[b] = si[b];
                        }
                    }

                    // Scatter the results back.
                    for (long m = 0; m < dim_k; ++m) {
                        const long off = offsets[m];
                        const Real* sr = &out_re[m * B];
                        const Real* si = &out_im[m * B];
                        for (long b = 0; b < bw; ++b) {
                            output_state(anchors[b] + off) = Complex(sr[b], si[b]);
                        }
                    }
                }
            }
        } else {
            // Fusion is only applied to statevectors, so a dense multi-qubit
            // gate should never reach the density-matrix paths.
            throw std::invalid_argument("Dense multi-qubit (fused) operators are only supported for statevectors, not density matrices.");
        }

    } else {
        std::stringstream ss;
        ss << "Unknown operator: " << get_id();
        throw std::invalid_argument(ss.str());
    }
}

MatrixFreeOperator::MatrixFreeOperator(const std::string& name, const std::vector<int>& control_qubits, const std::vector<int>& target_qubits, const DenseMatrix& base_matrix) {
    /*
    Main constructor for MatrixFreeOperator.

    Args:
        name (const std::string&): The name of the operator, e.g. "X", "H", "CNOT", etc.
        control_qubits (const std::vector<int>&): The control qubits for the operator, if it is a controlled operator.
        target_qubits (const std::vector<int>&): The target qubits for the operator.
        base_matrix (const DenseMatrix&): The base matrix for the operator, if it is a custom operator defined by a 2x2 unitary.

    Returns:
        MatrixFreeOperator: The resulting operator after construction.

    Throws:
        py::value_error: If the operator has more than 1 control qubit (other than CCX)
        py::value_error: If the operator has a number of target qubits not equal to 1 (other than SWAP)

    */
    this->name = name;
    this->control_qubits = control_qubits;
    this->target_qubits = target_qubits;
    this->base_matrix = base_matrix;

    // Get rid of superfluous names for common gates
    if (this->name == "CCX" || this->name == "Toffoli" || this->name == "CX" || this->name == "CNOT") {
        this->name = "X";
    }
    if (this->name == "CY" || this->name == "CCY") {
        this->name = "Y";
    }
    if (this->name == "CZ" || this->name == "CCZ") {
        this->name = "Z";
    }

    // A dense multi-qubit gate (e.g. produced by gate fusion): a full 2^k x 2^k
    // unitary acting on k target qubits with no separate control structure.
    bool is_dense_multi_qubit = this->control_qubits.empty() && this->base_matrix.rows() == this->base_matrix.cols() && this->base_matrix.rows() == (1LL << this->target_qubits.size());

    // Checks
    if (this->control_qubits.size() > 1 && !(this->name == "X" && this->control_qubits.size() == 2)) {
        throw py::value_error("MatrixFreeOperator only supports gates with 1 or fewer total control qubits (other than CCX).");
    }
    if (this->target_qubits.size() != 1 && this->name != "SWAP" && this->name != "M" && !is_dense_multi_qubit) {
        throw py::value_error("MatrixFreeOperator requires a gate with exactly 1 target qubit (other than SWAP, M, or a dense multi-qubit gate).");
    }
}

std::ostream& operator<<(std::ostream& os, const MatrixFreeOperator& op) {
    /*
    Output a human-readable representation of the operator, including its name and target/control qubits if applicable.
    Usage is like std::cout << op << std::endl;

    Args:
        os (std::ostream&): The output stream to write to.
        op (const MatrixFreeOperator&): The operator to output.

    Returns:
        std::ostream&: The output stream after writing the operator to it.
    */
    os << op.name << "(";
    for (size_t i = 0; i < op.target_qubits.size(); ++i) {
        os << op.target_qubits[i];
        if (i < op.target_qubits.size() - 1) {
            os << ",";
        }
    }
    os << ")";
    for (size_t i = 0; i < op.control_qubits.size(); ++i) {
        os << "_c" << op.control_qubits[i];
    }
    return os;
}

bool MatrixFreeOperator::operator==(const MatrixFreeOperator& other) const {
    /*
    Check if this operator is equal to another operator by comparing their names, target qubits, control qubits, and base matrices.

    Args:
        other (const MatrixFreeOperator&): The operator to compare to.

    Returns:
        bool: True if the operators are equal, false otherwise.
    */
    return name == other.name && target_qubits == other.target_qubits && control_qubits == other.control_qubits && base_matrix.isApprox(other.base_matrix);
}

std::string MatrixFreeOperator::get_id() const {
    /*
    Get a unique string identifier for the operator based on its name, target qubits and control qubits.

    Returns:
        std::string: The unique identifier for the operator.
    */
    std::stringstream id;
    id << *this;
    return id.str();
}

// GCOV_EXCL_BR_STOP