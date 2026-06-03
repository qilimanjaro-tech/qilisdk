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

// GCOV_EXCL_BR_START

const std::complex<double> imag(0.0, 1.0);
const std::complex<double> imag_conj(0.0, -1.0);
const double inv_sqrt_2 = 1.0 / std::sqrt(2.0);
constexpr double pi = 3.14159265358979323846;
const std::complex<double> t_phase = std::exp(std::complex<double>(0.0, pi / 4.0));
const std::complex<double> t_phase_conj = std::conj(t_phase);

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
    long half = N >> 1;
    long dim = 1LL << num_qubits;

    // If we have an X on qubit i, we swap the amplitudes of all basis states where qubit i is 0 with those where qubit i is 1
    if (name == "X" && control_qubits.empty()) {
        if (output_state.cols() == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
                long i = base + offset;
                long j = i + stride;
                std::swap(output_state(i), output_state(j));
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
                long r0 = base + offset;
                long r1 = r0 + stride;
                output_state.row(r0).swap(output_state.row(r1));
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
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
                    long block = k / stride;
                    long offset = k % stride;
                    long base = block * (stride << 1);
                    long r0 = base + offset;
                    long r1 = r0 + stride;
                    output_state.row(r0).swap(output_state.row(r1));
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long k = 0; k < half; ++k) {
                    long block = k / stride;
                    long offset = k % stride;
                    long base = block * (stride << 1);
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
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
                long i = base + offset;
                long j = i + stride;
                std::complex<double> temp = output_state(i);
                output_state(i) = output_state(j) * imag_conj;
                output_state(j) = temp * imag;
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
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
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
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
                    long block = k / stride;
                    long offset = k % stride;
                    long base = block * (stride << 1);
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
                    long block = k / stride;
                    long offset = k % stride;
                    long base = block * (stride << 1);
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
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
                long i = base + offset;
                output_state(i + stride) *= -1.0;
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long r = 0; r < half; ++r) {
                long block = r / stride;
                long offset = r % stride;
                long base = block * (stride << 1);
                long i = base + offset;
                output_state.row(i + stride) *= -1.0;
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long c = 0; c < half; ++c) {
                long block = c / stride;
                long offset = c % stride;
                long base = block * (stride << 1);
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
                    long block = r / stride;
                    long offset = r % stride;
                    long base = block * (stride << 1);
                    long i = base + offset;
                    output_state.row(i + stride) *= -1.0;
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long c = 0; c < half; ++c) {
                    long block = c / stride;
                    long offset = c % stride;
                    long base = block * (stride << 1);
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
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
                long i = base + offset;
                long j = i + stride;
                std::complex<double> temp_i = output_state(i);
                std::complex<double> temp_j = output_state(j);
                output_state(i) = (temp_i + temp_j) * inv_sqrt_2;
                output_state(j) = (temp_i - temp_j) * inv_sqrt_2;
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
                long r0 = base + offset;
                long r1 = r0 + stride;
                Eigen::RowVectorXcd temp0 = output_state.row(r0);
                Eigen::RowVectorXcd temp1 = output_state.row(r1);
                output_state.row(r0) = (temp0 + temp1) * inv_sqrt_2;
                output_state.row(r1) = (temp0 - temp1) * inv_sqrt_2;
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < half; ++k) {
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
                long c0 = base + offset;
                long c1 = c0 + stride;
                Eigen::VectorXcd temp0 = output_state.col(c0);
                Eigen::VectorXcd temp1 = output_state.col(c1);
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
                    long block = k / stride;
                    long offset = k % stride;
                    long base = block * (stride << 1);
                    long r0 = base + offset;
                    long r1 = r0 + stride;
                    Eigen::RowVectorXcd temp0 = output_state.row(r0);
                    Eigen::RowVectorXcd temp1 = output_state.row(r1);
                    output_state.row(r0) = (temp0 + temp1) * inv_sqrt_2;
                    output_state.row(r1) = (temp0 - temp1) * inv_sqrt_2;
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long k = 0; k < half; ++k) {
                    long block = k / stride;
                    long offset = k % stride;
                    long base = block * (stride << 1);
                    long c0 = base + offset;
                    long c1 = c0 + stride;
                    Eigen::VectorXcd temp0 = output_state.col(c0);
                    Eigen::VectorXcd temp1 = output_state.col(c1);
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
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
                long i = base + offset;
                output_state(i + stride) *= imag;
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long r = 0; r < half; ++r) {
                long block = r / stride;
                long offset = r % stride;
                long base = block * (stride << 1);
                long i = base + offset;
                output_state.row(i + stride) *= imag;
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long c = 0; c < half; ++c) {
                long block = c / stride;
                long offset = c % stride;
                long base = block * (stride << 1);
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
                    long block = r / stride;
                    long offset = r % stride;
                    long base = block * (stride << 1);
                    long i = base + offset;
                    output_state.row(i + stride) *= imag;
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long c = 0; c < half; ++c) {
                    long block = c / stride;
                    long offset = c % stride;
                    long base = block * (stride << 1);
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
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
                long i = base + offset;
                output_state(i + stride) *= t_phase;
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long r = 0; r < half; ++r) {
                long block = r / stride;
                long offset = r % stride;
                long base = block * (stride << 1);
                long i = base + offset;
                output_state.row(i + stride) *= t_phase;
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long c = 0; c < half; ++c) {
                long block = c / stride;
                long offset = c % stride;
                long base = block * (stride << 1);
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
                    long block = r / stride;
                    long offset = r % stride;
                    long base = block * (stride << 1);
                    long i = base + offset;
                    output_state.row(i + stride) *= t_phase;
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long c = 0; c < half; ++c) {
                    long block = c / stride;
                    long offset = c % stride;
                    long base = block * (stride << 1);
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
                long block = k / stride;
                long offset = k % stride;
                long i = block * (stride << 1) + offset;
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
                long block = k / stride;
                long offset = k % stride;
                long i = block * (stride << 1) + offset;
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
                long block = k / stride;
                long offset = k % stride;
                long i = block * (stride << 1) + offset;
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
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
                long i = base + offset;
                if (i & control_mask) {
                    long j = i ^ mask;
                    std::complex<double> temp_i = output_state(i);
                    std::complex<double> temp_j = output_state(j);
                    output_state(i) = base_matrix(0, 0) * temp_i + base_matrix(0, 1) * temp_j;
                    output_state(j) = base_matrix(1, 0) * temp_i + base_matrix(1, 1) * temp_j;
                }
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < long(half); ++k) {
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
                long r0 = base + offset;
                long r1 = r0 + stride;
                if (r0 & control_mask) {
                    Eigen::RowVectorXcd temp0 = output_state.row(r0);
                    Eigen::RowVectorXcd temp1 = output_state.row(r1);
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
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
                long c0 = base + offset;
                long c1 = c0 + stride;
                if (c0 & control_mask) {
                    Eigen::VectorXcd temp0 = output_state.col(c0);
                    Eigen::VectorXcd temp1 = output_state.col(c1);
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
                    long block = k / stride;
                    long offset = k % stride;
                    long base = block * (stride << 1);
                    long r0 = base + offset;
                    long r1 = r0 + stride;
                    if (r0 & control_mask) {
                        Eigen::RowVectorXcd temp0 = output_state.row(r0);
                        Eigen::RowVectorXcd temp1 = output_state.row(r1);
                        output_state.row(r0) = base_matrix(0, 0) * temp0 + base_matrix(0, 1) * temp1;
                        output_state.row(r1) = base_matrix(1, 0) * temp0 + base_matrix(1, 1) * temp1;
                    }
                }

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long k = 0; k < long(half); ++k) {
                    long block = k / stride;
                    long offset = k % stride;
                    long base = block * (stride << 1);
                    long c0 = base + offset;
                    long c1 = c0 + stride;
                    if (c0 & control_mask) {
                        Eigen::VectorXcd temp0 = output_state.col(c0);
                        Eigen::VectorXcd temp1 = output_state.col(c1);
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
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
                long i = base + offset;
                long j = i + stride;
                std::complex<double> temp_i = output_state(i);
                std::complex<double> temp_j = output_state(j);
                output_state(i) = base_matrix(0, 0) * temp_i + base_matrix(0, 1) * temp_j;
                output_state(j) = base_matrix(1, 0) * temp_i + base_matrix(1, 1) * temp_j;
            }
        } else if (application_type == MatrixFreeApplicationType::Left) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < long(half); ++k) {
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
                long r0 = base + offset;
                long r1 = r0 + stride;
                Eigen::RowVectorXcd temp0 = output_state.row(r0);
                Eigen::RowVectorXcd temp1 = output_state.row(r1);
                output_state.row(r0) = base_matrix(0, 0) * temp0 + base_matrix(0, 1) * temp1;
                output_state.row(r1) = base_matrix(1, 0) * temp0 + base_matrix(1, 1) * temp1;
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
            DenseMatrix base_matrix_conj = base_matrix.conjugate().transpose();
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (long k = 0; k < long(half); ++k) {
                long block = k / stride;
                long offset = k % stride;
                long base = block * (stride << 1);
                long c0 = base + offset;
                long c1 = c0 + stride;
                Eigen::VectorXcd temp0 = output_state.col(c0);
                Eigen::VectorXcd temp1 = output_state.col(c1);
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
                    long block = k / stride;
                    long offset = k % stride;
                    long base = block * (stride << 1);
                    long r0 = base + offset;
                    long r1 = r0 + stride;
                    Eigen::RowVectorXcd temp0 = output_state.row(r0);
                    Eigen::RowVectorXcd temp1 = output_state.row(r1);
                    output_state.row(r0) = base_matrix(0, 0) * temp0 + base_matrix(0, 1) * temp1;
                    output_state.row(r1) = base_matrix(1, 0) * temp0 + base_matrix(1, 1) * temp1;
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long k = 0; k < long(half); ++k) {
                    long block = k / stride;
                    long offset = k % stride;
                    long base = block * (stride << 1);
                    long c0 = base + offset;
                    long c1 = c0 + stride;
                    Eigen::VectorXcd temp0 = output_state.col(c0);
                    Eigen::VectorXcd temp1 = output_state.col(c1);
                    output_state.col(c0) = base_matrix_conj(0, 0) * temp0 + base_matrix_conj(1, 0) * temp1;
                    output_state.col(c1) = base_matrix_conj(0, 1) * temp0 + base_matrix_conj(1, 1) * temp1;
                }
            }
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

    // Checks
    if (this->control_qubits.size() > 1 && !(this->name == "X" && this->control_qubits.size() == 2)) {
        throw py::value_error("MatrixFreeOperator only supports gates with 1 or fewer total control qubits (other than CCX).");
    }
    if (this->target_qubits.size() != 1 && this->name != "SWAP" && this->name != "M") {
        throw py::value_error("MatrixFreeOperator requires a gate with exactly 1 target qubit (other than SWAP or M).");
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