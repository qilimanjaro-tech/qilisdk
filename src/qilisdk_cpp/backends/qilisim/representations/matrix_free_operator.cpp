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

const std::complex<double> imag(0.0, 1.0);
const std::complex<double> imag_conj(0.0, -1.0);
const double inv_sqrt_2 = 1.0 / std::sqrt(2.0);
const std::complex<double> t_phase = std::exp(std::complex<double>(0.0, M_PI / 4.0));
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
    long long mask = 1LL << (num_qubits - 1 - target_qubit);
    long N = output_state.rows();
    long stride = mask;
    long half = N >> 1;
    long dim = 1LL << num_qubits;

    // If we have an X on qubit i, we swap the amplitudes of all basis states where qubit i is 0 with those where qubit i is 1
    if (name == "X") {
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
    } else if (name == "Y") {
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
                for (long r = 0; r < dim; ++r) {
                    bool bit = (r & mask);
                    output_state.row(r) *= (bit ? imag_conj : imag);
                }
            }
        } else if (application_type == MatrixFreeApplicationType::Right) {
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
                    long c0 = base + offset;
                    long c1 = c0 + stride;
                    output_state.col(c0).swap(output_state.col(c1));
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long c = 0; c < dim; ++c) {
                    bool bit = (c & mask);
                    output_state.col(c) *= (bit ? imag : imag_conj);
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
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long r = 0; r < dim; ++r) {
                    bool bit = (r & mask);
                    output_state.row(r) *= (bit ? imag : imag_conj);
                }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (long c = 0; c < dim; ++c) {
                    bool bit = (c & mask);
                    output_state.col(c) *= (bit ? imag_conj : imag);
                }
            }
        }

        // If we have a Z on qubit i, we multiply the amplitude of all basis states where qubit i is 1 by -1
    } else if (name == "Z") {
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
    } else if (name == "H") {
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
    } else if (name == "S") {
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
    } else if (name == "T") {
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

        // If we have a CNOT with control qubit j and target qubit i, we swap the amplitudes of all basis states where qubit j is 1 and qubit i is 0 with those where qubit j is 1 and qubit i is 1
    } else if (name == "CNOT") {
        long control_mask = 1L << (num_qubits - 1 - control_qubit);
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

        // If we have a CZ with control qubit j and target qubit i, we multiply the amplitude of all basis states where qubit j is 1 and qubit i is 1 by -1
    } else if (name == "CZ") {
        long control_mask = 1L << (num_qubits - 1 - control_qubit);
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
                output_state.col(c0) = base_matrix(0, 0) * temp0 + base_matrix(0, 1) * temp1;
                output_state.col(c1) = base_matrix(1, 0) * temp0 + base_matrix(1, 1) * temp1;
            }
        } else if (application_type == MatrixFreeApplicationType::LeftAndRight) {
            DenseMatrix base_matrix_conj = base_matrix.conjugate();
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
                    output_state.col(c0) = base_matrix_conj(0, 0) * temp0 + base_matrix_conj(0, 1) * temp1;
                    output_state.col(c1) = base_matrix_conj(1, 0) * temp0 + base_matrix_conj(1, 1) * temp1;
                }
            }
        }
    } else {
        throw std::runtime_error("Unknown operator: " + name);
    }
}

MatrixFreeOperator::MatrixFreeOperator(const Gate& gate) {
    /*
    Construct a MatrixFreeOperator from a given gate by extracting the target and control qubits, and the name of the gate.

    Args:
        gate (const Gate&): The gate to construct the operator from.

    Returns:
        MatrixFreeOperator: The resulting operator after construction.

    Raises:
        std::invalid_argument: If the gate has more than 1 control qubits or does not have exactly 1 target qubit.
    */
    if (gate.get_control_qubits().size() > 1) {
        throw std::invalid_argument("MatrixFreeOperator only supports gates with 1 or fewer total control qubits.");
    }
    if (gate.get_target_qubits().size() != 1) {
        throw std::invalid_argument("MatrixFreeOperator requires a gate with exactly 1 target qubit.");
    }
    target_qubit = gate.get_target_qubits()[0];
    control_qubit = gate.get_control_qubits().empty() ? -1 : gate.get_control_qubits()[0];
    base_matrix = gate.get_base_matrix();
    name = gate.get_name();
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
    os << op.name;
    if (op.target_qubit != -1) {
        os << "(" << op.target_qubit << ")";
    }
    if (op.control_qubit != -1) {
        os << "_c" << op.control_qubit;
    }
    return os;
}