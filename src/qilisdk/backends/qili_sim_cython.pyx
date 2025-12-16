# cython: language_level=3
# distutils: language = c++
# distutils: extra_compile_args = -g
# Copyright 2025 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from qilisdk.backends.backend import Backend
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.time_evolution import TimeEvolution


from libc.math cimport fabs
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.string cimport string
from cython.operator cimport dereference, postincrement
from libcpp.complex cimport complex
import cython
from libc.math cimport cos, sin, sqrt
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.map cimport map
from libcpp cimport bool
import numpy as np
import cmath

cdef extern from "<complex>":
    double abs(double complex)

cdef double atol_ = 1e-12  # tolerance for treating values as zero

cdef class SparseMatrix:
    """
    A simple sparse matrix representation using compressed sparse row (CSR) format.
    """
    cdef int nrows_
    cdef int ncols_
    cdef vector[int] rows_
    cdef vector[int] cols_
    cdef vector[complex] values_

    def __cinit__(self, int nrows = 0, int ncols = 0):
        self.nrows_ = nrows
        self.ncols_ = ncols
        self.rows_ = vector[int](nrows + 1, 0)
        self.cols_ = vector[int]()
        self.values_ = vector[complex]()

    @staticmethod
    def from_dense(vector[vector[complex]] dense) -> SparseMatrix:
        """
        Construct a sparse matrix from a dense vector of vectors.
        """
        cdef SparseMatrix mat = SparseMatrix(dense.size(), dense[0].size())
        cdef int r, c
        cdef double complex val
        for r in range(mat.nrows_):
            for c in range(mat.ncols_):
                val = dense[r][c]
                if abs(val) > atol_:
                    mat.cols_.push_back(c)
                    mat.values_.push_back(val)
                    mat.rows_[r + 1] += 1
        # convert counts to offsets
        for r in range(1, mat.nrows_ + 1):
            mat.rows_[r] += mat.rows_[r - 1]

    cdef SparseMatrix from_tuples(self, vector[tuple[int, int, complex]] entries, int nrows, int ncols):
        """
        Construct a sparse matrix from a list of (row, col, value) tuples.
        Entries must be sorted by (row, col).
        """
        cdef SparseMatrix mat = SparseMatrix(nrows, ncols)
        cdef int currentRow = 0
        cdef int nnzCount = 0
        mat.rows_ = vector[int](nrows + 2, 0)
        mat.cols_ = vector[int]()
        mat.values_ = vector[complex]()

        # ensure entries are sorted
        cdef int i
        for i in range(1, len(entries)):
            if entries[i][0] < entries[i-1][0] or \
               (entries[i][0] == entries[i-1][0] and entries[i][1] < entries[i-1][1]):
                raise RuntimeError("Entries must be sorted by (row, col)")

        for e in entries:
            row, col, val = e
            while currentRow < row:
                mat.rows_[currentRow + 1] = nnzCount
                currentRow += 1
            mat.cols_.push_back(col)
            mat.values_.push_back(val)
            nnzCount += 1

        while currentRow <= nrows:
            mat.rows_[currentRow + 1] = nnzCount
            currentRow += 1

    cdef vector[tuple[int, int, complex]] to_tuples(self):
        """
        Convert the sparse matrix to a list of (row, col, value) tuples.
        """
        cdef int r, idx
        cdef list entries = []
        for r in range(self.nrows_):
            for idx in range(self.rows_[r], self.rows_[r + 1]):
                entries.append((r, self.cols_[idx], self.values_[idx]))
        return entries

    cdef int get_width(self):
        return self.ncols_

    cdef double complex get(self, int row, int col):
        cdef int idx
        for idx in range(self.rows_[row], self.rows_[row + 1]):
            if self.cols_[idx] == col:
                return self.values_[idx]
        return 0.0

    cdef insert(self, int row, int col, double complex value):
        cdef int start = self.rows_[row]
        cdef int end = self.rows_[row + 1]
        cdef int insertPos = start
        while insertPos < end and self.cols_[insertPos] < col:
            insertPos += 1
        self.cols_.insert(self.cols_.begin() + insertPos, col)
        self.values_.insert(self.values_.begin() + insertPos, value)

        for r in range(row + 1, self.rows_.size()):
            self.rows_[r] += 1

    cdef string get_dims(self):
        return f"{self.nrows_}x{self.ncols_}"

    cdef SparseMatrix mul(self, SparseMatrix other):
        """
        Multiply two sparse matrices using C++ map for accumulation.
        """
        if self.ncols_ != other.nrows_:
            raise RuntimeError(f"Matrix dimensions do not match for multiplication: {self.get_dims()} * {other.get_dims()}")
        cdef map[pair[int,int], complex] entries_map
        cdef int r, idxA, idxB, colA, colB
        cdef double complex valA, valB
        cdef double complex prod 

        for r in range(self.nrows_):
            for idxA in range(self.rows_[r], self.rows_[r+1]):
                colA = self.cols_[idxA]
                valA = self.values_[idxA]
                for idxB in range(other.rows_[colA], other.rows_[colA+1]):
                    colB = other.cols_[idxB]
                    valB = other.values_[idxB]
                    prod = valA * valB
                    entries_map[(r, colB)] += prod

        cdef list entries = []
        cdef map[pair[int,int], complex].iterator it = entries_map.begin()
        while it != entries_map.end():
            k = dereference(it).first
            v = dereference(it).second
            if abs(v) > atol_:
                entries.append((k.first, k.second, v))
            postincrement(it)

        return SparseMatrix(entries, self.nrows_, other.ncols_)

# Identity matrix constant
cdef SparseMatrix I = SparseMatrix.from_dense([[1, 0],
                                              [0, 1]])

cdef class Gate:
    """
    A quantum gate with type, control qubits, target qubits, and parameters.
    """

    cdef string gate_type
    cdef vector[int] control_qubits
    cdef vector[int] target_qubits
    cdef vector[double] parameters

    cdef Gate(self, string type_, vector[int] controls_, vector[int] targets_, vector[double] params_):
        print("Creating gate of type:", type_)
        self.gate_type = type_
        self.control_qubits = controls_
        self.target_qubits = targets_
        self.parameters = params_

    # ------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------

    cdef list tensor_product(self, vector[tuple[int, int, complex]] A, vector[tuple[int, int, complex]] B, int A_width, int B_width):
        """
        Compute the tensor product of two sparse matrices in tuple form.
        """
        cdef vector[tuple[int, int, complex]] result = []
        cdef int row, col
        cdef double complex val

        for (ar, ac, av) in A:
            for (br, bc, bv) in B:
                row = ar * B_width + br
                col = ac * B_width + bc
                val = av * bv
                result.push_back((row, col, val))

        return result

    cdef int permute_bits(self, int index, vector[int] perm):
        """
        Permute the bits of an index according to a permutation.
        """
        cdef int n = perm.size()
        cdef int out = 0
        cdef int old_q, new_q
        cdef int old_bit, new_bit
        cdef int bit

        for old_q in range(n):
            new_q = perm[old_q]
            old_bit = n - 1 - old_q
            new_bit = n - 1 - new_q
            bit = (index >> old_bit) & 1
            out |= (bit << new_bit)

        return out

    cdef SparseMatrix base_to_full(self, SparseMatrix base_gate, int num_qubits, vector[int] control_qubits, vector[int] target_qubits):
        """
        Expand a base gate matrix to the full register.
        """

        cdef int min_qubit = num_qubits
        cdef int q

        for q in control_qubits:
            if q < min_qubit:
                min_qubit = q
        for q in target_qubits:
            if q < min_qubit:
                min_qubit = q

        cdef int base_gate_qubits = target_qubits.size() + control_qubits.size()
        cdef int needed_before = min_qubit
        cdef int needed_after = num_qubits - needed_before - base_gate_qubits

        cdef vector[tuple[int, int, complex]] out_entries = base_gate.to_tuples()

        # --- Make controlled ---
        cdef int delta
        cdef vector[tuple[int, int, complex]] new_entries
        cdef int row, col
        cdef double complex val
        cdef int i

        for _ in control_qubits:
            delta = 1 << len(target_qubits)
            new_entries = []

            # for (row, col, val) in out_entries:
            for i in range(len(out_entries)):
                row = out_entries[i][0]
                col = out_entries[i][1]
                val = out_entries[i][2]
                new_entries.push_back((row + delta, col + delta, val))

            for i in range(delta):
                new_entries.push_back((i, i, 1.0))

            out_entries = new_entries

        # --- Tensor with identities ---
        cdef vector[tuple[int, int, complex]] identity_entries = [(0, 0, 1.0), (1, 1, 1.0)]
        cdef int gate_size = 1 << base_gate_qubits

        for _ in range(needed_before):
            out_entries = self.tensor_product(identity_entries,
                                               out_entries,
                                               2,
                                               gate_size)
            gate_size *= 2

        for _ in range(needed_after):
            out_entries = self.tensor_product(out_entries,
                                               identity_entries,
                                               gate_size,
                                               2)
            gate_size *= 2

        # --- Permutation ---
        # cdef list all_qubits = control_qubits + target_qubits
        cdef vector[int] all_qubits
        for q in control_qubits:
            all_qubits.push_back(q)
        for q in target_qubits:
            all_qubits.push_back(q)

        cdef vector[int] perm = vector[int](num_qubits)
        for i in range(num_qubits):
            perm[i] = i
        cdef int tmp

        for i in range(len(all_qubits)):
            if perm[needed_before + i] != all_qubits[i]:
                tmp = perm[needed_before + i]
                perm[needed_before + i] = perm[all_qubits[i]]
                perm[all_qubits[i]] = tmp

        # invert permutation
        cdef vector[int] inv_perm = vector[int](num_qubits)
        for i in range(num_qubits):
            inv_perm[perm[i]] = i
        perm = inv_perm

        # apply permutation
        cdef int old_row, old_col, new_row, new_col
        for i in range(len(out_entries)):
            old_row = out_entries[i][0]
            old_col = out_entries[i][1]
            val = out_entries[i][2]
            new_row = self.permute_bits(old_row, perm)
            new_col = self.permute_bits(old_col, perm)
            out_entries[i] = (new_row, new_col, val)

        # sort entries
        # out_entries.sort(key=lambda x: (x[0], x[1]))
        # need the following since Cython does not support lambda functions
        cdef int j
        cdef tuple[int, int, complex] temp
        for i in range(len(out_entries)):
            for j in range(0, len(out_entries) - i - 1):
                if (out_entries[j][0] > out_entries[j + 1][0]) or \
                   (out_entries[j][0] == out_entries[j + 1][0] and out_entries[j][1] > out_entries[j + 1][1]):
                    temp = out_entries[j]
                    out_entries[j] = out_entries[j + 1]
                    out_entries[j + 1] = temp

        cdef int full_size = 1 << num_qubits
        return SparseMatrix(out_entries, full_size, full_size)

    cdef SparseMatrix get_base_matrix(self):
        """
        Return the base matrix of the gate.
        """
        cdef double theta, phi, gamma
        cdef double cos_half, sin_half, scale

        if self.gate_type == "H":
            return SparseMatrix([[1 / sqrt(2), 1 / sqrt(2)],
                                 [1 / sqrt(2), -1 / sqrt(2)]])

        elif self.gate_type == "X":
            return SparseMatrix([[0, 1],
                                 [1, 0]])

        elif self.gate_type == "Y":
            return SparseMatrix([[0, -1j],
                                 [1j, 0]])

        elif self.gate_type == "Z":
            return SparseMatrix([[1, 0],
                                 [0, -1]])

        elif self.gate_type == "CNOT":
            return SparseMatrix([[0, 1],
                                 [1, 0]])

        elif self.gate_type == "RX":
            theta = self.parameters[0]
            cos_half = cos(theta / 2)
            sin_half = sin(theta / 2)
            return SparseMatrix([[cos_half, -1j * sin_half],
                                 [-1j * sin_half, cos_half]])

        elif self.gate_type == "RY":
            theta = self.parameters[0]
            cos_half = cos(theta / 2)
            sin_half = sin(theta / 2)
            return SparseMatrix([[cos_half, -sin_half],
                                 [sin_half, cos_half]])

        elif self.gate_type == "RZ":
            phi = self.parameters[0]
            return SparseMatrix([[cmath.exp(-1j * phi / 2), 0],
                                 [0, cmath.exp(1j * phi / 2)]])

        elif self.gate_type == "U1":
            phi = self.parameters[0]
            return SparseMatrix([[1, 0],
                                 [0, cmath.exp(1j * phi)]])

        elif self.gate_type == "U2":
            phi = self.parameters[0]
            gamma = self.parameters[1]
            scale = 1 / sqrt(2)
            return SparseMatrix([[scale, -scale * cmath.exp(1j * gamma)],
                                 [scale * cmath.exp(1j * phi),
                                  scale * cmath.exp(1j * (phi + gamma))]])

        elif self.gate_type == "U3":
            theta = self.parameters[0]
            phi = self.parameters[1]
            gamma = self.parameters[2]
            cos_half = cos(theta / 2)
            sin_half = sin(theta / 2)
            return SparseMatrix([[cos_half, -cmath.exp(1j * gamma) * sin_half],
                                 [cmath.exp(1j * phi) * sin_half,
                                  cmath.exp(1j * (phi + gamma)) * cos_half]])

        elif self.gate_type == "M":
            return SparseMatrix([[1, 0],
                                 [0, 1]])

        else:
            return SparseMatrix([[1, 0],
                                 [0, 1]])
        #     raise RuntimeError(f"Unsupported gate type: {gate_type}")

    cdef int size(self):
        return self.get_base_matrix().size()

    cdef SparseMatrix get_full_matrix(self, int num_qubits):
        cdef SparseMatrix base_gate = self.get_base_matrix()
        return self.base_to_full(base_gate,
                                 num_qubits,
                                 self.control_qubits,
                                 self.target_qubits)

        
class QiliSimCython(Backend):
    def __init__(self):
        super().__init__()

    def _execute_sampling(self, functional: Sampling) -> SamplingResult:

        print("here0")

        # Get the gates from the sampling
        gates = []
        cdef string gate_name
        cdef Gate new_gate    
        for gate in functional.circuit.gates:
            gate_name = gate.name.encode('utf-8')
            while gate_name.size() > 1 and gate_name[0] == 'C' and gate_name != "CNOT":
                gate_name = gate_name.substr(1)
            new_gate = Gate(gate_name, gate.control_qubits, gate.target_qubits, gate.parameters)
            print("New gate has type:", new_gate.gate_type)
            gates.append(new_gate)

        print("Gates loaded")

        # Get the number of qubits and shots
        cdef int num_qubits = functional.circuit.nqubits
        cdef int nshots = functional.nshots

        print("Qubits loaded")

        # Start with the zero state
        cdef int mat_size = 1 << num_qubits
        cdef SparseMatrix state = SparseMatrix([(0, 0, 1.0)], mat_size, 1)

        # Output the statevector
        print("Initial statevector:")
        cdef vector[tuple[int, int, complex]] state_entries_init = state.to_tuples()
        cdef tuple[int, int, complex] state_entry_init
        for state_entry_init in state_entries_init:
            print(f"({state_entry_init[0]}, {state_entry_init[1]}) = {state_entry_init[2]}")

        print("here1")

        # Apply each gate in the circuit
        cdef SparseMatrix gate_matrix = SparseMatrix(0, 0)
        cdef Gate gate_c
        for gate_c in gates:
            print("Gate type:", gate_c.gate_type)
            if gate_c.gate_type == "":
                continue
            print("Applying gate")
        #     gate_matrix = gate_c.get_full_matrix(num_qubits)
        #     state = gate_matrix.mul(state)

        print("here2")

        # Output the statevector
        print("Final statevector:")
        cdef vector[tuple[int, int, complex]] state_entries = state.to_tuples()
        cdef tuple[int, int, complex] state_entry
        for state_entry in state_entries:
            print(f"({state_entry[0]}, {state_entry[1]}) = {state_entry[2]}")

        # Get the probabilities
        cdef vector[tuple[int, int, double complex]] amplitude_entries = state.to_tuples()
        cdef vector[tuple[int, double]] prob_entries
        cdef double total_prob = 0.0
        cdef int row
        cdef double complex amp
        cdef double prob
        cdef double real_part, imag_part
        cdef tuple[int, int, double complex] entry
        for entry in amplitude_entries:
            row = entry[0]
            amp = entry[2]
            prob = abs(amp) ** 2
            if prob > atol_:
                prob_entries.push_back((row, prob))
                total_prob += prob

        print("here3")

        # Make sure probabilities sum to 1
        if fabs(total_prob - 1.0) > atol_:
            raise RuntimeError(f"Probabilities do not sum to 1 (sum = {total_prob})")

        # Sample from the final state
        cdef map[string, int] counts
        cdef int shot, state_index
        cdef double random_value, cumulative_prob
        cdef string bitstring
        cdef string one_str = "1"
        cdef string zero_str = "0"
        cdef int b
        cdef tuple[int, double] prob_entry
        import random
        for shot in range(nshots):
            random_value = random.uniform(0.0, 1.0)
            cumulative_prob = 0.0
            for prob_entry in prob_entries:
                state_index = prob_entry[0]
                prob = prob_entry[1]
                cumulative_prob += prob
                if random_value <= cumulative_prob:
                    # Convert state_index to bitstring
                    bitstring = ""
                    for b in range(num_qubits - 1, -1, -1):
                        if (state_index >> b) & 1:
                            bitstring += one_str
                        else:
                            bitstring += zero_str
                    counts[bitstring] += 1
                    break

        results = SamplingResult(nshots=nshots, samples=counts)
        return results
     