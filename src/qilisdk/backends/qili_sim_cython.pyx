# cython: language_level=3
# distutils: language = c++
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
# from libcpp.complex cimport complex, abs
import cython

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
    cdef SparseMatrix from_dense(vector[vector[complex]] dense):
        """
        Construct a sparse matrix from a dense vector of vectors.
        """
        cdef SparseMatrix mat = SparseMatrix(dense.size(), dense[0].size())
        cdef int r, c
        cdef complex val
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
        return mat

    @staticmethod
    cdef from_tuples(list entries, int nrows, int ncols):
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

        return mat

    cdef to_tuples(self):
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

    cdef get(self, int row, int col):
        cdef int idx
        for idx in range(self.rows_[row], self.rows_[row + 1]):
            if self.cols_[idx] == col:
                return self.values_[idx]
        return 0.0

    cdef insert(self, int row, int col, complex value):
        cdef int start = self.rows_[row]
        cdef int end = self.rows_[row + 1]
        cdef int insertPos = start
        while insertPos < end and self.cols_[insertPos] < col:
            insertPos += 1
        self.cols_.insert(self.cols_.begin() + insertPos, col)
        self.values_.insert(self.values_.begin() + insertPos, value)

        for r in range(row + 1, self.rows_.size()):
            self.rows_[r] += 1

    cdef get_dims(self):
        return f"{self.nrows_}x{self.ncols_}"

    cdef mul_cpp(self, SparseMatrix other):
        """
        Multiply two sparse matrices using C++ map for accumulation.
        """
        if self.ncols_ != other.nrows_:
            raise RuntimeError(f"Matrix dimensions do not match for multiplication: {self.get_dims()} * {other.get_dims()}")
        cdef map[pair[int,int], complex] entries_map
        cdef int r, idxA, idxB, colA, colB
        cdef complex valA, valB

        print("here3")
        for r in range(self.nrows_):
            for idxA in range(self.rows_[r], self.rows_[r+1]):
                colA = self.cols_[idxA]
                valA = self.values_[idxA]
                for idxB in range(other.rows_[colA], other.rows_[colA+1]):
                    colB = other.cols_[idxB]
                    valB = other.values_[idxB]
                    entries_map[(r, colB)] += valA * valB

        cdef list entries = []
        cdef map[pair[int,int], complex].iterator it = entries_map.begin()
        while it != entries_map.end():
            k = dereference(it).first
            v = dereference(it).second
            if abs(v) > atol_:
                entries.append((k.first, k.second, v))
            postincrement(it)

        return SparseMatrix.from_tuples(entries, self.nrows_, other.ncols_)

    def __mul__(self, other):
        return self.mul_cpp(other)

        
class QiliSimCython(Backend):
    def __init__(self):
        super().__init__()

    def _execute_sampling(self, functional: Sampling) -> SamplingResult:

        # Sparse matrix test
        cdef list entriesA = [(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0)]
        cdef list entriesB = [(0, 1, 4.0), (1, 1, 6.0), (2, 0, 5.0)]
        cdef SparseMatrix A = SparseMatrix.from_tuples(entriesA, 2, 3)
        cdef SparseMatrix B = SparseMatrix.from_tuples(entriesB, 3, 2)
        print("here")
        cdef SparseMatrix C = A * B
        print("here2")
        print("Matrix A entries:", A.to_tuples())
        print("Matrix B entries:", B.to_tuples())
        print("Matrix C = A * B entries:", C.to_tuples())

        return SamplingResult(nshots=1, samples={"001": 1})
     