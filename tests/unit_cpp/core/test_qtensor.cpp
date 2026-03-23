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

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include "../../../src/qilisdk_cpp/core/qtensor.h"

namespace py = pybind11;

static SparseMatrix make_ket0() {
    SparseMatrix m(2, 1);
    m.insert(0, 0) = 1.0;
    m.makeCompressed();
    return m;
}

static SparseMatrix make_ket1() {
    SparseMatrix m(2, 1);
    m.insert(1, 0) = 1.0;
    m.makeCompressed();
    return m;
}

static SparseMatrix make_dm_pure0() {
    SparseMatrix m(2, 2);
    m.insert(0, 0) = 1.0;
    m.makeCompressed();
    return m;
}

static SparseMatrix make_dm_mixed() {
    SparseMatrix m(2, 2);
    m.insert(0, 0) = 0.5;
    m.insert(1, 1) = 0.5;
    m.makeCompressed();
    return m;
}

static SparseMatrix make_pauli_x() {
    SparseMatrix m(2, 2);
    m.insert(0, 1) = 1.0;
    m.insert(1, 0) = 1.0;
    m.makeCompressed();
    return m;
}

static SparseMatrix make_identity2() {
    SparseMatrix m(2, 2);
    m.insert(0, 0) = 1.0;
    m.insert(1, 1) = 1.0;
    m.makeCompressed();
    return m;
}

static SparseMatrix make_asymmetric() {
    SparseMatrix m(2, 2);
    m.insert(0, 1) = 1.0;
    m.insert(1, 0) = 2.0;
    m.makeCompressed();
    return m;
}

static SparseMatrix make_non_hermitian() {
    SparseMatrix m(2, 2);
    m.insert(0, 1) = std::complex<double>(0, 1);
    m.insert(1, 0) = std::complex<double>(0, 1);
    m.makeCompressed();
    return m;
}

TEST(ValidateShapeTest, ZeroRows_Throws) {
    SparseMatrix m(0, 2);
    EXPECT_THROW(QTensorCpp q(m), py::value_error);
}

TEST(ValidateShapeTest, ZeroCols_Throws) {
    SparseMatrix m(2, 0);
    EXPECT_THROW(QTensorCpp q(m), py::value_error);
}

TEST(ValidateShapeTest, NonPowerOf2Rows_Throws) {
    SparseMatrix m(3, 1);
    EXPECT_THROW(QTensorCpp q(m), py::value_error);
}

TEST(ValidateShapeTest, NonPowerOf2Cols_Throws) {
    SparseMatrix m(1, 3);
    EXPECT_THROW(QTensorCpp q(m), py::value_error);
}

TEST(ValidateShapeTest, NonSquareNonVector_Throws) {
    SparseMatrix m(2, 4);
    EXPECT_THROW(QTensorCpp q(m), py::value_error);
}

TEST(ValidateShapeTest, ValidKet_NoThrow) {
    EXPECT_NO_THROW(QTensorCpp q(make_ket0()));
}

TEST(ValidateShapeTest, ValidBra_NoThrow) {
    SparseMatrix m(1, 2);
    EXPECT_NO_THROW(QTensorCpp q(m));
}

TEST(ValidateShapeTest, ValidOperator_NoThrow) {
    EXPECT_NO_THROW(QTensorCpp q(make_identity2()));
}

TEST(ValidateShapeTest, Scalar1x1_NoThrow) {
    SparseMatrix m(1, 1);
    EXPECT_NO_THROW(QTensorCpp q(m));
}

TEST(ConstructorTest, Default_NoThrow) {
    EXPECT_NO_THROW(QTensorCpp q);
}

TEST(ConstructorTest, RowsCols_ValidPowerOf2) {
    QTensorCpp q(2, 2);
    auto [r, c] = q.get_shape();
    EXPECT_EQ(r, 2);
    EXPECT_EQ(c, 2);
}

TEST(ConstructorTest, RowsCols_InvalidNonPowerOf2_Throws) {
    EXPECT_THROW(QTensorCpp q(3, 3), py::value_error);
}

TEST(ConstructorTest, FromSparseMatrix_Valid) {
    QTensorCpp q(make_ket0());
    EXPECT_TRUE(q.is_ket());
}

TEST(ConstructorTest, FromSparseMatrix_Invalid_Throws) {
    SparseMatrix bad(3, 1);
    EXPECT_THROW(QTensorCpp q(bad), py::value_error);
}

TEST(ConstructorTest, FromPyObject_QTensorCpp_Direct) {
    py::gil_scoped_acquire gil;
    QTensorCpp src(make_ket0());
    py::object obj = py::cast(src);
    QTensorCpp q(obj);
    EXPECT_TRUE(q.is_ket());
}

TEST(ConstructorTest, FromPyObject_WithQtensorCppAttr) {
    py::gil_scoped_acquire gil;
    QTensorCpp src(make_ket0());
    py::exec(R"(
class _FakeWrap:
    def __init__(self, cpp):
        self._qtensor_cpp = cpp
)");
    py::object wrap_cls = py::globals()["_FakeWrap"];
    py::object obj = wrap_cls(py::cast(src));
    QTensorCpp q(obj);
    EXPECT_TRUE(q.is_ket());
}

TEST(ConstructorTest, FromPyObject_EmptyList_Returns00) {
    py::gil_scoped_acquire gil;
    py::list empty;
    EXPECT_NO_THROW(QTensorCpp q(empty));
}

TEST(ConstructorTest, FromPyObject_ListOfLists_Valid) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
_list_ket = [[1.0+0j], [0.0+0j]]
    )");
    QTensorCpp q(py::globals()["_list_ket"]);
    EXPECT_TRUE(q.is_ket());
}

TEST(ConstructorTest, FromPyObject_ListNotListOfLists_Throws) {
    py::gil_scoped_acquire gil;
    py::exec(R"(_flat_list = [1, 2])");
    EXPECT_THROW(QTensorCpp q(py::globals()["_flat_list"]), py::value_error);
}

TEST(ConstructorTest, FromPyObject_RaggedList_Throws) {
    py::gil_scoped_acquire gil;
    py::exec(R"(_ragged = [[1.0+0j, 2.0+0j], [3.0+0j]])");
    EXPECT_THROW(QTensorCpp q(py::globals()["_ragged"]), py::value_error);
}

TEST(ConstructorTest, FromPyObject_ListRowNotList_Throws) {
    py::gil_scoped_acquire gil;
    py::exec(R"(_bad_row = [[1.0+0j], 2])");
    EXPECT_THROW(QTensorCpp q(py::globals()["_bad_row"]), py::value_error);
}

TEST(ConstructorTest, FromPyObject_ScipyCsr) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
import scipy.sparse as sp, numpy as np
_csr = sp.csr_matrix(np.eye(2, dtype=complex))
    )");
    EXPECT_NO_THROW(QTensorCpp q(py::globals()["_csr"]));
}

TEST(ConstructorTest, FromPyObject_ScipyCsc) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
import scipy.sparse as sp, numpy as np
_csc = sp.csc_matrix(np.eye(2, dtype=complex))
    )");
    EXPECT_NO_THROW(QTensorCpp q(py::globals()["_csc"]));
}

TEST(ConstructorTest, FromPyObject_ScipyCoo) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
import scipy.sparse as sp, numpy as np
_coo = sp.coo_matrix(np.eye(2, dtype=complex))
    )");
    EXPECT_NO_THROW(QTensorCpp q(py::globals()["_coo"]));
}

TEST(ConstructorTest, FromPyObject_Numpy) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
import numpy as np
_nparr = np.eye(2, dtype=complex)
    )");
    EXPECT_NO_THROW(QTensorCpp q(py::globals()["_nparr"]));
}

TEST(ConstructorTest, FromPyObject_InvalidType_Throws) {
    py::gil_scoped_acquire gil;
    py::exec(R"(_invalid_obj = 42)");
    EXPECT_THROW(QTensorCpp q(py::globals()["_invalid_obj"]), py::value_error);
}

TEST(AccessorsTest, AsScipy_ReturnsObject) {
    py::gil_scoped_acquire gil;
    QTensorCpp q(make_identity2());
    py::object scipy_m = q.as_scipy();
    EXPECT_TRUE(scipy_m.ptr() != nullptr);
}

TEST(AccessorsTest, AsNumpy_ReturnsObject) {
    py::gil_scoped_acquire gil;
    QTensorCpp q(make_identity2());
    py::object np_arr = q.as_numpy();
    EXPECT_TRUE(np_arr.ptr() != nullptr);
}

TEST(AccessorsTest, GetNqubits_Ket2) {
    QTensorCpp q(make_ket0());
    EXPECT_EQ(q.get_nqubits(), 1);
}

TEST(AccessorsTest, GetNqubits_Operator4x4) {
    SparseMatrix m(4, 4);
    QTensorCpp q(m);
    EXPECT_EQ(q.get_nqubits(), 2);
}

TEST(AccessorsTest, GetShape_Returns) {
    QTensorCpp q(make_identity2());
    auto [r, c] = q.get_shape();
    EXPECT_EQ(r, 2);
    EXPECT_EQ(c, 2);
}

TEST(AccessorsTest, AsString_ContainsShape) {
    QTensorCpp q(make_ket0());
    std::string s = q.as_string();
    EXPECT_NE(s.find("2x1"), std::string::npos);
}

TEST(AccessorsTest, AsDense_CorrectDimensions) {
    QTensorCpp q(make_identity2());
    DenseMatrix d = q.as_dense();
    EXPECT_EQ(d.rows(), 2);
    EXPECT_EQ(d.cols(), 2);
}

TEST(TypeCheckTest, IsKet_ColVector) {
    QTensorCpp q(make_ket0());
    EXPECT_TRUE(q.is_ket());
    EXPECT_FALSE(q.is_bra());
    EXPECT_FALSE(q.is_operator());
    EXPECT_FALSE(q.is_scalar());
}

TEST(TypeCheckTest, IsBra_RowVector) {
    SparseMatrix m(1, 2);
    m.insert(0, 0) = 1.0;
    m.makeCompressed();
    QTensorCpp q(m);
    EXPECT_TRUE(q.is_bra());
    EXPECT_FALSE(q.is_ket());
    EXPECT_FALSE(q.is_operator());
    EXPECT_FALSE(q.is_scalar());
}

TEST(TypeCheckTest, IsOperator_Square) {
    QTensorCpp q(make_identity2());
    EXPECT_TRUE(q.is_operator());
    EXPECT_FALSE(q.is_ket());
    EXPECT_FALSE(q.is_bra());
    EXPECT_FALSE(q.is_scalar());
}

TEST(TypeCheckTest, IsScalar_1x1) {
    SparseMatrix m(1, 1);
    m.insert(0, 0) = 1.0;
    m.makeCompressed();
    QTensorCpp q(m);
    EXPECT_TRUE(q.is_scalar());
    EXPECT_FALSE(q.is_ket());
    EXPECT_FALSE(q.is_bra());
    EXPECT_FALSE(q.is_operator());
}

TEST(IsSymmetricTest, NonSquare_False) {
    QTensorCpp q(make_ket0());
    EXPECT_FALSE(q.is_symmetric());
}

TEST(IsSymmetricTest, Symmetric_True) {
    QTensorCpp q(make_pauli_x());
    EXPECT_TRUE(q.is_symmetric());
}

TEST(IsSymmetricTest, Asymmetric_False) {
    QTensorCpp q(make_asymmetric());
    EXPECT_FALSE(q.is_symmetric());
}

TEST(IsSymmetricTest, Cached_SecondCallSameResult) {
    QTensorCpp q(make_pauli_x());
    EXPECT_TRUE(q.is_symmetric());
    EXPECT_TRUE(q.is_symmetric());
}

TEST(IsSelfAdjointTest, NonSquare_False) {
    QTensorCpp q(make_ket0());
    EXPECT_FALSE(q.is_self_adjoint());
}

TEST(IsSelfAdjointTest, NonSquare_False_overload) {
    QTensorCpp q(make_ket0());
    EXPECT_FALSE(q.is_hermitian());
}

TEST(IsSelfAdjointTest, Hermitian_True) {
    QTensorCpp q(make_identity2());
    EXPECT_TRUE(q.is_self_adjoint());
}

TEST(IsSelfAdjointTest, NonHermitian_False) {
    QTensorCpp q(make_non_hermitian());
    EXPECT_FALSE(q.is_self_adjoint());
}

TEST(IsSelfAdjointTest, Cached_SecondCallSameResult) {
    QTensorCpp q(make_identity2());
    EXPECT_TRUE(q.is_self_adjoint());
    EXPECT_TRUE(q.is_self_adjoint());
}

TEST(IsPSDTest, NonSquare_False) {
    QTensorCpp q(make_ket0());
    EXPECT_FALSE(q.is_positive_semidefinite());
}

TEST(IsPSDTest, IdentityMatrix_True) {
    QTensorCpp q(make_identity2());
    EXPECT_TRUE(q.is_positive_semidefinite());
}

TEST(IsPSDTest, NegativeDefinite_False) {
    SparseMatrix m(2, 2);
    m.insert(0, 0) = -1.0;
    m.insert(1, 1) = -1.0;
    m.makeCompressed();
    QTensorCpp q(m);
    EXPECT_FALSE(q.is_positive_semidefinite());
}

TEST(IsPSDTest, Cached_SecondCallSameResult) {
    QTensorCpp q(make_identity2());
    EXPECT_TRUE(q.is_positive_semidefinite());
    EXPECT_TRUE(q.is_positive_semidefinite());
}

TEST(IsPSDTest, EigenvaluesCachedPositive_True) {
    QTensorCpp q(make_dm_pure0());
    q.compute_eigendecomposition();
    EXPECT_TRUE(q.is_positive_semidefinite());
}

TEST(IsPSDTest, EigenvaluesCachedNegative_False) {
    SparseMatrix m(2, 2);
    m.insert(0, 0) = -1.0;
    m.insert(1, 1) = 1.0;
    m.makeCompressed();
    QTensorCpp q(m);
    q.compute_eigendecomposition();
    EXPECT_FALSE(q.is_positive_semidefinite());
}

TEST(EigendecompTest, AlreadyCached_EarlyReturn) {
    QTensorCpp q(make_identity2());
    q.compute_eigendecomposition();
    EXPECT_NO_THROW(q.compute_eigendecomposition());
    auto evals = q.get_eigenvalues();
    EXPECT_EQ(int(evals.size()), 2);
}

TEST(EigendecompTest, Ket_StoresSingleEigenvalue) {
    QTensorCpp q(make_ket0());
    q.compute_eigendecomposition();
    auto evals = q.get_eigenvalues();
    EXPECT_EQ(int(evals.size()), 1);
    EXPECT_NEAR(evals[0].real(), 1.0, 1e-10);
}

TEST(EigendecompTest, Bra_StoresSingleEigenvalue) {
    SparseMatrix m(1, 2);
    m.insert(0, 0) = 1.0;
    m.makeCompressed();
    QTensorCpp q(m);
    q.compute_eigendecomposition();
    auto evals = q.get_eigenvalues();
    EXPECT_EQ(int(evals.size()), 1);
}

TEST(EigendecompTest, SelfAdjoint_UsesSelfAdjointSolver) {
    QTensorCpp q(make_identity2());
    q.compute_eigendecomposition();
    auto evals = q.get_eigenvalues();
    EXPECT_EQ(int(evals.size()), 2);
    for (auto& e : evals) {
        EXPECT_NEAR(e.real(), 1.0, 1e-10);
    }
}

TEST(EigendecompTest, NonSelfAdjoint_UsesComplexSolver) {
    QTensorCpp q(make_non_hermitian());
    q.compute_eigendecomposition();
    auto evals = q.get_eigenvalues();
    EXPECT_EQ(int(evals.size()), 2);
}

TEST(DensityMatrixTest, ValidPureDM_True) {
    QTensorCpp q(make_dm_pure0());
    EXPECT_TRUE(q.is_density_matrix());
}

TEST(DensityMatrixTest, MixedDM_True) {
    QTensorCpp q(make_dm_mixed());
    EXPECT_TRUE(q.is_density_matrix());
}

TEST(DensityMatrixTest, NotOperator_False) {
    QTensorCpp q(make_ket0());
    EXPECT_FALSE(q.is_density_matrix());
}

TEST(DensityMatrixTest, NotSelfAdjoint_False) {
    QTensorCpp q(make_non_hermitian());
    EXPECT_FALSE(q.is_density_matrix());
}

TEST(DensityMatrixTest, WrongTrace_False) {
    QTensorCpp q(make_identity2());
    EXPECT_FALSE(q.is_density_matrix());
}

TEST(DensityMatrixTest, NotPSD_False) {
    SparseMatrix m(2, 2);
    m.insert(0, 0) = 1.5;
    m.insert(1, 1) = -0.5;
    m.makeCompressed();
    QTensorCpp q(m);
    EXPECT_FALSE(q.is_density_matrix());
}

TEST(IsPureTest, Ket_True) {
    QTensorCpp q(make_ket0());
    EXPECT_TRUE(q.is_pure());
}

TEST(IsPureTest, Bra_True) {
    SparseMatrix m(1, 2);
    m.insert(0, 0) = 1.0;
    m.makeCompressed();
    QTensorCpp q(m);
    EXPECT_TRUE(q.is_pure());
}

TEST(IsPureTest, PureDM_True) {
    QTensorCpp q(make_dm_pure0());
    EXPECT_TRUE(q.is_pure());
}

TEST(IsPureTest, MixedDM_False) {
    QTensorCpp q(make_dm_mixed());
    EXPECT_FALSE(q.is_pure());
}

TEST(IsPureTest, NotDensityMatrix_False) {
    QTensorCpp q(make_identity2());
    EXPECT_FALSE(q.is_pure());
}

TEST(IsPureTest, Cached_SecondCallSameResult) {
    QTensorCpp q(make_dm_pure0());
    EXPECT_TRUE(q.is_pure());
    EXPECT_TRUE(q.is_pure());
}

TEST(LinearAlgebraTest, Conjugate_ComplexEntries) {
    SparseMatrix m(2, 2);
    m.insert(0, 1) = std::complex<double>(0, 1);
    m.makeCompressed();
    QTensorCpp q(m);
    QTensorCpp conj = q.conjugate();
    EXPECT_NEAR(conj.get_data().coeff(0, 1).imag(), -1.0, 1e-10);
}

TEST(LinearAlgebraTest, Transpose_SwapsShape) {
    QTensorCpp q(make_ket0());
    QTensorCpp t = q.transpose();
    EXPECT_TRUE(t.is_bra());
}

TEST(LinearAlgebraTest, Adjoint_KetBecomesBra) {
    QTensorCpp q(make_ket0());
    QTensorCpp a = q.adjoint();
    EXPECT_TRUE(a.is_bra());
}

TEST(TraceTest, Identity2_TraceIs2) {
    QTensorCpp q(make_identity2());
    EXPECT_NEAR(q.trace().real(), 2.0, 1e-10);
}

TEST(TraceTest, Cached_SecondCallSameResult) {
    QTensorCpp q(make_identity2());
    auto t1 = q.trace();
    auto t2 = q.trace();  // hits cache
    EXPECT_EQ(t1, t2);
}

TEST(TraceTest, OffDiagonalNotCounted) {
    QTensorCpp q(make_pauli_x());
    EXPECT_NEAR(q.trace().real(), 0.0, 1e-10);
}

TEST(PartialTraceTest, PythonVersion_NonIntItem_Throws) {
    py::gil_scoped_acquire gil;
    py::exec(R"(_bad_keep = ["a"])");
    QTensorCpp q(make_dm_pure0());
    EXPECT_THROW(q.partial_trace_python(py::globals()["_bad_keep"]), py::value_error);
}

TEST(PartialTraceTest, FromKet_ConvertsToOperatorFirst) {
    SparseMatrix ket00(4, 1);
    ket00.insert(0, 0) = 1.0;
    ket00.makeCompressed();
    QTensorCpp q(ket00);
    QTensorCpp pt = q.partial_trace({1});
    EXPECT_EQ(pt.get_shape().first, 2);
}

TEST(PartialTraceTest, FromBra_ConvertsToOperatorFirst) {
    SparseMatrix bra00(1, 4);
    bra00.insert(0, 0) = 1.0;
    bra00.makeCompressed();
    QTensorCpp q(bra00);
    QTensorCpp pt = q.partial_trace({1});
    EXPECT_EQ(pt.get_shape().first, 2);
}

TEST(PartialTraceTest, InvalidQubitIndex_Throws) {
    QTensorCpp q(make_dm_pure0());
    EXPECT_THROW(q.partial_trace({5}), py::value_error);
}

TEST(PartialTraceTest, InvalidNegativeQubitIndex_Throws) {
    QTensorCpp q(make_dm_pure0());
    EXPECT_THROW(q.partial_trace({-1}), py::value_error);
}

TEST(PartialTraceTest, TraceOutAll_Returns1x1) {
    SparseMatrix dm4(4, 4);
    dm4.insert(0, 0) = 1.0;
    dm4.makeCompressed();
    QTensorCpp q(dm4);
    QTensorCpp pt = q.partial_trace({});
    EXPECT_EQ(pt.get_shape().first, 1);
}

TEST(PartialTraceTest, TwoQubit_TraceOutOneQubit) {
    SparseMatrix dm4(4, 4);
    dm4.insert(0, 0) = 1.0;
    dm4.makeCompressed();
    QTensorCpp q(dm4);
    QTensorCpp pt = q.partial_trace({0});
    EXPECT_EQ(pt.get_shape().first, 2);
}

TEST(PartialTraceTest, PythonVersion_ValidList_ReducesShape) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
import scipy.sparse as sp
import numpy as np
_dm4 = sp.csr_matrix(np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=complex))
    )");
    QTensorCpp q(py::globals()["_dm4"]);
    py::exec(R"(_keep = [0])");
    QTensorCpp pt = q.partial_trace_python(py::globals()["_keep"]);
    EXPECT_EQ(pt.get_shape().first, 2);
    EXPECT_EQ(pt.get_shape().second, 2);
}

TEST(NormTest, Auto_Operator_TracePath) {
    QTensorCpp q(make_dm_pure0());
    EXPECT_NEAR(q.norm("auto"), 1.0, 1e-10);
}

TEST(NormTest, Auto_Ket_FrobeniusPath) {
    QTensorCpp q(make_ket0());
    EXPECT_NEAR(q.norm("auto"), 1.0, 1e-10);
}

TEST(NormTest, Frobenius_SameAsL2) {
    QTensorCpp q(make_ket0());
    EXPECT_NEAR(q.norm("frobenius"), q.norm("l2"), 1e-10);
}

TEST(NormTest, L1Norm) {
    QTensorCpp q(make_ket0());
    EXPECT_NEAR(q.norm("l1"), 1.0, 1e-10);
}

TEST(NormTest, Trace_DensityMatrix) {
    QTensorCpp q(make_dm_pure0());
    EXPECT_NEAR(q.norm("trace"), 1.0, 1e-10);
}

TEST(NormTest, Nuclear) {
    QTensorCpp q(make_identity2());
    EXPECT_NEAR(q.norm("nuclear"), 2.0, 1e-10);
}

TEST(NormTest, Inf) {
    QTensorCpp q(make_pauli_x());
    EXPECT_NEAR(q.norm("inf"), 1.0, 1e-10);
}

TEST(NormTest, Unknown_Throws) {
    QTensorCpp q(make_ket0());
    EXPECT_THROW(q.norm("xyz"), py::value_error);
}

TEST(NormalizedTest, ZeroNorm_Throws) {
    SparseMatrix m(2, 1);  // all zeros
    m.makeCompressed();
    QTensorCpp q(m);
    EXPECT_THROW(q.normalized("l2"), py::value_error);
}

TEST(NormalizedTest, NonzeroNorm_Normalized) {
    QTensorCpp q(make_ket0());
    QTensorCpp n = q.normalized("l2");
    EXPECT_NEAR(n.norm("l2"), 1.0, 1e-10);
}

TEST(KetTest, Empty_Throws) {
    EXPECT_THROW(QTensorCpp::ket({}), py::value_error);
}

TEST(KetTest, InvalidBit_Throws) {
    EXPECT_THROW(QTensorCpp::ket({2}), py::value_error);
}

TEST(KetTest, Ket0) {
    QTensorCpp q = QTensorCpp::ket({0});
    EXPECT_NEAR(q.get_data().coeff(0, 0).real(), 1.0, 1e-10);
}

TEST(KetTest, Ket1) {
    QTensorCpp q = QTensorCpp::ket({1});
    EXPECT_NEAR(q.get_data().coeff(1, 0).real(), 1.0, 1e-10);
}

TEST(KetTest, Ket00_TwoQubits) {
    QTensorCpp q = QTensorCpp::ket({0, 0});
    EXPECT_EQ(q.get_shape().first, 4);
    EXPECT_NEAR(q.get_data().coeff(0, 0).real(), 1.0, 1e-10);
}

TEST(KetPythonTest, NonIntItem_Throws) {
    py::gil_scoped_acquire gil;
    py::exec(R"(_bad_ket = ["a"])");
    EXPECT_THROW(QTensorCpp::ket_python(py::globals()["_bad_ket"]), py::value_error);
}

TEST(KetPythonTest, Valid) {
    py::gil_scoped_acquire gil;
    py::exec(R"(_ket_state = [0, 1])");
    EXPECT_NO_THROW(QTensorCpp::ket_python(py::globals()["_ket_state"]));
}

TEST(BraTest, Empty_Throws) {
    EXPECT_THROW(QTensorCpp::bra({}), py::value_error);
}

TEST(BraTest, InvalidBit_Throws) {
    EXPECT_THROW(QTensorCpp::bra({-1}), py::value_error);
}

TEST(BraTest, Bra0) {
    QTensorCpp q = QTensorCpp::bra({0});
    EXPECT_TRUE(q.is_bra());
    EXPECT_NEAR(q.get_data().coeff(0, 0).real(), 1.0, 1e-10);
}

TEST(BraPythonTest, NonIntItem_Throws) {
    py::gil_scoped_acquire gil;
    py::exec(R"(_bad_bra = ["x"])");
    EXPECT_THROW(QTensorCpp::bra_python(py::globals()["_bad_bra"]), py::value_error);
}

TEST(BraPythonTest, Valid) {
    py::gil_scoped_acquire gil;
    py::exec(R"(_bra_state = [1, 0])");
    EXPECT_NO_THROW(QTensorCpp::bra_python(py::globals()["_bra_state"]));
}

TEST(ZeroTest, NegativeNqubits_Throws) {
    EXPECT_THROW(QTensorCpp::zero(-1, "ket"), py::value_error);
}

TEST(ZeroTest, Ket_Type) {
    QTensorCpp q = QTensorCpp::zero(1, "ket");
    EXPECT_TRUE(q.is_ket());
}

TEST(ZeroTest, Bra_Type) {
    QTensorCpp q = QTensorCpp::zero(1, "bra");
    EXPECT_TRUE(q.is_bra());
}

TEST(ZeroTest, Operator_Type) {
    QTensorCpp q = QTensorCpp::zero(1, "operator");
    EXPECT_TRUE(q.is_operator());
}

TEST(ZeroTest, InvalidType_Throws) {
    EXPECT_THROW(QTensorCpp::zero(1, "vector"), py::value_error);
}

TEST(IdentityTest, Is2x2Identity) {
    QTensorCpp q = QTensorCpp::identity(1);
    EXPECT_NEAR(q.get_data().coeff(0, 0).real(), 1.0, 1e-10);
    EXPECT_NEAR(q.get_data().coeff(1, 1).real(), 1.0, 1e-10);
    EXPECT_NEAR(q.get_data().coeff(0, 1).real(), 0.0, 1e-10);
}

TEST(GhzTest, TwoQubits_SuperpositionOf00And11) {
    QTensorCpp q = QTensorCpp::ghz(2);
    EXPECT_EQ(q.get_shape().first, 4);
    EXPECT_EQ(q.get_shape().second, 1);
    double c = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(q.get_data().coeff(0, 0).real(), c, 1e-10);
    EXPECT_NEAR(q.get_data().coeff(3, 0).real(), c, 1e-10);
}

TEST(TensorProductTest, Empty_Throws) {
    EXPECT_THROW(QTensorCpp::tensor_product({}), py::value_error);
}

TEST(TensorProductTest, KetKronKet_4Dim) {
    QTensorCpp q0(make_ket0());
    QTensorCpp q1(make_ket1());
    QTensorCpp tp = QTensorCpp::tensor_product({q0, q1});
    EXPECT_EQ(tp.get_shape().first, 4);
}

TEST(TensorProductTest, PythonVersion_InvalidType_Throws) {
    py::gil_scoped_acquire gil;
    py::exec(R"(_bad_list = [42])");
    EXPECT_THROW(QTensorCpp::tensor_product_python(py::globals()["_bad_list"].cast<py::list>()), py::value_error);
}

TEST(TensorProductTest, PythonVersion_QTensorCpp_Valid) {
    py::gil_scoped_acquire gil;
    py::list lst;
    lst.append(py::cast(QTensorCpp(make_ket1())));
    EXPECT_NO_THROW(QTensorCpp::tensor_product_python(lst));
}

TEST(TensorProductTest, PythonVersion_WrappedObject_Valid) {
    py::gil_scoped_acquire gil;
    QTensorCpp inner(make_ket1());
    py::exec(R"(
class _WrapTP:
    def __init__(self, cpp):
        self._qtensor_cpp = cpp
)");
    py::object wrap_cls = py::globals()["_WrapTP"];
    py::list lst;
    lst.append(wrap_cls(py::cast(inner)));
    EXPECT_NO_THROW(QTensorCpp::tensor_product_python(lst));
}

TEST(AddTest, TwoOperators) {
    QTensorCpp a(make_identity2()), b(make_identity2());
    QTensorCpp c = a.add(b);
    EXPECT_NEAR(c.get_data().coeff(0, 0).real(), 2.0, 1e-10);
}

TEST(AddPythonTest, QTensorCppDirect) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::object b_obj = py::cast(QTensorCpp(make_identity2()));
    QTensorCpp c = a.add_python(b_obj);
    EXPECT_NEAR(c.get_data().coeff(0, 0).real(), 2.0, 1e-10);
}

TEST(AddPythonTest, WrappedObject) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::exec(R"(
class _WrapAdd:
    def __init__(self, cpp):
        self._qtensor_cpp = cpp
)");
    py::object wrapped = py::globals()["_WrapAdd"](py::cast(QTensorCpp(make_identity2())));
    EXPECT_NO_THROW(a.add_python(wrapped));
}

TEST(AddPythonTest, ScalarOnScalar_Succeeds) {
    py::gil_scoped_acquire gil;
    SparseMatrix m(1, 1);
    m.insert(0, 0) = 1.0;
    m.makeCompressed();
    QTensorCpp q(m);
    py::object two = py::int_(2);
    EXPECT_NO_THROW(q.add_python(two));
}

TEST(AddPythonTest, ZeroScalarOnNonScalar_ReturnsSelf) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::object zero = py::float_(0.0);
    QTensorCpp c = a.add_python(zero);
    EXPECT_TRUE(a.equals(c));
}

TEST(AddPythonTest, NonzeroScalarOnNonScalar_Throws) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::object one = py::float_(1.0);
    EXPECT_THROW(a.add_python(one), py::type_error);
}

TEST(AddPythonTest, WrongType_Throws) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::exec(R"(_not_a_qt = "hello")");
    EXPECT_THROW(a.add_python(py::globals()["_not_a_qt"]), py::type_error);
}

TEST(SubTest, DirectCpp) {
    QTensorCpp a(make_identity2()), b(make_identity2());
    QTensorCpp c = a.sub(b);
    EXPECT_NEAR(c.get_data().coeff(0, 0).real(), 0.0, 1e-10);
}

TEST(SubPythonTest, QTensorCppDirect) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::object b_obj = py::cast(QTensorCpp(make_identity2()));
    EXPECT_NO_THROW(a.sub_python(b_obj));
}

TEST(SubPythonTest, WrappedObject) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::exec(R"(
class _WrapSub:
    def __init__(self, cpp):
        self._qtensor_cpp = cpp
)");
    py::object wrapped = py::globals()["_WrapSub"](py::cast(QTensorCpp(make_identity2())));
    EXPECT_NO_THROW(a.sub_python(wrapped));
}

TEST(SubPythonTest, WrongType_Throws) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::exec(R"(_sub_bad = 99)");
    EXPECT_THROW(a.sub_python(py::globals()["_sub_bad"]), py::type_error);
}

TEST(MulTest, ElementwiseCpp) {
    QTensorCpp a(make_identity2()), b(make_identity2());
    QTensorCpp c = a.mul(b);
    EXPECT_NEAR(c.get_data().coeff(0, 0).real(), 1.0, 1e-10);
}

TEST(MulTest, ScalarCpp) {
    QTensorCpp a(make_identity2());
    QTensorCpp c = a.mul(std::complex<double>(2.0, 0.0));
    EXPECT_NEAR(c.get_data().coeff(0, 0).real(), 2.0, 1e-10);
}

TEST(MulPythonTest, QTensorCppDirect) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::object b_obj = py::cast(QTensorCpp(make_identity2()));
    EXPECT_NO_THROW(a.mul_python(b_obj));
}

TEST(MulPythonTest, WrappedObject) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::exec(R"(
class _WrapMul:
    def __init__(self, cpp):
        self._qtensor_cpp = cpp
)");
    py::object wrapped = py::globals()["_WrapMul"](py::cast(QTensorCpp(make_identity2())));
    EXPECT_NO_THROW(a.mul_python(wrapped));
}

TEST(MulPythonTest, FloatScalar) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    EXPECT_NO_THROW(a.mul_python(py::float_(2.0)));
}

TEST(MulPythonTest, IntScalar) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    EXPECT_NO_THROW(a.mul_python(py::int_(3)));
}

TEST(MulPythonTest, WrongType_Throws) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::exec(R"(_mul_bad = "hello")");
    EXPECT_THROW(a.mul_python(py::globals()["_mul_bad"]), py::type_error);
}

TEST(MatmulTest, OperatorTimesOperator) {
    QTensorCpp a(make_identity2()), b(make_pauli_x());
    QTensorCpp c = a.matmul(b);
    EXPECT_NEAR(c.get_data().coeff(0, 1).real(), 1.0, 1e-10);
}

TEST(MatmulPythonTest, QTensorCppDirect) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::object b_obj = py::cast(QTensorCpp(make_pauli_x()));
    EXPECT_NO_THROW(a.matmul_python(b_obj));
}

TEST(MatmulPythonTest, WrappedObject) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::exec(R"(
class _WrapMM:
    def __init__(self, cpp):
        self._qtensor_cpp = cpp
)");
    py::object wrapped = py::globals()["_WrapMM"](py::cast(QTensorCpp(make_pauli_x())));
    EXPECT_NO_THROW(a.matmul_python(wrapped));
}

TEST(MatmulPythonTest, WrongType_Throws) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::exec(R"(_mm_bad = 5)");
    EXPECT_THROW(a.matmul_python(py::globals()["_mm_bad"]), py::type_error);
}

TEST(EqualsTest, SameMatrix_True) {
    QTensorCpp a(make_identity2()), b(make_identity2());
    EXPECT_TRUE(a.equals(b));
}

TEST(EqualsTest, DifferentMatrix_False) {
    QTensorCpp a(make_identity2()), b(make_pauli_x());
    EXPECT_FALSE(a.equals(b));
}

TEST(EqualsTest, DifferentShape_False) {
    QTensorCpp a(make_ket0()), b(make_identity2());
    EXPECT_FALSE(a.equals(b));
}

TEST(EqualsPythonTest, QTensorCppDirect_True) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::object b_obj = py::cast(QTensorCpp(make_identity2()));
    EXPECT_TRUE(a.equals_python(b_obj));
}

TEST(EqualsPythonTest, WrappedObject) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::exec(R"(
class _WrapEq:
    def __init__(self, cpp):
        self._qtensor_cpp = cpp
)");
    py::object wrapped = py::globals()["_WrapEq"](py::cast(QTensorCpp(make_identity2())));
    EXPECT_TRUE(a.equals_python(wrapped));
}

TEST(EqualsPythonTest, WrongType_False) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::exec(R"(_eq_bad = "hello")");
    EXPECT_FALSE(a.equals_python(py::globals()["_eq_bad"]));
}

TEST(OperatorOverloadsTest, Plus) {
    QTensorCpp a(make_identity2()), b(make_identity2());
    QTensorCpp c = a + b;
    EXPECT_NEAR(c.get_data().coeff(0, 0).real(), 2.0, 1e-10);
}

TEST(OperatorOverloadsTest, Minus) {
    QTensorCpp a(make_identity2()), b(make_identity2());
    QTensorCpp c = a - b;
    EXPECT_NEAR(c.get_data().coeff(0, 0).real(), 0.0, 1e-10);
}

TEST(OperatorOverloadsTest, Times_Matmul) {
    QTensorCpp a(make_identity2()), b(make_pauli_x());
    QTensorCpp c = a * b;
    EXPECT_NEAR(c.get_data().coeff(0, 1).real(), 1.0, 1e-10);
}

TEST(OperatorOverloadsTest, DivideByComplex) {
    QTensorCpp a(make_identity2());
    QTensorCpp c = a / std::complex<double>(2.0, 0.0);
    EXPECT_NEAR(c.get_data().coeff(0, 0).real(), 0.5, 1e-10);
}

TEST(OperatorOverloadsTest, DivideByDouble) {
    QTensorCpp a(make_identity2());
    QTensorCpp c = a / 2.0;
    EXPECT_NEAR(c.get_data().coeff(0, 0).real(), 0.5, 1e-10);
}

TEST(OperatorOverloadsTest, EqualityAndInequality) {
    QTensorCpp a(make_identity2()), b(make_identity2()), c(make_pauli_x());
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_TRUE(a != c);
    EXPECT_FALSE(a != b);
}

TEST(OperatorOverloadsTest, StreamOperator) {
    QTensorCpp a(make_ket0());
    std::ostringstream oss;
    EXPECT_NO_THROW(oss << a);
    EXPECT_NE(oss.str().find("QTensor"), std::string::npos);
}

TEST(OperatorOverloadsTest, Subscript) {
    QTensorCpp a(make_identity2());
    auto diag = a[{0, 0}];
    auto off = a[{0, 1}];
    EXPECT_NEAR(diag.real(), 1.0, 1e-10);
    EXPECT_NEAR(off.real(), 0.0, 1e-10);
}

TEST(OperatorOverloadsTest, MultiplyByComplexScalar) {
    QTensorCpp a(make_identity2());
    QTensorCpp c = a * std::complex<double>(3.0, 0.0);
    EXPECT_NEAR(c.get_data().coeff(0, 0).real(), 3.0, 1e-10);
}

TEST(OperatorOverloadsTest, MultiplyByDoubleScalar) {
    QTensorCpp a(make_identity2());
    QTensorCpp c = a * 3.0;
    EXPECT_NEAR(c.get_data().coeff(0, 0).real(), 3.0, 1e-10);
}

TEST(DivTest, ZeroScalar_Throws) {
    QTensorCpp a(make_identity2());
    EXPECT_THROW(a.div(0.0), py::value_error);
}

TEST(DivTest, NonzeroScalar_Succeeds) {
    QTensorCpp a(make_identity2());
    EXPECT_NO_THROW(a.div(2.0));
}

TEST(EigenGetterTest, BeforeCompute_Throws) {
    QTensorCpp q(make_identity2());
    EXPECT_THROW(q.get_eigenvalues(), py::value_error);
}

TEST(EigenGetterTest, AfterCompute_ReturnsValues) {
    QTensorCpp q(make_identity2());
    q.compute_eigendecomposition();
    EXPECT_NO_THROW(q.get_eigenvalues());
    EXPECT_EQ(int(q.get_eigenvalues().size()), 2);
}

TEST(EigenGetterTest, PythonVersion_AfterCompute) {
    py::gil_scoped_acquire gil;
    QTensorCpp q(make_identity2());
    q.compute_eigendecomposition();
    py::object evals = q.get_eigenvalues_python();
    EXPECT_EQ(py::len(evals), 2);
}

TEST(EigenGetterTest, EigenvectorsBeforeCompute_Throws) {
    QTensorCpp q(make_identity2());
    EXPECT_THROW(q.get_eigenvectors(), py::value_error);
}

TEST(EigenGetterTest, EigenvectorsAfterCompute_ReturnsVecs) {
    QTensorCpp q(make_identity2());
    q.compute_eigendecomposition();
    EXPECT_NO_THROW(q.get_eigenvectors());
    EXPECT_EQ(int(q.get_eigenvectors().size()), 2);
}

TEST(EigenGetterTest, EigenvectorsPythonVersion_AfterCompute) {
    py::gil_scoped_acquire gil;
    QTensorCpp q(make_identity2());
    q.compute_eigendecomposition();
    py::object evecs = q.get_eigenvectors_python();
    EXPECT_EQ(py::len(evecs), 2);
}

TEST(ClearCacheTest, AfterClear_EigenvaluesThrow) {
    QTensorCpp q(make_identity2());
    q.compute_eigendecomposition();
    q.clear_cache();
    EXPECT_THROW(q.get_eigenvalues(), py::value_error);
}

TEST(ClearCacheTest, AfterClear_TraceRecomputed) {
    QTensorCpp q(make_identity2());
    q.trace();  // caches trace
    q.clear_cache();
    // should recompute without error
    EXPECT_NO_THROW(q.trace());
}

TEST(ExpectationValueTest, ExactKet_ZOperator) {
    // <0|Z|0> = 1
    SparseMatrix Z(2, 2);
    Z.insert(0, 0) = 1.0;
    Z.insert(1, 1) = -1.0;
    Z.makeCompressed();
    QTensorCpp ket(make_ket0());
    QTensorCpp op_z(Z);
    auto ev = ket.expectation_value(op_z, 0);
    EXPECT_NEAR(ev.real(), 1.0, 1e-10);
}

TEST(ExpectationValueTest, ExactBra) {
    SparseMatrix Z(2, 2);
    Z.insert(0, 0) = 1.0;
    Z.insert(1, 1) = -1.0;
    Z.makeCompressed();
    SparseMatrix bra_m(1, 2);
    bra_m.insert(0, 0) = 1.0;
    bra_m.makeCompressed();
    QTensorCpp bra(bra_m);
    QTensorCpp op_z(Z);
    auto ev = bra.expectation_value(op_z, 0);
    EXPECT_NEAR(ev.real(), 1.0, 1e-10);
}

TEST(ExpectationValueTest, ExactOperator) {
    QTensorCpp dm(make_dm_pure0());
    QTensorCpp id(make_identity2());
    auto ev = dm.expectation_value(id, 0);
    EXPECT_NEAR(ev.real(), 1.0, 1e-10);
}

TEST(ExpectationValueTest, SampledZeroState_Throws) {
    QTensorCpp q(QTensorCpp::zero(2, "operator"));
    q.compute_eigendecomposition();
    QTensorCpp op(make_identity2());
    EXPECT_THROW(q.expectation_value(op, 10), py::value_error);
}

TEST(ExpectationValueTest, SampledNoEigendecomp_Throws) {
    QTensorCpp q(make_dm_pure0());
    QTensorCpp op(make_identity2());
    EXPECT_THROW(q.expectation_value(op, 10), py::value_error);
}

TEST(ExpectationValueTest, SampledWithEigendecomp_Succeeds) {
    QTensorCpp q(make_dm_pure0());
    q.compute_eigendecomposition();
    QTensorCpp op(make_identity2());
    EXPECT_NO_THROW(q.expectation_value(op, 10));
}

TEST(ExpectationValuePythonTest, QTensorCppDirect) {
    py::gil_scoped_acquire gil;
    QTensorCpp q(make_dm_pure0());
    py::object op_obj = py::cast(QTensorCpp(make_identity2()));
    EXPECT_NO_THROW(q.expectation_value_python(op_obj));
}

TEST(ExpectationValuePythonTest, WrappedObject) {
    py::gil_scoped_acquire gil;
    QTensorCpp q(make_dm_pure0());
    py::exec(R"(
class _WrapEV:
    def __init__(self, cpp):
        self._qtensor_cpp = cpp
)");
    py::object wrapped = py::globals()["_WrapEV"](py::cast(QTensorCpp(make_identity2())));
    EXPECT_NO_THROW(q.expectation_value_python(wrapped));
}

TEST(ExpectationValuePythonTest, WrongType_Throws) {
    py::gil_scoped_acquire gil;
    QTensorCpp q(make_dm_pure0());
    py::exec(R"(_ev_bad = 1)");
    EXPECT_THROW(q.expectation_value_python(py::globals()["_ev_bad"]), py::value_error);
}

TEST(CommutatorTest, IdentityAndX_ZeroCommutator) {
    QTensorCpp id(make_identity2()), x(make_pauli_x());
    QTensorCpp comm = id.commutator(x);
    EXPECT_TRUE(QTensorCpp(SparseMatrix(2, 2)).equals(comm));
}

TEST(CommutatorPythonTest, QTensorCppDirect) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::object b_obj = py::cast(QTensorCpp(make_pauli_x()));
    EXPECT_NO_THROW(a.commutator_python(b_obj));
}

TEST(CommutatorPythonTest, WrappedObject) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::exec(R"(
class _WrapComm:
    def __init__(self, cpp):
        self._qtensor_cpp = cpp
)");
    py::object wrapped = py::globals()["_WrapComm"](py::cast(QTensorCpp(make_pauli_x())));
    EXPECT_NO_THROW(a.commutator_python(wrapped));
}

TEST(CommutatorPythonTest, WrongType_Throws) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::exec(R"(_comm_bad = "x")");
    EXPECT_THROW(a.commutator_python(py::globals()["_comm_bad"]), py::value_error);
}

TEST(AnticommutatorTest, IdentityAndX) {
    QTensorCpp id(make_identity2()), x(make_pauli_x());
    QTensorCpp acomm = id.anticommutator(x);
    // {I, X} = 2X
    EXPECT_NEAR(acomm.get_data().coeff(0, 1).real(), 2.0, 1e-10);
}

TEST(AnticommutatorPythonTest, QTensorCppDirect) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::object b_obj = py::cast(QTensorCpp(make_pauli_x()));
    EXPECT_NO_THROW(a.anticommutator_python(b_obj));
}

TEST(AnticommutatorPythonTest, WrappedObject) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::exec(R"(
class _WrapAComm:
    def __init__(self, cpp):
        self._qtensor_cpp = cpp
)");
    py::object wrapped = py::globals()["_WrapAComm"](py::cast(QTensorCpp(make_pauli_x())));
    EXPECT_NO_THROW(a.anticommutator_python(wrapped));
}

TEST(AnticommutatorPythonTest, WrongType_Throws) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_identity2());
    py::exec(R"(_acomm_bad = None)");
    EXPECT_THROW(a.anticommutator_python(py::globals()["_acomm_bad"]), py::value_error);
}

TEST(ProbabilitiesTest, Ket0_FirstProb1) {
    QTensorCpp q(make_ket0());
    auto probs = q.probabilities();
    EXPECT_NEAR(probs[0], 1.0, 1e-10);
    EXPECT_NEAR(probs[1], 0.0, 1e-10);
}

TEST(ProbabilitiesTest, Bra0_FirstProb1) {
    SparseMatrix m(1, 2);
    m.insert(0, 0) = 1.0;
    m.makeCompressed();
    QTensorCpp q(m);
    auto probs = q.probabilities();
    EXPECT_NEAR(probs[0], 1.0, 1e-10);
    EXPECT_NEAR(probs[1], 0.0, 1e-10);
}

TEST(ProbabilitiesTest, OperatorDM_DiagonalEntries) {
    QTensorCpp q(make_dm_mixed());
    auto probs = q.probabilities();
    EXPECT_NEAR(probs[0], 0.5, 1e-10);
    EXPECT_NEAR(probs[1], 0.5, 1e-10);
}

TEST(ProbabilitiesTest, Scalar_Throws) {
    SparseMatrix m(1, 1);
    m.insert(0, 0) = 1.0;
    m.makeCompressed();
    QTensorCpp q(m);
    EXPECT_THROW(q.probabilities(), py::value_error);
}

TEST(ProbabilitiesPythonTest, ReturnsList) {
    py::gil_scoped_acquire gil;
    QTensorCpp q(make_ket0());
    py::list probs = q.probabilities_python();
    EXPECT_EQ(py::len(probs), 2);
}

TEST(IsUnitaryTest, NonOperator_False) {
    QTensorCpp q(make_ket0());
    EXPECT_FALSE(q.is_unitary());
}

TEST(IsUnitaryTest, PauliX_True) {
    QTensorCpp q(make_pauli_x());
    EXPECT_TRUE(q.is_unitary());
}

TEST(IsUnitaryTest, ScaledIdentity_False) {
    QTensorCpp q(make_identity2());
    QTensorCpp scaled = q * 2.0;
    EXPECT_FALSE(scaled.is_unitary());
}

TEST(IsUnitaryTest, Cached_SecondCallSameResult) {
    QTensorCpp q(make_pauli_x());
    EXPECT_TRUE(q.is_unitary());
    EXPECT_TRUE(q.is_unitary());  // hits cache
}

TEST(PurityTest, PureDM_Purity1) {
    QTensorCpp q(make_dm_pure0());
    EXPECT_NEAR(q.purity(), 1.0, 1e-10);
}

TEST(PurityTest, MixedDM_PurityLessThan1) {
    QTensorCpp q(make_dm_mixed());  // Tr((0.5I)^2) = 0.5
    EXPECT_NEAR(q.purity(), 0.5, 1e-10);
}

TEST(PurityTest, Cached_SecondCallSameResult) {
    QTensorCpp q(make_dm_pure0());
    double p1 = q.purity();
    double p2 = q.purity();  // hits cache
    EXPECT_EQ(p1, p2);
}

TEST(DotTest, Ket0DotKet0_Is1) {
    QTensorCpp a(make_ket0()), b(make_ket0());
    auto d = a.dot(b);
    EXPECT_NEAR(d.real(), 1.0, 1e-10);
}

TEST(DotTest, Ket0DotKet1_Is0) {
    QTensorCpp a(make_ket0()), b(make_ket1());
    auto d = a.dot(b);
    EXPECT_NEAR(std::abs(d), 0.0, 1e-10);
}

TEST(DotPythonTest, QTensorCppDirect) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_ket0());
    py::object b_obj = py::cast(QTensorCpp(make_ket0()));
    EXPECT_NO_THROW(a.dot_python(b_obj));
}

TEST(DotPythonTest, WrappedObject) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_ket0());
    py::exec(R"(
class _WrapDot:
    def __init__(self, cpp):
        self._qtensor_cpp = cpp
)");
    py::object wrapped = py::globals()["_WrapDot"](py::cast(QTensorCpp(make_ket0())));
    EXPECT_NO_THROW(a.dot_python(wrapped));
}

TEST(DotPythonTest, WrongType_Throws) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_ket0());
    py::exec(R"(_dot_bad = 1.0)");
    EXPECT_THROW(a.dot_python(py::globals()["_dot_bad"]), py::value_error);
}

TEST(FidelityTest, Ket0AndKet0_Is1) {
    QTensorCpp a(make_ket0()), b(make_ket0());
    EXPECT_NEAR(a.fidelity(b), 1.0, 1e-10);
}

TEST(FidelityTest, Ket0AndKet1_Is0) {
    QTensorCpp a(make_ket0()), b(make_ket1());
    EXPECT_NEAR(a.fidelity(b), 0.0, 1e-10);
}

TEST(FidelityTest, BraAndBra) {
    SparseMatrix bm(1, 2);
    bm.insert(0, 0) = 1.0;
    bm.makeCompressed();
    QTensorCpp a(bm), b(bm);
    EXPECT_NEAR(a.fidelity(b), 1.0, 1e-10);
}

TEST(FidelityTest, OperatorAndOperator) {
    QTensorCpp a(make_dm_pure0()), b(make_dm_pure0());
    EXPECT_NEAR(a.fidelity(b), 1.0, 1e-10);
}

TEST(FidelityTest, MixedTypes_Throws) {
    QTensorCpp a(make_ket0()), b(make_dm_pure0());
    EXPECT_THROW(a.fidelity(b), py::value_error);
}

TEST(FidelityPythonTest, QTensorCppDirect) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_ket0());
    py::object b_obj = py::cast(QTensorCpp(make_ket0()));
    EXPECT_NO_THROW(a.fidelity_python(b_obj));
}

TEST(FidelityPythonTest, WrappedObject) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_ket0());
    py::exec(R"(
class _WrapFid:
    def __init__(self, cpp):
        self._qtensor_cpp = cpp
)");
    py::object wrapped = py::globals()["_WrapFid"](py::cast(QTensorCpp(make_ket0())));
    EXPECT_NO_THROW(a.fidelity_python(wrapped));
}

TEST(FidelityPythonTest, WrongType_Throws) {
    py::gil_scoped_acquire gil;
    QTensorCpp a(make_ket0());
    py::exec(R"(_fid_bad = None)");
    EXPECT_THROW(a.fidelity_python(py::globals()["_fid_bad"]), py::value_error);
}

TEST(EntropyTest, VonNeumann_NotDM_Throws) {
    QTensorCpp q(make_identity2());  // trace=2, not a DM
    EXPECT_THROW(q.entropy_von_neumann(), py::value_error);
}

TEST(EntropyTest, VonNeumann_PureDM_Is0) {
    QTensorCpp q(make_dm_pure0());
    EXPECT_NEAR(q.entropy_von_neumann(), 0.0, 1e-8);
}

TEST(EntropyTest, VonNeumann_MaxMixed_IsLog2) {
    QTensorCpp q(make_dm_mixed());
    EXPECT_NEAR(q.entropy_von_neumann(), std::log(2.0), 1e-8);
}

TEST(EntropyTest, Renyi_NotDM_Throws) {
    QTensorCpp q(make_identity2());
    EXPECT_THROW(q.entropy_renyi(2.0), py::value_error);
}

TEST(EntropyTest, Renyi_AlphaZero_Throws) {
    QTensorCpp q(make_dm_mixed());
    EXPECT_THROW(q.entropy_renyi(0.0), py::value_error);
}

TEST(EntropyTest, Renyi_AlphaOne_Throws) {
    QTensorCpp q(make_dm_mixed());
    EXPECT_THROW(q.entropy_renyi(1.0), py::value_error);
}

TEST(EntropyTest, Renyi_Alpha2_Succeeds) {
    QTensorCpp q(make_dm_mixed());
    EXPECT_NO_THROW(q.entropy_renyi(2.0));
}

TEST(RankTest, Ket_Rank1) {
    QTensorCpp q(make_ket0());
    EXPECT_EQ(q.rank(), 1);
}

TEST(RankTest, Bra_Rank1) {
    SparseMatrix m(1, 2);
    m.insert(0, 0) = 1.0;
    m.makeCompressed();
    QTensorCpp q(m);
    EXPECT_EQ(q.rank(), 1);
}

TEST(RankTest, IdentityRank2) {
    QTensorCpp q(make_identity2());
    EXPECT_EQ(q.rank(), 2);
}

TEST(RankTest, WithEigenvaluesCached) {
    QTensorCpp q(make_dm_pure0());
    q.compute_eigendecomposition();
    EXPECT_EQ(q.rank(), 1);
}

TEST(RankTest, Cached_SecondCallSameResult) {
    QTensorCpp q(make_identity2());
    EXPECT_EQ(q.rank(), 2);
    EXPECT_EQ(q.rank(), 2);
}

TEST(ExpTest, NonOperator_Throws) {
    QTensorCpp q(make_ket0());
    EXPECT_THROW(q.exp(), py::value_error);
}

TEST(ExpTest, ZeroMatrix_SucceedsWithoutEigendecomp) {
    QTensorCpp q(make_identity2());
    EXPECT_NO_THROW(q.exp());
}

TEST(ExpTest, WithCachedSymmetricEigendecomp) {
    QTensorCpp q(make_identity2());
    q.compute_eigendecomposition();
    EXPECT_NO_THROW(q.exp());
}

TEST(LogTest, NonOperator_Throws) {
    QTensorCpp q(make_ket0());
    EXPECT_THROW(q.log(), py::value_error);
}

TEST(LogTest, WithoutEigendecomp) {
    QTensorCpp q(make_identity2());
    EXPECT_NO_THROW(q.log());
}

TEST(LogTest, WithCachedSymmetricEigendecomp) {
    QTensorCpp q(make_identity2());
    q.compute_eigendecomposition();
    EXPECT_NO_THROW(q.log());
}

TEST(SqrtTest, NonOperator_Throws) {
    QTensorCpp q(make_ket0());
    EXPECT_THROW(q.sqrt(), py::value_error);
}

TEST(SqrtTest, ZeroMatrix_SpecialCase) {
    QTensorCpp zero = QTensorCpp::zero(2, "operator");
    ASSERT_NO_THROW(zero.sqrt());
}

TEST(SqrtTest, WithCachedSelfAdjointEigendecomp) {
    QTensorCpp q(make_dm_pure0());
    q.compute_eigendecomposition();
    EXPECT_NO_THROW(q.sqrt());
}

TEST(SqrtTest, WithoutEigendecomp) {
    QTensorCpp q(make_identity2());
    EXPECT_NO_THROW(q.sqrt());
}

TEST(PowTest, NonOperator_Throws) {
    QTensorCpp q(make_ket0());
    EXPECT_THROW(q.pow(2.0), py::value_error);
}

TEST(PowTest, WithoutEigendecomp) {
    QTensorCpp q(make_identity2());
    EXPECT_NO_THROW(q.pow(2.0));
}

TEST(PowTest, WithCachedSelfAdjointEigendecomp) {
    QTensorCpp q(make_identity2());
    q.compute_eigendecomposition();
    EXPECT_NO_THROW(q.pow(2.0));
}

TEST(InverseTest, NonOperator_Throws) {
    QTensorCpp q(make_ket0());
    EXPECT_THROW(q.inverse(), py::value_error);
}

TEST(InverseTest, WithoutEigendecomp) {
    QTensorCpp q(make_identity2());
    EXPECT_NO_THROW(q.inverse());
}

TEST(InverseTest, WithCachedSelfAdjointEigendecomp) {
    QTensorCpp q(make_identity2());
    q.compute_eigendecomposition();
    QTensorCpp inv = q.inverse();
    EXPECT_NEAR(inv.get_data().coeff(0, 0).real(), 1.0, 1e-10);
}

TEST(AsDensityMatrixTest, Scalar_Throws) {
    SparseMatrix m(1, 1);
    m.insert(0, 0) = 1.0;
    m.makeCompressed();
    QTensorCpp q(m);
    EXPECT_THROW(q.as_density_matrix(), py::value_error);
}

TEST(AsDensityMatrixTest, NonPowerOf2Matrix_Throws) {
    SparseMatrix m(3, 3);
    m.insert(0, 0) = 1.0;
    m.makeCompressed();
    QTensorCpp q(m, true);
    EXPECT_THROW(q.as_density_matrix(), py::value_error);
}

TEST(AsDensityMatrixTest, NotOperator_Throws) {
    SparseMatrix m(3, 3);
    m.insert(0, 0) = 1.0;
    m.makeCompressed();
    EXPECT_THROW(QTensorCpp q(m, false), py::value_error);
}

TEST(AsDensityMatrixTest, Ket_ReturnsDM) {
    QTensorCpp q(make_ket0());
    QTensorCpp dm = q.as_density_matrix();
    EXPECT_TRUE(dm.is_density_matrix());
}

TEST(AsDensityMatrixTest, Bra_ReturnsDM) {
    SparseMatrix bra_m(1, 2);
    bra_m.insert(0, 0) = 1.0;
    bra_m.makeCompressed();
    QTensorCpp q(bra_m);
    QTensorCpp dm = q.as_density_matrix();
    EXPECT_TRUE(dm.is_density_matrix());
}

TEST(AsDensityMatrixTest, ValidDM_ReturnedAsIs) {
    QTensorCpp q(make_dm_pure0());
    QTensorCpp dm = q.as_density_matrix();
    EXPECT_TRUE(dm.is_density_matrix());
}

TEST(AsDensityMatrixTest, NonPSDOperator_Repaired) {
    SparseMatrix m(2, 2);
    m.insert(0, 0) = std::complex<double>(1.01, 0.0);
    m.insert(1, 1) = std::complex<double>(-0.01, 0.0);
    m.makeCompressed();
    QTensorCpp q(m);
    QTensorCpp dm;
    ASSERT_NO_THROW(dm = q.as_density_matrix());
    EXPECT_TRUE(dm.is_density_matrix());
}

TEST(AsDensityMatrixTest, LargeCorrectionOperator_Throws) {
    SparseMatrix m(2, 2);
    m.insert(0, 0) = std::complex<double>(100.0, 0.0);
    m.insert(1, 1) = std::complex<double>(-99.0, 0.0);
    m.makeCompressed();
    QTensorCpp q(m);
    EXPECT_THROW(q.as_density_matrix(), py::value_error);
}

TEST(ResetQubitsTest, EmptySet_ReturnsSelf) {
    QTensorCpp q(make_dm_pure0());
    QTensorCpp r = q.reset_qubits({});
    EXPECT_TRUE(q.equals(r));
}

TEST(ResetQubitsTest, NotDensityMatrix_Throws) {
    QTensorCpp q(make_ket0());
    EXPECT_THROW(q.reset_qubits({0}), py::value_error);
}

TEST(ResetQubitsTest, InvalidQubit_Throws) {
    QTensorCpp q(make_dm_pure0());
    EXPECT_THROW(q.reset_qubits({5}), py::value_error);
}

TEST(ResetQubitsTest, ResetOneQubit_InTwoQubitSystem) {
    SparseMatrix dm4(4, 4);
    dm4.insert(0, 0) = 1.0;
    dm4.makeCompressed();
    QTensorCpp q(dm4);
    EXPECT_NO_THROW(q.reset_qubits({0}));
}

TEST(ResetQubitsTest, ResetAllQubits) {
    QTensorCpp q(make_dm_pure0());
    QTensorCpp r = q.reset_qubits({0});
    EXPECT_TRUE(r.is_density_matrix());
}

TEST(ResetQubitsTest, PythonVersion) {
    py::gil_scoped_acquire gil;
    QTensorCpp q(make_dm_pure0());
    py::exec(R"(_reset_qubits = [0])");
    EXPECT_NO_THROW(q.reset_qubits_python(py::globals()["_reset_qubits"]));
}

TEST(ResetQubitsTest, MaximallyMixedState_InnerIteratorCovered) {
    SparseMatrix dm4(4, 4);
    dm4.insert(0, 0) = 0.25;
    dm4.insert(1, 1) = 0.25;
    dm4.insert(2, 2) = 0.25;
    dm4.insert(3, 3) = 0.25;
    dm4.makeCompressed();
    QTensorCpp q(dm4);
    QTensorCpp r = q.reset_qubits({0});
    EXPECT_TRUE(r.is_density_matrix());
}

TEST(ResetQubitsTest, ResetToZeroState) {
    SparseMatrix dm4(4, 4);
    dm4.insert(0, 0) = 0.5;
    dm4.insert(3, 3) = 0.5;
    dm4.makeCompressed();
    QTensorCpp q(dm4);
    QTensorCpp r = q.reset_qubits({0, 1});
    EXPECT_TRUE(r.is_density_matrix());
    EXPECT_NEAR(r.get_data().coeff(0, 0).real(), 1.0, 1e-10);
}
