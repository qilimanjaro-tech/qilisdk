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

// GCOV_EXCL_BR_START

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include "../../../src/qilisdk_cpp/backends/qilisim/utils/parsers.h"

namespace py = pybind11;

TEST(ParseHamiltonians, MatrixFreeEmptyThrows) {
    py::list empty;
    EXPECT_ANY_THROW(parse_hamiltonians_matrix_free(1, empty));
}

TEST(ParseHamiltonians, MatrixFreeSingleTerm) {
    py::gil_scoped_acquire gil;

    py::exec(R"(
        from qilisdk.analog.hamiltonian import X
        fake_H = X(0)
    )");

    py::object fake_H = py::globals()["fake_H"];
    py::list Hs;
    Hs.append(fake_H);

    std::vector<MatrixFreeHamiltonian> result = parse_hamiltonians_matrix_free(1, Hs);
    std::vector<MatrixFreeHamiltonian> expected = {MatrixFreeHamiltonian(1, MatrixFreeOperator("X", 0))};
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], expected[0]);
}

TEST(ParseHamiltonians, MatrixFreeOutOfRangeQubitThrows) {
    py::gil_scoped_acquire gil;

    // A Pauli acting on qubit 5, but only 1 qubit is declared. The parser must
    // reject the out-of-range index (validate_qubit_index) before it reaches the
    // matrix-free kernels.
    py::exec(R"(
        from qilisdk.analog.hamiltonian import X
        fake_H = X(5)
    )");

    py::object fake_H = py::globals()["fake_H"];
    py::list Hs;
    Hs.append(fake_H);

    EXPECT_ANY_THROW(parse_hamiltonians_matrix_free(1, Hs));
}

TEST(ParseHamiltonians, ParseEmptyHamiltoniansThrows) {
    py::list empty;
    EXPECT_ANY_THROW(parse_hamiltonians(empty, 1e-10));
}

TEST(ParseHamiltonians, SparseSingleHamiltonian) {
    py::gil_scoped_acquire gil;

    py::exec(R"(
        import scipy.sparse as sp
        import numpy as np

        class FakeSparseHamiltonian:
            def to_matrix(self):
                return sp.eye(2, format='csr')

        fake_sparse_H = FakeSparseHamiltonian()
    )");

    py::object fake_sparse_H = py::globals()["fake_sparse_H"];
    py::list Hs;
    Hs.append(fake_sparse_H);

    auto result = parse_hamiltonians(Hs, 1e-10);
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].rows(), 2);
    EXPECT_EQ(result[0].cols(), 2);
}

TEST(ParseNoiseModel, NoneReturnsEmpty) {
    py::gil_scoped_acquire gil;
    auto result = parse_noise_model(py::none(), 2, 1e-10);
    EXPECT_TRUE(result.is_empty());
}

TEST(ParseNoiseModel, GlobalStaticKraus) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp
        import numpy as np
        from qilisdk.noise.protocols import SupportsStaticKraus as _SSK

        class FakeKrausOp:
            def __init__(self, data):
                self.data = data

        class FakeKrausChannel:
            def __init__(self, ops):
                self.operators = ops

        class StaticKrausPass(_SSK):
            def as_kraus(self):
                data = sp.csr_matrix(np.eye(2, dtype=complex))
                return FakeKrausChannel([FakeKrausOp(data)])

        class FakeNoiseConfig:
            default_gate_time = 1.0

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = [StaticKrausPass()]
            per_qubit_noise = {}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}
        
        fake_nm_global_static_kraus = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_global_static_kraus"];

    EXPECT_FALSE(SupportsStaticKraus.ptr() == nullptr) << "SupportsStaticKraus is not initialized";

    auto result = parse_noise_model(fake_nm, 1, 1e-10);

    EXPECT_FALSE(result.is_empty());
    EXPECT_EQ(result.get_kraus_operators_global().size(), 1u);
}

TEST(ParseNoiseModel, GlobalTimeDerivedKraus) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np
        from qilisdk.noise.protocols import SupportsTimeDerivedKraus as _STDK

        class FakeKrausOp:
            def __init__(self, data): self.data = data

        class FakeKrausChannel:
            def __init__(self, ops): self.operators = ops

        class TimeDerivedKrausPass(_STDK):
            def as_kraus_from_duration(self, duration):
                data = sp.csr_matrix(np.eye(2, dtype=complex))
                return FakeKrausChannel([FakeKrausOp(data)])

        class FakeNoiseConfig:
            default_gate_time = 0.5

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = [TimeDerivedKrausPass()]
            per_qubit_noise = {}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_global_td_kraus = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_global_td_kraus"];
    auto result = parse_noise_model(fake_nm, 1, 1e-10);

    EXPECT_FALSE(result.is_empty());
    EXPECT_EQ(result.get_kraus_operators_global().size(), 1u);
}

TEST(ParseNoiseModel, GlobalStaticLindblad_SingleQubit_ExpandsToAll) {
    // Regression test for the global single-qubit Lindblad bug: a global single-qubit jump
    // operator must expand into one independent jump per qubit (L on qubit q, identity
    // elsewhere), NOT collapse into a single collective jump L^{⊗N}. The buggy code produced
    // one operator; the correct code produces nqubits operators.
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np

        class FakeJumpOp:
            def __init__(self, data): self.data = data

        class FakeLindblad:
            def __init__(self, ops): self.jump_operators_with_rates = ops

        class StaticLindbladPass:
            def as_lindblad(self):
                data = sp.csr_matrix(np.array([[0,1],[0,0]], dtype=complex))
                return FakeLindblad([FakeJumpOp(data)])

        class FakeNoiseConfig:
            default_gate_time = 1.0

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = [StaticLindbladPass()]
            per_qubit_noise = {}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_global_lindblad_1q = FakeNoiseModel()
    )");
    py::object fake_nm = py::globals()["fake_nm_global_lindblad_1q"];
    auto result = parse_noise_model(fake_nm, 3, 1e-10);
    EXPECT_EQ(result.get_jump_operators().size(), 3u);
}

// A global single-qubit Lindblad pass must produce exactly the same set of jump operators as
// attaching that same single-qubit Lindblad pass to every qubit individually. This is the
// global-vs-per-qubit equivalence that the buggy L^{⊗N} expansion violated.
TEST(ParseNoiseModel, GlobalStaticLindblad_SingleQubit_MatchesPerQubit) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np

        class FakeJumpOp:
            def __init__(self, data): self.data = data

        class FakeLindblad:
            def __init__(self, ops): self.jump_operators_with_rates = ops

        class StaticLindbladPass:
            def as_lindblad(self):
                data = sp.csr_matrix(np.array([[0,1],[0,0]], dtype=complex))
                return FakeLindblad([FakeJumpOp(data)])

        class FakeNoiseConfig:
            default_gate_time = 1.0

        class FakeNoiseModelGlobal:
            noise_config = FakeNoiseConfig()
            global_noise = [StaticLindbladPass()]
            per_qubit_noise = {}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        class FakeNoiseModelPerQubit:
            noise_config = FakeNoiseConfig()
            global_noise = []
            per_qubit_noise = {0: [StaticLindbladPass()], 1: [StaticLindbladPass()], 2: [StaticLindbladPass()]}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_equiv_global = FakeNoiseModelGlobal()
        fake_nm_equiv_per_qubit = FakeNoiseModelPerQubit()
    )");

    auto global_result = parse_noise_model(py::globals()["fake_nm_equiv_global"], 3, 1e-10);
    auto per_qubit_result = parse_noise_model(py::globals()["fake_nm_equiv_per_qubit"], 3, 1e-10);

    const auto& global_ops = global_result.get_jump_operators();
    const auto& per_qubit_ops = per_qubit_result.get_jump_operators();

    ASSERT_EQ(global_ops.size(), 3u);
    ASSERT_EQ(per_qubit_ops.size(), 3u);
    for (size_t i = 0; i < global_ops.size(); ++i) {
        EXPECT_TRUE(DenseMatrix(global_ops[i]).isApprox(DenseMatrix(per_qubit_ops[i]))) << "Global and per-qubit jump operator " << i << " differ";
    }
}

// A global Lindblad operator that is multi-qubit but does not span the whole system is
// ambiguous (which qubits does it act on?) and must be rejected rather than silently expanded.
TEST(ParseNoiseModel, GlobalLindblad_MultiQubitNonFullSystemThrows) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np

        class FakeJumpOp:
            def __init__(self, data): self.data = data

        class FakeLindblad:
            def __init__(self, ops): self.jump_operators_with_rates = ops

        class StaticLindbladPass:
            def as_lindblad(self):
                # 4x4 (2-qubit) operator on a 3-qubit system: neither single-qubit nor full-system.
                data = sp.csr_matrix(np.eye(4, dtype=complex))
                return FakeLindblad([FakeJumpOp(data)])

        class FakeNoiseConfig:
            default_gate_time = 1.0

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = [StaticLindbladPass()]
            per_qubit_noise = {}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_global_lindblad_ambiguous = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_global_lindblad_ambiguous"];
    EXPECT_THROW(parse_noise_model(fake_nm, 3, 1e-10), py::value_error);
}

TEST(ParseNoiseModel, GlobalLindblad_FullSystemOperator) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np

        class FakeJumpOp:
            def __init__(self, data): self.data = data

        class FakeLindblad:
            def __init__(self, ops): self.jump_operators_with_rates = ops

        class StaticLindbladPass:
            def as_lindblad(self):
                # 4x4 matches a 2-qubit system exactly
                data = sp.csr_matrix(np.eye(4, dtype=complex))
                return FakeLindblad([FakeJumpOp(data)])

        class FakeNoiseConfig:
            default_gate_time = 1.0

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = [StaticLindbladPass()]
            per_qubit_noise = {}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_global_lindblad_full = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_global_lindblad_full"];
    auto result = parse_noise_model(fake_nm, 2, 1e-10);

    EXPECT_EQ(result.get_jump_operators().size(), 1u);
}

TEST(ParseNoiseModel, GlobalTimeDerivedLindblad) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np
        from qilisdk.noise.protocols import SupportsTimeDerivedLindblad as _STDL

        class FakeJumpOp:
            def __init__(self, data): self.data = data

        class FakeLindblad:
            def __init__(self, ops): self.jump_operators_with_rates = ops

        class TimeDerivedLindbladPass(_STDL):
            def as_lindblad_from_duration(self, duration):
                data = sp.csr_matrix(np.array([[0,1],[0,0]], dtype=complex))
                return FakeLindblad([FakeJumpOp(data)])

        class FakeNoiseConfig:
            default_gate_time = 1.0

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = [TimeDerivedLindbladPass()]
            per_qubit_noise = {}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_global_td_lindblad = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_global_td_lindblad"];
    auto result = parse_noise_model(fake_nm, 2, 1e-10);
    // Single-qubit global Lindblad expands to one independent jump per qubit (see the fix in
    // GlobalStaticLindblad_SingleQubit_ExpandsToAll), so a 2-qubit system yields 2 operators.
    EXPECT_EQ(result.get_jump_operators().size(), 2u);
}

TEST(ParseNoiseModel, GlobalReadoutAssignment) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.noise.readout_assignment import ReadoutAssignment
        class ReadoutPass(ReadoutAssignment):
            p01 = 0.01
            p10 = 0.02
            def __init__(self):
                super().__init__(p01=self.p01, p10=self.p10)

        class FakeNoiseConfig:
            default_gate_time = 1.0

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = [ReadoutPass()]
            per_qubit_noise = {}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_global_readout = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_global_readout"];
    auto result = parse_noise_model(fake_nm, 2, 1e-10);

    auto [p01_q0, p10_q0] = result.get_relevant_readout_error(0);
    EXPECT_NEAR(p01_q0, 0.01, 1e-10);
    EXPECT_NEAR(p10_q0, 0.02, 1e-10);

    auto [p01_q1, p10_q1] = result.get_relevant_readout_error(1);
    EXPECT_NEAR(p01_q1, 0.01, 1e-10);
    EXPECT_NEAR(p10_q1, 0.02, 1e-10);
}

TEST(ParseNoiseModel, PerQubitStaticKraus) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np

        class FakeKrausOp:
            def __init__(self, data): self.data = data

        class FakeKrausChannel:
            def __init__(self, ops): self.operators = ops

        class StaticKrausPass:
            def as_kraus(self):
                data = sp.csr_matrix(np.eye(2, dtype=complex))
                return FakeKrausChannel([FakeKrausOp(data)])

        class FakeNoiseConfig:
            default_gate_time = 1.0

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = []
            per_qubit_noise = {0: [StaticKrausPass()], 1: [StaticKrausPass()]}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_per_qubit_kraus = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_per_qubit_kraus"];
    auto result = parse_noise_model(fake_nm, 2, 1e-10);

    const auto& per_qubit_map = result.get_kraus_operators_per_qubit();
    ASSERT_TRUE(per_qubit_map.count(0));
    EXPECT_EQ(per_qubit_map.at(0).size(), 1u);
    ASSERT_TRUE(per_qubit_map.count(1));
    EXPECT_EQ(per_qubit_map.at(1).size(), 1u);
}

TEST(ParseNoiseModel, PerQubitDynamicKraus) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np
        from qilisdk.noise.protocols import SupportsTimeDerivedKraus as _STDK

        class FakeKrausOp:
            def __init__(self, data): self.data = data
        class FakeKrausChannel:
            def __init__(self, ops): self.operators = ops
        class TimeDerivedKrausPass(_STDK):
            def as_kraus_from_duration(self, duration):
                data = sp.csr_matrix(np.eye(2, dtype=complex))
                return FakeKrausChannel([FakeKrausOp(data)])
        class FakeNoiseConfig:
            default_gate_time = 1.0
        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = []
            per_qubit_noise = {0: [TimeDerivedKrausPass()], 1: [TimeDerivedKrausPass()]}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_per_qubit_td_kraus = FakeNoiseModel()
    )");
    py::object fake_nm = py::globals()["fake_nm_per_qubit_td_kraus"];
    auto result = parse_noise_model(fake_nm, 2, 1e-10);
    const auto& per_qubit_map = result.get_kraus_operators_per_qubit();
    ASSERT_TRUE(per_qubit_map.count(0));
    EXPECT_EQ(per_qubit_map.at(0).size(), 1u);
    ASSERT_TRUE(per_qubit_map.count(1));
    EXPECT_EQ(per_qubit_map.at(1).size(), 1u);
}

TEST(ParseNoiseModel, PerQubitLindbladDynamic) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np
        from qilisdk.noise.protocols import SupportsTimeDerivedLindblad as _STDL

        class FakeJumpOp:
            def __init__(self, data): self.data = data

        class FakeLindblad:
            def __init__(self, ops): self.jump_operators_with_rates = ops

        class TimeDerivedLindbladPass(_STDL):
            def as_lindblad_from_duration(self, duration):
                data = sp.csr_matrix(np.array([[0,1],[0,0]], dtype=complex))
                return FakeLindblad([FakeJumpOp(data)])

        class FakeNoiseConfig:
            default_gate_time = 1.0

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = []
            per_qubit_noise = {1: [TimeDerivedLindbladPass()]}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_per_qubit_td_lindblad = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_per_qubit_td_lindblad"];
    auto result = parse_noise_model(fake_nm, 3, 1e-10);

    EXPECT_EQ(result.get_jump_operators().size(), 1u);
}

TEST(ParseNoiseModel, PerQubitLindbladExpandedToFullSystem) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np

        class FakeJumpOp:
            def __init__(self, data): self.data = data

        class FakeLindblad:
            def __init__(self, ops): self.jump_operators_with_rates = ops

        class StaticLindbladPass:
            def as_lindblad(self):
                data = sp.csr_matrix(np.array([[0,1],[0,0]], dtype=complex))
                return FakeLindblad([FakeJumpOp(data)])

        class FakeNoiseConfig:
            default_gate_time = 1.0

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = []
            per_qubit_noise = {1: [StaticLindbladPass()]}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_per_qubit_lindblad = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_per_qubit_lindblad"];
    auto result = parse_noise_model(fake_nm, 3, 1e-10);

    EXPECT_EQ(result.get_jump_operators().size(), 1u);
}

TEST(ParseNoiseModel, PerQubitReadoutAssignment) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.noise.readout_assignment import ReadoutAssignment
        class ReadoutPass(ReadoutAssignment):
            p01 = 0.05
            p10 = 0.03
            def __init__(self):
                super().__init__(p01=self.p01, p10=self.p10)

        class FakeNoiseConfig:
            default_gate_time = 1.0

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = []
            per_qubit_noise = {2: [ReadoutPass()]}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_per_qubit_readout = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_per_qubit_readout"];
    auto result = parse_noise_model(fake_nm, 3, 1e-10);

    auto [p01, p10] = result.get_relevant_readout_error(2);
    EXPECT_NEAR(p01, 0.05, 1e-10);
    EXPECT_NEAR(p10, 0.03, 1e-10);
}

TEST(ParseNoiseModel, PerQubitReadoutAssignment_NoBleedToOtherQubits) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.noise.readout_assignment import ReadoutAssignment
        class ReadoutPass(ReadoutAssignment):
            p01 = 0.05
            p10 = 0.03
            def __init__(self):
                super().__init__(p01=self.p01, p10=self.p10)

        class FakeNoiseConfig:
            default_gate_time = 1.0

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = []
            per_qubit_noise = {2: [ReadoutPass()]}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_per_qubit_readout_no_bleed = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_per_qubit_readout_no_bleed"];
    auto result = parse_noise_model(fake_nm, 3, 1e-10);

    auto [p01_q0, p10_q0] = result.get_relevant_readout_error(0);
    EXPECT_NEAR(p01_q0, 0.0, 1e-10);
    EXPECT_NEAR(p10_q0, 0.0, 1e-10);
}

TEST(ParseNoiseModel, PerGateStaticKraus) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np

        class FakeKrausOp:
            def __init__(self, data): self.data = data

        class FakeKrausChannel:
            def __init__(self, ops): self.operators = ops

        class StaticKrausPass:
            def as_kraus(self):
                data = sp.csr_matrix(np.eye(2, dtype=complex))
                return FakeKrausChannel([FakeKrausOp(data)])

        class FakeGate:
            __name__ = 'H'

        class FakeNoiseConfig:
            default_gate_time = 1.0
            def get_gate_time(self, gate_type): return self.default_gate_time

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = []
            per_qubit_noise = {}
            per_gate_noise = {FakeGate(): [StaticKrausPass()]}
            per_gate_per_qubit_noise = {}

        fake_nm_per_gate_kraus = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_per_gate_kraus"];
    auto result = parse_noise_model(fake_nm, 1, 1e-10);

    const auto& per_gate_map = result.get_kraus_operators_per_gate();
    ASSERT_TRUE(per_gate_map.count("H"));
    EXPECT_EQ(per_gate_map.at("H").size(), 1u);
}

TEST(ParseNoiseModel, PerGateTimeDerivedKraus) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np
        from qilisdk.noise.protocols import SupportsTimeDerivedKraus as _STDK

        class FakeKrausOp:
            def __init__(self, data): self.data = data

        class FakeKrausChannel:
            def __init__(self, ops): self.operators = ops

        class TimeDerivedKrausPass(_STDK):
            def as_kraus_from_duration(self, duration):
                data = sp.csr_matrix(np.eye(2, dtype=complex))
                return FakeKrausChannel([FakeKrausOp(data)])

        class FakeGate:
            __name__ = 'RZ'

        class FakeNoiseConfig:
            default_gate_time = 0.25
            def get_gate_time(self, gate_type): return self.default_gate_time

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = []
            per_qubit_noise = {}
            per_gate_noise = {FakeGate(): [TimeDerivedKrausPass()]}
            per_gate_per_qubit_noise = {}

        fake_nm_per_gate_td_kraus = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_per_gate_td_kraus"];
    auto result = parse_noise_model(fake_nm, 1, 1e-10);

    const auto& per_gate_map = result.get_kraus_operators_per_gate();
    ASSERT_TRUE(per_gate_map.count("RZ"));
    EXPECT_EQ(per_gate_map.at("RZ").size(), 1u);
}

TEST(ParseNoiseModel, PerGatePerQubitStaticKraus) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np

        class FakeKrausOp:
            def __init__(self, data): self.data = data

        class FakeKrausChannel:
            def __init__(self, ops): self.operators = ops

        class StaticKrausPass:
            def as_kraus(self):
                data = sp.csr_matrix(np.eye(2, dtype=complex))
                return FakeKrausChannel([FakeKrausOp(data)])

        class FakeGate:
            __name__ = 'CNOT'

        class FakeNoiseConfig:
            default_gate_time = 1.0
            def get_gate_time(self, gate_type): return self.default_gate_time

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = []
            per_qubit_noise = {}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {(FakeGate(), 0): [StaticKrausPass()]}

        fake_nm_per_gate_per_qubit_kraus = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_per_gate_per_qubit_kraus"];
    auto result = parse_noise_model(fake_nm, 2, 1e-10);

    const auto& map = result.get_kraus_operators_per_gate_qubit();
    auto key = std::make_pair(NoiseModelCpp::make_gate_key("X", 1), 0);
    ASSERT_TRUE(map.count(key));
    EXPECT_EQ(map.at(key).size(), 1u);
}

TEST(ParseNoiseModel, PerGatePerQubitTimeDerivedKraus) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np
        from qilisdk.noise.protocols import SupportsTimeDerivedKraus as _STDK

        class FakeKrausOp:
            def __init__(self, data): self.data = data

        class FakeKrausChannel:
            def __init__(self, ops): self.operators = ops

        class TimeDerivedKrausPass(_STDK):
            def as_kraus_from_duration(self, duration):
                data = sp.csr_matrix(np.eye(2, dtype=complex))
                return FakeKrausChannel([FakeKrausOp(data)])

        class FakeGate:
            __name__ = 'CZ'

        class FakeNoiseConfig:
            default_gate_time = 1.0
            def get_gate_time(self, gate_type): return self.default_gate_time

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = []
            per_qubit_noise = {}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {(FakeGate(), 1): [TimeDerivedKrausPass()]}

        fake_nm_per_gate_per_qubit_td_kraus = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_per_gate_per_qubit_td_kraus"];
    auto result = parse_noise_model(fake_nm, 2, 1e-10);

    const auto& map = result.get_kraus_operators_per_gate_qubit();
    auto key = std::make_pair(NoiseModelCpp::make_gate_key("Z", 1), 1);
    ASSERT_TRUE(map.count(key));
    EXPECT_EQ(map.at(key).size(), 1u);
}

TEST(ParseNoiseModel, GlobalTimeDerivedKrausExpandsPerGateWithCircuit) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np
        from qilisdk.noise.protocols import SupportsTimeDerivedKraus as _STDK

        class FakeKrausOp:
            def __init__(self, data): self.data = data

        class FakeKrausChannel:
            def __init__(self, ops): self.operators = ops

        class TimeDerivedKrausPass(_STDK):
            def as_kraus_from_duration(self, duration):
                data = sp.csr_matrix(np.eye(2, dtype=complex))
                return FakeKrausChannel([FakeKrausOp(data)])

        class FakeNoiseConfig:
            default_gate_time = 1.0
            def get_gate_time(self, gate_type):
                return 5e-6

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = [TimeDerivedKrausPass()]
            per_qubit_noise = {}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        class FakeCircuitGate:
            def __init__(self, name, control_qubits=()):
                self.name = name
                self.control_qubits = list(control_qubits)

        class FakeCircuit:
            # A plain X, an I, and a CNOT (base 'X' with 1 control). The CNOT must be kept
            # distinct from the plain X gate.
            gates = [FakeCircuitGate('X'), FakeCircuitGate('I'), FakeCircuitGate('CNOT', control_qubits=[0])]

        fake_nm_global_td_kraus_circuit = FakeNoiseModel()
        fake_circuit_global = FakeCircuit()
    )");

    py::object fake_nm = py::globals()["fake_nm_global_td_kraus_circuit"];
    py::object circuit = py::globals()["fake_circuit_global"];
    auto result = parse_noise_model(fake_nm, 1, 1e-10, circuit);
    EXPECT_TRUE(result.get_kraus_operators_global().empty());
    const auto& per_gate_map = result.get_kraus_operators_per_gate();
    EXPECT_EQ(per_gate_map.size(), 3u);  // 'X', 'I', and 'X#c1' (the CNOT) kept distinct
    ASSERT_TRUE(per_gate_map.count("X"));
    ASSERT_TRUE(per_gate_map.count("I"));
    ASSERT_TRUE(per_gate_map.count(NoiseModelCpp::make_gate_key("X", 1)));
    EXPECT_EQ(per_gate_map.at("X").size(), 1u);
    EXPECT_EQ(per_gate_map.at("I").size(), 1u);
}

TEST(ParseNoiseModel, PerQubitTimeDerivedKrausExpandsPerGateQubitWithCircuit) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np
        from qilisdk.noise.protocols import SupportsTimeDerivedKraus as _STDK

        class FakeKrausOp:
            def __init__(self, data): self.data = data

        class FakeKrausChannel:
            def __init__(self, ops): self.operators = ops

        class TimeDerivedKrausPass(_STDK):
            def as_kraus_from_duration(self, duration):
                data = sp.csr_matrix(np.eye(2, dtype=complex))
                return FakeKrausChannel([FakeKrausOp(data)])

        class FakeNoiseConfig:
            default_gate_time = 1.0
            def get_gate_time(self, gate_type):
                return 5e-6

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = []
            per_qubit_noise = {0: [TimeDerivedKrausPass()]}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        class FakeCircuitGate:
            def __init__(self, name, control_qubits=()):
                self.name = name
                self.control_qubits = list(control_qubits)

        class FakeCircuit:
            gates = [FakeCircuitGate('I')]

        fake_nm_per_qubit_td_kraus_circuit = FakeNoiseModel()
        fake_circuit_per_qubit = FakeCircuit()
    )");

    py::object fake_nm = py::globals()["fake_nm_per_qubit_td_kraus_circuit"];
    py::object circuit = py::globals()["fake_circuit_per_qubit"];
    auto result = parse_noise_model(fake_nm, 1, 1e-10, circuit);
    EXPECT_TRUE(result.get_kraus_operators_per_qubit().empty());
    const auto& map = result.get_kraus_operators_per_gate_qubit();
    auto key = std::make_pair(std::string("I"), 0);
    ASSERT_TRUE(map.count(key));
    EXPECT_EQ(map.at(key).size(), 1u);
}

TEST(ParseNoiseModel, NonSquareLindbladThrows) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np

        class FakeJumpOp:
            def __init__(self, data): self.data = data

        class FakeLindblad:
            def __init__(self, ops): self.jump_operators_with_rates = ops

        class BadLindbladPass:
            def as_lindblad(self):
                data = sp.csr_matrix(np.zeros((2, 4), dtype=complex))
                return FakeLindblad([FakeJumpOp(data)])

        class FakeNoiseConfig:
            default_gate_time = 1.0

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = [BadLindbladPass()]
            per_qubit_noise = {}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_nonsquare = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_nonsquare"];
    EXPECT_THROW(parse_noise_model(fake_nm, 2, 1e-10), py::value_error);
}

TEST(ParseNoiseModel, NonPowerOfTwoLindbladThrows) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np

        class FakeJumpOp:
            def __init__(self, data): self.data = data

        class FakeLindblad:
            def __init__(self, ops): self.jump_operators_with_rates = ops

        class BadLindbladPass:
            def as_lindblad(self):
                # 3x3 — square but not a power of two
                data = sp.csr_matrix(np.eye(3, dtype=complex))
                return FakeLindblad([FakeJumpOp(data)])

        class FakeNoiseConfig:
            default_gate_time = 1.0

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = [BadLindbladPass()]
            per_qubit_noise = {}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_nonpow2 = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_nonpow2"];
    EXPECT_THROW(parse_noise_model(fake_nm, 2, 1e-10), py::value_error);
}

TEST(ParseNoiseModel, AllEmptyPassesProduceEmptyModel) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        class FakeNoiseConfig:
            default_gate_time = 1.0

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = []
            per_qubit_noise = {}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_all_empty = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_all_empty"];
    auto result = parse_noise_model(fake_nm, 3, 1e-10);

    EXPECT_TRUE(result.is_empty());
    EXPECT_TRUE(result.get_kraus_operators_global().empty());
    EXPECT_TRUE(result.get_jump_operators().empty());
    EXPECT_TRUE(result.get_kraus_operators_per_qubit().empty());
    EXPECT_TRUE(result.get_kraus_operators_per_gate().empty());
    EXPECT_TRUE(result.get_kraus_operators_per_gate_qubit().empty());
}

TEST(ParseNoiseModel, MultipleGlobalKrausPassesAccumulate) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np

        class FakeKrausOp:
            def __init__(self, data): self.data = data

        class FakeKrausChannel:
            def __init__(self, ops): self.operators = ops

        class StaticKrausPass:
            def as_kraus(self):
                data = sp.csr_matrix(np.eye(2, dtype=complex))
                return FakeKrausChannel([FakeKrausOp(data)])

        class FakeNoiseConfig:
            default_gate_time = 1.0

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = [StaticKrausPass(), StaticKrausPass(), StaticKrausPass()]
            per_qubit_noise = {}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_multi_kraus = FakeNoiseModel()
    )");

    py::object fake_nm = py::globals()["fake_nm_multi_kraus"];
    auto result = parse_noise_model(fake_nm, 1, 1e-10);

    EXPECT_EQ(result.get_kraus_operators_global().size(), 3u);
}

TEST(ParseObservablesMatrixFree, HamiltonianObservable) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.analog.hamiltonian import X
        fake_obs_hamiltonian = [X(0)]
    )");

    py::object fake_obs = py::globals()["fake_obs_hamiltonian"];
    auto result = parse_observables_matrix_free(1, fake_obs);

    ASSERT_EQ(result.size(), 1u);
}

TEST(ParseObservablesMatrixFree, PauliOperatorObservable) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.analog.hamiltonian import PauliZ
        fake_obs_pauli = [PauliZ(1)]
    )");

    py::object fake_obs = py::globals()["fake_obs_pauli"];
    auto result = parse_observables_matrix_free(1, fake_obs);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], MatrixFreeHamiltonian(1, MatrixFreeOperator("Z", 1)));
}

TEST(ParseObservablesMatrixFree, MultipleMixedObservables) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.analog.hamiltonian import X, PauliX
        fake_obs_mixed = [X(0), PauliX(1)]
    )");

    py::object fake_obs = py::globals()["fake_obs_mixed"];
    auto result = parse_observables_matrix_free(2, fake_obs);

    EXPECT_EQ(result.size(), 2u);
}

TEST(ParseObservablesMatrixFree, QTensorObservableThrows) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.core.qtensor import QTensor
        import scipy.sparse as sp, numpy as np
        fake_obs_qtensor = [QTensor(sp.csr_matrix(np.eye(2, dtype=complex)))]
    )");

    py::object fake_obs = py::globals()["fake_obs_qtensor"];
    EXPECT_THROW(parse_observables_matrix_free(1, fake_obs), py::value_error);
}

TEST(ParseObservablesMatrixFree, UnrecognizedTypeThrows) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        class UnknownObs:
            pass
        fake_obs_unknown = [UnknownObs()]
    )");

    py::object fake_obs = py::globals()["fake_obs_unknown"];
    EXPECT_THROW(parse_observables_matrix_free(1, fake_obs), py::value_error);
}

TEST(ParseObservablesMatrixFree, EmptyObservablesReturnsEmpty) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        fake_obs_empty = []
    )");

    py::object fake_obs = py::globals()["fake_obs_empty"];
    auto result = parse_observables_matrix_free(1, fake_obs);
    EXPECT_TRUE(result.empty());
}

TEST(ParseObservables, HamiltonianObservable) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.analog.hamiltonian import X
        fake_parse_obs_hamiltonian = [X(0)]
    )");

    py::object fake_obs = py::globals()["fake_parse_obs_hamiltonian"];
    auto result = parse_observables(fake_obs, 2, 1e-10);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].rows(), 4);
    EXPECT_EQ(result[0].cols(), 4);
}

TEST(ParseObservables, HamiltonianObservable_ExpandedToFullSystem) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.analog.hamiltonian import X
        fake_parse_obs_hamiltonian_expand = [X(1)]
    )");

    py::object fake_obs = py::globals()["fake_parse_obs_hamiltonian_expand"];
    auto result = parse_observables(fake_obs, 2, 1e-10);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].rows(), 4);
    EXPECT_EQ(result[0].cols(), 4);
}

TEST(ParseObservables, PauliOperatorObservable) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.analog.hamiltonian import PauliZ
        import numpy as np
        fake_parse_obs_pauli = [PauliZ(0)]
    )");

    py::object fake_obs = py::globals()["fake_parse_obs_pauli"];
    auto result = parse_observables(fake_obs, 1, 1e-10);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].rows(), 2);
    EXPECT_EQ(result[0].cols(), 2);
}

TEST(ParseObservables, PauliOperatorObservable_ExpandedToFullSystem) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.analog.hamiltonian import PauliZ
        fake_parse_obs_pauli_expand = [PauliZ(1)]
    )");

    py::object fake_obs = py::globals()["fake_parse_obs_pauli_expand"];
    auto result = parse_observables(fake_obs, 2, 1e-10);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].rows(), 4);
    EXPECT_EQ(result[0].cols(), 4);
}

TEST(ParseObservables, QTensorObservable) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.core.qtensor import QTensor
        import scipy.sparse as sp, numpy as np
        fake_parse_obs_qtensor = [QTensor(sp.csr_matrix(np.eye(4, dtype=complex)))]
    )");

    py::object fake_obs = py::globals()["fake_parse_obs_qtensor"];
    auto result = parse_observables(fake_obs, 2, 1e-10);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].rows(), 4);
    EXPECT_EQ(result[0].cols(), 4);
}

TEST(ParseObservables, UnrecognizedTypeThrows) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        class UnknownObs:
            pass
        fake_parse_obs_unknown = [UnknownObs()]
    )");

    py::object fake_obs = py::globals()["fake_parse_obs_unknown"];
    EXPECT_THROW(parse_observables(fake_obs, 1, 1e-10), py::value_error);
}

TEST(ParseObservables, EmptyObservablesReturnsEmpty) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        fake_parse_obs_empty_list = []
    )");

    py::object fake_obs = py::globals()["fake_parse_obs_empty_list"];
    auto result = parse_observables(fake_obs, 1, 1e-10);
    EXPECT_TRUE(result.empty());
}

TEST(ParseTimeSteps, BasicSteps) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        fake_steps_basic = [0.1, 0.2, 0.3]
    )");

    py::object fake_steps = py::globals()["fake_steps_basic"];
    auto result = parse_time_steps(fake_steps);

    ASSERT_EQ(result.size(), 3u);
    EXPECT_NEAR(result[0], 0.1, 1e-10);
    EXPECT_NEAR(result[1], 0.2, 1e-10);
    EXPECT_NEAR(result[2], 0.3, 1e-10);
}

TEST(ParseTimeSteps, SingleStep) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        fake_steps_single = [1.0]
    )");

    py::object fake_steps = py::globals()["fake_steps_single"];
    auto result = parse_time_steps(fake_steps);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 1.0, 1e-10);
}

TEST(ParseTimeSteps, EmptyStepsThrows) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        fake_steps_empty = []
    )");

    py::object fake_steps = py::globals()["fake_steps_empty"];
    EXPECT_THROW(parse_time_steps(fake_steps), py::value_error);
}

TEST(ParseCoefficients, BasicCoefficients) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        class FakeSchedule:
            # coefficients[h_key][t] -> float
            coefficients = {
                'H0': {0.1: 1.0, 0.2: 2.0, 0.3: 3.0},
                'H1': {0.1: 4.0, 0.2: 5.0, 0.3: 6.0},
            }

        fake_schedule_basic = FakeSchedule()
        fake_ham_keys_basic = ['H0', 'H1']
        fake_steps_coeff = [0.1, 0.2, 0.3]
    )");

    py::object schedule = py::globals()["fake_schedule_basic"];
    py::list h_keys = py::globals()["fake_ham_keys_basic"].cast<py::list>();
    py::object steps = py::globals()["fake_steps_coeff"];

    auto result = parse_coefficients(schedule, h_keys, steps);

    ASSERT_EQ(result.size(), 2u);
    ASSERT_EQ(result[0].size(), 3u);
    EXPECT_NEAR(result[0][0], 1.0, 1e-10);
    EXPECT_NEAR(result[0][1], 2.0, 1e-10);
    EXPECT_NEAR(result[0][2], 3.0, 1e-10);
    ASSERT_EQ(result[1].size(), 3u);
    EXPECT_NEAR(result[1][0], 4.0, 1e-10);
    EXPECT_NEAR(result[1][1], 5.0, 1e-10);
    EXPECT_NEAR(result[1][2], 6.0, 1e-10);
}

TEST(ParseCoefficients, SingleHamiltonianSingleStep) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        class FakeScheduleSingle:
            coefficients = {'H0': {0.5: 7.0}}

        fake_schedule_single = FakeScheduleSingle()
        fake_ham_keys_single = ['H0']
        fake_steps_single_coeff = [0.5]
    )");

    py::object schedule = py::globals()["fake_schedule_single"];
    py::list h_keys = py::globals()["fake_ham_keys_single"].cast<py::list>();
    py::object steps = py::globals()["fake_steps_single_coeff"];

    auto result = parse_coefficients(schedule, h_keys, steps);

    ASSERT_EQ(result.size(), 1u);
    ASSERT_EQ(result[0].size(), 1u);
    EXPECT_NEAR(result[0][0], 7.0, 1e-10);
}

TEST(ParseInitialState, BasicDensityMatrix) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np

        class FakeQTensor:
            def __init__(self):
                self.data = sp.csr_matrix(np.eye(2, dtype=complex))

        fake_initial_state_basic = FakeQTensor()
    )");

    py::object fake_state = py::globals()["fake_initial_state_basic"];
    auto result = parse_initial_state(fake_state, 1e-10, 1);

    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 2);
}

TEST(ParseInitialState, FourByFourState) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import scipy.sparse as sp, numpy as np

        class FakeQTensor4:
            def __init__(self):
                self.data = sp.csr_matrix(np.eye(4, dtype=complex))

        fake_initial_state_4x4 = FakeQTensor4()
    )");

    py::object fake_state = py::globals()["fake_initial_state_4x4"];
    auto result = parse_initial_state(fake_state, 1e-10, 2);

    EXPECT_EQ(result.rows(), 4);
    EXPECT_EQ(result.cols(), 4);
}

TEST(ParseMeasurements, ExplicitMeasurements) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        class FakeMGate:
            name = 'M'
            target_qubits = [0, 2]

        class FakeCircuitMeasure:
            nqubits = 3
            gates = [FakeMGate()]

        fake_circuit_measure = FakeCircuitMeasure()
    )");

    py::object fake_circuit = py::globals()["fake_circuit_measure"];
    auto result = parse_measurements(fake_circuit);

    ASSERT_EQ(result.size(), 3u);
    EXPECT_TRUE(result[0]);
    EXPECT_FALSE(result[1]);
    EXPECT_TRUE(result[2]);
}

TEST(ParseMeasurements, NoMeasurementsDefaultsToAll) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        class FakeHGate:
            name = 'H'
            target_qubits = [0]

        class FakeCircuitNoMeasure:
            nqubits = 3
            gates = [FakeHGate()]

        fake_circuit_no_measure = FakeCircuitNoMeasure()
    )");

    py::object fake_circuit = py::globals()["fake_circuit_no_measure"];
    auto result = parse_measurements(fake_circuit);

    ASSERT_EQ(result.size(), 3u);
    EXPECT_TRUE(result[0]);
    EXPECT_TRUE(result[1]);
    EXPECT_TRUE(result[2]);
}

TEST(ParseMeasurements, EmptyCircuitDefaultsToAll) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        class FakeCircuitEmpty:
            nqubits = 2
            gates = []

        fake_circuit_empty_measure = FakeCircuitEmpty()
    )");

    py::object fake_circuit = py::globals()["fake_circuit_empty_measure"];
    auto result = parse_measurements(fake_circuit);

    ASSERT_EQ(result.size(), 2u);
    EXPECT_TRUE(result[0]);
    EXPECT_TRUE(result[1]);
}

TEST(ParseGates, BasicGate) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np

        class FakeHGate:
            name = 'H'
            control_qubits = []
            target_qubits = [0]
            is_parameterized = False
            @property
            def matrix(self):
                s = 1 / np.sqrt(2)
                return np.array([[s, s], [s, -s]], dtype=complex)
            def get_parameters(self):
                return {}

        class FakeCircuitBasicGate:
            nqubits = 1
            gates = [FakeHGate()]

        fake_circuit_basic_gate = FakeCircuitBasicGate()
    )");

    py::object fake_circuit = py::globals()["fake_circuit_basic_gate"];
    auto result = parse_gates(fake_circuit, 1e-10, py::none());

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].get_name(), "H");
    EXPECT_TRUE(result[0].get_control_qubits().empty());
    ASSERT_EQ(result[0].get_target_qubits().size(), 1u);
    EXPECT_EQ(result[0].get_target_qubits()[0], 0);
}

TEST(ParseGates, MeasurementGateIsNotSkipped) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np

        class FakeMGate:
            name = 'M'
            target_qubits = [0]
            control_qubits = []
            is_parameterized = False
            @property
            def matrix(self): 
                return np.eye(2, dtype=complex)
            def get_parameters(self): return {}

        class FakeCircuitMGate:
            nqubits = 1
            gates = [FakeMGate()]

        fake_circuit_m_gate = FakeCircuitMGate()
    )");

    py::object fake_circuit = py::globals()["fake_circuit_m_gate"];
    auto result = parse_gates(fake_circuit, 1e-10, py::none());

    EXPECT_FALSE(result.empty());
}

TEST(ParseGates, ParameterizedGate) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np

        class FakeRZGate:
            name = 'RZ'
            control_qubits = []
            target_qubits = [0]
            is_parameterized = True
            _theta = 0.5
            @property
            def matrix(self):
                return np.array([
                    [np.exp(-1j * self._theta / 2), 0],
                    [0, np.exp(1j * self._theta / 2)]
                ], dtype=complex)
            def get_parameters(self):
                return {'theta': self._theta}
            def set_parameters(self, params):
                self._theta = params['theta']

        class FakeCircuitParamGate:
            nqubits = 1
            gates = [FakeRZGate()]

        fake_circuit_param_gate = FakeCircuitParamGate()
    )");

    py::object fake_circuit = py::globals()["fake_circuit_param_gate"];
    auto result = parse_gates(fake_circuit, 1e-10, py::none());

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].get_name(), "RZ");
    ASSERT_EQ(result[0].get_parameters().size(), 1u);
    EXPECT_EQ(result[0].get_parameters()[0].first, "theta");
    EXPECT_NEAR(result[0].get_parameters()[0].second, 0.5, 1e-10);
}

TEST(ParseGates, ControlledGate) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np

        class FakeCXGate:
            name = 'CX'
            control_qubits = [0]
            target_qubits = [1]
            is_parameterized = False
            @property
            def matrix(self):
                return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
            def get_parameters(self): return {}

        class FakeCircuitControlledGate:
            nqubits = 2
            gates = [FakeCXGate()]

        fake_circuit_controlled_gate = FakeCircuitControlledGate()
    )");

    py::object fake_circuit = py::globals()["fake_circuit_controlled_gate"];
    auto result = parse_gates(fake_circuit, 1e-10, py::none());

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].get_name(), "X");
    ASSERT_EQ(result[0].get_control_qubits().size(), 1u);
    EXPECT_EQ(result[0].get_control_qubits()[0], 0);
    ASSERT_EQ(result[0].get_target_qubits().size(), 1u);
    EXPECT_EQ(result[0].get_target_qubits()[0], 1);
}

// Same as above but for CY
TEST(ParseGates, ControlledYGate) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np

        class FakeCYGate:
            name = 'CY'
            control_qubits = [0]
            target_qubits = [1]
            is_parameterized = False
            @property
            def matrix(self):
                return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,-1j],[0,0,1j,0]], dtype=complex)
            def get_parameters(self): return {}

        class FakeCircuitControlledYGate:
            nqubits = 2
            gates = [FakeCYGate()]

        fake_circuit_controlled_y_gate = FakeCircuitControlledYGate()
    )");

    py::object fake_circuit = py::globals()["fake_circuit_controlled_y_gate"];
    auto result = parse_gates(fake_circuit, 1e-10, py::none());

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].get_name(), "Y");
    ASSERT_EQ(result[0].get_control_qubits().size(), 1u);
    EXPECT_EQ(result[0].get_control_qubits()[0], 0);
    ASSERT_EQ(result[0].get_target_qubits().size(), 1u);
    EXPECT_EQ(result[0].get_target_qubits()[0], 1);
}

// Same as above but for CZ
TEST(ParseGates, ControlledZGate) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np
        class FakeCZGate:
            name = 'CZ'
            control_qubits = [0]
            target_qubits = [1]
            is_parameterized = False
            @property
            def matrix(self):
                return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=complex)
            def get_parameters(self): return {}

        class FakeCircuitControlledZGate:
            nqubits = 2
            gates = [FakeCZGate()]
        fake_circuit_controlled_z_gate = FakeCircuitControlledZGate()
    )");

    py::object fake_circuit = py::globals()["fake_circuit_controlled_z_gate"];
    auto result = parse_gates(fake_circuit, 1e-10, py::none());

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].get_name(), "Z");
    ASSERT_EQ(result[0].get_control_qubits().size(), 1u);
    EXPECT_EQ(result[0].get_control_qubits()[0], 0);
    ASSERT_EQ(result[0].get_target_qubits().size(), 1u);
    EXPECT_EQ(result[0].get_target_qubits()[0], 1);
}

TEST(ParseGates, EmptyCircuitReturnsNoGates) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        class FakeCircuitNoGates:
            nqubits = 2
            gates = []

        fake_circuit_no_gates = FakeCircuitNoGates()
    )");

    py::object fake_circuit = py::globals()["fake_circuit_no_gates"];
    auto result = parse_gates(fake_circuit, 1e-10, py::none());

    EXPECT_TRUE(result.empty());
}

TEST(ParseGates, ParameterPerturbationAppliedGlobalNoise) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np

        class FakePerturbation:
            def perturb(self, value):
                return value + 0.1  # always adds 0.1

        class FakeRZGatePert:
            name = 'RZ'
            control_qubits = []
            target_qubits = [0]
            is_parameterized = True
            _theta = 0.5
            @property
            def matrix(self):
                return np.eye(2, dtype=complex)
            def get_parameters(self):
                return {'theta': self._theta}
            def set_parameters(self, params):
                self._theta = params['theta']

        class FakeNoiseModelPert:
            global_perturbations = {'theta': [FakePerturbation()]}
            per_gate_perturbations = {}

        class FakeCircuitPert:
            nqubits = 1
            gates = [FakeRZGatePert()]

        fake_circuit_pert = FakeCircuitPert()
        fake_noise_model_pert = FakeNoiseModelPert()
    )");

    py::object fake_circuit = py::globals()["fake_circuit_pert"];
    py::object fake_noise_model = py::globals()["fake_noise_model_pert"];
    auto result = parse_gates(fake_circuit, 1e-10, fake_noise_model);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0].get_parameters()[0].second, 0.6, 1e-10);
}

TEST(ParseGates, ParameterPerturbationAppliedPerGate) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np

        class FakePerturbation:
            def perturb(self, value):
                return value * 2  # always doubles the parameter

        class RZ:
            name = 'RZ'
            control_qubits = []
            target_qubits = [0]
            is_parameterized = True
            _theta = 0.25
            @property
            def matrix(self):
                return np.eye(2, dtype=complex)
            def get_parameters(self):
                return {'theta': self._theta}
            def set_parameters(self, params):
                self._theta = params['theta']

        class FakeNoiseModelPertPerGate:
            global_perturbations = {}
            per_gate_perturbations = {(RZ, 'theta'): [FakePerturbation()]}

        class FakeCircuitPertPerGate:
            nqubits = 1
            gates = [RZ()]

        fake_circuit_pert_per_gate = FakeCircuitPertPerGate()
        fake_noise_model_pert_per_gate = FakeNoiseModelPertPerGate()
    )");

    py::object fake_circuit = py::globals()["fake_circuit_pert_per_gate"];
    py::object fake_noise_model = py::globals()["fake_noise_model_pert_per_gate"];
    auto result = parse_gates(fake_circuit, 1e-10, fake_noise_model);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0].get_parameters()[0].second, 0.5, 1e-10);
}

TEST(ParseSolverParams, EmptyDictReturnsDefaults) {
    py::gil_scoped_acquire gil;
    py::dict empty_params;
    EXPECT_NO_THROW({
        auto config = parse_solver_params(empty_params);
        EXPECT_GE(config.get_num_threads(), 1);
    });
}

TEST(ParseSolverParams, NumThreadsZeroClampedToOne) {
    py::gil_scoped_acquire gil;
    py::dict params;
    params["num_threads"] = py::int_(0);
    auto config = parse_solver_params(params);
    EXPECT_EQ(config.get_num_threads(), 1);
}

TEST(ParseSolverParams, NegativeNumThreadsClampedToOne) {
    py::gil_scoped_acquire gil;
    py::dict params;
    params["num_threads"] = py::int_(-4);
    auto config = parse_solver_params(params);
    EXPECT_EQ(config.get_num_threads(), 1);
}

TEST(ParseSolverParams, AllFieldsParsedCorrectly) {
    py::gil_scoped_acquire gil;
    py::dict params;
    params["max_cache_size"] = py::int_(128);
    params["seed"] = py::int_(42);
    params["atol"] = py::float_(1e-8);
    params["arnoldi_dim"] = py::int_(20);
    params["num_arnoldi_substeps"] = py::int_(4);
    params["evolution_method"] = py::str("arnoldi");
    params["digital_method"] = py::str("statevector");
    params["monte_carlo"] = py::bool_(true);
    params["adaptive_tol"] = py::float_(1e-2);
    params["num_monte_carlo_trajectories"] = py::int_(500);
    params["num_threads"] = py::int_(4);
    params["store_intermediate_results"] = py::bool_(true);
    params["normalize_after_each_gate"] = py::bool_(true);
    params["combine_single_qubit_gates"] = py::bool_(false);
    params["fuse_gates"] = py::bool_(true);
    params["max_fused_qubits"] = py::int_(3);
    params["measurement_collapse"] = py::bool_(true);
    params["gpu"] = py::bool_(true);

    auto config = parse_solver_params(params);

    EXPECT_EQ(config.get_max_cache_size(), 128);
    EXPECT_EQ(config.get_seed(), 42);
    EXPECT_NEAR(config.get_atol(), 1e-8, 1e-15);
    EXPECT_EQ(config.get_arnoldi_dim(), 20);
    EXPECT_EQ(config.get_num_arnoldi_substeps(), 4);
    EXPECT_EQ(config.get_time_evolution_method(), "arnoldi");
    EXPECT_EQ(config.get_digital_method(), "statevector");
    EXPECT_TRUE(config.get_monte_carlo());
    EXPECT_EQ(config.get_num_monte_carlo_trajectories(), 500);
    EXPECT_EQ(config.get_num_threads(), 4);
    EXPECT_TRUE(config.get_store_intermediate_results());
    EXPECT_TRUE(config.get_normalize_after_gate());
    EXPECT_FALSE(config.get_combine_single_qubit_gates());
    EXPECT_TRUE(config.get_fuse_gates());
    EXPECT_EQ(config.get_max_fused_qubits(), 3);
    EXPECT_TRUE(config.get_measurement_collapse());
    EXPECT_NEAR(config.get_adaptive_tol(), 1e-2, 1e-15);
    EXPECT_TRUE(config.get_gpu());
}

TEST(ParseSolverParams, PartialFieldsParsed) {
    py::gil_scoped_acquire gil;
    py::dict params;
    params["seed"] = py::int_(99);
    params["num_threads"] = py::int_(2);

    auto config = parse_solver_params(params);

    EXPECT_EQ(config.get_seed(), 99);
    EXPECT_EQ(config.get_num_threads(), 2);
}

// py::object construct_result_object(const DenseMatrix& state_dense, const py::object& readout, NoiseModelCpp& noise_model_cpp, int n_qubits, const QiliSimConfig& config, const std::vector<bool>& qubits_to_measure) {
TEST(ConstructResults, EmptyResults) {
    DenseMatrix state = DenseMatrix::Zero(2, 1);
    py::list readout = py::list();
    NoiseModelCpp noise_model_cpp;
    int n_qubits = 1;
    QiliSimConfig config;
    std::vector<bool> qubits_to_measure = {true};
    EXPECT_NO_THROW({ auto results = construct_result_object(state, readout, noise_model_cpp, n_qubits, config, qubits_to_measure); });
}

TEST(ConstructResults, StateTomographyNotExactThrows) {
    DenseMatrix state = DenseMatrix::Zero(2, 1);

    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.readout import StateTomographyReadout
        readout = [StateTomographyReadout(method="not_exact")]
    )");
    py::list readout = py::globals()["readout"];

    NoiseModelCpp noise_model_cpp;
    int n_qubits = 1;
    QiliSimConfig config;
    std::vector<bool> qubits_to_measure = {true};
    EXPECT_THROW({ auto results = construct_result_object(state, readout, noise_model_cpp, n_qubits, config, qubits_to_measure); }, py::value_error);
}

TEST(ConstructResults, BadReadoutTypeThrows) {
    DenseMatrix state = DenseMatrix::Zero(2, 1);

    py::list readout = py::list();  // should be a Readout object, not a list
    readout.append(42);             // just to make it non-empty

    NoiseModelCpp noise_model_cpp;
    int n_qubits = 1;
    QiliSimConfig config;
    std::vector<bool> qubits_to_measure = {true};
    EXPECT_THROW({ auto results = construct_result_object(state, readout, noise_model_cpp, n_qubits, config, qubits_to_measure); }, py::value_error);
}

TEST(ConstructResultsExponentialAnsatz, WithExpectationReadout_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.readout import ExpectationReadout
from qilisdk.analog.hamiltonian import Z
_ea_ro_exp = [ExpectationReadout(observables=[Z(0)])]
    )");
    ExponentialAnsatz state(1, 1, 50, 0);
    py::list readout = py::globals()["_ea_ro_exp"].cast<py::list>();
    EXPECT_NO_THROW({ auto result = construct_result_object(state, readout, 1); });
}

TEST(ConstructResultsExponentialAnsatz, WithSamplingReadout_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.readout import SamplingReadout
_ea_ro_samp = [SamplingReadout(nshots=10)]
    )");
    ExponentialAnsatz state(1, 1, 50, 0);
    py::list readout = py::globals()["_ea_ro_samp"].cast<py::list>();
    EXPECT_NO_THROW({ auto result = construct_result_object(state, readout, 1); });
}

TEST(ConstructResultsExponentialAnsatz, WithUnsupportedReadout_Throws) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.readout import StateTomographyReadout
_ea_ro_bad = [StateTomographyReadout()]
    )");
    ExponentialAnsatz state(1, 1, 50, 0);
    py::list readout = py::globals()["_ea_ro_bad"].cast<py::list>();
    EXPECT_THROW({ auto result = construct_result_object(state, readout, 1); }, py::value_error);
}

TEST(ParseInitialState, WithQTensorObject_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.core.qtensor import QTensor
import scipy.sparse as sp, numpy as np
_parse_is_qtensor = QTensor(sp.csr_matrix(np.array([[1.+0j], [0.]], dtype=complex)))
    )");
    py::object qs = py::globals()["_parse_is_qtensor"];
    SparseMatrix result;
    EXPECT_NO_THROW(result = parse_initial_state(qs, 1e-12, 1));
    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 1);
}

TEST(ParseInitialState, WithInitialStateEnum_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.core.qtensor import InitialState
_parse_is_uniform = InitialState.UNIFORM
    )");
    py::object is_obj = py::globals()["_parse_is_uniform"];
    SparseMatrix result;
    EXPECT_NO_THROW(result = parse_initial_state(is_obj, 1e-12, 1));
}

TEST(ParseInitialState, UnknownType_ThrowsValueError) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
_unknown_initial_state = 42
    )");
    py::object obj = py::globals()["_unknown_initial_state"];
    EXPECT_THROW(parse_initial_state(obj, 1e-12, 1), py::value_error);
}

TEST(ParseSolverParams, OrderShotsWarmupsAreApplied) {
    py::gil_scoped_acquire gil;
    py::dict params;
    params["variational_order"] = py::int_(3);
    params["variational_shots"] = py::int_(200);
    params["variational_warmups"] = py::int_(50);
    QiliSimConfig config;
    EXPECT_NO_THROW(config = parse_solver_params(params));
    EXPECT_EQ(config.get_order(), 3);
    EXPECT_EQ(config.get_shots(), 200);
    EXPECT_EQ(config.get_warmups(), 50);
}

// Build a noise model whose only global pass is a real LindbladGenerator. ``rates`` is a Python
// snippet for the rate list (mixing constants and callables exercises both make_sqrt_rate_series
// branches). The generator's real ``is_time_dependent`` / ``jump_operators`` / ``rates`` are used.
static py::object make_global_lindblad_noise_model(const std::string& rates_py, const std::string& global_name) {
    py::exec(R"(
        import numpy as np
        from qilisdk.core import QTensor
        from qilisdk.noise import LindbladGenerator

        _sigma_minus = QTensor(np.array([[0, 1], [0, 0]], dtype=complex))

        class FakeNoiseConfig:
            default_gate_time = 1.0
    )");
    py::exec("_n_ops = len(" + rates_py + ")");
    int n_ops = py::globals()["_n_ops"].cast<int>();
    std::string ops = "[";
    for (int i = 0; i < n_ops; ++i) {
        ops += "_sigma_minus, ";
    }
    ops += "]";
    py::exec(global_name +
             " = type('FakeNoiseModel', (), {"
             "'noise_config': FakeNoiseConfig(), "
             "'global_noise': [LindbladGenerator(" +
             ops + ", rates=" + rates_py +
             ")], "
             "'per_qubit_noise': {}, 'per_gate_noise': {}, 'per_gate_per_qubit_noise': {}})()");
    return py::globals()[global_name.c_str()];
}

TEST(ParseNoiseModel, GlobalTimeDependentLindbladRate) {
    py::gil_scoped_acquire gil;
    // Mixed list: a constant rate (folded as a constant series) and a callable rate(t).
    py::object fake_nm = make_global_lindblad_noise_model("[0.16, lambda t: 0.25]", "fake_nm_td_global");
    std::vector<double> step_list = {0.0, 1.0, 2.0};
    auto result = parse_noise_model(fake_nm, 1, 1e-10, py::none(), &step_list);

    EXPECT_TRUE(result.has_time_dependent_rates());
    ASSERT_EQ(result.get_jump_operators().size(), 2u);
    ASSERT_EQ(result.get_jump_rate_series().size(), 2u);
    // Each series holds sqrt(rate) sampled at every step: sqrt(0.16)=0.4 and sqrt(0.25)=0.5.
    ASSERT_EQ(result.get_jump_rate_series()[0].size(), 3u);
    EXPECT_NEAR(result.get_jump_rate_series()[0][0], 0.4, 1e-9);
    EXPECT_NEAR(result.get_jump_rate_series()[1][2], 0.5, 1e-9);
}

TEST(ParseNoiseModel, PerQubitTimeDependentLindbladRate) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        import numpy as np
        from qilisdk.core import QTensor
        from qilisdk.noise import LindbladGenerator

        class FakeNoiseConfig:
            default_gate_time = 1.0

        class FakeNoiseModel:
            noise_config = FakeNoiseConfig()
            global_noise = []
            per_qubit_noise = {0: [LindbladGenerator([QTensor(np.array([[0,1],[0,0]], dtype=complex))], rates=[lambda t: 0.25])]}
            per_gate_noise = {}
            per_gate_per_qubit_noise = {}

        fake_nm_td_per_qubit = FakeNoiseModel()
    )");
    py::object fake_nm = py::globals()["fake_nm_td_per_qubit"];
    std::vector<double> step_list = {0.0, 1.0};
    auto result = parse_noise_model(fake_nm, 2, 1e-10, py::none(), &step_list);

    EXPECT_TRUE(result.has_time_dependent_rates());
    ASSERT_EQ(result.get_jump_operators().size(), 1u);
    // Single-qubit operator on qubit 0 of a 2-qubit system is expanded to 4x4.
    EXPECT_EQ(result.get_jump_operators()[0].rows(), 4);
    ASSERT_EQ(result.get_jump_rate_series()[0].size(), 2u);
    EXPECT_NEAR(result.get_jump_rate_series()[0][0], 0.5, 1e-9);
}

TEST(ParseNoiseModel, TimeDependentLindbladNegativeRateThrows) {
    py::gil_scoped_acquire gil;
    py::object fake_nm = make_global_lindblad_noise_model("[lambda t: -1.0]", "fake_nm_td_negative");
    std::vector<double> step_list = {0.0, 1.0};
    EXPECT_ANY_THROW(parse_noise_model(fake_nm, 1, 1e-10, py::none(), &step_list));
}

TEST(ParseNoiseModel, TimeDependentLindbladNonFiniteRateThrows) {
    py::gil_scoped_acquire gil;
    py::object fake_nm = make_global_lindblad_noise_model("[lambda t: float('nan')]", "fake_nm_td_nan");
    std::vector<double> step_list = {0.0, 1.0};
    EXPECT_ANY_THROW(parse_noise_model(fake_nm, 1, 1e-10, py::none(), &step_list));
}

TEST(ParseNoiseModel, TimeDependentLindbladWithoutStepListThrows) {
    py::gil_scoped_acquire gil;
    // No step_list (e.g. digital propagation) must reject a time-dependent rate.
    py::object fake_nm = make_global_lindblad_noise_model("[lambda t: 0.25]", "fake_nm_td_no_steps");
    EXPECT_ANY_THROW(parse_noise_model(fake_nm, 1, 1e-10));
}

TEST(ParseNoiseModel, TimeDependentLindbladCallableNonNumericThrows) {
    py::gil_scoped_acquire gil;
    // A callable rate(t) that returns a non-numeric value is rejected.
    py::object fake_nm = make_global_lindblad_noise_model("[lambda t: 'not a number']", "fake_nm_td_nonnum");
    std::vector<double> step_list = {0.0, 1.0};
    EXPECT_ANY_THROW(parse_noise_model(fake_nm, 1, 1e-10, py::none(), &step_list));
}

TEST(ParseNoiseModel, TimeDependentLindbladConstantInvalidRateThrows) {
    py::gil_scoped_acquire gil;
    // A non-finite constant rate alongside a callable rate is rejected by the constant-rate branch.
    py::object fake_nm = make_global_lindblad_noise_model("[lambda t: 0.1, float('nan')]", "fake_nm_td_const_invalid");
    std::vector<double> step_list = {0.0, 1.0};
    EXPECT_ANY_THROW(parse_noise_model(fake_nm, 1, 1e-10, py::none(), &step_list));
}

TEST(ParseSolverParams, StabilizerMaxStatesApplied) {
    py::gil_scoped_acquire gil;
    py::dict params;
    params["stabilizer_max_states"] = py::int_(16);
    QiliSimConfig config;
    EXPECT_NO_THROW(config = parse_solver_params(params));
    EXPECT_EQ(config.get_stabilizer_max_states(), 16);
}

TEST(ParseInitialStateStabilizer, NoneReturnsZeroState) {
    py::gil_scoped_acquire gil;
    auto state = parse_initial_state_stabilizer(py::none(), 2);
    EXPECT_EQ(state.get_nqubits(), 2);
    EXPECT_EQ(state.get_states().size(), 1u);
}

TEST(ParseInitialStateStabilizer, UniformAppliesHadamards) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.core.qtensor import InitialState
        _stab_is_uniform = InitialState.UNIFORM
    )");
    py::object is_obj = py::globals()["_stab_is_uniform"];
    auto state = parse_initial_state_stabilizer(is_obj, 2);
    EXPECT_EQ(state.get_nqubits(), 2);
    // After H on every qubit each qubit is in a superposition (random Z outcome).
    EXPECT_NE(state.get_states()[0].find_x_pivot(0), -1);
    EXPECT_NE(state.get_states()[0].find_x_pivot(1), -1);
}

TEST(ParseInitialStateStabilizer, OneAppliesXGates) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.core.qtensor import InitialState
        _stab_is_one = InitialState.ONE
    )");
    py::object is_obj = py::globals()["_stab_is_one"];
    auto state = parse_initial_state_stabilizer(is_obj, 2);
    EXPECT_EQ(state.get_nqubits(), 2);
    // |11>: every qubit has Z eigenvalue -1.
    EXPECT_TRUE(state.get_states()[0].z_eigenvalue(0));
    EXPECT_TRUE(state.get_states()[0].z_eigenvalue(1));
}

TEST(ParseInitialStateStabilizer, OtherNamedStateFallsBackToZero) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.core.qtensor import InitialState
        _stab_is_zero = InitialState.ZERO
    )");
    py::object is_obj = py::globals()["_stab_is_zero"];
    auto state = parse_initial_state_stabilizer(is_obj, 3);
    EXPECT_EQ(state.get_nqubits(), 3);
    // The unhandled named state falls back to the |00...0> stabilizer state.
    EXPECT_FALSE(state.get_states()[0].z_eigenvalue(0));
    EXPECT_FALSE(state.get_states()[0].z_eigenvalue(2));
}

TEST(ParseInitialStateStabilizer, UnrecognizedTypeThrows) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        _stab_is_bad = 123
    )");
    py::object obj = py::globals()["_stab_is_bad"];
    EXPECT_THROW(parse_initial_state_stabilizer(obj, 1), py::value_error);
}

TEST(ConstructResultsStabilizer, SamplingReadout_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.readout import SamplingReadout
        _stab_ro_samp = [SamplingReadout(nshots=10)]
    )");
    StabilizerStateSum state(1);
    py::list readout = py::globals()["_stab_ro_samp"].cast<py::list>();
    NoiseModelCpp noise_model_cpp;
    QiliSimConfig config;
    std::vector<bool> qubits_to_measure = {true};
    EXPECT_NO_THROW({ auto result = construct_result_object(state, readout, noise_model_cpp, 1, config, qubits_to_measure); });
}

TEST(ConstructResultsStabilizer, ExpectationReadout_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.readout import ExpectationReadout
        from qilisdk.analog.hamiltonian import X
        _stab_ro_exp = [ExpectationReadout(observables=[X(0)])]
    )");
    // |0> has <X(0)> = 0; this exercises the stabilizer ExpectationReadout branch.
    StabilizerStateSum state(1);
    py::list readout = py::globals()["_stab_ro_exp"].cast<py::list>();
    NoiseModelCpp noise_model_cpp;
    QiliSimConfig config;
    std::vector<bool> qubits_to_measure = {true};
    EXPECT_NO_THROW({ auto result = construct_result_object(state, readout, noise_model_cpp, 1, config, qubits_to_measure); });
}

TEST(ConstructResultsStabilizer, StateTomographyReadoutExact_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.readout import StateTomographyReadout
        _stab_ro_tomo = [StateTomographyReadout()]
    )");
    StabilizerStateSum state(1);
    py::list readout = py::globals()["_stab_ro_tomo"].cast<py::list>();
    NoiseModelCpp noise_model_cpp;
    QiliSimConfig config;
    std::vector<bool> qubits_to_measure = {true};
    EXPECT_NO_THROW({ auto result = construct_result_object(state, readout, noise_model_cpp, 1, config, qubits_to_measure); });
}

TEST(ConstructResultsStabilizer, StateTomographyReadoutNonExact_Throws) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.readout import StateTomographyReadout
        _stab_ro_tomo_bad = [StateTomographyReadout(method="mle")]
    )");
    StabilizerStateSum state(1);
    py::list readout = py::globals()["_stab_ro_tomo_bad"].cast<py::list>();
    NoiseModelCpp noise_model_cpp;
    QiliSimConfig config;
    std::vector<bool> qubits_to_measure = {true};
    EXPECT_THROW({ auto result = construct_result_object(state, readout, noise_model_cpp, 1, config, qubits_to_measure); }, py::value_error);
}

TEST(ConstructResultsStabilizer, UnsupportedReadoutThrows) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
        from qilisdk.readout import ReadoutMethod
        _stab_ro_bad = [ReadoutMethod()]
    )");
    StabilizerStateSum state(1);
    py::list readout = py::globals()["_stab_ro_bad"].cast<py::list>();
    NoiseModelCpp noise_model_cpp;
    QiliSimConfig config;
    std::vector<bool> qubits_to_measure = {true};
    EXPECT_THROW({ auto result = construct_result_object(state, readout, noise_model_cpp, 1, config, qubits_to_measure); }, py::value_error);
}

TEST(GateNameParser, GatesControlCounts) {
    EXPECT_EQ(gate_num_controls("H"), 0);
    EXPECT_EQ(gate_num_controls("X"), 0);
    EXPECT_EQ(gate_num_controls("Y"), 0);
    EXPECT_EQ(gate_num_controls("Z"), 0);
    EXPECT_EQ(gate_num_controls("CNOT"), 1);
    EXPECT_EQ(gate_num_controls("CX"), 1);
    EXPECT_EQ(gate_num_controls("CY"), 1);
    EXPECT_EQ(gate_num_controls("CZ"), 1);
    EXPECT_EQ(gate_num_controls("CCX"), 2);
    EXPECT_EQ(gate_num_controls("CCY"), 2);
    EXPECT_EQ(gate_num_controls("CCZ"), 2);
    EXPECT_EQ(gate_num_controls("Toffoli"), 2);
}

TEST(GateNameParser, GateNamesNormalized) {
    EXPECT_EQ(normalize_gate_name("H"), "H");
    EXPECT_EQ(normalize_gate_name("X"), "X");
    EXPECT_EQ(normalize_gate_name("Y"), "Y");
    EXPECT_EQ(normalize_gate_name("Z"), "Z");
    EXPECT_EQ(normalize_gate_name("CNOT"), "X");
    EXPECT_EQ(normalize_gate_name("CX"), "X");
    EXPECT_EQ(normalize_gate_name("CY"), "Y");
    EXPECT_EQ(normalize_gate_name("CZ"), "Z");
    EXPECT_EQ(normalize_gate_name("CCX"), "X");
    EXPECT_EQ(normalize_gate_name("CCY"), "Y");
    EXPECT_EQ(normalize_gate_name("CCZ"), "Z");
    EXPECT_EQ(normalize_gate_name("Toffoli"), "X");
}

// GCOV_EXCL_BR_STOP