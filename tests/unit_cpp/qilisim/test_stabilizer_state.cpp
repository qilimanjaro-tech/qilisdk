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

#include "../../../src/qilisdk_cpp/backends/qilisim/digital/gate.h"
#include "../../../src/qilisdk_cpp/backends/qilisim/representations/stabilizer_state.h"

#include <cmath>
#include <complex>
#include <map>
#include <sstream>
#include <string>

namespace {

// Minimal sparse matrices to satisfy Gate constructor (not used for stabilizer logic)
SparseMatrix make2x2Identity() {
    SparseMatrix m(2, 2);
    m.insert(0, 0) = std::complex<double>(1, 0);
    m.insert(1, 1) = std::complex<double>(1, 0);
    m.makeCompressed();
    return m;
}

Gate makeGate(const std::string& name, const std::vector<int>& controls, const std::vector<int>& targets) {
    return Gate(name, make2x2Identity(), controls, targets, {});
}

}  // namespace

// ──────────────────────────────────────────────────────────────────────────────
// StabilizerState construction
// ──────────────────────────────────────────────────────────────────────────────

TEST(StabilizerStateConstructor, InitializesNQubits) {
    StabilizerState s(3);
    EXPECT_EQ(s.get_nqubits(), 3);
}

TEST(StabilizerStateConstructor, InitialStateIsAllZero) {
    StabilizerState s(3);
    // |000⟩ has deterministic Z eigenvalue +1 on every qubit
    for (int q = 0; q < 3; ++q) {
        EXPECT_EQ(s.find_x_pivot(q), -1) << "qubit " << q << " should be deterministic in |0⟩";
        EXPECT_FALSE(s.z_eigenvalue(q)) << "qubit " << q << " should have eigenvalue +1";
    }
}

TEST(StabilizerStateConstructor, InitialZBitsAreIdentity) {
    StabilizerState s(2);
    // z_bits[q][q] should be set, all others clear
    for (int q = 0; q < 2; ++q) {
        EXPECT_TRUE(s.get_z_bits()[q][q]);
        for (int r = 0; r < 2; ++r) {
            if (r != q)
                EXPECT_FALSE(s.get_z_bits()[q][r]);
        }
        EXPECT_FALSE(s.get_x_bits()[q].any());
    }
}

TEST(StabilizerStateConstructor, InitialPhasesAreZero) {
    StabilizerState s(4);
    EXPECT_FALSE(s.get_phases().any());
}

TEST(StabilizerStateConstructor, ThrowsWhenTooManyQubits) {
    EXPECT_THROW(StabilizerState(MAX_ROWS_STABILIZER), std::invalid_argument);
}

// ──────────────────────────────────────────────────────────────────────────────
// apply_gate: single-qubit Cliffords
// ──────────────────────────────────────────────────────────────────────────────

TEST(ApplyGate, H_ThenH_IsIdentity) {
    StabilizerState s(1);
    Gate h = makeGate("H", {}, {0});
    s.apply_gate(h);
    s.apply_gate(h);
    // Back to Z stabilizer, no X, phase +1
    EXPECT_EQ(s.find_x_pivot(0), -1);
    EXPECT_FALSE(s.z_eigenvalue(0));
}

TEST(ApplyGate, H_FlipsToXStabilizer) {
    // After H on |0⟩ we get |+⟩ which is stabilized by X, not Z
    StabilizerState s(1);
    Gate h = makeGate("H", {}, {0});
    s.apply_gate(h);
    // Now measurement outcome on Z is random (X pivot exists)
    EXPECT_NE(s.find_x_pivot(0), -1);
}

TEST(ApplyGate, X_FlipsPhaseOfZStabilizer) {
    // X maps Z -> -Z, so eigenvalue becomes -1 on qubit 0
    StabilizerState s(1);
    Gate x = makeGate("X", {}, {0});
    s.apply_gate(x);
    EXPECT_TRUE(s.z_eigenvalue(0));
}

TEST(ApplyGate, XX_IsIdentity) {
    StabilizerState s(1);
    Gate x = makeGate("X", {}, {0});
    s.apply_gate(x);
    s.apply_gate(x);
    EXPECT_FALSE(s.z_eigenvalue(0));
}

TEST(ApplyGate, Z_FlipsPhaseOfXStabilizer) {
    // Z maps X -> -X. Prepare |+⟩ with H first, then Z.
    StabilizerState s(1);
    Gate h = makeGate("H", {}, {0});
    Gate z = makeGate("Z", {}, {0});
    s.apply_gate(h);
    // phases before Z should be 0 for row 0
    EXPECT_FALSE(s.get_phases()[0]);
    s.apply_gate(z);
    // The X stabilizer row should now have phase set
    EXPECT_TRUE(s.get_phases()[0]);
}

TEST(ApplyGate, S_MapsXtoY) {
    // S maps X -> Y (X+Z combined) and Z -> Z
    StabilizerState s(1);
    Gate h = makeGate("H", {}, {0});
    Gate sg = makeGate("S", {}, {0});
    s.apply_gate(h);   // stabilizer is now X
    s.apply_gate(sg);  // should become Y = iXZ
    // Y anticommutes with Z, so find_x_pivot is still non-negative
    EXPECT_NE(s.find_x_pivot(0), -1);
    // After S the x_bit and z_bit for qubit 0 row 0 should both be set
    int p = s.find_x_pivot(0);
    EXPECT_TRUE(s.get_x_bits()[0][p]);
    EXPECT_TRUE(s.get_z_bits()[0][p]);
}

TEST(ApplyGate, S4_IsIdentity) {
    // S^4 = I
    StabilizerState s(1);
    Gate sg = makeGate("S", {}, {0});
    Gate h = makeGate("H", {}, {0});
    s.apply_gate(h);
    for (int i = 0; i < 4; ++i)
        s.apply_gate(sg);
    s.apply_gate(h);
    EXPECT_FALSE(s.z_eigenvalue(0));
    EXPECT_EQ(s.find_x_pivot(0), -1);
}

TEST(ApplyGate, Y_FlipsBothPhases) {
    // Y maps X -> -X and Z -> -Z
    StabilizerState s(1);
    Gate y = makeGate("Y", {}, {0});
    s.apply_gate(y);
    EXPECT_TRUE(s.z_eigenvalue(0));
}

TEST(ApplyGate, SWAP_ExchangesQubits) {
    StabilizerState s(2);
    Gate x = makeGate("X", {}, {0});  // flip qubit 0 to |1⟩
    Gate sw = makeGate("SWAP", {}, {0, 1});
    s.apply_gate(x);
    s.apply_gate(sw);
    // After swap: qubit 1 should be |1⟩, qubit 0 should be |0⟩
    EXPECT_FALSE(s.z_eigenvalue(0));
    EXPECT_TRUE(s.z_eigenvalue(1));
}

TEST(ApplyGate, CNOT_EntanglesQubits) {
    // CNOT(0→1) on |00⟩: qubit 0 stays |0⟩, qubit 1 stays |0⟩ -- no change
    StabilizerState s(2);
    Gate cnot = makeGate("X", {0}, {1});
    s.apply_gate(cnot);
    EXPECT_FALSE(s.z_eigenvalue(0));
    EXPECT_FALSE(s.z_eigenvalue(1));
}

TEST(ApplyGate, CNOT_FlipsTarget_WhenControlIsOne) {
    // X(0) then CNOT(0→1): qubit 0 is |1⟩, after CNOT qubit 1 also flips to |1⟩
    StabilizerState s(2);
    Gate x0 = makeGate("X", {}, {0});
    Gate cnot = makeGate("X", {0}, {1});
    s.apply_gate(x0);
    s.apply_gate(cnot);
    EXPECT_TRUE(s.z_eigenvalue(0));
    EXPECT_TRUE(s.z_eigenvalue(1));
}

TEST(ApplyGate, BellState_TwoRandomQubits) {
    // H(0), CNOT(0→1) creates Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2
    // Both qubits should be random when measured independently
    StabilizerState s(2);
    Gate h = makeGate("H", {}, {0});
    Gate cnot = makeGate("X", {0}, {1});
    s.apply_gate(h);
    s.apply_gate(cnot);
    EXPECT_NE(s.find_x_pivot(0), -1);
}

TEST(ApplyGate, CZ_OnComputationalBasis_NoChange) {
    // CZ on |00⟩ has no effect on eigenvalues
    StabilizerState s(2);
    Gate cz = makeGate("Z", {0}, {1});
    s.apply_gate(cz);
    EXPECT_FALSE(s.z_eigenvalue(0));
    EXPECT_FALSE(s.z_eigenvalue(1));
}

// ──────────────────────────────────────────────────────────────────────────────
// rowsum
// ──────────────────────────────────────────────────────────────────────────────

TEST(Rowsum, SameRow_Gives_PlusOnePhase) {
    // Multiplying a stabilizer by itself: g^2 = I (+1 phase, all bits clear)
    StabilizerState s(2);
    Gate x = makeGate("X", {}, {0});
    s.apply_gate(x);
    // row 0 has Z on qubit 0 with phase 1 (eigenvalue -1)
    s.rowsum(0, 0);
    // phase of row 0 should become +1 after g*g = I
    EXPECT_FALSE(s.get_phases()[0]);
}

TEST(Rowsum, MultiplyIdentityRow_NoChange) {
    // In |00⟩, row 1 is Z on qubit 1 with phase 0.
    // rowsum(0, 1) multiplies row 0 by row 1: Z0 * Z1 -> no X bits, z bits combined
    StabilizerState s(2);
    s.rowsum(0, 1);
    // Row 0 should now have Z on both qubit 0 and qubit 1
    EXPECT_TRUE(s.get_z_bits()[0][0]);
    EXPECT_TRUE(s.get_z_bits()[1][0]);
    EXPECT_FALSE(s.get_phases()[0]);
}

// ──────────────────────────────────────────────────────────────────────────────
// project_z
// ──────────────────────────────────────────────────────────────────────────────

TEST(ProjectZ, ProjectOntoZero_GivesZeroEigenvalue) {
    StabilizerState s(1);
    Gate h = makeGate("H", {}, {0});
    s.apply_gate(h);  // |+⟩, random outcome
    s.project_z(0, false);
    EXPECT_EQ(s.find_x_pivot(0), -1);
    EXPECT_FALSE(s.z_eigenvalue(0));
}

TEST(ProjectZ, ProjectOntoOne_GivesOneEigenvalue) {
    StabilizerState s(1);
    Gate h = makeGate("H", {}, {0});
    s.apply_gate(h);
    s.project_z(0, true);
    EXPECT_EQ(s.find_x_pivot(0), -1);
    EXPECT_TRUE(s.z_eigenvalue(0));
}

TEST(ProjectZ, ProjectBellStateToZero_CollapsesBothQubits) {
    // Bell state: project qubit 0 to |0⟩, qubit 1 must also be |0⟩
    StabilizerState s(2);
    Gate h = makeGate("H", {}, {0});
    Gate cnot = makeGate("X", {0}, {1});
    s.apply_gate(h);
    s.apply_gate(cnot);
    s.project_z(0, false);
    EXPECT_FALSE(s.z_eigenvalue(0));
    EXPECT_FALSE(s.z_eigenvalue(1));
}

TEST(ProjectZ, ProjectBellStateToOne_CollapsesBothQubits) {
    StabilizerState s(2);
    Gate h = makeGate("H", {}, {0});
    Gate cnot = makeGate("X", {0}, {1});
    s.apply_gate(h);
    s.apply_gate(cnot);
    s.project_z(0, true);
    EXPECT_TRUE(s.z_eigenvalue(0));
    EXPECT_TRUE(s.z_eigenvalue(1));
}

// ──────────────────────────────────────────────────────────────────────────────
// find_x_pivot / z_eigenvalue
// ──────────────────────────────────────────────────────────────────────────────

TEST(FindXPivot, ReturnsMinusOne_ForComputationalBasis) {
    StabilizerState s(3);
    for (int q = 0; q < 3; ++q)
        EXPECT_EQ(s.find_x_pivot(q), -1);
}

TEST(FindXPivot, ReturnsValidRow_AfterH) {
    StabilizerState s(2);
    Gate h = makeGate("H", {}, {1});
    s.apply_gate(h);
    EXPECT_EQ(s.find_x_pivot(0), -1);
    EXPECT_GE(s.find_x_pivot(1), 0);
}

TEST(ZEigenvalue, ReturnsFalse_ForInitialState) {
    StabilizerState s(4);
    for (int q = 0; q < 4; ++q)
        EXPECT_FALSE(s.z_eigenvalue(q));
}

TEST(ZEigenvalue, ReturnsTrue_AfterXGate) {
    StabilizerState s(2);
    Gate x = makeGate("X", {}, {1});
    s.apply_gate(x);
    EXPECT_FALSE(s.z_eigenvalue(0));
    EXPECT_TRUE(s.z_eigenvalue(1));
}

// ──────────────────────────────────────────────────────────────────────────────
// sample
// ──────────────────────────────────────────────────────────────────────────────

TEST(Sample, InitialStateSamplesAllZeros) {
    StabilizerState s(3);
    std::string result = s.sample();
    EXPECT_EQ(result, "000");
}

TEST(Sample, AfterXGate_SamplesOne) {
    StabilizerState s(2);
    Gate x = makeGate("X", {}, {0});
    s.apply_gate(x);
    std::string result = s.sample();
    EXPECT_EQ(result[0], '1');
    EXPECT_EQ(result[1], '0');
}

TEST(Sample, RandomState_ReturnsBinaryString) {
    StabilizerState s(2);
    Gate h = makeGate("H", {}, {0});
    s.apply_gate(h);
    std::string result = s.sample();
    ASSERT_EQ(result.size(), 2u);
    for (char c : result)
        EXPECT_TRUE(c == '0' || c == '1');
}

TEST(Sample, BellState_SamplesAreCorrelated) {
    srand(42);
    int both_zero = 0, both_one = 0, mixed = 0;
    for (int trial = 0; trial < 200; ++trial) {
        StabilizerState s(2);
        Gate h = makeGate("H", {}, {0});
        Gate cnot = makeGate("X", {0}, {1});
        s.apply_gate(h);
        s.apply_gate(cnot);
        std::string r = s.sample();
        if (r == "00")
            both_zero++;
        else if (r == "11")
            both_one++;
        else
            mixed++;
    }
    EXPECT_EQ(mixed, 0) << "Bell state should never yield mixed 01 or 10 outcomes";
    EXPECT_GT(both_zero, 0);
    EXPECT_GT(both_one, 0);
}

TEST(Sample, Length_MatchesNQubits) {
    for (int n = 1; n <= 5; ++n) {
        StabilizerState s(n);
        EXPECT_EQ(s.sample().size(), static_cast<size_t>(n));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// StabilizerStateSum
// ──────────────────────────────────────────────────────────────────────────────

TEST(StabilizerStateSum, Constructor_SingleTerm) {
    StabilizerStateSum sss(2);
    EXPECT_EQ(sss.get_states().size(), 1u);
    EXPECT_EQ(sss.get_coefficients().size(), 1u);
    EXPECT_EQ(sss.get_nqubits(), 2);
}

TEST(StabilizerStateSum, Sample_AllZeros_ForInitialState) {
    srand(0);
    StabilizerStateSum sss(3);
    auto counts = sss.sample(100);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.begin()->first, "000");
    EXPECT_EQ(counts.begin()->second, 100);
}

TEST(StabilizerStateSum, Sample_AfterX_ReturnsOne) {
    StabilizerStateSum sss(2);
    Gate x = makeGate("X", {}, {0});
    sss.apply_gate(x);
    auto counts = sss.sample(50);
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts.begin()->first[0], '1');
    EXPECT_EQ(counts.begin()->first[1], '0');
}

TEST(StabilizerStateSum, ApplyGate_Clifford_DoesNotExpandSum) {
    StabilizerStateSum sss(2);
    Gate h = makeGate("H", {}, {0});
    Gate cnot = makeGate("X", {0}, {1});
    sss.apply_gate(h);
    sss.apply_gate(cnot);
    // Clifford gates do not expand the sum
    EXPECT_EQ(sss.get_states().size(), 1u);
}

TEST(StabilizerStateSum, ApplyGate_T_ExpandsSum) {
    // T on |+⟩ (after H) should branch since qubit is in superposition
    StabilizerStateSum sss(1);
    Gate h = makeGate("H", {}, {0});
    Gate t = makeGate("T", {}, {0});
    sss.apply_gate(h);
    sss.apply_gate(t);
    EXPECT_EQ(sss.get_states().size(), 2u);
    EXPECT_EQ(sss.get_coefficients().size(), 2u);
}

TEST(StabilizerStateSum, ApplyGate_T_OnComputational_DoesNotExpand) {
    // T on |0⟩ is deterministic (eigenvalue 0), no branching
    StabilizerStateSum sss(1);
    Gate t = makeGate("T", {}, {0});
    sss.apply_gate(t);
    EXPECT_EQ(sss.get_states().size(), 1u);
}

TEST(StabilizerStateSum, ApplyGate_T_OnOne_DoesNotExpand) {
    // T on |1⟩ is deterministic (eigenvalue 1), no branching, coefficient gets phase
    StabilizerStateSum sss(1);
    Gate x = makeGate("X", {}, {0});
    Gate t = makeGate("T", {}, {0});
    sss.apply_gate(x);
    sss.apply_gate(t);
    EXPECT_EQ(sss.get_states().size(), 1u);
    // Coefficient should have been multiplied by e^{iπ/4}
    auto c = sss.get_coefficients()[0];
    EXPECT_NEAR(std::abs(c), 1.0, 1e-9);
    EXPECT_NEAR(std::arg(c), M_PI / 4.0, 1e-9);
}

TEST(StabilizerStateSum, ApplyGate_Toffoli_DeterministicControl0_NoExpand) {
    // CCX with control[0] = |0⟩: deterministic, no branching
    StabilizerStateSum sss(3);
    Gate ccx = makeGate("X", {0, 1}, {2});
    sss.apply_gate(ccx);
    EXPECT_EQ(sss.get_states().size(), 1u);
    // With control 0 = |0⟩, nothing should flip
    EXPECT_FALSE(sss.get_states()[0].z_eigenvalue(2));
}

TEST(StabilizerStateSum, ApplyGate_Toffoli_BothControlsOne_FlipsTarget) {
    // X(0), X(1), then CCX(0,1->2): target should flip
    StabilizerStateSum sss(3);
    Gate x0 = makeGate("X", {}, {0});
    Gate x1 = makeGate("X", {}, {1});
    Gate ccx = makeGate("X", {0, 1}, {2});
    sss.apply_gate(x0);
    sss.apply_gate(x1);
    sss.apply_gate(ccx);
    EXPECT_EQ(sss.get_states().size(), 1u);
    EXPECT_TRUE(sss.get_states()[0].z_eigenvalue(2));
}

TEST(StabilizerStateSum, ApplyGate_Toffoli_RandomControl_ExpandsSum) {
    // H on qubit 0, then CCX: control 0 is in superposition -> branching
    StabilizerStateSum sss(3);
    Gate h = makeGate("H", {}, {0});
    Gate x1 = makeGate("X", {}, {1});
    Gate ccx = makeGate("X", {0, 1}, {2});
    sss.apply_gate(h);
    sss.apply_gate(x1);
    sss.apply_gate(ccx);
    EXPECT_EQ(sss.get_states().size(), 2u);
}

TEST(StabilizerStateSum, ApplyGate_UnsupportedGate_Throws) {
    StabilizerStateSum sss(1);
    Gate bad = makeGate("RZ", {}, {0});
    EXPECT_THROW(sss.apply_gate(bad), std::runtime_error);
}

TEST(StabilizerStateSum, Stream_ContainsTermCount) {
    StabilizerStateSum sss(2);
    std::ostringstream oss;
    oss << sss;
    EXPECT_NE(oss.str().find("1"), std::string::npos);
}

TEST(StabilizerStateSum, Sample_TotalCountMatchesNshots) {
    srand(7);
    StabilizerStateSum sss(2);
    Gate h = makeGate("H", {}, {0});
    sss.apply_gate(h);
    const int nshots = 100;
    auto counts = sss.sample(nshots);
    int total = 0;
    for (auto& kv : counts)
        total += kv.second;
    EXPECT_EQ(total, nshots);
}

// GCOV_EXCL_BR_STOP
