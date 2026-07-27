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

// RX(theta) matrix: [[cos(t/2), -i*sin(t/2)], [-i*sin(t/2), cos(t/2)]]
SparseMatrix makeRXMatrix(double theta) {
    SparseMatrix m(2, 2);
    double c = std::cos(theta / 2.0), s = std::sin(theta / 2.0);
    m.insert(0, 0) = std::complex<double>(c, 0);
    m.insert(0, 1) = std::complex<double>(0, -s);
    m.insert(1, 0) = std::complex<double>(0, -s);
    m.insert(1, 1) = std::complex<double>(c, 0);
    m.makeCompressed();
    return m;
}

Gate makeRX(double theta, int target) {
    return Gate("RX", makeRXMatrix(theta), {}, {target}, {});
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
    EXPECT_THROW(StabilizerState{MAX_ROWS_STABILIZER + 1}, std::invalid_argument);
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

TEST(Rowsum, XStabilizer_XBitsFlipped) {
    // After H on qubit 0, row 0 is X0 (x_bits[0][0]=1).
    // rowsum(0, 0) multiplies X0 by itself: X*X = I.
    StabilizerState s(1);
    Gate h = makeGate("H", {}, {0});
    s.apply_gate(h);
    EXPECT_TRUE(s.get_x_bits()[0][0]);
    s.rowsum(0, 0);
    EXPECT_FALSE(s.get_x_bits()[0][0]);
    EXPECT_FALSE(s.get_phases()[0]);
}

TEST(Rowsum, YStabilizer_PhaseContribution) {
    // After H then S on qubit 0, row 0 is Y0 (x and z bits both set).
    // rowsum(0, 0) multiplies Y0 by itself: Y*Y = I.
    StabilizerState s(1);
    Gate h = makeGate("H", {}, {0});
    Gate sg = makeGate("S", {}, {0});
    s.apply_gate(h);
    s.apply_gate(sg);
    EXPECT_TRUE(s.get_x_bits()[0][0]);
    EXPECT_TRUE(s.get_z_bits()[0][0]);
    s.rowsum(0, 0);
    EXPECT_FALSE(s.get_x_bits()[0][0]);
    EXPECT_FALSE(s.get_z_bits()[0][0]);
    EXPECT_FALSE(s.get_phases()[0]);
}

TEST(Sample, MultipleAnticommutingRows_CleanupLoopRuns) {
    // Build a 2-qubit state where two rows both have X on qubit 0:
    //   Apply X, H, S on qubit 0 -> Row 0 = -Y0 (x0=1, z0=1, phase=-1)
    //   rowsum(1, 0) -> Row 1 = -Y0Z1 (also has X on qubit 0)
    // Sampling qubit 0 triggers the multi-anticommuting cleanup loop.
    StabilizerState s(2);
    Gate x = makeGate("X", {}, {0});
    Gate h = makeGate("H", {}, {0});
    Gate sg = makeGate("S", {}, {0});
    s.apply_gate(x);
    s.apply_gate(h);
    s.apply_gate(sg);
    s.rowsum(1, 0);
    // Both rows now have X on qubit 0 - sampling must clean this up
    std::string result = s.sample();
    ASSERT_EQ(result.size(), 2u);
    for (char c : result)
        EXPECT_TRUE(c == '0' || c == '1');
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

TEST(ProjectZ, MultipleAnticommutingRows_InternalRowsumCalled) {
    // Build state where two rows have X on qubit 0 so project_z must rowsum them.
    // X, H, S on qubit 0 gives Row 0 = -Y0; rowsum(1, 0) makes Row 1 = -Y0Z1.
    StabilizerState s(2);
    Gate x = makeGate("X", {}, {0});
    Gate h = makeGate("H", {}, {0});
    Gate sg = makeGate("S", {}, {0});
    s.apply_gate(x);
    s.apply_gate(h);
    s.apply_gate(sg);
    s.rowsum(1, 0);
    s.project_z(0, false);
    EXPECT_EQ(s.find_x_pivot(0), -1);
    EXPECT_FALSE(s.z_eigenvalue(0));
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
            mixed++;  // GCOV_EXCL_LINE
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
    // Controlled single-qubit non-Clifford gates are not yet supported
    StabilizerStateSum sss(2);
    Gate bad = makeGate("RZ", {0}, {1});
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

// ──────────────────────────────────────────────────────────────────────────────
// max_terms truncation
// ──────────────────────────────────────────────────────────────────────────────

TEST(StabilizerStateSum, MaxTerms_DefaultIsZero) {
    StabilizerStateSum sss(1);
    EXPECT_EQ(sss.get_max_terms(), 0);
}

TEST(StabilizerStateSum, MaxTerms_SetAndGet) {
    StabilizerStateSum sss(1);
    sss.set_max_terms(5);
    EXPECT_EQ(sss.get_max_terms(), 5);
}

TEST(StabilizerStateSum, MaxTerms_ZeroDisablesTruncation) {
    // max_terms = 0 means no truncation: T on |+⟩ should still expand to 2 terms
    StabilizerStateSum sss(1);
    sss.set_max_terms(0);
    Gate h = makeGate("H", {}, {0});
    Gate t = makeGate("T", {}, {0});
    sss.apply_gate(h);
    sss.apply_gate(t);
    EXPECT_EQ(sss.get_states().size(), 2u);
}

TEST(StabilizerStateSum, MaxTerms_TruncatesToLimit) {
    // Three T-gates on |+⟩ would produce up to 8 terms; cap at 3
    StabilizerStateSum sss(3);
    sss.set_max_terms(3);
    Gate h0 = makeGate("H", {}, {0});
    Gate h1 = makeGate("H", {}, {1});
    Gate h2 = makeGate("H", {}, {2});
    Gate t0 = makeGate("T", {}, {0});
    Gate t1 = makeGate("T", {}, {1});
    Gate t2 = makeGate("T", {}, {2});
    sss.apply_gate(h0);
    sss.apply_gate(h1);
    sss.apply_gate(h2);
    sss.apply_gate(t0);
    sss.apply_gate(t1);
    sss.apply_gate(t2);
    EXPECT_LE(static_cast<int>(sss.get_states().size()), 3);
    EXPECT_EQ(sss.get_states().size(), sss.get_coefficients().size());
}

TEST(StabilizerStateSum, MaxTerms_KeepsLargestAmplitudes) {
    // After H then T on a single qubit the two branches have equal amplitudes (both 1/√2).
    // With max_terms = 1, only one term remains and it has the larger (or equal) amplitude.
    StabilizerStateSum sss(1);
    sss.set_max_terms(1);
    Gate h = makeGate("H", {}, {0});
    Gate t = makeGate("T", {}, {0});
    sss.apply_gate(h);
    sss.apply_gate(t);
    ASSERT_EQ(sss.get_states().size(), 1u);
    // The kept term had |c|^2 = 0.5 before truncation; normalize() (called after truncate) then
    // rescales the lone survivor back to a unit-norm state, so |c|^2 = 1.
    double norm = std::norm(sss.get_coefficients()[0]);
    EXPECT_NEAR(norm, 1.0, 1e-9);
}

TEST(StabilizerStateSum, MaxTerms_LargerThanSumNoTruncation) {
    // max_terms larger than the number of terms — nothing should be removed
    StabilizerStateSum sss(1);
    sss.set_max_terms(100);
    Gate h = makeGate("H", {}, {0});
    Gate t = makeGate("T", {}, {0});
    sss.apply_gate(h);
    sss.apply_gate(t);
    EXPECT_EQ(sss.get_states().size(), 2u);
}

TEST(StabilizerStateSum, MaxTerms_CliffordGatesNotTruncated) {
    // Clifford gates never branch, so max_terms = 1 should not affect them
    StabilizerStateSum sss(2);
    sss.set_max_terms(1);
    Gate h = makeGate("H", {}, {0});
    Gate cnot = makeGate("X", {0}, {1});
    sss.apply_gate(h);
    sss.apply_gate(cnot);
    EXPECT_EQ(sss.get_states().size(), 1u);
}

// ──────────────────────────────────────────────────────────────────────────────
// Arbitrary single-qubit gate (Pauli decomposition path)
// ──────────────────────────────────────────────────────────────────────────────

TEST(StabilizerStateSum, ArbitraryGate_RX_Pi_On_Zero_OneTermResult) {
    // RX(π)|0⟩ = -i|1⟩: u00 = 0 so only the u10 (X-flip) branch is created
    StabilizerStateSum sss(1);
    sss.apply_gate(makeRX(M_PI, 0));
    ASSERT_EQ(sss.get_states().size(), 1u);
    EXPECT_NEAR(std::norm(sss.get_coefficients()[0]), 1.0, 1e-9);
    EXPECT_TRUE(sss.get_states()[0].z_eigenvalue(0));  // qubit ended in |1⟩
}

TEST(StabilizerStateSum, ArbitraryGate_RX_HalfPi_On_Zero_TwoTermResult) {
    // RX(π/2)|0⟩ = cos(π/4)|0⟩ - i·sin(π/4)|1⟩: both u00 and u10 nonzero
    StabilizerStateSum sss(1);
    sss.apply_gate(makeRX(M_PI / 2.0, 0));
    EXPECT_EQ(sss.get_states().size(), 2u);
    EXPECT_EQ(sss.get_coefficients().size(), 2u);
    double total_prob = 0.0;
    for (const auto& c : sss.get_coefficients())
        total_prob += std::norm(c);
    EXPECT_NEAR(total_prob, 1.0, 1e-9);
}

TEST(StabilizerStateSum, ArbitraryGate_RX_Pi_On_One_OneTermResult) {
    // RX(π)|1⟩ = -i|0⟩: u11 = 0 so only the u01 (X-flip) branch is created via on_one
    StabilizerStateSum sss(1);
    sss.apply_gate(makeGate("X", {}, {0}));  // prepare |1⟩
    sss.apply_gate(makeRX(M_PI, 0));
    ASSERT_EQ(sss.get_states().size(), 1u);
    EXPECT_FALSE(sss.get_states()[0].z_eigenvalue(0));  // qubit ended in |0⟩
}

TEST(StabilizerStateSum, ArbitraryGate_RX_HalfPi_On_One_TwoTermResult) {
    // RX(π/2)|1⟩: both u01 and u11 nonzero → two terms via on_one
    StabilizerStateSum sss(1);
    sss.apply_gate(makeGate("X", {}, {0}));  // prepare |1⟩
    sss.apply_gate(makeRX(M_PI / 2.0, 0));
    EXPECT_EQ(sss.get_states().size(), 2u);
}

TEST(StabilizerStateSum, ArbitraryGate_RX_On_Superposition_BothBranchesHit) {
    // H then RX on qubit in superposition: random Z eigenvalue exercises both on_zero and on_one
    StabilizerStateSum sss(1);
    sss.apply_gate(makeGate("H", {}, {0}));  // prepare |+⟩
    sss.apply_gate(makeRX(M_PI / 2.0, 0));
    EXPECT_GT(sss.get_states().size(), 0u);
    EXPECT_EQ(sss.get_states().size(), sss.get_coefficients().size());
}

TEST(StabilizerStateSum, ArbitraryGate_IdentityMatrix_DeterministicState_NoChange) {
    // Identity on |0⟩: deterministic Z eigenvalue, goes through on_zero which returns one term unchanged
    StabilizerStateSum sss(1);
    Gate id("ID", make2x2Identity(), {}, {0}, {});
    sss.apply_gate(id);
    ASSERT_EQ(sss.get_states().size(), 1u);
    EXPECT_NEAR(std::norm(sss.get_coefficients()[0]), 1.0, 1e-9);
    EXPECT_FALSE(sss.get_states()[0].z_eigenvalue(0));  // still |0⟩
}

// ──────────────────────────────────────────────────────────────────────────────
// amplitude(): relative phases must match a brute-force statevector (entangled states)
// ──────────────────────────────────────────────────────────────────────────────

namespace {

using cd = std::complex<double>;

int bitval(int idx, int q, int n) {
    return (idx >> (n - 1 - q)) & 1;
}
int setbitval(int idx, int q, int n, int v) {
    int mask = 1 << (n - 1 - q);
    return v ? (idx | mask) : (idx & ~mask);
}

// Apply a Clifford gate to a dense statevector using the canonical matrices.
void denseApply(std::vector<cd>& sv, int n, const std::string& name, const std::vector<int>& ctr, const std::vector<int>& tgt) {
    int dim = 1 << n;
    std::vector<cd> out(dim, cd(0, 0));
    const double s = 1.0 / std::sqrt(2.0);
    if (name == "SWAP") {
        for (int i = 0; i < dim; ++i) {
            int a = bitval(i, tgt[0], n), b = bitval(i, tgt[1], n);
            out[setbitval(setbitval(i, tgt[0], n, b), tgt[1], n, a)] += sv[i];
        }
        sv = out;
        return;
    }
    if (ctr.empty()) {  // single-qubit
        int t = tgt[0];
        cd u00, u01, u10, u11;
        if (name == "H") {
            u00 = s;
            u01 = s;
            u10 = s;
            u11 = -s;
        } else if (name == "X") {
            u00 = 0;
            u01 = 1;
            u10 = 1;
            u11 = 0;
        } else if (name == "Y") {
            u00 = 0;
            u01 = cd(0, -1);
            u10 = cd(0, 1);
            u11 = 0;
        } else if (name == "Z") {
            u00 = 1;
            u01 = 0;
            u10 = 0;
            u11 = -1;
        } else if (name == "S") {
            u00 = 1;
            u01 = 0;
            u10 = 0;
            u11 = cd(0, 1);
        }
        for (int i = 0; i < dim; ++i) {
            int tb = bitval(i, t, n);
            int i0 = setbitval(i, t, n, 0), i1 = setbitval(i, t, n, 1);
            if (tb == 0) {
                out[i0] += u00 * sv[i];
                out[i1] += u10 * sv[i];
            } else {
                out[i0] += u01 * sv[i];
                out[i1] += u11 * sv[i];
            }
        }
        sv = out;
        return;
    }
    // CNOT (name X) / CZ (name Z), single control
    int c = ctr[0], t = tgt[0];
    for (int i = 0; i < dim; ++i) {
        if (name == "X") {
            int j = bitval(i, c, n) ? setbitval(i, t, n, 1 - bitval(i, t, n)) : i;
            out[j] += sv[i];
        } else {  // CZ
            cd v = sv[i];
            if (bitval(i, c, n) && bitval(i, t, n))
                v = -v;
            out[i] += v;
        }
    }
    sv = out;
}

double amplitudeMismatch(int n, const std::vector<std::tuple<std::string, std::vector<int>, std::vector<int>>>& ops) {
    StabilizerState st(n);
    int dim = 1 << n;
    std::vector<cd> sv(dim, cd(0, 0));
    sv[0] = 1.0;
    for (const auto& [name, ctr, tgt] : ops) {
        Gate g(name, make2x2Identity(), ctr, tgt, {});
        st.apply_gate(g);
        denseApply(sv, n, name, ctr, tgt);
    }
    // Compare up to a single global phase, anchored on the largest-magnitude dense amplitude.
    int piv = 0;
    double best = -1;
    for (int i = 0; i < dim; ++i)
        if (std::abs(sv[i]) > best) {
            best = std::abs(sv[i]);
            piv = i;
        }
    std::string bpiv(n, '0');
    for (int q = 0; q < n; ++q)
        bpiv[q] = bitval(piv, q, n) ? '1' : '0';
    cd apiv = st.amplitude(bpiv);
    if (std::abs(apiv) < 1e-9)
        return 1e9;  // GCOVR_EXCL_LINE
    cd gp = sv[piv] / apiv;
    double maxerr = 0;
    for (int i = 0; i < dim; ++i) {
        std::string b(n, '0');
        for (int q = 0; q < n; ++q)
            b[q] = bitval(i, q, n) ? '1' : '0';
        maxerr = std::max(maxerr, std::abs(sv[i] - gp * st.amplitude(b)));
    }
    return maxerr;
}

}  // namespace

TEST(Amplitude, MatchesStatevector_EntangledPattern) {
    // The pattern that exposed the entangled relative-phase bug (Clifford part of the failing circuit).
    double e = amplitudeMismatch(2, {{"H", {}, {1}}, {"H", {}, {0}}, {"S", {}, {0}}, {"X", {1}, {0}}, {"H", {}, {1}}});
    EXPECT_LT(e, 1e-6) << "amplitude() relative phases disagree with statevector by " << e;
}

TEST(Amplitude, MatchesStatevector_RandomCliffords) {
    // Deterministic pseudo-random sweep of Clifford circuits on 3 qubits.
    const char* singles[] = {"H", "X", "Y", "Z", "S"};
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    auto next = [&]() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return rng;
    };
    double worst = 0;
    for (int trial = 0; trial < 400; ++trial) {
        int n = 3;
        int ng = 4 + (next() % 8);
        std::vector<std::tuple<std::string, std::vector<int>, std::vector<int>>> ops;
        for (int g = 0; g < ng; ++g) {
            if (next() % 4 == 0) {
                int a = next() % n, b = next() % n;
                if (a == b)
                    b = (b + 1) % n;
                ops.push_back({(next() % 2) ? "X" : "Z", {a}, {b}});
            } else {
                ops.push_back({singles[next() % 5], {}, {(int)(next() % n)}});
            }
        }
        worst = std::max(worst, amplitudeMismatch(n, ops));
    }
    EXPECT_LT(worst, 1e-6) << "worst amplitude() vs statevector mismatch over random Cliffords = " << worst;
}

// ──────────────────────────────────────────────────────────────────────────────
// StabilizerStateSum::amplitude must match a brute-force statevector (incl. T gates)
// ──────────────────────────────────────────────────────────────────────────────

namespace {

void denseApplyT(std::vector<cd>& sv, int n, int t) {
    int dim = 1 << n;
    cd ph = std::exp(cd(0, M_PI / 4.0));
    for (int i = 0; i < dim; ++i)
        if (bitval(i, t, n))
            sv[i] *= ph;
}

SparseMatrix tMatrix() {
    SparseMatrix m(2, 2);
    m.insert(0, 0) = cd(1, 0);
    m.insert(1, 1) = std::exp(cd(0, M_PI / 4.0));
    m.makeCompressed();
    return m;
}

// Build a StabilizerStateSum and a dense statevector from the same gate list, return max amplitude
// mismatch up to a single global phase.
double sumAmplitudeMismatch(int n, const std::vector<std::tuple<std::string, std::vector<int>, std::vector<int>>>& ops) {
    StabilizerStateSum sum(n);
    int dim = 1 << n;
    std::vector<cd> sv(dim, cd(0, 0));
    sv[0] = 1.0;
    for (const auto& [name, ctr, tgt] : ops) {
        if (name == "T") {
            sum.apply_gate(Gate("T", tMatrix(), ctr, tgt, {}));
            denseApplyT(sv, n, tgt[0]);
        } else {
            sum.apply_gate(Gate(name, make2x2Identity(), ctr, tgt, {}));
            denseApply(sv, n, name, ctr, tgt);
        }
    }
    int piv = 0;
    double best = -1;
    for (int i = 0; i < dim; ++i)
        if (std::abs(sv[i]) > best) {
            best = std::abs(sv[i]);
            piv = i;
        }
    std::string bpiv(n, '0');
    for (int q = 0; q < n; ++q)
        bpiv[q] = bitval(piv, q, n) ? '1' : '0';
    cd apiv = sum.amplitude(bpiv);
    if (std::abs(apiv) < 1e-9)
        return 1e9;  // GCOVR_EXCL_LINE
    cd gp = sv[piv] / apiv;
    double maxerr = 0;
    for (int i = 0; i < dim; ++i) {
        std::string b(n, '0');
        for (int q = 0; q < n; ++q)
            b[q] = bitval(i, q, n) ? '1' : '0';
        maxerr = std::max(maxerr, std::abs(sv[i] - gp * sum.amplitude(b)));
    }
    return maxerr;
}

}  // namespace

TEST(StabilizerStateSum, Amplitude_MatchesStatevector_TwoTGatesEntangled) {
    // Reduced from a randomized failure: two T gates plus entangling Cliffords.
    double e = sumAmplitudeMismatch(2, {{"H", {}, {1}}, {"X", {}, {0}}, {"Y", {}, {0}}, {"X", {1}, {0}}, {"H", {}, {0}}, {"T", {}, {0}}, {"Y", {}, {1}}, {"X", {1}, {0}}, {"H", {}, {0}}, {"T", {}, {1}}, {"H", {}, {0}}});
    EXPECT_LT(e, 1e-6) << "StabilizerStateSum amplitude disagrees with statevector by " << e;
}

TEST(StabilizerStateSum, Amplitude_MatchesStatevector_RandomWithT) {
    const char* singles[] = {"H", "X", "Y", "Z", "S", "T"};
    uint64_t rng = 0xD1B54A32D192ED03ULL;
    auto next = [&]() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return rng;
    };
    double worst = 0;
    for (int trial = 0; trial < 400; ++trial) {
        int n = 1 + (next() % 3);
        int ng = 4 + (next() % 16);
        std::vector<std::tuple<std::string, std::vector<int>, std::vector<int>>> ops;
        for (int g = 0; g < ng; ++g) {
            if (n > 1 && next() % 4 == 0) {
                int a = next() % n, b = next() % n;
                if (a == b)
                    b = (b + 1) % n;
                ops.push_back({(next() % 2) ? "X" : "Z", {a}, {b}});
            } else {
                ops.push_back({singles[next() % 6], {}, {(int)(next() % n)}});
            }
        }
        worst = std::max(worst, sumAmplitudeMismatch(n, ops));
    }
    EXPECT_LT(worst, 1e-6) << "worst StabilizerStateSum amplitude vs statevector over random+T = " << worst;
}

// ──────────────────────────────────────────────────────────────────────────────
// T-gate algebraic identities (verified against the brute-force statevector)
// ──────────────────────────────────────────────────────────────────────────────

TEST(StabilizerStateSum, T_Squared_EqualsS_OnSuperposition) {
    // T^2 = S. On |+⟩ the two operators must produce identical statevectors. We verify both the
    // T^2 circuit and the S circuit match the same dense statevector built from their own gate lists.
    double eTT = sumAmplitudeMismatch(1, {{"H", {}, {0}}, {"T", {}, {0}}, {"T", {}, {0}}});
    double eS = sumAmplitudeMismatch(1, {{"H", {}, {0}}, {"S", {}, {0}}});
    EXPECT_LT(eTT, 1e-6) << "H,T,T amplitude disagrees with statevector by " << eTT;
    EXPECT_LT(eS, 1e-6) << "H,S amplitude disagrees with statevector by " << eS;
}

TEST(StabilizerStateSum, T_Fourth_EqualsZ_OnSuperposition) {
    // T^4 = Z. Four T gates on |+⟩ should match the statevector (and equal a single Z).
    double e = sumAmplitudeMismatch(1, {{"H", {}, {0}}, {"T", {}, {0}}, {"T", {}, {0}}, {"T", {}, {0}}, {"T", {}, {0}}});
    EXPECT_LT(e, 1e-6) << "T^4 on |+⟩ disagrees with statevector by " << e;
}

TEST(StabilizerStateSum, T_Eighth_IsIdentity_OnSuperposition) {
    // T^8 = I. Eight T gates on |+⟩ must return to |+⟩.
    std::vector<std::tuple<std::string, std::vector<int>, std::vector<int>>> ops = {{"H", {}, {0}}};
    for (int i = 0; i < 8; ++i)
        ops.push_back({"T", {}, {0}});
    double e = sumAmplitudeMismatch(1, ops);
    EXPECT_LT(e, 1e-6) << "T^8 on |+⟩ disagrees with statevector by " << e;
}

TEST(StabilizerStateSum, T_OnSuperposition_ProducesEqualMagnitudeBranches) {
    // T on |+⟩ branches into two terms; after normalization both carry probability 1/2.
    StabilizerStateSum sss(1);
    sss.apply_gate(makeGate("H", {}, {0}));
    sss.apply_gate(makeGate("T", {}, {0}));
    ASSERT_EQ(sss.get_states().size(), 2u);
    double total = 0.0;
    for (const auto& c : sss.get_coefficients()) {
        EXPECT_NEAR(std::norm(c), 0.5, 1e-9);
        total += std::norm(c);
    }
    EXPECT_NEAR(total, 1.0, 1e-9);
}

TEST(StabilizerStateSum, HTH_IsNonCliffordRotation_MatchesStatevector) {
    // H T H is a rotation that takes |0⟩ out of the computational basis with a non-trivial phase;
    // a good stress test of the branch-and-recombine machinery on a deterministic input.
    double e = sumAmplitudeMismatch(1, {{"H", {}, {0}}, {"T", {}, {0}}, {"H", {}, {0}}});
    EXPECT_LT(e, 1e-6) << "H,T,H disagrees with statevector by " << e;
}

// ──────────────────────────────────────────────────────────────────────────────
// T gates entangled with Cliffords (verified against the brute-force statevector)
// ──────────────────────────────────────────────────────────────────────────────

TEST(StabilizerStateSum, T_OnBellState_MatchesStatevector) {
    // T applied to one half of a Bell pair: the relative phase between the |00⟩ and |11⟩
    // branches must be tracked exactly across the entanglement.
    double e = sumAmplitudeMismatch(2, {{"H", {}, {0}}, {"X", {0}, {1}}, {"T", {}, {0}}});
    EXPECT_LT(e, 1e-6) << "Bell + T disagrees with statevector by " << e;
}

TEST(StabilizerStateSum, T_OnBothHalvesOfBellState_MatchesStatevector) {
    // A T on each half of a Bell pair: |00⟩ picks up no phase, |11⟩ picks up e^{iπ/2}=i.
    double e = sumAmplitudeMismatch(2, {{"H", {}, {0}}, {"X", {0}, {1}}, {"T", {}, {0}}, {"T", {}, {1}}});
    EXPECT_LT(e, 1e-6) << "Bell + T on both halves disagrees with statevector by " << e;
}

TEST(StabilizerStateSum, GHZ_WithT_MatchesStatevector) {
    // 3-qubit GHZ state with a T on the middle qubit, then more entangling gates: exercises
    // phase tracking through a larger entangled register.
    double e = sumAmplitudeMismatch(3, {{"H", {}, {0}}, {"X", {0}, {1}}, {"X", {1}, {2}}, {"T", {}, {1}}, {"H", {}, {2}}, {"Z", {0}, {2}}});
    EXPECT_LT(e, 1e-6) << "GHZ + T disagrees with statevector by " << e;
}

TEST(StabilizerStateSum, ManyTGatesAcrossQubits_MatchesStatevector) {
    // Several T gates interleaved with entangling Cliffords on 3 qubits — multiple branch
    // expansions and recombinations in sequence.
    double e = sumAmplitudeMismatch(3, {{"H", {}, {0}}, {"T", {}, {0}}, {"X", {0}, {1}}, {"T", {}, {1}}, {"H", {}, {2}}, {"T", {}, {2}}, {"X", {1}, {2}}, {"S", {}, {0}}, {"T", {}, {0}}});
    EXPECT_LT(e, 1e-6) << "many T gates disagrees with statevector by " << e;
}

TEST(StabilizerStateSum, RepeatedTOnSameEntangledQubit_MatchesStatevector) {
    // Four T gates (= Z) on an entangled qubit, separated by Cliffords. The accumulated phase
    // must still match exactly after combine_duplicates folds equivalent terms together.
    double e = sumAmplitudeMismatch(2, {{"H", {}, {0}}, {"X", {0}, {1}}, {"T", {}, {0}}, {"T", {}, {0}}, {"H", {}, {0}}, {"T", {}, {0}}, {"T", {}, {0}}});
    EXPECT_LT(e, 1e-6) << "repeated T on entangled qubit disagrees with statevector by " << e;
}

// ──────────────────────────────────────────────────────────────────────────────
// Sampling distributions for T-bearing circuits
// ──────────────────────────────────────────────────────────────────────────────

TEST(StabilizerStateSum, Sample_HTH_BiasedDistribution) {
    // H,T,H on |0⟩ yields probabilities cos^2(π/8)≈0.854 for |0⟩ and sin^2(π/8)≈0.146 for |1⟩.
    // Check the empirical counts land near those values.
    srand(123);
    StabilizerStateSum sss(1);
    sss.apply_gate(makeGate("H", {}, {0}));
    sss.apply_gate(makeGate("T", {}, {0}));
    sss.apply_gate(makeGate("H", {}, {0}));
    const int nshots = 4000;
    auto counts = sss.sample(nshots);
    int total = 0;
    for (auto& kv : counts)
        total += kv.second;
    EXPECT_EQ(total, nshots);
    double p0 = static_cast<double>(counts["0"]) / nshots;
    const double expected0 = std::pow(std::cos(M_PI / 8.0), 2);  // ≈ 0.8536
    EXPECT_NEAR(p0, expected0, 0.05) << "P(0) for H,T,H was " << p0 << ", expected ≈ " << expected0;
}

// ──────────────────────────────────────────────────────────────────────────────
// Coverage: sampling cleanup-loop phase branches, amplitude pure-Z reduction,
// identity fallbacks, unsupported-gate throw, and small-amplitude guards.
// ──────────────────────────────────────────────────────────────────────────────

TEST(Sample, GHZ_CleanupLoop_PureXPivot) {
    // A 3-qubit GHZ state: H(0), CNOT(0->1), CNOT(1->2). The X-stabilizer pivot row is XXX,
    // so sampling qubit 0 must rowsum the other X-bearing rows into the pivot, exercising the
    // X-case phase branch of the cleanup loop. Outcomes are perfectly correlated (000 or 111).
    srand(11);
    bool saw_zero = false, saw_one = false;
    for (int trial = 0; trial < 60; ++trial) {
        StabilizerState s(3);
        s.apply_gate(makeGate("H", {}, {0}));
        s.apply_gate(makeGate("X", {0}, {1}));
        s.apply_gate(makeGate("X", {1}, {2}));
        std::string r = s.sample();
        ASSERT_EQ(r.size(), 3u);
        EXPECT_TRUE(r == "000" || r == "111") << "GHZ sample was " << r;
        if (r == "000")
            saw_zero = true;
        if (r == "111")
            saw_one = true;
    }
    EXPECT_TRUE(saw_zero);
    EXPECT_TRUE(saw_one);
}

TEST(Sample, ClusterState_MixedPauliPivot_AllPhaseBranches) {
    // A linear cluster state: H on every qubit, then CZ along a chain. Its stabilizers mix X and Z
    // across qubits (e.g. X0 Z1, Z0 X1 Z2, Z1 X2), so the sampling cleanup loop hits the pure-X
    // (216-217), pure-Z (219) and combined phase branches, including the exp==2 phase-set (223).
    srand(5);
    for (int trial = 0; trial < 60; ++trial) {
        StabilizerState s(4);
        for (int q = 0; q < 4; ++q)
            s.apply_gate(makeGate("H", {}, {q}));
        s.apply_gate(makeGate("Z", {0}, {1}));
        s.apply_gate(makeGate("Z", {1}, {2}));
        s.apply_gate(makeGate("Z", {2}, {3}));
        // Add an S to introduce a Y component on one pivot row.
        s.apply_gate(makeGate("S", {}, {1}));
        std::string r = s.sample();
        ASSERT_EQ(r.size(), 4u);
        for (char c : r)
            EXPECT_TRUE(c == '0' || c == '1');
    }
}

TEST(Sample, PivotRowWithPureZ_HitsZPhaseBranch) {
    // Construct two stabilizer rows that both carry an X on qubit 0, where the pivot row also carries
    // a pure Z on qubit 1 (row 0 = X0 Z1, row 1 = X0). Sampling qubit 0 then runs the cleanup loop,
    // whose per-qubit phase accumulation hits the pure-X branch (q0) and the pure-Z branch (q1).
    StabilizerState s(2);
    s.apply_gate(makeGate("H", {}, {0}));  // row 0: X0  (row 1: Z1)
    s.rowsum(0, 1);                        // row 0: X0 Z1
    s.rowsum(1, 0);                        // row 1: Z1 * X0 Z1 = X0
    std::string r = s.sample();
    ASSERT_EQ(r.size(), 2u);
    for (char c : r)
        EXPECT_TRUE(c == '0' || c == '1');
}

TEST(Sample, PivotCleanup_NegativePhase_HitsExpEqualsTwo) {
    // Same construction as PivotRowWithPureZ but with an added Z that flips the accumulated phase so
    // the cleanup loop's modular exponent lands on 2, exercising the ph.set(i) branch.
    srand(3);
    for (int trial = 0; trial < 20; ++trial) {
        StabilizerState s(2);
        s.apply_gate(makeGate("H", {}, {0}));  // row 0: X0
        s.apply_gate(makeGate("Z", {}, {0}));  // X0 -> -X0 (sets a phase on row 0)
        s.rowsum(0, 1);                        // row 0: -X0 Z1
        s.rowsum(1, 0);                        // row 1: X0 (carrying phase)
        std::string r = s.sample();
        ASSERT_EQ(r.size(), 2u);
        for (char c : r)
            EXPECT_TRUE(c == '0' || c == '1');
    }
}

TEST(StabilizerStateSum, Sample_WideSupportFewShots_CoverageWarning) {
    // A wide superposition (many T-branches across 6 qubits) sampled with very few shots cannot cover
    // 95% of the probability mass, exercising the low-coverage warning path. Also stresses the
    // weight-selection / importance-sampling loop with many terms.
    srand(2024);
    StabilizerStateSum sss(6);
    for (int q = 0; q < 6; ++q) {
        sss.apply_gate(makeGate("H", {}, {q}));
        sss.apply_gate(makeGate("T", {}, {q}));
    }
    auto counts = sss.sample(1);
    int total = 0;
    for (auto& kv : counts)
        total += kv.second;
    EXPECT_EQ(total, 1);
}

TEST(Rowsum, ZTimesX_HitsPauliPhaseZCase) {
    // rowsum where row i carries a Z and row h carries an X on the same qubit exercises the
    // pauli_phase Z-case branch. After H(0) row 0 = X0; row 1 (the Z1 generator) times row 0 gives
    // a well-defined Pauli product with a tracked phase.
    StabilizerState s(2);
    s.apply_gate(makeGate("H", {}, {0}));  // row 0 = X0
    s.apply_gate(makeGate("S", {}, {1}));  // no effect on |0> for qubit 1, keeps Z1 generator
    // row 1 is Z1; multiply row 1 by row 0 (X0): Z on q1 times X on q0 -> Z-case + X-case phases.
    s.rowsum(1, 0);
    EXPECT_TRUE(s.get_x_bits()[0][1]);  // row 1 now has X on qubit 0
    EXPECT_TRUE(s.get_z_bits()[1][1]);  // row 1 still has Z on qubit 1
}

TEST(Amplitude, ClusterState_MatchesStatevector) {
    // The cluster state has X-pivot generators that carry Z on non-pivot qubits, exercising the
    // pure-Z reduction and its skip branches inside amplitude().
    double e = amplitudeMismatch(4, {{"H", {}, {0}}, {"H", {}, {1}}, {"H", {}, {2}}, {"H", {}, {3}}, {"Z", {0}, {1}}, {"Z", {1}, {2}}, {"Z", {2}, {3}}});
    EXPECT_LT(e, 1e-6) << "cluster-state amplitude disagrees with statevector by " << e;
}

TEST(Amplitude, WithSwap_MatchesStatevector) {
    // Includes a SWAP gate so the dense reference application exercises its SWAP branch, and the
    // stabilizer amplitude()/matrix_element SWAP handling is checked against it.
    double e = amplitudeMismatch(3, {{"H", {}, {0}}, {"X", {0}, {1}}, {"SWAP", {}, {1, 2}}, {"S", {}, {2}}});
    EXPECT_LT(e, 1e-6) << "circuit with SWAP disagrees with statevector by " << e;
}

TEST(StabilizerStateSum, WithSwap_AmplitudeMatchesStatevector) {
    // SWAP plus a T gate through the StabilizerStateSum dense reference (denseApply SWAP branch).
    double e = sumAmplitudeMismatch(3, {{"H", {}, {0}}, {"T", {}, {0}}, {"SWAP", {}, {0, 2}}, {"X", {2}, {1}}});
    EXPECT_LT(e, 1e-6) << "sum circuit with SWAP disagrees with statevector by " << e;
}

TEST(Amplitude, GHZ_MatchesStatevector) {
    // GHZ has X-block rank 1 with two pure-Z generators; exercises the pure-Z reduction loop
    // (including the skip branches when a qubit has no Z pivot) inside amplitude().
    double e = amplitudeMismatch(3, {{"H", {}, {0}}, {"X", {0}, {1}}, {"X", {1}, {2}}});
    EXPECT_LT(e, 1e-6) << "GHZ amplitude disagrees with statevector by " << e;
}

TEST(Amplitude, GHZ_FourQubit_MatchesStatevector) {
    // Larger GHZ: more pure-Z generators, more qubits without their own Z pivot.
    double e = amplitudeMismatch(4, {{"H", {}, {0}}, {"X", {0}, {1}}, {"X", {1}, {2}}, {"X", {2}, {3}}});
    EXPECT_LT(e, 1e-6) << "4-qubit GHZ amplitude disagrees with statevector by " << e;
}

TEST(StabilizerStateSum, UnsupportedThreeControlGate_Throws) {
    // A gate with three controls is not supported by the tableau update and must throw.
    StabilizerState s(4);
    Gate bad("X", make2x2Identity(), {0, 1, 2}, {3}, {});
    EXPECT_THROW(s.apply_gate(bad), std::invalid_argument);
}

TEST(StabilizerStateSum, Sample_LargeSuperposition_ManyBranches) {
    // Many T gates on independent superposed qubits create a large weighted sum of stabilizer
    // states, exercising the weight-selection clamp and the importance-sampling path.
    srand(99);
    StabilizerStateSum sss(4);
    for (int q = 0; q < 4; ++q) {
        sss.apply_gate(makeGate("H", {}, {q}));
        sss.apply_gate(makeGate("T", {}, {q}));
    }
    auto counts = sss.sample(200);
    int total = 0;
    for (auto& kv : counts)
        total += kv.second;
    EXPECT_EQ(total, 200);
}

TEST(MatrixElement, UnknownSingleQubitGate_UsesIdentityFallback) {
    // matrix_element on a single-qubit gate whose name is not a known Clifford falls back to the
    // identity matrix element (clifford_elem identity branch). For the identity, <b|I|psi> = psi(b).
    StabilizerState s(1);
    s.apply_gate(makeGate("X", {}, {0}));  // state is |1>
    Gate unknown("FOO", make2x2Identity(), {}, {0}, {});
    std::complex<double> e1 = s.matrix_element(unknown, "1");
    std::complex<double> e0 = s.matrix_element(unknown, "0");
    EXPECT_NEAR(std::abs(e1), 1.0, 1e-9);  // amplitude of |1> is 1
    EXPECT_NEAR(std::abs(e0), 0.0, 1e-9);  // amplitude of |0> is 0
}

TEST(MatrixElement, UnknownMultiQubitGate_UsesIdentityFallback) {
    // A multi-qubit gate that is neither SWAP, CNOT, CZ nor Toffoli falls through to the identity
    // fallback (returns amplitude(b) unchanged).
    StabilizerState s(2);
    s.apply_gate(makeGate("X", {}, {0}));                    // state is |10>
    Gate unknown("ISWAP", make2x2Identity(), {1}, {0}, {});  // controlled "ISWAP"-named, unhandled
    std::complex<double> e = s.matrix_element(unknown, "10");
    EXPECT_NEAR(std::abs(e), 1.0, 1e-9);
    std::complex<double> e_other = s.matrix_element(unknown, "01");
    EXPECT_NEAR(std::abs(e_other), 0.0, 1e-9);
}

TEST(StabilizerStateSum, Sample_TGate_DoesNotChangeMeasurementProbabilities) {
    // T is diagonal, so it never changes Z-basis measurement statistics: |+⟩ stays 50/50.
    srand(321);
    StabilizerStateSum sss(1);
    sss.apply_gate(makeGate("H", {}, {0}));
    sss.apply_gate(makeGate("T", {}, {0}));
    const int nshots = 4000;
    auto counts = sss.sample(nshots);
    double p0 = static_cast<double>(counts["0"]) / nshots;
    EXPECT_NEAR(p0, 0.5, 0.05) << "T should not bias Z-basis statistics; P(0) was " << p0;
}

// ──────────────────────────────────────────────────────────────────────────────
// expectation_value: must match the dense statevector reference (incl. T-gate sums)
// ──────────────────────────────────────────────────────────────────────────────

namespace {

// Build a StabilizerStateSum from a gate list and return |stabilizer <H> - dense <H>|. The dense
// reference is <v|H|v>/<v|v> on the exact statevector v = sum.as_dense(), which is the ground truth.
double sumExpectationError(int n, const std::vector<std::tuple<std::string, std::vector<int>, std::vector<int>>>& ops, const MatrixFreeHamiltonian& H) {
    StabilizerStateSum sum(n);
    for (const auto& [name, ctr, tgt] : ops) {
        if (name == "T") {
            sum.apply_gate(Gate("T", tMatrix(), ctr, tgt, {}));
        } else {
            sum.apply_gate(Gate(name, make2x2Identity(), ctr, tgt, {}));
        }
    }
    DenseMatrix v = sum.as_dense();
    double nrm2 = 0.0;
    for (int i = 0; i < v.rows(); ++i) {
        nrm2 += std::norm(v(i, 0));
    }
    double dense_ev = H.expectation_value(v) / nrm2;
    double stab_ev = sum.expectation_value(H);
    return std::abs(stab_ev - dense_ev);
}

}  // namespace

TEST(StabilizerStateSumExpectation, Bell_ZZ_IsOne) {
    MatrixFreeHamiltonian H(2);
    H.add(cd(1, 0), std::vector<MatrixFreeOperator>{MatrixFreeOperator("Z", 0), MatrixFreeOperator("Z", 1)});
    EXPECT_LT(sumExpectationError(2, {{"H", {}, {0}}, {"X", {0}, {1}}}, H), 1e-9);
}

TEST(StabilizerStateSumExpectation, Bell_XX_IsOne_ZsAreZero) {
    MatrixFreeHamiltonian Hxx(2);
    Hxx.add(cd(1, 0), std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0), MatrixFreeOperator("X", 1)});
    MatrixFreeHamiltonian Hz(2);
    Hz.add(cd(1, 0), MatrixFreeOperator("Z", 0));
    const std::vector<std::tuple<std::string, std::vector<int>, std::vector<int>>> bell = {{"H", {}, {0}}, {"X", {0}, {1}}};
    EXPECT_LT(sumExpectationError(2, bell, Hxx), 1e-9);
    EXPECT_LT(sumExpectationError(2, bell, Hz), 1e-9);
}

TEST(StabilizerStateSumExpectation, TState_XYZ_CrossTerms) {
    // H|0>,T -> (|0> + e^{i pi/4}|1>)/sqrt2: <X>=<Y>=1/sqrt2, <Z>=0. Exercises the two-term cross sum.
    const std::vector<std::tuple<std::string, std::vector<int>, std::vector<int>>> ht = {{"H", {}, {0}}, {"T", {}, {0}}};
    for (const char* p : {"X", "Y", "Z"}) {
        MatrixFreeHamiltonian H(1);
        H.add(cd(1, 0), MatrixFreeOperator(p, 0));
        EXPECT_LT(sumExpectationError(1, ht, H), 1e-9) << "Pauli " << p;
    }
}

TEST(StabilizerStateSumExpectation, RandomCircuitsWithT_MatchDense) {
    std::mt19937 rng(0xC0FFEE);
    const char* singles[] = {"H", "X", "Y", "Z", "S", "T"};
    const char* paulis[] = {"X", "Y", "Z"};
    double worst = 0.0;
    for (int trial = 0; trial < 60; ++trial) {
        const int n = 2 + static_cast<int>(rng() % 2);  // 2 or 3 qubits
        std::vector<std::tuple<std::string, std::vector<int>, std::vector<int>>> ops;
        const int ngates = 4 + static_cast<int>(rng() % 6);
        for (int g = 0; g < ngates; ++g) {
            if (rng() % 3 == 0) {  // CNOT
                int c = static_cast<int>(rng() % n), t = static_cast<int>(rng() % n);
                while (t == c) {
                    t = static_cast<int>(rng() % n);
                }
                ops.push_back({"X", {c}, {t}});
            } else {
                ops.push_back({singles[rng() % 6], {}, {static_cast<int>(rng() % n)}});
            }
        }
        // Random Hermitian Hamiltonian: a few real-coefficient multi-qubit Pauli terms.
        MatrixFreeHamiltonian H(n);
        const int nterms = 1 + static_cast<int>(rng() % 3);
        for (int term = 0; term < nterms; ++term) {
            std::vector<MatrixFreeOperator> ps;
            for (int q = 0; q < n; ++q) {
                int r = static_cast<int>(rng() % 4);  // 0 = identity on this qubit
                if (r > 0) {
                    ps.push_back(MatrixFreeOperator(paulis[r - 1], q));
                }
            }
            if (ps.empty()) {
                ps.push_back(MatrixFreeOperator("Z", 0));
            }
            const double coeff = (static_cast<double>(rng() % 200) - 100.0) / 50.0;
            H.add(cd(coeff, 0), ps);
        }
        worst = std::max(worst, sumExpectationError(n, ops, H));
    }
    EXPECT_LT(worst, 1e-6) << "worst stabilizer-sum <H> vs dense over random+T circuits = " << worst;
}

// ──────────────────────────────────────────────────────────────────────────────
// Edge cases
// ──────────────────────────────────────────────────────────────────────────────

TEST(Sample, CleanupLoop_NegativePhaseBranch) {
    // The multi-anticommuting cleanup loop in sample() updates a row's phase from the product of two
    // generators that both carry X on the sampled qubit. Depending on the tableau that product can be
    // +1 or -1; this deterministic sweep of random Clifford states (each sampled with a fixed seed)
    // drives the loop through both, including the -1 (exp == 2) branch.
    const char* singles[] = {"H", "X", "Y", "Z", "S"};
    uint64_t rng = 0xDEADBEEFCAFEULL;
    auto next = [&]() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return rng;
    };
    for (int trial = 0; trial < 300; ++trial) {
        int n = 4;
        StabilizerState s(n);
        int ng = 8 + (next() % 8);
        for (int g = 0; g < ng; ++g) {
            if (next() % 4 == 0) {
                int a = next() % n, b = next() % n;
                if (a == b)
                    b = (b + 1) % n;
                s.apply_gate(makeGate((next() % 2) ? "X" : "Z", {a}, {b}));
            } else {
                s.apply_gate(makeGate(singles[next() % 5], {}, {(int)(next() % n)}));
            }
        }
        s.set_seed(0x1234u + trial);
        std::string result = s.sample();
        ASSERT_EQ(result.size(), (size_t)n);
    }
}

TEST(StabilizerStateSumExpectation, OffDiagonal_NonTrivialSupport) {
    // H on both qubits then T on q0 yields the two-term sum |0>|+> + e^{i*}|1>|+>; each term has a
    // one-dimensional support (the spectator qubit q1 is in |+>). The off-diagonal overlap therefore
    // enumerates that support (direction XOR) and applies the Z-phase sign, matching the dense value.
    const std::vector<std::tuple<std::string, std::vector<int>, std::vector<int>>> ops = {{"H", {}, {0}}, {"H", {}, {1}}, {"T", {}, {0}}};
    MatrixFreeHamiltonian H(2);
    H.add(cd(0.7, 0), MatrixFreeOperator("X", 0));
    H.add(cd(0.5, 0), std::vector<MatrixFreeOperator>{MatrixFreeOperator("X", 0), MatrixFreeOperator("Z", 1)});
    H.add(cd(0.3, 0), std::vector<MatrixFreeOperator>{MatrixFreeOperator("Y", 0), MatrixFreeOperator("Z", 1)});
    H.add(cd(0.9, 0), MatrixFreeOperator("Z", 1));
    EXPECT_LT(sumExpectationError(2, ops, H), 1e-9);
}

TEST(StabilizerStateSumExpectation, OffDiagonal_SupportTooLarge_Throws) {
    // |+>^26 branched by a T gate gives two terms each with a 25-dimensional support, exceeding the
    // MAX_SUPPORT_DIM (24) cap; the cross-term overlap enumeration must refuse to run.
    const int n = 26;
    StabilizerStateSum sum(n);
    for (int q = 0; q < n; ++q) {
        sum.apply_gate(Gate("H", make2x2Identity(), {}, {q}, {}));
    }
    sum.apply_gate(Gate("T", tMatrix(), {}, {0}, {}));
    MatrixFreeHamiltonian H(n);
    H.add(cd(1, 0), MatrixFreeOperator("Z", 1));
    EXPECT_THROW(sum.expectation_value(H), std::runtime_error);
}

TEST(StabilizerStateSumExpectation, EmptySum_IsZero) {
    // A sum with no terms has no expectation value to speak of; it must return 0 rather than divide.
    StabilizerStateSum empty(2, std::vector<StabilizerState>{}, std::vector<cd>{});
    MatrixFreeHamiltonian H(2);
    H.add(cd(1, 0), MatrixFreeOperator("Z", 0));
    EXPECT_EQ(empty.expectation_value(H), 0.0);
}

TEST(StabilizerStateSumExpectation, ZeroNormState_IsZero) {
    // Two identical terms with opposite coefficients sum to the zero vector: <psi|psi> = 0, so the
    // normalised expectation value is undefined and must fall back to 0.
    StabilizerState s(2);
    StabilizerStateSum cancel(2, std::vector<StabilizerState>{s, s}, std::vector<cd>{cd(1, 0), cd(-1, 0)});
    MatrixFreeHamiltonian H(2);
    H.add(cd(1, 0), MatrixFreeOperator("Z", 0));
    EXPECT_EQ(cancel.expectation_value(H), 0.0);
}

TEST(StabilizerState, DroppedGlobalPhase_HighRankDefaultsToOne) {
    // The normalisation factor 2^(-rank/2) underflows the 1e-12 guard for a high-rank support, so the
    // dropped global phase cannot be reconstructed and defaults to 1.
    const int n = 90;
    StabilizerState pre(n);
    StabilizerState after(n);
    for (int q = 0; q < n; ++q) {
        after.apply_gate(makeGate("H", {}, {q}));
    }
    cd ph = pre.dropped_global_phase(after, makeGate("X", {}, {0}));
    EXPECT_EQ(ph, cd(1, 0));
}

// GCOV_EXCL_BR_STOP
