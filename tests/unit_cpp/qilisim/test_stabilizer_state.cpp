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
    EXPECT_THROW(StabilizerState{MAX_ROWS_STABILIZER}, std::invalid_argument);
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

int bitval(int idx, int q, int n) { return (idx >> (n - 1 - q)) & 1; }
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
        if (name == "H") { u00 = s; u01 = s; u10 = s; u11 = -s; }
        else if (name == "X") { u00 = 0; u01 = 1; u10 = 1; u11 = 0; }
        else if (name == "Y") { u00 = 0; u01 = cd(0, -1); u10 = cd(0, 1); u11 = 0; }
        else if (name == "Z") { u00 = 1; u01 = 0; u10 = 0; u11 = -1; }
        else if (name == "S") { u00 = 1; u01 = 0; u10 = 0; u11 = cd(0, 1); }
        for (int i = 0; i < dim; ++i) {
            int tb = bitval(i, t, n);
            int i0 = setbitval(i, t, n, 0), i1 = setbitval(i, t, n, 1);
            if (tb == 0) { out[i0] += u00 * sv[i]; out[i1] += u10 * sv[i]; }
            else { out[i0] += u01 * sv[i]; out[i1] += u11 * sv[i]; }
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
            if (bitval(i, c, n) && bitval(i, t, n)) v = -v;
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
        if (std::abs(sv[i]) > best) { best = std::abs(sv[i]); piv = i; }
    std::string bpiv(n, '0');
    for (int q = 0; q < n; ++q) bpiv[q] = bitval(piv, q, n) ? '1' : '0';
    cd apiv = st.amplitude(bpiv);
    if (std::abs(apiv) < 1e-9) return 1e9;
    cd gp = sv[piv] / apiv;
    double maxerr = 0;
    for (int i = 0; i < dim; ++i) {
        std::string b(n, '0');
        for (int q = 0; q < n; ++q) b[q] = bitval(i, q, n) ? '1' : '0';
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
    auto next = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    double worst = 0;
    for (int trial = 0; trial < 400; ++trial) {
        int n = 3;
        int ng = 4 + (next() % 8);
        std::vector<std::tuple<std::string, std::vector<int>, std::vector<int>>> ops;
        for (int g = 0; g < ng; ++g) {
            if (next() % 4 == 0) {
                int a = next() % n, b = next() % n;
                if (a == b) b = (b + 1) % n;
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
        if (bitval(i, t, n)) sv[i] *= ph;
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
        if (std::abs(sv[i]) > best) { best = std::abs(sv[i]); piv = i; }
    std::string bpiv(n, '0');
    for (int q = 0; q < n; ++q) bpiv[q] = bitval(piv, q, n) ? '1' : '0';
    cd apiv = sum.amplitude(bpiv);
    if (std::abs(apiv) < 1e-9) return 1e9;
    cd gp = sv[piv] / apiv;
    double maxerr = 0;
    for (int i = 0; i < dim; ++i) {
        std::string b(n, '0');
        for (int q = 0; q < n; ++q) b[q] = bitval(i, q, n) ? '1' : '0';
        maxerr = std::max(maxerr, std::abs(sv[i] - gp * sum.amplitude(b)));
    }
    return maxerr;
}

}  // namespace

TEST(StabilizerStateSum, Amplitude_MatchesStatevector_TwoTGatesEntangled) {
    // Reduced from a randomized failure: two T gates plus entangling Cliffords.
    double e = sumAmplitudeMismatch(2, {
        {"H", {}, {1}}, {"X", {}, {0}}, {"Y", {}, {0}}, {"X", {1}, {0}}, {"H", {}, {0}},
        {"T", {}, {0}}, {"Y", {}, {1}}, {"X", {1}, {0}}, {"H", {}, {0}}, {"T", {}, {1}}, {"H", {}, {0}}});
    EXPECT_LT(e, 1e-6) << "StabilizerStateSum amplitude disagrees with statevector by " << e;
}

TEST(StabilizerStateSum, Amplitude_MatchesStatevector_RandomWithT) {
    const char* singles[] = {"H", "X", "Y", "Z", "S", "T"};
    uint64_t rng = 0xD1B54A32D192ED03ULL;
    auto next = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    double worst = 0;
    for (int trial = 0; trial < 400; ++trial) {
        int n = 1 + (next() % 3);
        int ng = 4 + (next() % 16);
        std::vector<std::tuple<std::string, std::vector<int>, std::vector<int>>> ops;
        for (int g = 0; g < ng; ++g) {
            if (n > 1 && next() % 4 == 0) {
                int a = next() % n, b = next() % n;
                if (a == b) b = (b + 1) % n;
                ops.push_back({(next() % 2) ? "X" : "Z", {a}, {b}});
            } else {
                ops.push_back({singles[next() % 6], {}, {(int)(next() % n)}});
            }
        }
        worst = std::max(worst, sumAmplitudeMismatch(n, ops));
    }
    EXPECT_LT(worst, 1e-6) << "worst StabilizerStateSum amplitude vs statevector over random+T = " << worst;
}

// GCOV_EXCL_BR_STOP
