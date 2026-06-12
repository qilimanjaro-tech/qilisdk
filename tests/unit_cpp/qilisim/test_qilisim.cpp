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
#include "../../../src/qilisdk_cpp/backends/qilisim/qilisim.h"

namespace py = pybind11;

static py::dict empty_solver_params() {
    return py::dict();
}

class ExecuteSamplingTest : public ::testing::Test {
   protected:
    QiliSimCpp sim;
};

TEST_F(ExecuteSamplingTest, NotSamplingType_None_ThrowsValueError) {
    py::gil_scoped_acquire gil;
    EXPECT_THROW(sim.execute_digital_propagation(py::none(), py::list(), py::none(), py::none(), empty_solver_params()), py::value_error);
}

TEST_F(ExecuteSamplingTest, NotSamplingType_Object_ThrowsValueError) {
    py::gil_scoped_acquire gil;
    py::exec(R"(not_a_sampling = object())");
    EXPECT_THROW(sim.execute_digital_propagation(py::globals()["not_a_sampling"], py::list(), py::none(), py::none(), empty_solver_params()), py::value_error);
}

TEST_F(ExecuteSamplingTest, ZeroNqubits_ThrowsValueError) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.digital_propagation import DigitalPropagation

class _FakeCircuitZero:
    nqubits = 0
    gates = []

class _ZeroNqubits(DigitalPropagation):
    def __init__(self):
        object.__init__(self)
        self.circuit = _FakeCircuitZero()

_zero_nqubits = _ZeroNqubits()
    )");
    EXPECT_THROW(sim.execute_digital_propagation(py::globals()["_zero_nqubits"], py::list(), py::none(), py::none(), empty_solver_params()), py::value_error);
}

TEST_F(ExecuteSamplingTest, StandardMethod_EmptyCircuit_NoNoise_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.digital_propagation import DigitalPropagation
from qilisdk.digital.circuit import Circuit

_c_empty = Circuit(nqubits=2)
_samp_empty = DigitalPropagation(circuit=_c_empty)
    )");
    EXPECT_NO_THROW(sim.execute_digital_propagation(py::globals()["_samp_empty"], py::list(), py::none(), py::none(), empty_solver_params()));
}

TEST_F(ExecuteSamplingTest, StandardMethod_XGate_NoNoise_NoInitialState_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.digital_propagation import DigitalPropagation
from qilisdk.digital.circuit import Circuit
from qilisdk.digital.gates import X

_c_x = Circuit(nqubits=1)
_c_x.add(X(0))
_samp_x = DigitalPropagation(circuit=_c_x)
    )");
    EXPECT_NO_THROW(sim.execute_digital_propagation(py::globals()["_samp_x"], py::list(), py::none(), py::none(), empty_solver_params()));
}

TEST_F(ExecuteSamplingTest, StandardMethod_XGate_NoNoise_WithInitialState_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.digital_propagation import DigitalPropagation
from qilisdk.digital.circuit import Circuit
from qilisdk.digital.gates import X
from qilisdk.core.qtensor import QTensor
import scipy.sparse as sp, numpy as np

_c_is = Circuit(nqubits=1)
_c_is.add(X(0))
_samp_is = DigitalPropagation(circuit=_c_is)
_init_sv = QTensor(sp.csr_matrix(np.array([[1.0+0j], [0.0]], dtype=complex)))
    )");
    EXPECT_NO_THROW(sim.execute_digital_propagation(py::globals()["_samp_is"], py::list(), py::none(), py::globals()["_init_sv"], empty_solver_params()));
}

TEST_F(ExecuteSamplingTest, MatrixFreeMethod_XGate_NoNoise_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.digital_propagation import DigitalPropagation
from qilisdk.digital.circuit import Circuit
from qilisdk.digital.gates import X

_c_mf = Circuit(nqubits=1)
_c_mf.add(X(0))
_samp_mf = DigitalPropagation(circuit=_c_mf)
    )");
    py::dict p;
    p["sampling_method"] = py::str("statevector_matrix_free");
    EXPECT_NO_THROW(sim.execute_digital_propagation(py::globals()["_samp_mf"], py::list(), py::none(), py::none(), p));
}

TEST_F(ExecuteSamplingTest, NonNormalizedGate_NormalizationBranchExecuted) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
import numpy as np
from qilisdk.functionals.digital_propagation import DigitalPropagation

class _ShrinkGate:
    name = 'Shrink'
    control_qubits = []
    target_qubits = [0]
    is_parameterized = False
    @property
    def matrix(self):
        return np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
    def get_parameters(self):
        return {}

class _FakeCircuitShrink:
    nqubits = 1
    gates = [_ShrinkGate()]

class _ShrinkSampling(DigitalPropagation):
    def __init__(self):
        object.__init__(self)
        self.circuit = _FakeCircuitShrink()

_shrink_samp = _ShrinkSampling()
    )");
    EXPECT_NO_THROW(sim.execute_digital_propagation(py::globals()["_shrink_samp"], py::list(), py::none(), py::none(), empty_solver_params()));
}

TEST_F(ExecuteSamplingTest, Result_HasSamples) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.digital_propagation import DigitalPropagation
from qilisdk.readout import SamplingReadout
from qilisdk.digital.circuit import Circuit
from qilisdk.digital.gates import X

_c_res = Circuit(nqubits=1)
_c_res.add(X(0))
_samp_res = DigitalPropagation(circuit=_c_res)
_readout_res = [SamplingReadout(nshots=50)]
    )");
    py::object result;
    ASSERT_NO_THROW(result = sim.execute_digital_propagation(py::globals()["_samp_res"], py::globals()["_readout_res"], py::none(), py::none(), empty_solver_params()));
    EXPECT_TRUE(py::hasattr(result, "sampling"));
}

TEST_F(ExecuteSamplingTest, Result_HasIntermediateSamples) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.digital_propagation import DigitalPropagation
from qilisdk.readout import SamplingReadout
from qilisdk.digital.circuit import Circuit
from qilisdk.digital.gates import X, M

_c_intermediate = Circuit(nqubits=1)
_c_intermediate.add(X(0))
_c_intermediate.add(M(0))
_c_intermediate.add(X(0))
_c_intermediate.add(M(0))
_samp_intermediate = DigitalPropagation(circuit=_c_intermediate)
_readout_intermediate = [SamplingReadout(nshots=50)]
    )");
    py::object result;
    ASSERT_NO_THROW(result = sim.execute_digital_propagation(py::globals()["_samp_intermediate"], py::globals()["_readout_intermediate"], py::none(), py::none(), empty_solver_params()));
    EXPECT_TRUE(py::hasattr(result, "intermediate_results"));
    py::list inter_results = result.attr("intermediate_results");
    EXPECT_TRUE(inter_results.size() > 0);
}

TEST_F(ExecuteSamplingTest, CollapseTest) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.digital_propagation import DigitalPropagation
from qilisdk.readout import SamplingReadout
from qilisdk.digital.circuit import Circuit
from qilisdk.digital.gates import H, M

_c_intermediate = Circuit(nqubits=1)
_c_intermediate.add(H(0))
_c_intermediate.add(M(0))
_c_intermediate.add(H(0))
_c_intermediate.add(M(0))
_samp_intermediate = DigitalPropagation(circuit=_c_intermediate)
_solver_params = {"measurement_collapse": True}
_readout_intermediate = [SamplingReadout(nshots=1000)]
    )");
    py::object result;
    ASSERT_NO_THROW(result = sim.execute_digital_propagation(py::globals()["_samp_intermediate"], py::globals()["_readout_intermediate"], py::none(), py::none(), py::globals()["_solver_params"]));
    EXPECT_TRUE(py::hasattr(result, "intermediate_results"));
    py::list inter_results = result.attr("intermediate_results");
    EXPECT_TRUE(inter_results.size() > 0);
    EXPECT_LT(result.attr("get_samples")().cast<py::dict>()["0"].cast<int>(), 999);
}

TEST_F(ExecuteSamplingTest, CollapseTestDensityMatrix) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.digital_propagation import DigitalPropagation
from qilisdk.readout import SamplingReadout
from qilisdk.digital.circuit import Circuit
from qilisdk.digital.gates import H, M
from qilisdk.core.qtensor import QTensor
import scipy.sparse as sp, numpy as np

_initial_state = QTensor(sp.csr_matrix(np.array([[0.5+0j, 0], [0, 0.5+0j]], dtype=complex)))

_c_intermediate = Circuit(nqubits=1)
_c_intermediate.add(H(0))
_c_intermediate.add(M(0))
_c_intermediate.add(H(0))
_c_intermediate.add(M(0))
_samp_intermediate = DigitalPropagation(circuit=_c_intermediate)
_solver_params = {"measurement_collapse": True}
_readout_intermediate = [SamplingReadout(nshots=1000)]
    )");
    py::object result;
    ASSERT_NO_THROW(result = sim.execute_digital_propagation(py::globals()["_samp_intermediate"], py::globals()["_readout_intermediate"], py::none(), py::globals()["_initial_state"], py::globals()["_solver_params"]));
    EXPECT_TRUE(py::hasattr(result, "intermediate_results"));
    py::list inter_results = result.attr("intermediate_results");
    EXPECT_TRUE(inter_results.size() > 0);
    EXPECT_LT(result.attr("get_samples")().cast<py::dict>()["0"].cast<int>(), 999);
}

class ExecuteTimeEvolutionTest : public ::testing::Test {
   protected:
    QiliSimCpp sim;
};

TEST_F(ExecuteTimeEvolutionTest, NotTimeEvolutionType_None_ThrowsValueError) {
    py::gil_scoped_acquire gil;
    EXPECT_THROW(sim.execute_analog_evolution(py::none(), py::list(), py::none(), empty_solver_params()), py::value_error);
}

TEST_F(ExecuteTimeEvolutionTest, NotTimeEvolutionType_Object_ThrowsValueError) {
    py::gil_scoped_acquire gil;
    py::exec(R"(not_a_te = object())");
    EXPECT_THROW(sim.execute_analog_evolution(py::globals()["not_a_te"], py::list(), py::none(), empty_solver_params()), py::value_error);
}

TEST_F(ExecuteTimeEvolutionTest, StandardMethod_NoNoise_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.analog.schedule import Schedule
from qilisdk.analog.hamiltonian import Z
from qilisdk.core.qtensor import QTensor
import scipy.sparse as sp, numpy as np

_sched_std = Schedule(hamiltonians={"h0": Z(0)}, dt=0.1, total_time=0.3)
_rho_std = QTensor(sp.csr_matrix(np.array([[1.+0j,0],[0,0]], dtype=complex)))
_te_std = AnalogEvolution(schedule=_sched_std, initial_state=_rho_std)
    )");
    EXPECT_NO_THROW(sim.execute_analog_evolution(py::globals()["_te_std"], py::list(), py::none(), empty_solver_params()));
}

TEST_F(ExecuteTimeEvolutionTest, StandardMethod_WithStoreIntermediateResults_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.analog.schedule import Schedule
from qilisdk.analog.hamiltonian import Z
from qilisdk.core.qtensor import QTensor
import scipy.sparse as sp, numpy as np

_sched_interm = Schedule(hamiltonians={"h0": Z(0)}, dt=0.1, total_time=0.3)
_rho_interm = QTensor(sp.csr_matrix(np.array([[1.+0j,0],[0,0]], dtype=complex)))
_te_interm = AnalogEvolution(
    schedule=_sched_interm,
    initial_state=_rho_interm,
    store_intermediate_results=True,
)
    )");
    py::object result;
    ASSERT_NO_THROW(result = sim.execute_analog_evolution(py::globals()["_te_interm"], py::list(), py::none(), empty_solver_params()));
    EXPECT_TRUE(result.attr("intermediate_results").cast<py::list>().size() > 0);
}

TEST_F(ExecuteTimeEvolutionTest, MatrixFreeMethod_NoNoise_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.analog.schedule import Schedule
from qilisdk.analog.hamiltonian import Z
from qilisdk.core.qtensor import QTensor
import scipy.sparse as sp, numpy as np

_sched_mf = Schedule(hamiltonians={"h0": Z(0)}, dt=0.1, total_time=0.3)
_rho_mf = QTensor(sp.csr_matrix(np.array([[1.+0j,0],[0,0]], dtype=complex)))
_te_mf = AnalogEvolution(schedule=_sched_mf, initial_state=_rho_mf)
    )");
    py::dict p;
    p["evolution_method"] = py::str("integrate_rk4_matrix_free");
    EXPECT_NO_THROW(sim.execute_analog_evolution(py::globals()["_te_mf"], py::list(), py::none(), p));
}

TEST_F(ExecuteTimeEvolutionTest, MatrixFreeMethod_WithStoreIntermediateResults_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.analog.schedule import Schedule
from qilisdk.analog.hamiltonian import Z
from qilisdk.core.qtensor import QTensor
import scipy.sparse as sp, numpy as np

_sched_mf_interm = Schedule(hamiltonians={"h0": Z(0)}, dt=0.1, total_time=0.3)
_rho_mf_interm = QTensor(sp.csr_matrix(np.array([[1.+0j,0],[0,0]], dtype=complex)))
_te_mf_interm = AnalogEvolution(
    schedule=_sched_mf_interm,
    initial_state=_rho_mf_interm,
    store_intermediate_results=True,
)
    )");
    py::dict p;
    p["evolution_method"] = py::str("integrate_rk4_matrix_free");
    EXPECT_NO_THROW(sim.execute_analog_evolution(py::globals()["_te_mf_interm"], py::list(), py::none(), p));
}

TEST_F(ExecuteTimeEvolutionTest, NoiseModel_EmptyGlobalPerturbations_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.analog.schedule import Schedule
from qilisdk.analog.hamiltonian import Z
from qilisdk.core.qtensor import QTensor
from qilisdk.noise.noise_model import NoiseModel
import scipy.sparse as sp, numpy as np

_sched_nm_empty = Schedule(hamiltonians={"h0": Z(0)}, dt=0.1, total_time=0.3)
_rho_nm_empty = QTensor(sp.csr_matrix(np.array([[1.+0j,0],[0,0]], dtype=complex)))
_te_nm_empty = AnalogEvolution(
    schedule=_sched_nm_empty, initial_state=_rho_nm_empty
)

_nm_empty = NoiseModel()
    )");
    EXPECT_NO_THROW(sim.execute_analog_evolution(py::globals()["_te_nm_empty"], py::list(), py::globals()["_nm_empty"], empty_solver_params()));
}

TEST_F(ExecuteTimeEvolutionTest, NoiseModel_GlobalPerturbations_NonMatchingKey_LoopBodyEntered) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.analog.schedule import Schedule
from qilisdk.analog.hamiltonian import Z
from qilisdk.core.qtensor import QTensor
from qilisdk.noise.noise_model import NoiseModel
import scipy.sparse as sp, numpy as np

_sched_nm_nonmatch = Schedule(hamiltonians={"h0": Z(0)}, dt=0.1, total_time=0.3)
_rho_nm_nonmatch = QTensor(sp.csr_matrix(np.array([[1.+0j,0],[0,0]], dtype=complex)))
_te_nm_nonmatch = AnalogEvolution(
    schedule=_sched_nm_nonmatch, initial_state=_rho_nm_nonmatch
)

class _Perturbation:
    def perturb(self, v): return v * 2.0

_nm_nonmatch = NoiseModel()
_nm_nonmatch.global_perturbations["nonexistent_param_xyz"].append(_Perturbation())
    )");
    EXPECT_NO_THROW(sim.execute_analog_evolution(py::globals()["_te_nm_nonmatch"], py::list(), py::globals()["_nm_nonmatch"], empty_solver_params()));
}

TEST_F(ExecuteTimeEvolutionTest, NoiseModel_GlobalPerturbations_MatchingKey_PerturbationApplied) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.analog.hamiltonian import Z
from qilisdk.core.qtensor import QTensor
import scipy.sparse as sp, numpy as np

_rho_nm_match = QTensor(sp.csr_matrix(np.array([[1.+0j,0],[0,0]], dtype=complex)))

class _ScalePert:
    def perturb(self, v): return v * 0.5

# Build a fake schedule whose get_parameters() returns {"amp": 1.0} so we can
# supply a matching noise perturbation key.
class _FakeScheduleWithParam:
    nqubits = 1
    tlist = [0.1, 0.2, 0.3]
    coefficients = {"h0": {0.1: 1.0, 0.2: 1.0, 0.3: 1.0}}
    _params = {"amp": 1.0}
    def get_parameters(self): return dict(self._params)
    def set_parameters(self, params): self._params.update(params)

    class _Ham:
        def keys(self): return ["h0"]
        def values(self): return [Z(0)]

    hamiltonians = _Ham()

class _TEMatchingParam(AnalogEvolution):
    def __init__(self):
        object.__init__(self)
        self.schedule = _FakeScheduleWithParam()
        self._initial_state = _rho_nm_match
        self.store_intermediate_results = False

_te_nm_match = _TEMatchingParam()

from qilisdk.noise.noise_model import NoiseModel

_nm_match = NoiseModel()
_nm_match.global_perturbations["amp"].append(_ScalePert())
    )");
    EXPECT_NO_THROW(sim.execute_analog_evolution(py::globals()["_te_nm_match"], py::list(), py::globals()["_nm_match"], empty_solver_params()));
}

TEST_F(ExecuteTimeEvolutionTest, HamiltonianCountMismatch_MatrixFree_ThrowsValueError) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.analog.hamiltonian import Z
from qilisdk.core.qtensor import QTensor
import scipy.sparse as sp, numpy as np

_rho_mm_mf = QTensor(sp.csr_matrix(np.array([[1.+0j,0],[0,0]], dtype=complex)))

class _WeirdHams_MF:
    def keys(self): return ["h0"]       # 1 key  -> parameters_list.size() == 1
    def values(self): return [Z(0), Z(0)]  # 2 vals -> hamiltonians.size()  == 2

class _FakeSched_MM_MF:
    nqubits = 1
    tlist = [0.1, 0.2, 0.3]
    coefficients = {"h0": {0.1: 1.0, 0.2: 1.0, 0.3: 1.0}}
    hamiltonians = _WeirdHams_MF()
    def get_parameters(self): return {}
    def set_parameters(self, _): pass

class _TE_MM_MF(AnalogEvolution):
    def __init__(self):
        object.__init__(self)
        self.schedule = _FakeSched_MM_MF()
        self._initial_state = _rho_mm_mf
        self.store_intermediate_results = False

_te_mm_mf = _TE_MM_MF()
    )");
    py::dict p;
    p["evolution_method"] = py::str("integrate_matrix_free");
    EXPECT_THROW(sim.execute_analog_evolution(py::globals()["_te_mm_mf"], py::list(), py::none(), p), py::value_error);
}

TEST_F(ExecuteTimeEvolutionTest, HamiltonianCountMismatch_Standard_ThrowsValueError) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.analog.hamiltonian import Z
from qilisdk.core.qtensor import QTensor
import scipy.sparse as sp, numpy as np

_rho_mm_std = QTensor(sp.csr_matrix(np.array([[1.+0j,0],[0,0]], dtype=complex)))

class _WeirdHams_Std:
    def keys(self): return ["h0"]       # 1 key
    def values(self): return [Z(0), Z(0)]  # 2 vals

class _FakeSched_MM_Std:
    nqubits = 1
    tlist = [0.1, 0.2, 0.3]
    coefficients = {"h0": {0.1: 1.0, 0.2: 1.0, 0.3: 1.0}}
    hamiltonians = _WeirdHams_Std()
    def get_parameters(self): return {}
    def set_parameters(self, _): pass

class _TE_MM_Std(AnalogEvolution):
    def __init__(self):
        object.__init__(self)
        self.schedule = _FakeSched_MM_Std()
        self._initial_state = _rho_mm_std
        self.store_intermediate_results = False

_te_mm_std = _TE_MM_Std()
    )");
    EXPECT_THROW(sim.execute_analog_evolution(py::globals()["_te_mm_std"], py::list(), py::none(), empty_solver_params()), py::value_error);
}

TEST_F(ExecuteTimeEvolutionTest, Result_HasStateAndExpectedValues) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.readout import StateTomographyReadout, ExpectationReadout
from qilisdk.analog.schedule import Schedule
from qilisdk.analog.hamiltonian import Z
from qilisdk.core.qtensor import QTensor
import scipy.sparse as sp, numpy as np

_sched_shape = Schedule(hamiltonians={"h0": Z(0)}, dt=0.1, total_time=0.3)
_rho_shape = QTensor(sp.csr_matrix(np.array([[1.+0j,0],[0,0]], dtype=complex)))
_te_shape = AnalogEvolution(
    schedule=_sched_shape, initial_state=_rho_shape, store_intermediate_results=True
)
_readout_shape = [StateTomographyReadout(), ExpectationReadout(observables=[Z(0)])]
    )");
    py::object result;
    ASSERT_NO_THROW(result = sim.execute_analog_evolution(py::globals()["_te_shape"], py::globals()["_readout_shape"], py::none(), empty_solver_params()));
    EXPECT_TRUE(py::hasattr(result, "state_tomography"));
    EXPECT_TRUE(py::hasattr(result, "expectation"));
}

class ExecuteReservoirTest : public ::testing::Test {
   protected:
    QiliSimCpp sim;
};

TEST_F(ExecuteReservoirTest, ReservoirRuns) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.quantum_reservoirs import QuantumReservoir, ReservoirLayer
from qilisdk.analog.schedule import Schedule
from qilisdk.analog.hamiltonian import Z
from qilisdk.core.qtensor import QTensor
import scipy.sparse as sp, numpy as np

_sched_res = Schedule(hamiltonians={"h0": Z(0)}, dt=0.1, total_time=0.3)
_layer_res = ReservoirLayer(evolution_dynamics=_sched_res)
_rho_res = QTensor(sp.csr_matrix(np.array([[1.+0j, 0], [0, 0]], dtype=complex)))
_reservoir = QuantumReservoir(
    initial_state=_rho_res,
    reservoir_layer=_layer_res,
    input_per_layer=[{}]
)
    )");
    py::object result;
    ASSERT_NO_THROW(result = sim.execute_quantum_reservoir(py::globals()["_reservoir"], py::list(), py::none(), empty_solver_params()));
    EXPECT_TRUE(py::hasattr(result, "readout_results"));
    EXPECT_TRUE(py::hasattr(result, "intermediate_results"));
}

TEST_F(ExecuteTimeEvolutionTest, VariationalExponential_NonUniformInitialState_Throws) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.analog.schedule import Schedule
from qilisdk.analog.hamiltonian import X, Z
from qilisdk.core.qtensor import QTensor, ket
import scipy.sparse as sp, numpy as np

_sched_var_bad = Schedule(
    hamiltonians={"h_x": X(0), "h_z": Z(0)},
    coefficients={"h_x": {(0, 2): lambda t: 1 - t/2}, "h_z": {(0, 2): lambda t: t/2}},
    dt=1,
)
_rho_var_bad = QTensor(sp.csr_matrix(np.array([[1.+0j], [0.]], dtype=complex)))
_te_var_bad = AnalogEvolution(schedule=_sched_var_bad, initial_state=_rho_var_bad)
    )");
    py::dict p;
    p["evolution_method"] = py::str("variational_exponential");
    p["order"] = py::int_(1);
    p["shots"] = py::int_(50);
    p["warmups"] = py::int_(0);
    EXPECT_THROW(sim.execute_analog_evolution(py::globals()["_te_var_bad"], py::list(), py::none(), p), py::value_error);
}

TEST_F(ExecuteTimeEvolutionTest, VariationalExponential_FirstHamiltonianNotXOnly_Throws) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.analog.schedule import Schedule
from qilisdk.analog.hamiltonian import Z
from qilisdk.core.qtensor import InitialState

_sched_var_nox = Schedule(
    hamiltonians={"h_z1": Z(0), "h_z2": Z(0)},
    coefficients={"h_z1": {(0, 2): lambda t: 1 - t/2}, "h_z2": {(0, 2): lambda t: t/2}},
    dt=1,
)
_te_var_nox = AnalogEvolution(schedule=_sched_var_nox, initial_state=InitialState.UNIFORM)
    )");
    py::dict p;
    p["evolution_method"] = py::str("variational_exponential");
    p["order"] = py::int_(1);
    p["shots"] = py::int_(50);
    p["warmups"] = py::int_(0);
    EXPECT_THROW(sim.execute_analog_evolution(py::globals()["_te_var_nox"], py::list(), py::none(), p), py::value_error);
}

TEST_F(ExecuteTimeEvolutionTest, VariationalExponential_LastHamiltonianNotZOnly_Throws) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.analog.schedule import Schedule
from qilisdk.analog.hamiltonian import X
from qilisdk.core.qtensor import InitialState

_sched_var_noz = Schedule(
    hamiltonians={"h_x1": X(0), "h_x2": X(0)},
    coefficients={"h_x1": {(0, 2): lambda t: 1 - t/2}, "h_x2": {(0, 2): lambda t: t/2}},
    dt=1,
)
_te_var_noz = AnalogEvolution(schedule=_sched_var_noz, initial_state=InitialState.UNIFORM)
    )");
    py::dict p;
    p["evolution_method"] = py::str("variational_exponential");
    p["order"] = py::int_(1);
    p["shots"] = py::int_(50);
    p["warmups"] = py::int_(0);
    EXPECT_THROW(sim.execute_analog_evolution(py::globals()["_te_var_noz"], py::list(), py::none(), p), py::value_error);
}

TEST_F(ExecuteTimeEvolutionTest, VariationalExponential_ValidXToZSchedule_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.analog.schedule import Schedule
from qilisdk.analog.hamiltonian import X, Z
from qilisdk.readout import ExpectationReadout
from qilisdk.core.qtensor import InitialState

_sched_var_ok = Schedule(
    hamiltonians={"h_x": X(0), "h_z": Z(0)},
    coefficients={"h_x": {(0, 2): lambda t: 1 - t/2}, "h_z": {(0, 2): lambda t: t/2}},
    dt=1,
)
_te_var_ok = AnalogEvolution(schedule=_sched_var_ok, initial_state=InitialState.UNIFORM)
_readout_var = [ExpectationReadout(observables=[Z(0)])]
    )");
    py::dict p;
    p["evolution_method"] = py::str("variational_exponential");
    p["order"] = py::int_(1);
    p["shots"] = py::int_(50);
    p["warmups"] = py::int_(0);
    EXPECT_NO_THROW(sim.execute_analog_evolution(py::globals()["_te_var_ok"], py::globals()["_readout_var"], py::none(), p));
}

TEST_F(ExecuteTimeEvolutionTest, VariationalExponential_SamplingReadout_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.analog.schedule import Schedule
from qilisdk.analog.hamiltonian import X, Z
from qilisdk.readout import SamplingReadout
from qilisdk.core.qtensor import InitialState

_sched_var_samp = Schedule(
    hamiltonians={"h_x": X(0), "h_z": Z(0)},
    coefficients={"h_x": {(0, 2): lambda t: 1 - t/2}, "h_z": {(0, 2): lambda t: t/2}},
    dt=1,
)
_te_var_samp = AnalogEvolution(schedule=_sched_var_samp, initial_state=InitialState.UNIFORM)
_readout_var_samp = [SamplingReadout(nshots=20)]
    )");
    py::dict p;
    p["evolution_method"] = py::str("variational_exponential");
    p["order"] = py::int_(1);
    p["shots"] = py::int_(50);
    p["warmups"] = py::int_(0);
    EXPECT_NO_THROW(sim.execute_analog_evolution(py::globals()["_te_var_samp"], py::globals()["_readout_var_samp"], py::none(), p));
}

TEST_F(ExecuteTimeEvolutionTest, IntegrateRK4Method_NoNoise_Succeeds) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.analog.schedule import Schedule
from qilisdk.analog.hamiltonian import Z
from qilisdk.core.qtensor import QTensor
import scipy.sparse as sp, numpy as np

_sched_rk4 = Schedule(hamiltonians={"h0": Z(0)}, dt=0.1, total_time=0.3)
_rho_rk4 = QTensor(sp.csr_matrix(np.array([[1.+0j,0],[0,0]], dtype=complex)))
_te_rk4 = AnalogEvolution(schedule=_sched_rk4, initial_state=_rho_rk4)
    )");
    py::dict p;
    p["evolution_method"] = py::str("integrate_rk4");
    EXPECT_NO_THROW(sim.execute_analog_evolution(py::globals()["_te_rk4"], py::list(), py::none(), p));
}

TEST_F(ExecuteTimeEvolutionTest, HamiltonianCountMismatch_IntegrateRK4_ThrowsValueError) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.analog.hamiltonian import Z
from qilisdk.core.qtensor import QTensor
import scipy.sparse as sp, numpy as np

_rho_mm_rk4 = QTensor(sp.csr_matrix(np.array([[1.+0j,0],[0,0]], dtype=complex)))

class _WeirdHams_RK4:
    def keys(self): return ["h0"]
    def values(self): return [Z(0), Z(0)]

class _FakeSched_MM_RK4:
    nqubits = 1
    tlist = [0.1, 0.2, 0.3]
    coefficients = {"h0": {0.1: 1.0, 0.2: 1.0, 0.3: 1.0}}
    hamiltonians = _WeirdHams_RK4()
    def get_parameters(self): return {}
    def set_parameters(self, _): pass

class _TE_MM_RK4(AnalogEvolution):
    def __init__(self):
        object.__init__(self)
        self.schedule = _FakeSched_MM_RK4()
        self._initial_state = _rho_mm_rk4
        self.store_intermediate_results = False

_te_mm_rk4 = _TE_MM_RK4()
    )");
    py::dict p;
    p["evolution_method"] = py::str("integrate_rk4");
    EXPECT_THROW(sim.execute_analog_evolution(py::globals()["_te_mm_rk4"], py::list(), py::none(), p), py::value_error);
}

TEST_F(ExecuteTimeEvolutionTest, UnknownEvolutionMethod_ThrowsValueError) {
    py::gil_scoped_acquire gil;
    py::exec(R"(
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.analog.hamiltonian import Z
from qilisdk.core.qtensor import QTensor
import scipy.sparse as sp, numpy as np

_rho_unknown = QTensor(sp.csr_matrix(np.array([[1.+0j,0],[0,0]], dtype=complex)))

class _Hams_Unknown:
    def keys(self): return ["h0"]
    def values(self): return [Z(0)]

class _FakeSched_Unknown:
    nqubits = 1
    tlist = [0.1, 0.2, 0.3]
    coefficients = {"h0": {0.1: 1.0, 0.2: 1.0, 0.3: 1.0}}
    hamiltonians = _Hams_Unknown()
    def get_parameters(self): return {}
    def set_parameters(self, _): pass

class _TE_Unknown(AnalogEvolution):
    def __init__(self):
        object.__init__(self)
        self.schedule = _FakeSched_Unknown()
        self._initial_state = _rho_unknown
        self.store_intermediate_results = False

_te_unknown = _TE_Unknown()
    )");
    py::dict p;
    p["evolution_method"] = py::str("not_a_real_method");
    EXPECT_THROW(sim.execute_analog_evolution(py::globals()["_te_unknown"], py::list(), py::none(), p), py::value_error);
}

// GCOV_EXCL_BR_STOP
