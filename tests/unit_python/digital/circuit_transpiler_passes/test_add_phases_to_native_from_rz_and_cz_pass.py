import numpy as np
import pytest

from qilisdk.digital import Circuit, Rmw
from qilisdk.digital.circuit_transpiler_passes import (
    AddPhasesToNativeFromRZAndCZPass,
    NativeSingleQubitGateBasis,
    TwoQubitGateBasis,
)
from qilisdk.digital.circuit_transpiler_passes.numeric_helpers import _wrap_angle
from qilisdk.digital.gates import CZ, RZ, Gate, M, X


@pytest.fixture(name="phase_correction_provider")
def fixture_phase_correction_provider():
    """A correction-provider analogous to the qililab DigitalCompilationSettings fixture.

    - CZ(0,1): control=0 -> 0.1, target=1 -> 0.2 (and symmetric for CZ(1,0))
    - CZ(0,2): control=0 -> 0.0 (no usable correction), target=2 -> 0.1
              (exercises the no-correction-on-one-side branch)
    """
    table = {
        (0, 1): (0.1, 0.2),
        (1, 0): (0.2, 0.1),
        (0, 2): (0.0, 0.1),
    }

    def provider(control: int, target: int) -> tuple[float, float]:
        return table.get((control, target), (0.0, 0.0))

    return provider


class TestAddPhasesToNativeFromRZAndCZ:
    def test_run_updated_behavior(self, phase_correction_provider) -> None:
        """End-to-end test:
        - phases from RZ and CZ are accumulated and applied to subsequent Rmw pulses
        - RZ gates are removed (virtualized)
        - shift/frame is NOT reset after an Rmw
        - CZ(0,2) provides no q0 correction, exercising that branch
        """
        step = AddPhasesToNativeFromRZAndCZPass(phase_correction_provider=phase_correction_provider)

        test_gates: list[Gate] = [
            Rmw(0, theta=1, phase=1),
            CZ(0, 1),
            RZ(1, phi=0.6),
            M(0),
            RZ(0, phi=0.3),
            Rmw(0, theta=3, phase=0),
            CZ(0, 2),
            CZ(1, 0),
            Rmw(1, theta=2, phase=-2),
            RZ(1, phi=0),
        ]

        expected_gates: list[Gate] = [
            Rmw(0, theta=1, phase=1),
            CZ(0, 1),
            M(0),
            Rmw(0, theta=3, phase=-0.4),
            CZ(0, 2),
            CZ(1, 0),
            Rmw(1, theta=2, phase=-3.0),
        ]

        circuit = Circuit(3)
        for g in test_gates:
            circuit.add(g)

        transpiled = step.run(circuit)
        assert len(transpiled.gates) == len(expected_gates), "RZ must be virtualized/removed"

        for g_exp, g_tr in zip(expected_gates, transpiled.gates):
            assert g_exp.name == g_tr.name
            assert g_exp.qubits == g_tr.qubits
            if isinstance(g_exp, Rmw):
                assert np.isclose(g_exp.theta, g_tr.theta)
                assert np.isclose(
                    _wrap_angle(g_exp.phase),
                    _wrap_angle(g_tr.phase),
                )

    def test_frame_persists_across_multiple_rmw(self, phase_correction_provider) -> None:
        step = AddPhasesToNativeFromRZAndCZPass(phase_correction_provider=phase_correction_provider)
        c = Circuit(1)
        c.add(RZ(0, phi=0.5))
        c.add(Rmw(0, theta=1.0, phase=0.0))
        c.add(Rmw(0, theta=1.0, phase=0.0))

        out = step.run(c)
        assert len(out.gates) == 2
        assert all(isinstance(g, Rmw) for g in out.gates)
        phases = [g.phase for g in out.gates]
        assert np.allclose(phases, [-0.5, -0.5])

    def test_cz_corrections_applied_to_both_qubits(self, phase_correction_provider) -> None:
        step = AddPhasesToNativeFromRZAndCZPass(phase_correction_provider=phase_correction_provider)
        c = Circuit(2)
        c.add(CZ(0, 1))
        c.add(Rmw(0, theta=0.1, phase=0.0))
        c.add(Rmw(1, theta=0.1, phase=0.0))

        out = step.run(c)
        assert [g.name for g in out.gates] == ["CZ", "Rmw", "Rmw"]
        assert np.isclose(out.gates[1].phase, -0.1)
        assert np.isclose(out.gates[2].phase, -0.2)

    def test_cz_with_partial_corrections(self, phase_correction_provider) -> None:
        step = AddPhasesToNativeFromRZAndCZPass(phase_correction_provider=phase_correction_provider)
        c = Circuit(3)
        c.add(CZ(0, 2))  # provider returns (0.0, 0.1)
        c.add(Rmw(0, theta=0.3, phase=0.05))
        c.add(Rmw(2, theta=0.3, phase=-0.07))

        out = step.run(c)
        assert [g.name for g in out.gates] == ["CZ", "Rmw", "Rmw"]
        assert np.isclose(out.gates[1].phase, 0.05)
        assert np.isclose(out.gates[2].phase, -0.07 - 0.1)

    def test_default_provider_is_noop(self) -> None:
        step = AddPhasesToNativeFromRZAndCZPass()
        c = Circuit(2)
        c.add(CZ(0, 1))
        c.add(Rmw(0, theta=0.1, phase=0.0))
        c.add(Rmw(1, theta=0.1, phase=0.0))

        out = step.run(c)
        assert [g.name for g in out.gates] == ["CZ", "Rmw", "Rmw"]
        assert np.isclose(out.gates[1].phase, 0.0)
        assert np.isclose(out.gates[2].phase, 0.0)

    def test_trailing_rz_is_removed_and_m_is_untouched(self, phase_correction_provider) -> None:
        step = AddPhasesToNativeFromRZAndCZPass(phase_correction_provider=phase_correction_provider)
        c = Circuit(3)
        c.add(RZ(2, phi=0.5))
        c.add(M(2))
        c.add(RZ(2, phi=0.0))

        out = step.run(c)
        assert len(out.gates) == 1
        assert isinstance(out.gates[0], M)
        assert out.gates[0].qubits == (2,)

    def test_unsupported_gate_raises(self, phase_correction_provider) -> None:
        step = AddPhasesToNativeFromRZAndCZPass(phase_correction_provider=phase_correction_provider)
        c = Circuit(1)
        c.add(X(0))

        with pytest.raises(ValueError, match="Unsupported gate X"):
            step.run(c)

    def test_rejects_invalid_basis_arguments(self) -> None:
        with pytest.raises(TypeError):
            AddPhasesToNativeFromRZAndCZPass(single_qubit_basis="Rmw")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            AddPhasesToNativeFromRZAndCZPass(two_qubit_basis="CZ")  # type: ignore[arg-type]

    def test_exposes_configured_basis_properties(self) -> None:
        p = AddPhasesToNativeFromRZAndCZPass(
            single_qubit_basis=NativeSingleQubitGateBasis.Rmw,
            two_qubit_basis=TwoQubitGateBasis.CZ,
        )
        assert p.single_qubit_basis == NativeSingleQubitGateBasis.Rmw
        assert p.two_qubit_basis == TwoQubitGateBasis.CZ
