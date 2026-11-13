# QiliSDK

A framework for writing digital, analog and hybrid quantum algorithms once and running them on any backend: a program is a functional that a backend executes, and a readout says what to measure.

Terms shared across Qilimanjaro's stack are in the [shared glossary](https://github.com/qilimanjaro-tech/claude-skills/blob/main/onboarding/qilimanjaro-agent-starter-kit/org-context/qilimanjaro-glossary.md); this file holds QiliSDK's own. The module pages under `docs/modules/` explain each term in depth.

## Language

### Execution

**Functional**:
An executable routine that a backend runs and that returns a result: `DigitalPropagation`, `AnalogEvolution`, `VariationalProgram` or `QuantumReservoir`.
_Avoid_: Job, program

**Backend**:
What executes a functional with a readout: QiliSim, or the optional CUDA-Q and QuTiP backends.
_Avoid_: Device, runner

**QiliSim**:
QiliSDK's own backend, a simulator written in C++ that runs digital circuits and analog time evolution on CPU or GPU.

**Readout**:
What to extract from an execution: samples, expectation values of observables, or the state.
_Avoid_: Measurement (a measurement is a gate in a circuit)

**Functional result**:
The container for the outputs a backend returns after executing a functional.

### Digital

**Circuit**:
An ordered list of gates acting on a fixed number of qubits.

**Gate**:
A quantum operation in a circuit, possibly parameterized and possibly controlled.

**Ansatz**:
A template for a parameterized circuit, used by variational algorithms.

**Transpiler pass**:
One step of the circuit transpiler, which rewrites a circuit into a new one (layout, routing, decomposition, gate fusion).

### Analog

**Hamiltonian**:
A linear combination of Pauli operators.

**Schedule**:
A set of time-dependent coefficients applied to a collection of Hamiltonians, which an analog evolution follows.
_Avoid_: Annealing path, pulse

### Problems and variables

**Model**:
An optimization problem: an objective plus constraints over variables, which converts into quantum jobs.

**QUBO**:
A model in quadratic unconstrained binary optimization form.

**Variable**:
An optimization variable of a model, continuous or binary, with a configurable binary encoding.
_Avoid_: Parameter (a symbol you set or optimize)

**Parameter**:
A symbolic scalar that parameterizes gates, Hamiltonians and schedules, and that an optimizer can tune.
_Avoid_: Variable (a decision variable of a model)

**QTensor**:
QiliSDK's representation of a quantum state or operator, as a sparse matrix with quantum-information utilities.
_Avoid_: State vector, density matrix (both are QTensors)

### Variational and noise

**Variational program**:
A parameterized functional, an optimizer and a cost function, bundled into a variational loop.

**Cost function**:
What maps a functional result to the scalar a variational program minimizes.

**Noise model**:
A container of noise sources and parameter perturbations applied when a backend executes a functional.

**Quantum reservoir**:
A functional that applies a sequence of inputs to a reservoir layer, running one pass of dynamics and measurement per input.

### Services

**SpeQtrum**:
Qilimanjaro's cloud for running experiments on remote devices; QiliSDK ships an optional client that authenticates, inspects devices and jobs, and submits work.
