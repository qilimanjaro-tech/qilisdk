# QiliSDK

[![Python Versions](https://img.shields.io/pypi/pyversions/qilisdk.svg)](https://pypi.org/project/qilisdk/)
[![PyPI Version](https://img.shields.io/pypi/v/qilisdk.svg)](https://pypi.org/project/qilisdk/)
[![License](https://img.shields.io/pypi/l/qilisdk.svg)](#license)

**QiliSDK** is a Python framework for writing digital and analog quantum algorithms and executing them across multiple quantum backends. Its modular design streamlines the development process and enables easy integration with a variety of quantum platforms.

---

## Table of Contents
- [QiliSDK](#qilisdk)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Base Installation](#base-installation)
    - [Optional Extras](#optional-extras)
  - [Usage](#usage)
    - [Digital Quantum Circuits](#digital-quantum-circuits)
    - [Hamiltonian and Analog Operations](#hamiltonian-and-analog-operations)
    - [Optimizers](#optimizers)
    - [Quantum-as-a-Service (QaaS)](#quantum-as-a-service-qaas)
    - [CUDA-Accelerated Simulation](#cuda-accelerated-simulation)
    - [Time Evolution](#time-evolution)
    - [Variational Quantum Eigensolver (VQE)](#variational-quantum-eigensolver-vqe)
    - [Open QASM Serialization](#open-qasm-serialization)
    - [YAML Serialization](#yaml-serialization)
  - [Development](#development)
    - [Prerequisites](#prerequisites)
    - [Setup \& Dependency Management](#setup--dependency-management)
    - [Testing](#testing)
    - [Linting \& Formatting](#linting--formatting)
    - [Type Checking](#type-checking)
    - [Changelog Management](#changelog-management)
    - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

---

## Installation

QiliSDK is available via [PyPI](https://pypi.org/project/qilisdk/). You can install the core package as well as optional extras for additional features.

### Base Installation

Install the core QiliSDK package using pip:

```bash
pip install qilisdk
```

### Optional Extras

QiliSDK supports optional modules for additional functionality:

- **QaaS (Quantum-as-a-Service):**
  To interface with Qilimanjaro’s cloud-based quantum services, install the QaaS extra:

  ```bash
  pip install qilisdk[qaas]
  ```

- **CUDA Acceleration:**
  For GPU-accelerated quantum simulation using NVIDIA GPUs, install the CUDA extra:

  ```bash
  pip install qilisdk[cuda]
  ```

  You can also install both optional dependencies using a single command:

  ```bash
  pip install qilisdk[cuda,qaas]
  ```

---

## Usage

QiliSDK is designed to simplify both digital and analog quantum computing workflows. This guide provides example code snippets for creating quantum circuits, performing Hamiltonian arithmetic, optimization, simulation on various backends (including CUDA and cloud-based QaaS), and using additional utility functions.

### Digital Quantum Circuits

Create and simulate quantum circuits with a flexible gate framework. The following example demonstrates how to assemble a circuit, adjust gate parameters, and execute it with the Qibo backend:

```python
import numpy as np
from qilisdk.digital import Circuit, H, RX, CNOT, M

# Create a circuit with 2 qubits
circuit = Circuit(2)
circuit.add(H(0))              # Apply Hadamard on qubit 0
circuit.add(RX(0, theta=np.pi))  # Apply RX rotation on qubit 0
circuit.add(CNOT(0, 1))          # Add a CNOT gate between qubit 0 and 1

# Retrieve the current gate parameters
print("Initial parameters:", circuit.get_parameter_values())

# Update circuit parameters (e.g., update RX rotation angle)
circuit.set_parameter_values([2 * np.pi])

# Execute the circuit simulation using CudaBackend
backend = CudaBackend()
results = backend.execute(circuit)

# Display the simulation output and measurement probabilities
print("Simulation Results:")
print(results)
print("Probabilities:", results.probabilities)
```

### Hamiltonian and Analog Operations

Utilize the `Hamiltonian` class for Pauli operator arithmetic. Build expressions, iterate through terms, and even round-trip using the parse feature:

```python
from qilisdk.analog import Hamiltonian, X, Y, Z, I

# Build a Hamiltonian expression using Pauli operators
H_expr = (Z(0) + X(0)) * (Z(0) - X(0))
print("Hamiltonian Expression:", H_expr)

# Iterate over Hamiltonian terms
for coeff, ops in H_expr:
    print("Coefficient:", coeff, "Operators:", ops)

# Parse a Hamiltonian from its string representation
parsed_H = Hamiltonian.parse("-2j Y(0)")
assert H_expr == parsed_H, "The Hamiltonian does not match after parsing."
```

### Optimizers

Run optimization routines using the `SciPyOptimizer` and integrate them with a variational algorithm (VQE). In the example below, a simple quadratic cost function is minimized:

```python
from qilisdk.common import SciPyOptimizer

def cost_function(params):
    # A simple quadratic cost: minimum at [1, 1, 1]
    return sum((p - 1) ** 2 for p in params)

# Initialize the optimizer with the BFGS method
optimizer = SciPyOptimizer(method="BFGS")
initial_parameters = [0, 0, 0]
result = optimizer.optimize(cost_function, initial_parameters)

print("Optimal cost:", result.optimal_cost)
print("Optimal Parameters:", result.optimal_parameters)
```

### Quantum-as-a-Service (QaaS)

QiliSDK now includes a draft backend for interfacing with Qilimanjaro's QaaS platform. This module supports secure login and a unified interface for both digital circuits and analog evolutions:

```python
from qilisdk.extras import QaaSBackend

# Login to QaaSBackend with credentials (or use environment variables)
# This only needs to be run once.
QaaSBackend.login(username="your_username", apikey="your_apikey")

# Instantiate QaaSBackend
qaas_backend = QaaSBackend()

# Execute a pre-built circuit (see Digital Quantum Circuits section)
results = qaas_backend.execute(circuit)
print("QaaS Simulation Results:", results)
```

### CUDA-Accelerated Simulation

For users with NVIDIA GPUs, the `CudaBackend` provides GPU-accelerated simulation using several simulation methods. For example, using the TENSOR_NETWORK method:

```python
from qilisdk.extras import CudaBackend
from qilisdk.digital import Circuit, H, M, DigitalSimulationMethod

# Build a single-qubit circuit
circuit = Circuit(nqubits=1)
circuit.add(H(0))
circuit.add(M(0))

# Initialize CudaBackend with the TENSOR_NETWORK simulation method
cuda_backend = CudaBackend(digital_simulation_method=DigitalSimulationMethod.TENSOR_NETWORK)
results = cuda_backend.execute(circuit, nshots=1000)
print("CUDA Backend Results:", results)
```

### Time Evolution

For analog simulations, the new `TimeEvolution` and `Schedule` classes allow you to simulate time-dependent quantum dynamics. The following example uses a linear schedule to interpolate between two Hamiltonians on a CUDA backend:

```python
import numpy as np
from qilisdk.analog import TimeEvolution, Schedule, tensor, ket, X, Z, Y
from qilisdk.extras import CudaBackend

T = 10       # Total evolution time
dt = 0.1     # Time-step
steps = np.linspace(0, T, int(T / dt))
nqubits = 1

# Define two Hamiltonians for the simulation
H1 = sum(X(i) for i in range(nqubits))
H2 = sum(Z(i) for i in range(nqubits))

# Create a schedule for the time evolution
schedule = Schedule(
    T,
    dt,
    hamiltonians={"h1": H1, "h2": H2},
    schedule={
        t: {"h1": 1 - steps[t] / T, "h2": steps[t] / T}
        for t in range(len(steps))
    },
)

# Prepare an initial state (equal superposition)
state = tensor([(ket(0) + ket(1)).unit() for _ in range(nqubits)]).unit()

# Perform time evolution on the CUDA backend with observables to monitor
time_evolution = TimeEvolution(
    backend=CudaBackend(),
    schedule=schedule,
    initial_state=state,
    observables=[Z(0), X(0), Y(0)],
)
results = time_evolution.evolve(store_intermediate_results=True)
print("Time Evolution Results:", results)
```

### Variational Quantum Eigensolver (VQE)

The VQE algorithm integrates ansatz design, cost function evaluation, and classical optimization. Below is an illustrative example that sets up a VQE instance using a hardware-efficient ansatz and the SciPy optimizer:

```python
import numpy as np
from qilisdk.common import SciPyOptimizer
from qilisdk.digital import HardwareEfficientAnsatz, VQE
from qilisdk.extras import CudaBackend


# Define problem parameters
n_items = 4
weights = [np.random.randint(1, 5) for _ in range(n_items)]
values  = [np.random.randint(1, 10) for _ in range(n_items)]
max_weight_perc = 0.6
max_w = int(max_weight_perc * sum(weights))
nqubits = n_items

# Initialize a hardware-efficient ansatz
ansatz = HardwareEfficientAnsatz(
    n_qubits=nqubits,
    connectivity="Full",
    layers=1,
    one_qubit_gate="U3",
    two_qubit_gate="CNOT"
)

# Use a CUDA backend for simulation
backend = CudaBackend()

LM = sum(values)

def cost_function(result):
    # Get the most probable outcomes from the digital result
    most_probable = result.get_probabilities()
    final_cost = 0
    for bitstring, prob in most_probable:
        x_n = [int(bit) for bit in bitstring]
        total_weight = sum(weights[i] * x_n[i] for i in range(n_items))
        penalty = (total_weight - max_w) if total_weight > max_w else 0
        reward = -sum(values[i] * x_n[i] for i in range(n_items))
        final_cost += prob * (LM * penalty + reward)
    return final_cost

optimizer = SciPyOptimizer(
    method="Powell",
    bounds=[(0, np.pi)] * ansatz.nparameters
)

# Initialize and execute the VQE algorithm
vqe = VQE(ansatz, [0.5] * ansatz.nparameters, cost_function)
results = vqe.execute(backend, optimizer, store_intermediate_results=True)

print("Optimal Cost:", results.optimal_cost)
print("Optimal Parameters:", results.optimal_parameters)
print("Intermediate Optimization Steps:")
for intermediate in results.intermediate_results:
    print("Cost:", intermediate.cost)
```

### Open QASM Serialization

Serialize and deserialize quantum circuits using Open QASM 2.0 grammar. The utility functions below allow conversion to a QASM string or file and vice versa:

```python
from qilisdk.digital import Circuit, CNOT, RX, H, M
from qilisdk.utils import to_qasm2, from_qasm2, to_qasm2_file, from_qasm2_file

# Create a sample circuit
circuit = Circuit(3)
circuit.add(H(0))
circuit.add(CNOT(0, 1))
circuit.add(RX(2, theta=3.1415))
circuit.add(M(0, 1, 2))

# Serialize to QASM string
qasm_code = to_qasm2(circuit)
print("Generated QASM:")
print(qasm_code)

# Deserialize back to a circuit
reconstructed_circuit = from_qasm2(qasm_code)

# Save to and load from a file
to_qasm2_file(circuit, "circuit.qasm")
reconstructed_circuit = from_qasm2_file("circuit.qasm")
```

### YAML Serialization

Easily save and restore circuits, hamiltonians, simulation and execution results, and virtually any other class in YAML format using the provided serialization functions:

```python
from qilisdk.digital import Circuit, H, CNOT, M
from qilisdk.utils import serialize, deserialize, serialize_to, deserialize_from

circuit = Circuit(2)
circuit.add(H(0))
circuit.add(CNOT(0, 1))
circuit.add(M(0, 1))

# Serialize a circuit to a YAML string
yaml_string = serialize(circuit)

# Deserialize back into a Circuit object
restored_circuit = deserialize(yaml_string, cls=Circuit)

# Alternatively, work with files:
serialize_to(circuit, 'circuit.yml')
restored_circuit = deserialize_from('circuit.yml', cls=Circuit)
```

---

## Development

This section covers how to set up a local development environment for qilisdk, run tests, enforce code style, manage dependencies, and contribute to the project. We use a number of tools to maintain code quality and consistency:

- **[uv](https://pypi.org/project/uv/)** for dependency management and packaging.
- **[ruff](https://beta.ruff.rs/docs/)** for linting and code formatting.
- **[mypy](http://mypy-lang.org/)** for static type checking.
- **[towncrier](https://github.com/twisted/towncrier)** for automated changelog generation.

### Prerequisites

- Python **3.10+** (we test against multiple versions, but 3.10 is the minimum for local dev).
- [Git](https://git-scm.com/) for version control.
- [uv](https://pypi.org/project/uv/) for dependency management.

### Setup & Dependency Management

1. **Clone the repository**:
   ```bash
   git clone https://github.com/qilimanjaro-tech/qilisdk.git
   cd qilisdk
   ```

2. **Install [uv](https://pypi.org/project/uv/) globally** (if not already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Sync dependencies**:
   - We maintain a `pyproject.toml` listing all dev and optional requirements.
   - To install the dev environment locally, run:
     ```bash
     uv sync
     ```
     This sets up a virtual environment and installs all pinned dependencies (including `ruff`, `mypy`, `towncrier`, etc.).
   - To install extra dependencies such as `qibo-backend`, run:
     ```bash
     uv sync --extra qibo-backend -extra ...
     ```
     This sets up a virtual environment and installs all pinned dependencies (previous), plus the specified extras.

4. **Activate the virtual environment**:
   - uv typically creates and manages its own environment, e.g., `.venv/`.
   - Run:
     ```bash
     source .venv/bin/activate
     ```
     *(Exact command can vary depending on your shell and OS.)*

Now you can run all development commands (tests, linting, etc.) within this environment.

### Testing

TODO: to_be_filled

### Linting & Formatting

We enforce code style and best practices using [**ruff**](https://beta.ruff.rs/docs/). ruff handles:

- Lint checks (similar to flake8, pylint).
- Formatting (similar to black or isort).
- Automated fixes for certain issues.

To check linting:

```bash
ruff check
```

To automatically fix lint issues (where possible):

```bash
ruff check --fix
```

To automatically format your code:

```bash
ruff format
```

*(We recommend running `ruff check --fix` and `ruff format` before committing any changes.)*

### Type Checking

We use [**mypy**](http://mypy-lang.org/) for static type checking. This helps ensure our code is type-safe and maintainable.

```bash
mypy qilisdk
```

If you have extra modules or tests you want type-checked, specify them:

```bash
mypy qilisdk tests
```

*(We encourage developers to annotate new functions, classes, and methods with type hints.)*

### Changelog Management

We manage our changelog using [**towncrier**](https://github.com/twisted/towncrier). Instead of editing `CHANGELOG.md` directly, **each pull request** includes a small *news fragment* file in the `changes/` directory describing the user-facing changes.

For example, if you create a PR with id #123 adding a new feature, you add:
```
changes/123.feature.rst
```
Inside this file, you briefly describe the new feature:
```rst
Added a new `cool_feature` in the `qilisdk.extras` module.
```
Instead of manually creating the file, you can run:
```bash
towncrier create --no-edit
```
When we cut a new release, we update the version in `pyproject.toml` file and run:
```bash
towncrier
```
This aggregates all the news fragments into the `CHANGELOG.md` under the new version and removes the used fragments.

### Contributing

We welcome contributions! Here’s the workflow:

1. **Fork** this repository and create a feature branch.
2. **Write** your changes (code, docs, or tests).
3. **Add a news fragment** (if applicable) in `changes/` describing the user-facing impact.
4. **Run** the following checks locally:
   ```bash
   ruff check --fix
   ruff format
   mypy qilisdk
   pytest tests
   ```
5. **Commit** and push your branch to your fork. `pre-commit` will also run the checks automatically.
6. **Open a Pull Request** against the `main` branch here.

Our CI will run tests, linting, and type checks. Please make sure your branch passes these checks before requesting a review.

---

## License

This project is licensed under the [Apache License](LICENSE).

---

## Acknowledgments

- Thanks to all the contributors who help develop qilisdk!
- [uv](https://pypi.org/project/uv/) for making dependency management smoother.
- [ruff](https://beta.ruff.rs/docs/), [mypy](http://mypy-lang.org/), and [towncrier](https://github.com/twisted/towncrier) for their amazing tooling.

---

Feel free to open [issues](https://github.com/qilimanjaro-tech/qilisdk/issues) or [pull requests](https://github.com/qilimanjaro-tech/qilisdk/pulls) if you have questions or contributions. Happy coding!
