# QiliSDK

[![Python Versions](https://img.shields.io/pypi/pyversions/qilisdk.svg)](https://pypi.org/project/qilisdk/)
[![PyPI Version](https://img.shields.io/pypi/v/qilisdk.svg)](https://pypi.org/project/qilisdk/)
[![License](https://img.shields.io/pypi/l/qilisdk.svg)](#license)
[![Docs](https://img.shields.io/badge/docs-latest-pink.svg)](https://qilimanjaro-tech.github.io/qilisdk/main/index.html)

**QiliSDK** is a Python framework for writing digital and analog quantum algorithms and executing them across multiple quantum backends. Its modular design streamlines the development process and enables easy integration with a variety of quantum platforms.

---

## Table of Contents
- [QiliSDK](#qilisdk)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Base Installation](#base-installation)
    - [Optional Dependencies: speqtrum, cuda, qutip](#optional-dependencies-speqtrum-cuda-qutip)
  - [Usage](#usage)
    - [Digital Quantum Circuits](#digital-quantum-circuits)
    - [Hamiltonians](#hamiltonians)
    - [Optimizers](#optimizers)
    - [Qilimanjaro SpeQtrum](#qilimanjaro-speqtrum)
    - [Sample a quantum Circuit Using a CUDA-Accelerated Simulator](#sample-a-quantum-circuit-using-a-cuda-accelerated-simulator)
    - [Time Evolution using Qutip](#time-evolution-using-qutip)
    - [Variational Programs](#variational-programs)
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

### Optional Dependencies: speqtrum, cuda, qutip

QiliSDK supports optional modules for additional functionality:

- **SpeQtrum:**
  To interface with Qilimanjaro’s cloud-based quantum services, install the Speqtrum optional dependency:

  ```bash
  pip install qilisdk[speqtrum]
  ```

- **CUDA Accelerated Simulator Backend:**
  For GPU-accelerated quantum simulation using NVIDIA GPUs, install the CUDA extra:

  ```bash
  pip install qilisdk[cuda]
  ```

- **Qutip Simulator Backend:**
  For GPU-accelerated quantum simulation using NVIDIA GPUs, install the CUDA extra:

  ```bash
  pip install qilisdk[qutip]
  ```

   **Note:**  You can also install both optional dependencies using a single command:

  ```bash
  pip install qilisdk[cuda,qutip,speqtrum]
  ```

---

## Usage

QiliSDK is designed to simplify both digital and analog quantum computing workflows. This guide provides example code snippets for creating quantum circuits, performing Hamiltonian arithmetic, optimization, simulation on various backends (including CUDA and cloud-based using Speqtrum), and using additional utility functions.

### Digital Quantum Circuits

Create and simulate quantum circuits with a flexible gate framework. The following example demonstrates how to assemble a circuit, adjust gate parameters:

```python
import numpy as np
from qilisdk.digital import Circuit, H, RX, CNOT, M

# Create a circuit with 2 qubits
circuit = Circuit(2)
circuit.add(H(0))  # Apply Hadamard on qubit 0
circuit.add(RX(0, theta=np.pi))  # Apply RX rotation on qubit 0
circuit.add(CNOT(0, 1))  # Add a CNOT gate between qubit 0 and 1

# Retrieve the current gate parameters
print("Initial parameters:", circuit.get_parameter_values())

# Update circuit parameters (e.g., update RX rotation angle)
circuit.set_parameter_values([2 * np.pi])
```

### Hamiltonians

Utilize the `Hamiltonian` class for Pauli operator arithmetic. Build expressions, iterate through terms:

```python
from qilisdk.analog import Hamiltonian, X, Y, Z, I

# Build a Hamiltonian expression using Pauli operators
H_expr = (Z(0) + X(0)) * (Z(0) - X(0))
print("Hamiltonian Expression:", H_expr)

# Iterate over Hamiltonian terms
for coeff, ops in H_expr:
    print("Coefficient:", coeff, "Operators:", ops)

print("-"*10)
# export hamiltonians to matrices
print("Sparse Matrix from Hamiltonian: \n", H_expr.to_matrix()) # sparse matrix  
# the returned matrix is sparse by default to get the dense representation call .toarray()
```


### Optimizers

Run optimization routines using the `SciPyOptimizer` and integrate them with a variational algorithm (VQE). In the example below, a simple quadratic cost function is minimized:

```python
from qilisdk.optimizers import SciPyOptimizer


def cost_function(params):
    # A simple quadratic cost: minimum at [1, 1, 1]
    return sum((p - 1) ** 2 for p in params)


# Initialize the optimizer with the BFGS method
optimizer = SciPyOptimizer(method="BFGS")
initial_parameters = [0, 0, 0]
parameter_bounds = [(0, 1), (0, 1), (0, 1)]
result = optimizer.optimize(cost_function, initial_parameters, parameter_bounds)

print("Optimal cost:", result.optimal_cost)
print("Optimal Parameters:", result.optimal_parameters)
```

### Qilimanjaro SpeQtrum

QiliSDK includes a client for interacting with Qilimanjaro's SpeQtrum platform. This module supports secure login and a unified interface for both digital circuits and analog evolutions:

```python
from qilisdk.backends import CudaBackend, CudaSamplingMethod
from qilisdk.digital import Circuit, H, M
from qilisdk.functionals import Sampling
from qilisdk.speqtrum import SpeQtrum


# Build a single-qubit circuit
circuit = Circuit(nqubits=1)
circuit.add(H(0))
circuit.add(M(0))

# Login to QaaSBackend with credentials (or use environment variables)
# This only needs to be run once.
SpeQtrum.login(username="YOUR_USERNAME", apikey="YOUR_APIKEY")

# Instantiate QaaSBackend
client = SpeQtrum()

# Execute a pre-built circuit (see Digital Quantum Circuits section)
# make sure to select the device (you can list available devices using ``client.list_devices()``)
job_id = client.submit(Sampling(circuit, 1000), device="SELECTED_DEVICE")
print("job id:", job_id)
print("job status:", client.get_job(job_id).status)
print("job result:", client.get_job(job_id).result)
```

### Sample a quantum Circuit Using a CUDA-Accelerated Simulator

For users with NVIDIA GPUs, the `CudaBackend` provides GPU-accelerated simulation using several simulation methods. For example, using the TENSOR_NETWORK method:

```python
from qilisdk.backends import CudaBackend, CudaSamplingMethod
from qilisdk.digital import Circuit, H, M
from qilisdk.functionals import Sampling

# Build a single-qubit circuit
circuit = Circuit(nqubits=1)
circuit.add(H(0))
circuit.add(M(0))

# Initialize CudaBackend with the TENSOR_NETWORK simulation method
backend = CudaBackend(sampling_method=CudaSamplingMethod.TENSOR_NETWORK)
results = backend.execute(Sampling(circuit))
print("CUDA Backend Results:", results)
```

### Time Evolution using Qutip

For analog simulations, the `TimeEvolution` and unified `Schedule` classes allow you to simulate time-dependent quantum dynamics. The following example uses callable coefficients defined over an interval to interpolate between two Hamiltonians on a Qutip backend:

```python
from qilisdk.analog import Schedule, X, Z, Y
from qilisdk.core import ket, tensor_prod
from qilisdk.backends import QutipBackend
from qilisdk.functionals import TimeEvolution

# Define total time and timestep
T = 100.0
dt = 0.1
nqubits = 1

# Define Hamiltonians
Hx = sum(X(i) for i in range(nqubits))
Hz = sum(Z(i) for i in range(nqubits))

# Build a time‑dependent schedule
schedule = Schedule(
    hamiltonians={"hx": Hx, "hz": Hz},
    coefficients={
        "hx": {(0.0, T): lambda t: 1 - t / T},
        "hz": {(0.0, T): lambda t: t / T},
    },
    dt=dt,
)

# draw the schedule
schedule.draw()

# Prepare an equal superposition initial state
initial_state = tensor_prod([(ket(0) - ket(1)).unit() for _ in range(nqubits)]).unit()

# Create the TimeEvolution functional
time_evolution = TimeEvolution(
    schedule=schedule,
    initial_state=initial_state,
    observables=[Z(0), X(0), Y(0)],
    nshots=100,
    store_intermediate_results=False,
)

# Execute on Qutip backend and inspect results
backend = QutipBackend()
results = backend.execute(time_evolution)
print(results)
```

### Variational Programs

The `VariationalProgram` allows the user to optimized parameterized functionals, including `Sampling` and `TimeEvolution`.


Here you find an example of building a Variational Quantum Eigensolver (VQE). To do this we need a circuit ansatz, cost function to optimize, and classical optimizer. Below is an illustrative example that sets up a VQE instance using a hardware-efficient ansatz and the SciPy optimizer:

```python
from qilisdk.digital.gates import U2, CNOT
from qilisdk.optimizers import SciPyOptimizer
from qilisdk.core.model import Model
from qilisdk.core.variables import LEQ, BinaryVariable
from qilisdk.digital.ansatz import HardwareEfficientAnsatz
from qilisdk.functionals import VariationalProgram, Sampling
from qilisdk.backends import CudaBackend
from qilisdk.cost_functions import ModelCostFunction


## Define the model

model = Model("Knapsack")

values = [2, 3, 7]
weights = [1, 3, 3]
max_weight = 4
b = [BinaryVariable(f"b{i}") for i in range(len(values))]

obj = sum(b[i] * values[i] for i in range(len(values)))
model.set_objective(obj, "obj")

con = LEQ(sum(b[i] * weights[i] for i in range(len(weights))), max_weight)

model.add_constraint("max_weight", con)

## Define the Ansatz:
nqubits = 3
ansatz = HardwareEfficientAnsatz(
    nqubits=nqubits, layers=2, connectivity="Linear", structure="grouped", one_qubit_gate=U2, two_qubit_gate=CNOT
)

## Define the Optimizer
optimizer = SciPyOptimizer(method="Powell")

## Build the VQE object
vqe = VariationalProgram(
    functional=Sampling(ansatz),
    optimizer=optimizer,
    cost_function=ModelCostFunction(model),
)

## Define the Backend 
backend = CudaBackend()

## Execute the VQE to find the optimal parameters
result = backend.execute(vqe)

## Sample the circuit using the optimal parameters
ansatz.set_parameter_values(result.optimal_parameters)
results = backend.execute(Sampling(ansatz))

## Print the probabilities
print(results.get_probabilities())
```

### Open QASM Serialization

Serialize and deserialize quantum circuits using Open QASM 2.0 grammar. The utility functions below allow conversion to a QASM string or file and vice versa:

```python
from qilisdk.digital import Circuit, CNOT, RX, H, M
from qilisdk.utils.openqasm2 import to_qasm2, from_qasm2, to_qasm2_file, from_qasm2_file

# Create a sample circuit
circuit = Circuit(3)
circuit.add(H(0))
circuit.add(CNOT(0, 1))
circuit.add(RX(2, theta=3.1415))
circuit.add(M(0, 1, 2))

print("Initial Circuit:")
circuit.draw()
# Serialize to QASM string
qasm_code = to_qasm2(circuit)
print("Generated QASM:")
print(qasm_code)

# Deserialize back to a circuit
reconstructed_circuit = from_qasm2(qasm_code)

# Save to and load from a file
to_qasm2_file(circuit, "circuit.qasm")
reconstructed_circuit = from_qasm2_file("circuit.qasm")
print()
print("Reconstructed Circuit:")
reconstructed_circuit.draw()


```

### YAML Serialization

Easily save and restore circuits, Hamiltonians, simulation and execution results, and virtually any other class in YAML format using the provided serialization functions:

```python
from qilisdk.digital import Circuit, H, CNOT, M
from qilisdk.utils.serialization import serialize, deserialize, serialize_to, deserialize_from

circuit = Circuit(2)
circuit.add(H(0))
circuit.add(CNOT(0, 1))
circuit.add(M(0, 1))

print("Initial Circuit:")
circuit.draw()

# Serialize a circuit to a YAML string
yaml_string = serialize(circuit)

# Deserialize back into a Circuit object
restored_circuit = deserialize(yaml_string, cls=Circuit)

# Alternatively, work with files:
serialize_to(circuit, "circuit.yml")
restored_circuit = deserialize_from("circuit.yml", cls=Circuit)
print()
print("Reconstructed Circuit:")
restored_circuit.draw()
```

### OpenFermion Integration

`QiliSDK` can translate ``QubitOperator`` objects from ``OpenFermion`` to ``QiliSDK``'s ``Hamiltonian`` Objects and vice versa.

This code is available under an optional dependency that can be installed using ``pip install qilisdk[openfermion]``.

here is an example of the usage: 
```python
from openfermion.hamiltonians import jellium_model
from openfermion.transforms import fourier_transform, jordan_wigner
from openfermion.utils import Grid

from qilisdk.utils.openfermion import openfermion_to_qilisdk, qilisdk_to_openfermion

# Let's look at a very small model of jellium in 1D.
grid = Grid(dimensions=1, length=3, scale=1.0)
spinless = True

# Get the momentum Hamiltonian.
momentum_hamiltonian = jellium_model(grid, spinless)
momentum_qubit_operator = jordan_wigner(momentum_hamiltonian)
momentum_qubit_operator.compress()

# Fourier transform the Hamiltonian to the position basis.
position_hamiltonian = fourier_transform(momentum_hamiltonian, grid, spinless)
position_qubit_operator = jordan_wigner(position_hamiltonian)
position_qubit_operator.compress()

qilisdk_ham = openfermion_to_qilisdk(position_qubit_operator)
openfermion_ham = qilisdk_to_openfermion(qilisdk_ham)

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
   - To install extra dependencies such as `CudaBackend`, run:
     ```bash
     uv sync --extra cuda -extra ...
     ```
     This sets up a virtual environment and installs all pinned dependencies (previous), plus the specified extras.
     you can also install all the optional dependencies and groups by running;
     ```bash
     uv sync --all-extras --all-groups
     ```

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
Added a new `cool_feature` in the `qilisdk.backend` module.
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
