# Qilisdk 0.1.3 (2025-05-07)

### Bugfixes

- Made `pydantic` pass to be a mandatory requirement, and not only for qaas as before. Solving a problem with installation overseen in previous PRs. 

  ([PR #29](https://github.com/qilimanjaro-tech/qilisdk/pulls/29))

- Made several small changes to the `QuantumObject` class and logic. The two main changes are:
  - The first concerns the trace norm which was incorrectly implemented before.
  - The second concerns changing the modulus of 2 check for the Hilbert Space size, to a a pow(2) check.

  ([PR #30](https://github.com/qilimanjaro-tech/qilisdk/pulls/30))

- Solved problems with ``Cudaq`` backend:
  - Updated ``Cudaq`` to version 0.10.0 to fix issues encountered in version 0.9.1
  - Migrated ``CudaBackend`` to use the new version of ``Cudaq``

  ([PR #31](https://github.com/qilimanjaro-tech/qilisdk/pulls/31))

### Misc

- Transformed hardcoded `PUBLIC URL` into an environment variable lookup that defaults to the hardcoded value

  ([PR #32](https://github.com/qilimanjaro-tech/qilisdk/pulls/32))


# Qilisdk 0.1.2 (2025-04-22)

### Misc

- Improved `QaaSBacked` functionality to include methods for executing digital and analog algorithms.
  
  [PR #27](https://github.com/qilimanjaro-tech/qilisdk/pulls/27)


# Qilisdk 0.1.1 (2025-04-11)

### Misc

- Improved README documentation with comprehensive usage examples, and restructured module imports to expose core user-facing symbols for a more intuitive experience.

  [PR #25](https://github.com/qilimanjaro-tech/qilisdk/pulls/25)


# Qilisdk 0.1.0 (2025-04-10)

### Features

- Introduces a new Circuit class for assembling quantum gates, along with a QiboBackend for simulating the constructed circuits:

  - **Added Circuit class**
    - Enables creation and composition of multiple quantum gates in a structured way.
    - Offers parameter management with methods like get_parameter_values() and set_parameter_values(), allowing dynamic adjustment of rotation angles or other gate parameters.

  - **Implemented common quantum gates**
    - Single-qubit rotations (RX, RY, RZ)
    - Parameterizable gates (U1, U2, U3)
    - Two-qubit control gates (CNOT, CZ)
    - Single-qubit standard gates (X, Y, Z, H, S, T)
    - Measurement gate (M) for reading out qubit states.

  - **Introduced QiboBackend**
    - Integrates with the Qibo framework to simulate circuits defined using the Circuit class.
    - Returns simulation results via a new DigitalResults class with a user-friendly __repr__ for quick inspection.

  Below is a usage example demonstrating circuit creation, parameter updates, and simulation with Qibo:
  ```python
  import numpy as np
  from qilisdk.digital import CNOT, RX, U3, Circuit, H, M, X

  # Create circuit
  circuit = Circuit(2)
  circuit.add(H(0))
  circuit.add(RX(0, theta=np.pi))
  circuit.add(CNOT(0, 1))

  # Get circuit's parameters
  circuit.get_parameter_values() # returns [3.141592653589793]

  # Set circuit's parameters
  circuit.set_parameter_values([2 * np.pi])

  # Create QiboBackend and execute circuit
  backend = QiboBackend()
  results = backend.execute(circuit)

  # Print the results using the __repr__ method
  print(results)

  # Or access specific information using properties
  print(results.probabilities)
  ```

  ([PR #2](https://github.com/qilimanjaro-tech/qilisdk/pulls/2))

- Introduces the `Hamiltonian` class as a central component for Pauli-based operator arithmetic, with a flyweight pattern for single-qubit operators. Internally stores terms as a dictionary mapping tuples of `PauliOperator` objects to complex coefficients.

  **Key Features**

  1. **Flyweight Pauli Operators**
     - `Z`, `X`, `Y`, `I` constructors backed by a cache to avoid repeated instantiation of identical operators.
     - Concrete classes `PauliZ`, `PauliX`, `PauliY`, and `PauliI` handle core matrix definitions and qubit association.

  2. **Dictionary-Based Storage**
     - Each Hamiltonian term is keyed by a tuple of `PauliOperator` objects (e.g. `(Z(0), Y(1))`) associated with a complex coefficient.
     - Uses `defaultdict(complex)` for coefficient accumulation.

  3. **Arithmetic Operations**
     - Addition, subtraction, multiplication, and division with other Hamiltonians, individual Pauli operators, or scalars (`int`, `float`, `complex`).
     - Merges like terms automatically (e.g., `(Z(0) + Z(0))` becomes `2*Z(0)`).
     - Distributes terms for Hamiltonian–Hamiltonian multiplication, applying a Pauli product table to compute phases and resulting operators on a per-qubit basis.

  4. **Simplification**
     - Removes terms below a configurable threshold (`_EPS`).
     - Aggregates any single-qubit identity operators `I(q)` into `I(0)` for a canonical form.

  5. **Iteration Protocol**
     - Exposes `__iter__` returning `(coefficient, list_of_operators)` pairs, facilitating term-by-term processing without a separate parsing function.

  6. **Equality Check**
     - Implements `__eq__` to compare Hamiltonians and detect pure scalar cases (`I(0) * c`).
     - Recognizes an effectively “zero” Hamiltonian and treats it consistently.

  7. **String Representation**
     - Renders numeric coefficients in a concise format (omitting unnecessary decimals).
     - Correctly handles purely real or imaginary parts, negative signs, and identity terms.
     - Arranges single identity terms `I(0)` to appear first.

  8. **Parse Method**
     - Implements a `@classmethod` named `parse` that reconstructs a Hamiltonian from its string representation.
     - Allows round-trip conversions (str(ham) → parse(...) → ham) for saving, logging, or user input.

  ### Code Example

  ```python
  from qilisdk.analog import Hamiltonian, X, Y, Z, I

  # Create a Hamiltonian using Pauli Operators
  H = (Z(0) + X(0)) * (Z(0) - X(0))

  # Access each term
  for coeff, ops in product:
      print("Coefficient:", coeff, "Operators:", ops)

  # Check equality
  assert H == -2j * Y(0), "Both expressions yield the same Hamiltonian."

  print(H)  # -2j Y(0)

  # Parse a string to Hamiltonian
  parsed_hamiltonian = Hamiltonian.parse("-2j Y(0)")

  # Check equality
  assert H == parsed_hamiltonian, "Hamiltonians are equal"
  ```

  This release provides a robust framework for Pauli-operator arithmetic, scalar integration, and canonical simplification, forming a foundation for higher-level quantum analog functionality.

  ([PR #3](https://github.com/qilimanjaro-tech/qilisdk/pulls/3))

- Added the `Optimizer` abstract base class and its concrete subclass `SciPyOptimizer`. The `SciPyOptimizer` class wraps `scipy.optimize.minimize` to optimize cost functions while supporting extra keyword arguments such as Jacobian, bounds, etc. This implementation provides a structured way to perform optimization and access optimal parameters via the `optimal_parameters` property.

  ### Code Example

  ```python
  from optimizer import SciPyOptimizer

  # Define a simple cost function, e.g., a quadratic function with minimum at [1, 1, 1]
  def cost_function(params):
      return sum((p - 1) ** 2 for p in params)

  # Create an instance of the SciPyOptimizer using the BFGS method.
  optimizer = SciPyOptimizer(method="BFGS")

  # Optimize the cost function starting from an initial guess.
  initial_parameters = [0, 0, 0]
  result = optimizer.optimize(cost_function, initial_parameters)

  # Print the result and the optimal parameters stored in the instance.
  print("Optimization result:", result)
  print("Optimal Parameters:", optimizer.optimal_parameters)
  ```

  ([PR #4](https://github.com/qilimanjaro-tech/qilisdk/pulls/4))

- This release introduces a **draft** version of the new `QaaSBackend` as part of the optional `qaas` module (install with `pip install qilisdk[qaas]`). The `QaaSBackend` enables users to interface with Qilimanjaro's cloud-based Quantum-as-a-Service (QaaS) platform via synchronous HTTP calls, providing a unified backend that supports both digital and analog workflows.

  In this update, the backend allows for secure login and logout operations using credentials sourced through method parameters or environment variables. The returned authenticated token is stored securily with `keyring`.

  `QaaSBackend` implements both `DigitalBackend` and `AnalogBackend` interfaces, paving the way for future extensions that support executing digital circuits and performing analog evolutions. It also offers functionalities such as list available devices. ([PR #5](https://github.com/qilimanjaro-tech/qilisdk/pulls/5))

- Added the `Ansatz` abstract class and `HardwareEfficientAnsatz` implementation. Users can retrieve the underlying circuit for a given set of parameter values by calling the `get_circuit(parameters: list[float])` method.

  ([PR #10](https://github.com/qilimanjaro-tech/qilisdk/pulls/10))

- Introduces a new backend for quantum circuit simulation leveraging CUDA. The backend supports multiple simulation methods:
    - `STATE_VECTOR` (automatically selects GPU if available, otherwise falls back to CPU)
    - `TENSOR_NETWORK`
    - `MATRIX_PRODUCT_STATE`

    The `CudaBackend` handles basic gates (X, Y, Z, H, S, T, RX, RY, RZ, U1, U2, U3), adjoint gates, controlled gates with one control qubit (we found a bug for more qubits, see https://github.com/NVIDIA/cuda-quantum/issues/2731), and measurement operations.

  **Example usage:**

  ```python
  from qilisdk.extras.cuda_backend import CudaBackend
  from qilisdk.digital import Circuit, H, M, DigitalSimulationMethod

  # Create a quantum circuit with one qubit.
  circuit = Circuit(nqubits=1)
  circuit.add(H(0))  # Apply a Hadamard gate.
  circuit.add(M(0))  # Measure the qubit.

  # Initialize the CudaBackend with the TENSOR_NETWORK simulation method.
  backend = CudaBackend(simulation_method=DigitalSimulationMethod.TENSOR_NETWORK)

  # Execute the circuit with 1000 shots and print the results.
  results = backend.execute(circuit, nshots=1000)
  print("Results:", results)
  ```

  ([PR #11](https://github.com/qilimanjaro-tech/qilisdk/pulls/11))

- Implement utility functions `to_qasm2`, `to_qasm2_file`, `from_qasm2` and `from_qasm2_file` to serialize and deserialize a `Circuit` using Open QASM 2.0 grammar.


  ### Code Example

  ```python
  from qilisdk.digital.circuit import Circuit
  from qilisdk.digital.gates import CNOT, RX, H, M
  from qilisdk.utils.openqasm2 import from_qasm2, from_qasm2_file, to_qasm2, to_qasm2_file

  # Create a sample circuit.
  circuit = Circuit(3)
  circuit.add(H(0))
  circuit.add(CNOT(0, 1))
  circuit.add(RX(2, theta=3.1415))
  circuit.add(M(0, 1, 2))

  # Convert to QASM.
  qasm_code = to_qasm2(circuit)
  print("Generated QASM:")
  print(qasm_code)

  # Reconstruct the circuit from the QASM.
  reconstructed_circuit = from_qasm2(qasm_code)

  # Convert to QASM and save to file
  to_qasm2_file(circuit, "circuit.qasm")

  # Reconstruct the circuit from the QASM file.
  reconstructed_circuit = from_qasm2_file("circuit.qasm")
  ```

  ([PR #12](https://github.com/qilimanjaro-tech/qilisdk/pulls/12))

- **Enhanced Gate Framework**

  This PR refactors the quantum gate hierarchy to provide a more modern and flexible framework. The following key improvements were introduced:

  1. **Adjoint Operation:**
      A new `Adjoint` class has been introduced to provide the Hermitian conjugate (adjoint) of any unitary gate. When you call the `adjoint()` method on a unitary gate, it returns an instance of `Adjoint` whose matrix is the conjugate transpose of the original gate's matrix. This operation is essential for many quantum algorithms that require reversing a gate's effect.

      *Example:*
      ```python
      x_adjoint = X(0).adjoint()
      print(x_adjoint.name)  # Output: X†
      print(x_adjoint.matrix)  # Displays the conjugate-transposed matrix of X
      ```

  2. **Exponential Operation:**
      The new `Exponential` class computes the matrix exponential of a unitary gate's matrix. This feature is particularly useful for simulating continuous time evolution in quantum systems.

      *Example:*
      ```python
      x_exponential = X(0).exponential()
      print(x_exponential.name)  # Output: e^X
      print(x_exponential.matrix)  # Displays the matrix exponential of X's matrix
      ```

  3. **Controlled Operation:**
      The refactored framework now includes a generic `Controlled` class that wraps around any unitary gate to add control qubits. When you invoke the `controlled()` method on a unitary gate, you get a new controlled gate instance that:
      - Checks for duplicate control qubits.
      - Ensures that the control qubits do not overlap with the target qubits of the underlying gate.
      - Preserves the type of the controlled gate using generics, providing better static type safety.

      *Example:*
      ```python
      # Creating an X gate and then its controlled version with control qubits 1 and 2.
      controlled_x = X(0).controlled(1, 2)
      print(controlled_x.name)  # Output: CCX
      print(controlled_x.matrix)  # Displays the controlled gate matrix computed from X's matrix
      ```

  4. **Dynamic Matrix Updates:**
      With the new framework, the `matrix` property of all modified operations (Adjoint, Exponential, Controlled) is always up-to-date. Whenever you update the parameters of a gate (via `set_parameters` or `set_parameter_values`), the underlying matrix is re-generated automatically. This ensures that any dependent modified gate also reflects the updated parameters.

      *Example:*
      ```python
      # Create an RX gate with an initial theta value.
      rx_gate = RX(0, theta=3.14)
      print("Initial RX matrix:")
      print(rx_gate.matrix)

      # Update the rotation parameter.
      rx_gate.set_parameter_values([1.57])
      print("Updated RX matrix:")
      print(rx_gate.matrix)

      # If you create a controlled version, its matrix will update as well:
      controlled_rx = RX(0, theta=3.14).controlled(1)
      print("Controlled-RX matrix:")
      print(controlled_rx.matrix)

      # Update the rotation parameter.
      controlled_rx.set_parameter_values([1.57])
      print("Updated Controlled-RX matrix:")
      print(controlled_rx.matrix)
      ```

  5. **Generic and Type-Safe Enhancements:**
      The new design leverages Python generics (with a type variable, e.g., `TUnitary`) to ensure that operations such as `adjoint`, `exponential`, and `controlled` correctly tie the modified gate to its underlying unitary type. This enhancement allows static type checkers to catch mismatches early and provides a cleaner, more robust API.

  6. **Unified Method Signatures for Gate Modifications:**
      The `Unitary` class now provides convenience methods for creating these modified gates:
      - `controlled(*control_qubits: int) -> Controlled[Self]`
      - `adjoint() -> Adjoint[Self]`
      - `exponential() -> Exponential[Self]`

      These methods allow you to easily derive new gate instances from existing ones while keeping the underlying type information intact.

  7. **Unified and Immutable Parameter Handling:**
      The new `Unitary` class now accepts a `parameters` dictionary during initialization and provides a read-only property that returns these values. The properties `parameter_names` and `parameter_values` are computed from the underlying parameters rather than being stored separately.

      This makes it simpler to update and validate parameter values:
      ```python
      rx_gate = RX(0, theta=3.14)
      print(rx_gate.parameters)  # Output: {'theta': 3.14}
      rx_gate.set_parameter_values([1.57])
      ```

  **Impact:**
  These features collectively enhance the modularity and expressiveness of our quantum gate framework. Users can now effortlessly generate new gate variants (controlled, adjoint, and exponential) with robust type safety and clear, intuitive API calls, laying a solid foundation for advanced quantum circuit design.

  ([PR #14](https://github.com/qilimanjaro-tech/qilisdk/pulls/14))

- This release introduces the new `TimeEvolution` analog algorithm along with supporting classes that empower users to simulate time-dependent quantum systems with greater flexibility and precision. The update brings a schedule-based approach that allows for dynamic interpolation between different Hamiltonians over time, making it easier to model complex evolution scenarios. New classes such as `Schedule` and `TimeEvolution` simplify the process of defining the evolution parameters, specifying initial states, and monitoring observables. For instance, the following code example demonstrates how to set up a one-qubit system with two Hamiltonians—gradually transitioning from one to the other using a linear schedule—and execute the simulation on a CUDA backend for enhanced performance:

  ```python
  T = 10
  dt = 0.1
  steps = np.linspace(0, T, int(T / dt))

  nqubits = 1

  H1 = sum(X(i) for i in range(nqubits))
  H2 = sum(Z(i) for i in range(nqubits))

  schedule = Schedule(
      T,
      dt,
      hamiltonians={
          "h1": H1,
          "h2": H2
      },
      schedule={
          t: {
              "h1": 1 - steps[t] / T,
              "h2": steps[t] / T,
          }
          for t in range(len(steps))
      },
  )

  state = tensor([(ket(0) + ket(1)).unit() for _ in range(nqubits)]).unit()
  dm = (state @ state.dag()).unit()

  time_evolution = TimeEvolution(
      backend=CudaBackend(),
      schedule=schedule,
      initial_state=state,
      observables=[Z(0), X(0), Y(0), Z(nqubits - 1), X(nqubits - 1), Y(nqubits - 1)],
  )

  results = time_evolution.evolve(store_intermediate_results=True)
  ```

  ([PR #16](https://github.com/qilimanjaro-tech/qilisdk/pulls/16))

- This release introduces a new Variational Quantum Eigensolver (VQE) algorithm designed to approximate the ground state energy of quantum systems via a hybrid quantum–classical approach. The VQE algorithm now provides users with a structured and detailed output that includes the estimated ground state energy (optimal cost) and the corresponding optimal ansatz parameters. In addition, users can optionally enable the recording of intermediate optimization results—each stored as an instance of `OptimizerResult`—to facilitate detailed analysis and debugging of the optimization process. Notable enhancements in this release include:
  - A new `VQE` class that integrates seamlessly with user-defined ansatzes, backends, and cost functions, and produces comprehensive results encapsulated in a `VQEResult`.
  - A detailed `VQEResult` output that contains both the final optimal values and, if requested, a list of intermediate results (each as an `OptimizerResult` instance) showcasing the progression of the optimization.
  - Enhanced optimizer functionality via an updated `SciPyOptimizer`, which now supports a `store_intermediate_results` flag to record intermediate steps as part of its return structure.

  Below is an example demonstrating how to use the new VQE implementation:

  ```python
  import numpy as np
  from qilisdk.common import algorithm
  from qilisdk.common.optimizer import SciPyOptimizer
  from qilisdk.digital.vqe import VQE
  from qilisdk.digital.ansatz import HardwareEfficientAnsatz
  from qilisdk.digital.digital_result import DigitalResult
  from qilisdk.extras.cuda_backend.cuda_backend import CudaBackend

  # Define the problem parameters.
  n_items = 4  # Number of items in the optimization problem.
  max_weight_perc = 0.6  # Maximum allowed weight percentage.
  # Randomly generate weights for each item.
  weights = [np.random.randint(1, 5) for _ in range(n_items)]
  # Randomly generate values for each item.
  values = [np.random.randint(1, 10) for _ in range(n_items)]
  # Compute the maximum allowed weight as a fraction of the total weight.
  max_w = int(max_weight_perc * sum(weights))

  # Initialize a hardware-efficient ansatz.
  # 'n_qubits' is set to the number of items for this example.
  nqubits = n_items
  ansatz = HardwareEfficientAnsatz(
      n_qubits=nqubits,
      connectivity="Full",
      layers=1,
      one_qubit_gate="U3",
      two_qubit_gate="CNOT"
  )

  # Set up the backend, for instance using a CUDA backend for hardware acceleration.
  backend = CudaBackend()

  # LM is defined as the total sum of values and used within the cost function.
  LM = sum(values)

  def cost_function(x: DigitalResult) -> float:
      """
      Compute the cost based on the most probable measurement outcomes.
      Each outcome is evaluated by:
        - Calculating a penalty (H_a) if the total weight exceeds the maximum allowed weight.
        - Calculating a reward (H_b) for the selected items, given as a negative cost.
      The final cost is computed as the weighted sum (by the probability of each outcome) of these contributions.
      """
      # Get the most probable measurement outcomes.
      most_probable = x.get_probabilities()

      final_cost = 0
      for n in range(len(most_probable)):
          # Convert the measurement bitstring into a list of integers.
          x_n = [int(i) for i in list(most_probable[n][0])]
          # Calculate the total weight for this outcome.
          t = sum(weights[i] * x_n[i] for i in range(n_items))
          # Penalty if the total weight exceeds the allowed maximum.
          H_a = (t - max_w) if t > max_w else 0
          # Reward (negative cost) based on the values for selected items.
          H_b = -sum(values[i] * x_n[i] for i in range(n_items))
          # Final cost is accumulated using the probability of the outcome.
          final_cost = three_most_probable[n][1] * (LM * H_a + H_b)

      return final_cost

  # Set up the SciPy optimizer using the "Powell" method and define parameter bounds.
  optimizer = SciPyOptimizer(
      method="Powell",
      bounds=[(0, np.pi) for _ in range(ansatz.nparameters)]
  )

  # Create the VQE algorithm instance with an initial guess for the parameters.
  algorithm = VQE(
      ansatz,
      [0.5 for _ in range(ansatz.nparameters)],
      cost_function
  )

  # Execute the VQE algorithm using the specified backend and optimizer.
  # The flag store_intermediate_results is set to True to capture each optimization step.
  results = algorithm.execute(backend, optimizer, store_intermediate_results=True)

  # Display the optimization results.
  print("Optimal Cost:", results.optimal_cost)
  print("Optimal Parameters:", results.optimal_parameters)
  print("Intermediate Optimization Results:")
  for intermediate in results.intermediate_results:
      print("Cost:", intermediate.cost)
  ```

  ([PR #21](https://github.com/qilimanjaro-tech/qilisdk/pulls/21))

- Added `serialize()`, `serialize_to()`, `deserialize()`, `deserialize_from()` functions to enable a unified method for serializing and deserializing classes to and from YAML memory strings and files.

  ```python
  from qilisdk.utils import serialize, deserialize, serialize_to, deserialize_from
  from qilisdk.digital import Circuit, H, CNOT, M

  circuit = Circuit(2)
  circuit.add(H(0))
  circuit.add(CNOT(0, 1))
  circuit.add(M(0, 1))

  # Serialize Circuit to a memory string and deserialize from it.
  yaml_string = serialize(circuit)
  deserialized_circuit = deserialize(yaml_string)

  # Specify the class for deserialization using the `cls` parameter.
  deserialized_circuit = deserialize(yaml_string, cls=Circuit)

  # Serialize to and deserialize from a file.
  serialize_to(circuit, 'circuit.yml')
  deserialized_circuit = deserialize_from('circuit.yml', cls=Circuit)
  ```

  ([PR #22](https://github.com/qilimanjaro-tech/qilisdk/pulls/22))

### Misc

- Drafted initial directory structure and configuration:

  - **Module Separation**
    Separated modules into `common`, `analog`, `digital`, `utils`, and `extras`, with the last one reserved for optional features.

  - **Dynamic Dependency Detection**
    Implemented a custom mechanism that checks which dependencies are installed. If a dependency required for an optional feature is missing, a stub is generated that raises an exception when used.

  - **Local Testing with Tox**
    Configured Tox to run tests locally. To install Tox, run:
    ```bash
    uv tool install tox --with tox-uv
    ```
    Then, execute all tests using:
    ```bash
    tox -p
    ```
    To run tests for a specific environment use the `-e` option. For example, in order to run tests for the core features in Python 3.12 you should run:
    ```
    tox -p -e py312-core
    ```
    You can find all declared environments in `pyproject.toml`.

  - **GitHub Actions Workflows**
    - **Publish**: Deploys the package to PyPI
    - **Code Quality**: Performs linting, formatting, and type checking
    - **Tests**: Runs tests with pytest
    - **Tests with Tox**: Runs tests with Tox (included for performance comparison)

  - **Changelog Management**
    Configured Towncrier to manage and maintain the project changelog.

  [PR #1](https://github.com/qilimanjaro-tech/qilisdk/pulls/1)
