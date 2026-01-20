# qilisdk 0.1.7 (2025-12-04)

## Features

- SpeQtrum's synchronous client now runs every request through a shared HTTPX client that refreshes bearer tokens automatically, ensuring submissions, listings, and polling continue seamlessly after 401s while still applying the SDK user agent. API failures now raise a dedicated SpeQtrum error that includes the message returned by the service and is logged in a human-friendly way, and job detail payloads, logs, and results are decoded defensively so malformed base64 no longer crashes the SDK. ([PR #94](https://github.com/qilimanjaro-tech/qilisdk/pulls/94))
- Added a ``QiliSDK`` to ``OpenFermion`` converter. This allows to translate ``QubitOperator`` from ``OpenFermion`` to ``QiliSDK``'s ``Hamiltonian`` Objects and vice versa.

  This optional can be installed using ``pip install qilisdk[openfermion]``.

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
  ([PR #96](https://github.com/qilimanjaro-tech/qilisdk/pulls/96))
- Raised the SDK baseline to Python 3.11 and aligned the local version file, mypy config, CI tests, docs build, and publishing workflow up to Python 3.13. Linux and Windows installs use customized NumPy and SciPy, linked to Intel's high-performance oneAPI Math Kernel Library (Intel MKL). ([PR #99](https://github.com/qilimanjaro-tech/qilisdk/pulls/99))
- Constraint-aware execution now threads through sampling, time evolution, variational programs, and optimizer loops, rejecting parameter sets that violate declared constraints; tests and serialization use the new APIs.

  ```python
  from qilisdk.core import GreaterThanOrEqual, Parameter
  from qilisdk.functionals import Sampling, VariationalProgram

  # Constrain theta >= phi for a variational run; violations short-circuit the optimizer step.
  theta = Parameter("theta", 0.4, bounds=(0, 1))
  phi = Parameter("phi", 0.3, bounds=(0, 1))
  constraints = [GreaterThanOrEqual(theta, phi)]

  vp = VariationalProgram(
      functional=Sampling(...),
      optimizer=...,
      cost_function=...,
      parameter_constraints=constraints,
  )
  ```
  ([PR #100](https://github.com/qilimanjaro-tech/qilisdk/pulls/100))
- Rebuilt analog scheduling around a single, flexible `Schedule`/`Interpolator`: `LinearSchedule` is removed, coefficients can be defined as callables or time intervals with step/linear interpolation, max-time rescaling, shared parameter tracking, and updated visualization/backends.

  Centralized parameter management in `Parameterizable` so Hamiltonians, circuits, schedules, and gates all inherit consistent parameter getters/setters, constraint checking, and validation; variables add comparison helpers, caching, and math maps.

  ```python
  from qilisdk.analog import Schedule, X, Z
  from qilisdk.analog.schedule import Interpolation
  from qilisdk.core import Parameter

  T = 10.0
  schedule = Schedule(
      hamiltonians={"driver": X(0), "problem": Z(0)},
      coefficients={
          # Interval syntax expands to sampled points; callables can reference time directly.
          "driver": {(0, T): lambda t: 1 - t / T},
          "problem": {(0, T): lambda t: t / T},
      },
      dt=0.5,
      interpolation=Interpolation.LINEAR,
  )
  schedule.scale_max_time(Parameter("T_max", 8.0))
  schedule.draw()
  ```
  ([PR #100](https://github.com/qilimanjaro-tech/qilisdk/pulls/100))
- Introduced the first version of the `CircuitTranspiler` alongside the `DecomposeMultiControlledGatesPass` that decomposes any multi-controlled single-qubit gate, enabling both CUDA and QuTiP backends to execute circuits containing such constructs. The pass is wired directly into both backends today while the full transpiler pipeline is staged for future expansion. ([PR #101](https://github.com/qilimanjaro-tech/qilisdk/pulls/101))

## Bugfixes

- Fixed `Schedule.add_hamiltonian` so time-dependent coefficient functions populate every discrete time step.

  Added regression tests to cover function-driven schedules and validate that only parameters appear in the generated coefficients. ([PR #95](https://github.com/qilimanjaro-tech/qilisdk/pulls/95))

## Improved Documentation

- Documentation refresh across the project:

  - README now links to the hosted docs and the SpeQtrum snippet walks through building a circuit, logging in, picking a device, and submitting a `Sampling` job.
  - Digital fundamentals gained a rendered circuit figure, QuTiP execution example for `HardwareEfficientAnsatz`, a clarified measurement note, and a new “build your own ansatz” walkthrough with code.
  - Analog fundamentals had their operator headings cleaned up, and the shared custom CSS now ships Plus Jakarta Sans, gradient headers, and dark-mode inline-code styling; the circuit diagram image is tracked under `_static/`.
  - The `time_evolution.ipynb` example notebook was stripped of bulky outputs/metadata to keep the repo lightweight.
  - Installation docs call out CUDA GPU requirements and defer hands-on content to a brand-new Quickstart guide (added to the index) that hosts the digital and analog walkthroughs plus backend-specific notes.
  - Backend fundamentals now highlight how to install optional extras and link back to the Installation guide for details.

  ([PR #97](https://github.com/qilimanjaro-tech/qilisdk/pulls/97))
- Removed `main` branch from documentation. Sorted tags by descending order. (latest is first) ([PR #104](https://github.com/qilimanjaro-tech/qilisdk/pulls/104))
- Docs publishing now points the root of the site to the latest release docs and keeps "main" as a clearly labeled development build, with the version picker reorganized into separate Releases and Development sections so visitors land on the most current tag while still finding the dev docs easily. ([PR #109](https://github.com/qilimanjaro-tech/qilisdk/pulls/109))
- Fixed a display bug in which inline code blocks in headers were invisible. This was due to the previous rule only accounting for "a" elements in headers and not ".pre" span elements. The reason there was a ".pre" block in the header is because of a broken class link, which will be fixed along with others in a different PR. ([PR #114](https://github.com/qilimanjaro-tech/qilisdk/pulls/114))
- Fixed a number of broken/missing API links in the docs and in general made the docs consistent with the API reference.

  Also:
   - Remove references to optimize() since it has since been moved into execute()
   - Fixed a broken output box
   - Fixed several lists that weren't displaying correctly
   - Removed an unused import from a code block in core.rst
   - Various other typos, such as missing "a"s before certain terms (keeping consistency with previous uses of said terms)
   - Wrapped some very long lines (as per the style throughout the rest of the files) ([PR #120](https://github.com/qilimanjaro-tech/qilisdk/pulls/120))

## Misc

- [PR #104](https://github.com/qilimanjaro-tech/qilisdk/pulls/104), [PR #106](https://github.com/qilimanjaro-tech/qilisdk/pulls/106)


# qilisdk 0.1.6 (2025-10-31)

## Features

- Added ``job_name`` to speqtrum's ``submit`` method to allow the user to give a custom name to their job. ([PR #86](https://github.com/qilimanjaro-tech/qilisdk/pulls/86))
- SpeQtrum jobs now retain their concrete result types from submission through retrieval, so every handle produced by `submit` returns the right `FunctionalResult` whether you are sampling, evolving in time, or orchestrating a variational loop. Waiting for a fresh job still takes a single call, but the typed detail that comes back no longer needs manual casting:

  ```python
  job_handle = speqtrum.submit(Sampling(circuit), device="cuda_state_vector")
  job = speqtrum.wait_for_job(job_handle)
  results = job.get_results()  # Your IDE will now know this is a SamplingResult
  ```

  The same strong typing applies when inspecting historical runs. Recreate a handle with the appropriate class method, pass it to `get_job`, and the returned `TypedJobDetail` exposes the precise payload you expect:

  ```python
  job_handle = JobHandle.sampling(1200853)
  job = speqtrum.get_job(job_handle)
  results = job.get_results()  # Your IDE will now know this is a SamplingResult
  ```

  Variational programs are the only scenario that needs an extra hint: because they wrap another functional, recreate the handle with the inner functional’s result type so the execution results can be validated against your expectations:

  ```python
  job_handle = JobHandle.variational_program(1200853, result_type=SamplingResult)
  job = speqtrum.get_job(job_handle)
  results = job.get_results()  # Your IDE will now know this is a VariationalProgramResult
  optimal = results.optimal_execution_results  # Your IDE will now know this is a SamplingResult
  ```

  Passing a bare integer job identifier to `wait_for_job` or `get_job` remains valid for quick checks and backwards-compatibility, but doing so skips the handle metadata and yields an untyped `JobDetail`, so you will need to inspect or cast the result manually. ([PR #87](https://github.com/qilimanjaro-tech/qilisdk/pulls/87))

## Bugfixes

- Fixed the `HardwareEfficientAnsatz` so the first layer of single-qubit gates is not duplicated, restoring the expected gate and parameter counts. ([PR #90](https://github.com/qilimanjaro-tech/qilisdk/pulls/90))
- Fixed the scheduler so numpy floats are accepted without errors. Fixed the QuTiP backend by aligning the schedule-to-backend mapping structure. ([PR #91](https://github.com/qilimanjaro-tech/qilisdk/pulls/91))

## Improved Documentation

- Updating the documentation and README to correspond to the latest version of QiliSDK. ([PR #80](https://github.com/qilimanjaro-tech/qilisdk/pulls/80))
- Added extra doc-strings, and fixed outdated doc-strings. ([PR #84](https://github.com/qilimanjaro-tech/qilisdk/pulls/84))

## Misc

- [PR #88](https://github.com/qilimanjaro-tech/qilisdk/pulls/88), [PR #92](https://github.com/qilimanjaro-tech/qilisdk/pulls/92)


# qilisdk 0.1.5 (2025-10-17)

## Features

- This PR adds foundational support for Variables, Terms, Constraints, Models, and QUBO processing, along with corresponding unit tests and exception definitions.

  - Variables: 
    
      - in the variables module we provide classes and functions to define Spin, Binary, and Generic Variables. 
      ```python
      b = BinaryVariable("b")
      s = SpinVariable("s")
      v = Variable("v", domain=Domain.REAL, bounds=(-2, 2), encoding=BitWise, precision=1e-3)
      ```
      - We define three different ways to encode continuous variables into binary encoding:
          - **Bitwise**: encodes the continuous variable's domain into a bit wise binary string. 
              ```python
              v = Variable("v", domain=Domain.POSITIVE_INTEGER, bounds=(0, 7), encoding=Bitwise)
              v.to_binary()

              >>> v(0) + (2) * v(1) + (4) * v(2) 
              ```
          - **OneHot**: encodes the continuous variable's domain using one-hot encoding.
              ```python
              v = Variable("v", domain=Domain.POSITIVE_INTEGER, bounds=(0, 7), encoding=OneHot)
              v.to_binary()

              >>> v(1) + (2) * v(2) + (3) * v(3) + (4) * v(4) + (5) * v(5) + (6) * v(6) + (7) * v(7) 
              ```
              Note: `v(0)` doesn't appear in the expression because it has the coefficient `0`.
          - **DomainWall**: encodes the continuous variable's domain using domain wall encoding.
              ```python
              v = Variable("v", domain=Domain.POSITIVE_INTEGER, bounds=(0, 7), encoding=DomainWall)
              v.to_binary()

              >>> v(0) + v(1) + v(2) + v(3) + v(4) + v(5) + v(6)
              ```
      - using these variables we added the ability to construct:
          - Mathematical expressions:
           ```python
              x = Variable("x", domain=Domain.POSITIVE_INTEGER, bounds=(0, 7), encoding=Bitwise)
              y = Variable("y", domain=Domain.POSITIVE_INTEGER, bounds=(2, 10), encoding=Bitwise)
              term = 3 * x + 4 * y + x ** 2 + 5 * x * y
              print(term)

              >>> (3) * x + (4) * y + (x^2) + (5) * (x * y) 
          ```
          - Comparison terms: 
          ```python
              x = Variable("x", domain=Domain.POSITIVE_INTEGER, bounds=(0, 7), encoding=Bitwise)
              y = Variable("y", domain=Domain.POSITIVE_INTEGER, bounds=(2, 10), encoding=Bitwise)

              term1 = 3 * x + 4 * y 
              term2 = x**2 + 3
              comparison = EQ(term1, term2)
              print(comparison)

              >>> (3) * x + (4) * y + (-1.0) * (x^2)  == (3.0)             
          ```
          comparison terms can be: 
          | Comparison Term              | Code Representation     | Short-form Representation     |
          | :--------------------------- | :---------------------: | ----------------------------: |
          | Equal to                     |  ``Equal``              | ``EQ``                        |
          | Not Equal to                 |  ``NotEqual``           | ``NEQ``                       |
          | Greater than                 |  ``GreaterThan``        | ``GT``                        |
          | Greater than or equal to     |  ``GreaterThanOrEqual`` | ``GEQ``                       |
          | Less than                    |  ``LessThan``           | ``LT``                        |
          | Less than or equal to        |  ``LessThanOrEqual``    | ``LEQ``                       |

          Note: both the short-form and long-form representations are valid to use in the code. 

  - Models: 
      - This module provides tools to construct mathematical models and QUBO models. 
    
      Example of Constructing a model: 
      ```python
      from qilisdk.common.variables import BinaryVariable, LEQ
      from qilisdk.common.model import Model

      num_items = 4
      values = [1, 3, 5, 2]
      weights = [3, 2, 4, 5]
      max_weight = 6
      bin_vars = [BinaryVariable(f"b{i}") for i in range(num_items)]

      model = Model("Knapsack")

      objective = sum(values[i] * bin_vars[i] for i in range(num_items))
      model.set_objective(objective)

      constraint = LEQ(sum(weights[i] * bin_vars[i] for i in range(num_items)), max_weight)
      model.add_constraint("maximum weight", constraint)

      print(model)

      >>> Model name: Knapsack 
      >>> objective (obj): 
      >>>     minimize : 
      >>>     b0 + (3) * b1 + (5) * b2 + (2) * b3  
      >>> 
      >>> subject to the constraint/s: 
      >>>     maximum weight: (3) * b0 + (2) * b1 + (4) * b2 + (5) * b3  <= (6.0)  
      ```

      - QUBO (Quadratic Unconstrained Binary Optimization) models can be constructed from models directly:
      ```python
      qubo_model = model.to_qubo()
      print(qubo_model)

      Model name: QUBO_Knapsack 
      >>> objective (obj): 
      >>> 	 minimize : 
      >>> 	 b0 + (3) * b1 + (5) * b2 + (2) * b3  

      >>> subject to the constraint/s: 
      >>> 	 maximum weight: (-27.0) * b0 + (12.0) * (b0 * b1) + (24.0) * (b0 * b2) + (30.0) * (b0 * b3) + (6.0) * (b0 * maximum weight_slack(0)) + (12.0) * (b0 * maximum weight_slack(1)) + (18.0) * (b0 * maximum weight_slack(2)) + (-20.0) * b1 + (16.0) * (b1 * b2) + (20.0) * (b1 * b3) + (4.0) * (b1 * maximum weight_slack(0)) + (8.0) * (b1 * maximum weight_slack(1)) + (12.0) * (b1 * maximum weight_slack(2)) + (-32.0) * b2 + (40.0) * (b2 * b3) + (8.0) * (b2 * maximum weight_slack(0)) + (16.0) * (b2 * maximum weight_slack(1)) + (24.0) * (b2 * maximum weight_slack(2)) + (-35.0) * b3 + (10.0) * (b3 * maximum weight_slack(0)) + (20.0) * (b3 * maximum weight_slack(1)) + (30.0) * (b3 * maximum weight_slack(2)) + (-11.0) * maximum weight_slack(0) + (-20.0) * maximum weight_slack(1) + (-27.0) * maximum weight_slack(2) + (4.0) * (maximum weight_slack(0) * maximum weight_slack(1)) + (6.0) * (maximum weight_slack(0) * maximum weight_slack(2)) + (12.0) * (maximum weight_slack(1) * maximum weight_slack(2))  == (-36.0)  
      >>> 
      >>> With Lagrange Multiplier/s: 
      >>> 	 maximum weight : 100 
      ```

      Note: QUBO models can be directly translated into Ising Hamiltonians: 
      ```python
      qubo_model.to_hamiltonian()
      >>> 3305.5 - 1200.5 Z(0) - 801.5 Z(1) - 1602.5 Z(2) - 2001 Z(3) + 300 Z(0) Z(1) + 600 Z(0) Z(2) + 750 Z(0) Z(3) - 400 Z(4) + 150 Z(0) Z(4) - 800 Z(5) + 300 Z(0) Z(5) - 1200 Z(6) + 450 Z(0) Z(6) + 400 Z(1) Z(2) + 500 Z(1) Z(3) + 100 Z(1) Z(4) + 200 Z(1) Z(5) + 300 Z(1) Z(6) + 1000 Z(2) Z(3) + 200 Z(2) Z(4) + 400 Z(2) Z(5) + 600 Z(2) Z(6) + 250 Z(3) Z(4) + 500 Z(3) Z(5) + 750 Z(3) Z(6) + 100 Z(4) Z(5) + 150 Z(4) Z(6) + 300 Z(5) Z(6)
      ```

  ([PR #18](https://github.com/qilimanjaro-tech/qilisdk/pulls/18))
- Replacing the callable cost function with the abstract model. The Abstract Model allows the user to write an objective function that is subjected to a set of constraints. Then this is used to evaluate the cost of a given sample. 


  Example: 
  ```python
  from qilisdk.common import SciPyOptimizer
  from qilisdk.common.model import QUBO, Model
  from qilisdk.common.variables import LEQ, BinaryVariable
  from qilisdk.digital.ansatz import HardwareEfficientAnsatz
  from qilisdk.digital.digital_result import DigitalResult
  from qilisdk.digital.vqe import VQE
  from qilisdk.extras import CudaBackend


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
  n_qubits = 3
  ansatz = HardwareEfficientAnsatz(
      n_qubits=n_qubits, layers=2, connectivity="Linear", structure="grouped", one_qubit_gate="U2", two_qubit_gate="CNOT"
  )

  ## Build the VQE object
  vqe = VQE(
      ansatz=ansatz,
      initial_params=[0 for _ in range(ansatz.nparameters)],
      model=model,
  )

  ## Define the Backend and the Optimizer
  backend = CudaBackend()
  optimizer = SciPyOptimizer(method="Powell")

  ## Execute the VQE to find the optimal parameters
  result = vqe.execute(backend, optimizer, nshots=1000)

  ## Sample the circuit using the optimal parameters
  circuit = ansatz.get_circuit(result.optimal_parameters)
  results = backend.execute(circuit)

  ## Print the probabilities
  print(results.get_probabilities())
  ``` ([PR #42](https://github.com/qilimanjaro-tech/qilisdk/pulls/42))
- The `VQEResult` object now reports, alongside the optimal cost, optimal parameters, and any intermediate data, the full statistics obtained by executing the ansatz with the optimal parameters on the backend. Two new fields are returned:

  * **`optimal_probabilities`** – the normalized probability of measuring each computational-basis bit-string when the circuit is executed with the optimizer’s best parameters (e.g., `'101': 0.248`, `'110': 0.243`, …).
  * **`optimal_samples`** – the corresponding raw counts collected from the shot-based execution that produced the probability distribution (e.g., `'101': 248`, `'110': 243`, …).

  ```
  VQEResult(
    Optimal Cost = -5.701,
    Optimal Parameters=[
   2.4765672929916778,
   1.3670931144817295,
   1.8739044290336593,
   1.9418187198696006,
   1.9415620118578056,
   1.3477569502527096,
   1.4586266908332153,
   2.399963229728653,
   1.0369721676199404,
   1.243262268356951,
   0.4583521910031868,
   0.8000040258451777,
   1.964761900107407,
   0.12234702880731152,
   0.7672651631073605,
   2.1237307626244775,
   0.9610328434183917,
   1.1962549279434016],
    Intermediate Results=[]
    Optimal Probabilities={
   '000': 0.007,
   '001': 0.234,
   '010': 0.237,
   '011': 0.001,
   '100': 0.026,
   '101': 0.248,
   '110': 0.243,
   '111': 0.004})
   Optimal Samples={
   '000': 7,
   '001': 234,
   '010': 237,
   '011': 1,
   '100': 26,
   '101': 248,
   '110': 243,
   '111': 4})
  ```

  ([PR #43](https://github.com/qilimanjaro-tech/qilisdk/pulls/43))
- Introduced various improvements in serialization and deserialization of objects to YAML. Changed QaaSBackend's models and methods to comply with new QaaS API. ([PR #47](https://github.com/qilimanjaro-tech/qilisdk/pulls/47))
- You can now pass a custom callable to both `list_devices` and `list_jobs`, enabling flexible client‑side filtering without touching the server API. Two new convenience helpers extend the job‑handling workflow: `get_job_details` fetches a complete record (payload, results, logs, and error information) for a given job ID, while `wait_for_job` blocks the caller until that job reaches a terminal state—`COMPLETED`, `ERROR`, or `CANCELLED`. In addition, the execution helpers (`execute`, `evolve`, `run_vqe`, and `run_time_evolution`) have been streamlined to return only the integer job identifier, making immediate responses lighter and encouraging explicit follow‑up calls when you actually need detailed data. ([PR #48](https://github.com/qilimanjaro-tech/qilisdk/pulls/48))
- Adding a Qutip simulation backend for both analog and digital simulations. 

  Qutip backends can be created by simply running:

  ```python
  from qilisdk.extras.qutip import QutipBackend

  backend = QutipBackend()
  ``` ([PR #49](https://github.com/qilimanjaro-tech/qilisdk/pulls/49))
- Introducing **Functionals**, a unified abstraction for quantum work units. The library now includes two functional families: **`Sampling`**, which executes gate‑based circuits and returns shot counts, and **`TimeEvolution`**, which drives Hamiltonian schedules and yields evolved states or expectation values. All back‑ends have been consolidated into the `qilisdk.backends` module and are now Functional‑aware—each advertises the families it supports and executes them via `backend.execute(functional)`. Meanwhile, **QaaS** has been refactored into a standalone API client located in the `qilisdk.qaas` module, providing endpoints for submitting Functionals to remote devices, monitoring job status, and listing available jobs and hardware. ([PR #50](https://github.com/qilimanjaro-tech/qilisdk/pulls/50))
- The QiliSDK settings system has been refactored to use Pydantic v2 and the `pydantic-settings` package, enabling centralized, validated configuration through environment variables. Each setting is defined in a structured `QiliSDKSettings` class, automatically loaded via a singleton `get_settings()` accessor. Environment variables follow a consistent `QILISDK_` prefix, and fields are now cleanly documented and type-safe. This change simplifies configuration management, eliminates scattered `os.environ` usage, and improves testability and documentation integration with AutoAPI. ([PR #55](https://github.com/qilimanjaro-tech/qilisdk/pulls/55))
- The entire project now funnels its diagnostics through a single Loguru pipeline. On import, `qilisdk.__init__` invokes `_logging.configure_logging()`, which reads `logging_config.yaml` (or any file you point to) and installs sinks, colour schemes, and per-library filters before attaching an intercept handler that routes **all** standard-library loggers to Loguru. Back-end drivers and the synchronous QaaS client have been instrumented with rich, level-appropriate messages—`DEBUG` traces payloads and HTTP timings, `INFO` announces lifecycle events, `SUCCESS` confirms completed actions, while `WARNING` and `ERROR` expose anomalies and failures. Configuration is now entirely declarative and lives in one place, `_logging.py`; downstream projects that merely `import qilisdk` inherit the same configuration.  A new environment variable, **`QILISDK_LOGGING_CONFIG_PATH`**, has been added to `QiliSDKSettings`, letting users override the default YAML location without code changes. ([PR #58](https://github.com/qilimanjaro-tech/qilisdk/pulls/58))
- Updating the structure of how variational algorithms are executed. We added a new `VariationalProgram` class that takes in a parameterized Functional, an Optimizer, and a model representing the cost function, then uses these elements to optimize the parameters of the functional. 

  Moreover, due to these changes the VQE class was removed and now all variational programs will be constructed and executed using VariationalProgram. Here is an example of optimizing a VQE to solve the knapsack problem: 


  ```python
  import numpy as np

  from qilisdk.backends import QutipBackend
  from qilisdk.common import BinaryVariable, LessThanOrEqual, Model, ObjectiveSense
  from qilisdk.digital import CNOT, U2, HardwareEfficientAnsatz
  from qilisdk.functionals import Sampling, VariationalProgram
  from qilisdk.optimizers import SciPyOptimizer

  values = [2, 3, 7]
  weights = [1, 3, 3]
  max_weight = 4
  binary_var = [BinaryVariable(f"b{i}") for i in range(len(values))]

  model = Model("Knapsack")

  model.set_objective(sum(binary_var[i] * values[i] for i in range(len(values))), sense=ObjectiveSense.MAXIMIZE)

  model.add_constraint("max_weights", LessThanOrEqual(sum(binary_var[i] * weights[i] for i in range(len(weights))), max_weight))


  n_qubits = 3
  ansatz = HardwareEfficientAnsatz(
      n_qubits=n_qubits, layers=3, connectivity="Linear", structure="grouped", one_qubit_gate=U2, two_qubit_gate=CNOT
  )
  circuit = ansatz.get_circuit([np.random.uniform(0, np.pi) for _ in range(ansatz.nparameters)])

  optimizer = SciPyOptimizer(method="Powell")

  backend = QutipBackend()
  result = backend.execute(VariationalProgram(functional=Sampling(circuit), optimizer=optimizer, cost_model=model))

  print(result)
  ```
  ([PR #62](https://github.com/qilimanjaro-tech/qilisdk/pulls/62))
- Added a high-level `Circuit.draw(style: CircuitStyle = CircuitStyle(), filepath: str | None = None)` that renders the circuit via `MatplotlibCircuitRenderer.plot()` and optionally saves it (format inferred from the file extension). Crucially, passing the `style` parameter **bypasses all library defaults** for visuals—theme, colors, fonts (including the default PlusJakartaSans), DPI, spacing, and math/label formatting — without mutating global Matplotlib `rcParams`. This makes figures reproducible and sandboxed: the style you pass is the style you get, regardless of notebook or app settings. For example, `circuit.draw(style=CircuitStyle(theme=dark, fontsize=12, fontfname=None, fontfamily="DejaVu Sans"), filepath="circuit.svg")` forces a dark theme and DejaVu Sans font locally to this call. If `style` is omitted, the library’s default style is used. ([PR #63](https://github.com/qilimanjaro-tech/qilisdk/pulls/63))
- You can now parameterize both ``Hamiltonian`` and ``Schedule`` objects using Parameter objects. This makes it easier to define flexible, variational quantum programs.

  **Parameterized Hamiltonians**

  You can insert symbolic parameters directly into Hamiltonian definitions, then set or update them later:

  ```python
  from qilisdk.common import Parameter
  from qilisdk.analog import Z, Y, X

  p = [Parameter(f"p({i})", 2) for i in range(2)]

  t = 2 * p[0]

  H = 2 * Z(1) + t * Y(0) 
  H2 = 3 * X(0) + p[1] * Y(0)

  H3 = H + H2

  print(H3) 
  # Output: 2 Z(1) + 6 Y(0) + 3 X(0)

  H3.set_parameters({"p(0)": 3})

  print(H3) 
  # Output: 2 Z(1) + 8 Y(0) + 3 X(0)

  # get hamiltonian parameters
  H3.get_parameters()
  ```

  **Parameterized Schedules**
  You can also define schedules whose weights are parameterized functions of time.

  ```python
  import numpy as np
  from qilisdk.analog import Schedule
  from qilisdk.analog.hamiltonian import X, Z
  from qilisdk.common import Parameter

  val = [0.3 , 0.7]
  p = [Parameter(f"p({i})", val[i]) for i in range(2)]

  dt = 0.1
  T = 10
  steps = np.linspace(0, T, int(T / dt))

  # Define two Hamiltonians
  h1 = X(0) + X(1) + X(2)
  h2 = Z(0) - 1 * Z(1) - 2 * Z(2) + 3 * Z(0) * Z(1)

  schedule = Schedule(
      T=T,
      dt=dt
  )


  def parameterized_schedule(t) -> float:
      if steps[t] < 2:
          return p[0] / 2 * (steps[t])
      if steps[t] < 4:
          return p[0]
      if steps[t] < 6:
          return p[0] + (p[1] - p[0]) / 2 * (steps[t] - 4)
      if steps[t] < 8:
          return p[1]
      return p[1] + (1 - p[1]) / 2 * (steps[t] - 8)


  schedule.add_hamiltonian("h1", h1, lambda t: (1 - steps[t] / T))
  schedule.add_hamiltonian("h2", h2, parameterized_schedule)

  # set parameters
  schedule.set_parameters({
      "p(0)" : 0.2,
      "p(1)" : 0.5
  })

  # get parameters
  print(schedule.get_parameters())
  ```

  **Mixing Hamiltonians and Schedules**

  You can freely combine parameterized ``Hamiltonians`` and parameterized ``Schedules``, making them ideal building blocks for Variational Programs.



  ```python
  import numpy as np
  from qilisdk.analog import Schedule
  from qilisdk.analog.hamiltonian import X, Z
  from qilisdk.common import Parameter

  val = [0.3 , 0.7]
  p = [Parameter(f"p({i})", val[i]) for i in range(2)]
  h_p = [Parameter(f"h_p({i})", 0.5 * np.pi) for i in range(3)]

  dt = 0.1
  T = 10
  steps = np.linspace(0, T, int(T / dt))

  # Define two Hamiltonians
  h1 = X(0) + X(1) + X(2)
  h2 = sum(h_p[i] * Z(i) for i in range(3)) + 3 * Z(0) * Z(1) # parameterized hamiltonian

  schedule = Schedule(
      T=T,
      dt=dt
  )


  def parameterized_schedule(t) -> float:
      if steps[t] < 2:
          return p[0] / 2 * (steps[t])
      if steps[t] < 4:
          return p[0]
      if steps[t] < 6:
          return p[0] + (p[1] - p[0]) / 2 * (steps[t] - 4)
      if steps[t] < 8:
          return p[1]
      return p[1] + (1 - p[1]) / 2 * (steps[t] - 8)


  schedule.add_hamiltonian("h1", h1, lambda t: (1 - steps[t] / T))
  schedule.add_hamiltonian("h2", h2, parameterized_schedule)

  # set parameters
  schedule.set_parameters({
      "p(0)" : 0.2,
      "p(1)" : 0.5
  })

  # get parameters
  print(schedule.get_parameters())
  ```
  **Cost Function Restructuring**

  We’ve refactored cost function handling into a dedicated cost function module.
  - Introduced a new ``ModelCostFunction`` class that allows you to create cost functions directly from abstract models.

  ```python

  from qilisdk.common import BinaryVariable, LEQ, Model, ObjectiveSense
  from qilisdk.cost_functions import ModelCostFunction

  values = [2, 3, 7]
  weights = [1, 3, 3]
  max_weight = 4
  binary_var = [BinaryVariable(f"b{i}") for i in range(len(values))]

  model = Model("Knapsack")

  model.set_objective(sum(binary_var[i] * values[i] for i in range(len(values))), sense=ObjectiveSense.MAXIMIZE)

  model.add_constraint("max_weights", LEQ(sum(binary_var[i] * weights[i] for i in range(len(weights))), max_weight))

  model_cost_function = ModelCostFunction(model)

  ```

  ([PR #64](https://github.com/qilimanjaro-tech/qilisdk/pulls/64))
- Ansatz now **inherits** from Circuit.

  We refactored the hierarchy so that `Ansatz` is an abstract subclass of `Circuit` rather than a separate builder/factory. Concrete templates like `HardwareEfficientAnsatz` are therefore circuits themselves. This simplifies downstream code: anything that accepted a `Circuit` now accepts an ansatz instance directly, without an extra “build” or “get\_circuit” step. As part of this change, the circuit is constructed during initialization, configurable attributes were made private with read-only properties (`layers`, `connectivity`, `structure`, `one_qubit_gate`, `two_qubit_gate`), and parameter handling became deterministic: all single-qubit parameters default to `0.0`; The `structure` flag is now meaningful at build time: `grouped` schedules `U(all) → E(all)` per layer, while `interposed` applies `U(q) → E(all)` for each qubit within a layer. Measurement gates are no longer appended automatically; add them explicitly if needed. The `nparameters` computation remains available via the parent `Circuit`.

  **Example**

  ```python
  from qilisdk.digital import HardwareEfficientAnsatz, U3, CNOT

  ansatz = HardwareEfficientAnsatz(
      nqubits=4,
      layers=2,
      connectivity="linear",        # or "circular" / "full"   
      structure="grouped",          # or "interposed"
      one_qubit_gate=U3,            # or U1 / U2
      two_qubit_gate=CNOT,          # or CZ
  )
  ansatz.draw()                     # ansatz is a Circuit

  # add measurements explicitly if your workflow requires them
  ```
  ([PR #66](https://github.com/qilimanjaro-tech/qilisdk/pulls/66))
- Adding Observable cost function that now could be used with VariationalPrograms. 

  You can construct an Observable cost function by using ``Hamiltonian``, ``PauliOperators``, or ``QuantumObjects``.

  The cost is the expected value of the observable given the final state after the simulation. 


  ```python
  from qilisdk.analog import Z, PauliZ
  from qilisdk.common import QuantumObject
  from qilisdk.cost_functions import ObservableCostFunction
  import numpy as np


  h = Z(0) + 2 * Z(1)
  cost_function = ObservableCostFunction(h)

  ## or

  cost_function = ObservableCostFunction(PauliZ(0))

  ## or

  cost_function = ObservableCostFunction(QuantumObject(np.array([[1, 0], [0, -1]])))
  ```

  Usage: 
  ```python
  from qilisdk.analog import Z
  from qilisdk.common import ket, tensor_prod
  from qilisdk.cost_functions import ObservableCostFunction
  from qilisdk.functionals import TimeEvolutionResult

  n = 2

  H = sum(Z(i) for i in range(n))

  ocf = ObservableCostFunction(H)

  te_results = TimeEvolutionResult(
      final_expected_values=np.array([[-0.9, 0]]),
      expected_values=None,
      final_state=tensor_prod([ket(1), ket(1)]),
      intermediate_states=None,
  )
  cost = ocf.compute_cost(te_results)
  # Output: -2
  ```
  ([PR #68](https://github.com/qilimanjaro-tech/qilisdk/pulls/68))
- The core type has been renamed from `QuantumObject` to `QTensor` to better reflect its role as a quantum tensor and to clarify semantics across states and operators. Internally, the storage switches to CSR (`csr_matrix`) and the hot paths were rewritten to be sparse‑first: `ptrace` no longer densifies and instead (i) remaps COO indices for operators, and (ii) forms the reduced state via `M @ M†` directly for pure states, avoiding construction of full $N\times N$ density matrices. State constructors are faster and leaner: `ket`/`bra` compute the basis index in one pass rather than chaining Kronecker products. Expectation values avoid densification (`trace(Oρ)` via element‑wise multiply–sum), density‑matrix validation uses sparse eigenvalue checks (`eigsh`) with a small‑dimensional dense fallback, and `nqubits` is cached. Together these changes substantially reduce memory and improve runtime for large sparse workloads—most notably for partial traces, state preparation, and expectation values—while keeping the public surface largely intact (helpers `basis_state`, `ket`, `bra`, `tensor_prod`, `expect_val` now operate on `QTensor`). Minor behavior notes: `.data` is CSR; stricter power‑of‑two shape validation; `norm('tr')` returns `1.0` for valid density matrices and scalar norms return `|z|`; `expm` converts to CSC internally; `tensor_prod` rejects empty inputs; and `ket()` requires at least one qubit. Update code by replacing references to `QuantumObject` with `QTensor`. ([PR #69](https://github.com/qilimanjaro-tech/qilisdk/pulls/69))
- Add your info here ([PR #71](https://github.com/qilimanjaro-tech/qilisdk/pulls/71))
- Implemented Identity gate for circuits and added support in CudaBackend and QutipBackend. ([PR #75](https://github.com/qilimanjaro-tech/qilisdk/pulls/75))
- Introduced the `ExperimentFunctional` and `ExperimentResult` classes, establishing a unified framework for defining and analyzing quantum characterization experiments in SpeQtrum.

  This update integrates two foundational single-qubit experiments — Rabi and T1.

  Example of execution:

  ```python
  import numpy as np
  from qilisdk.speqtrum import SpeQtrum
  from qilisdk.speqtrum.experiments import RabiExperiment

  # Authenticate with the SpeQtrum Quantum-as-a-Service (QaaS) platform
  SpeQtrum.login(apikey="...", username="...")

  # Create a SpeQtrum client instance for job submission
  client = SpeQtrum()

  # Define and submit a Rabi experiment on qubit 0,
  # sweeping the drive duration across a range of values
  job_id = client.submit(
      RabiExperiment(qubit=0, drive_duration_values=np.arange(...)),
      device_id=...,
  )

  # Wait for the remote job to complete and retrieve job details
  job_details = client.wait_for_job(job_id)

  # Access the processed RabiExperimentResult object
  results = job_details.result.rabi_experiment_result

  # Plot the experimental S21 response (amplitude vs. drive duration)
  results.plot()
  ``` ([PR #76](https://github.com/qilimanjaro-tech/qilisdk/pulls/76))
- Implemented schedule plotting with theme options consistent with those available for circuits, along with new schedule style classes to improve customization. A new `LinearSchedule` has been added, implementing linear interpolation between user-defined coefficients across the schedule. In addition, both `Schedule` and `LinearSchedule` now enforce that the total duration `T` must be divisible by the time step `dt`. ([PR #77](https://github.com/qilimanjaro-tech/qilisdk/pulls/77))

## Bugfixes

- The `parse` method has been made far more forgiving of extra or missing spaces around coefficients and operators. Internally, all whitespace inside any parentheses—whether around a complex coefficient like `(2.5 + 3j)` or an operator index like `Y( 0 )`—is now stripped out, and runs of spaces elsewhere are collapsed to single spaces. A post‐processing regex also ensures there’s always exactly one space between a closing parenthesis and the next Pauli operator token. As a result, variants such as `(2.5 + 3j)Y(0)`, `(2.5+3j)   Y(0)`, and `(2.5 + 3j) Y(0)` will all parse correctly without altering any downstream logic. ([PR #45](https://github.com/qilimanjaro-tech/qilisdk/pulls/45))
- Qutip backend dimension mismatch bug fix. This happened when executing time evolution with more than one qubit as the dimensions of the state and the observables did not match. for instance, for three qubits, the state had the dimension [1, 8] while the observables had [[2, 2, 2],[2, ,2, 2]]. the bug was solved by explicitly specifying the dimensions of all the Qobj objects when constructed. ([PR #53](https://github.com/qilimanjaro-tech/qilisdk/pulls/53))
- Fixed an ndarray shape inconsistency when creating TimeEvolutionResult. ([PR #73](https://github.com/qilimanjaro-tech/qilisdk/pulls/73))
- Fixed an issue where an exception was thrown if you tried to logout twice. ([PR #74](https://github.com/qilimanjaro-tech/qilisdk/pulls/74))
- We moved the dt and T in the `Schedule` class back to being floats instead of integers. Added helpful methods to the hamiltonian class (`commutator`, `anitcommutator`, `vector_norm`, `frobenius_norm`, `trace`) ([PR #78](https://github.com/qilimanjaro-tech/qilisdk/pulls/78))

## Improved Documentation

- Added initial Sphinx‐based documentation for QiliSDK, complete with a fully-configured `conf.py` that injects our source path, enables key extensions (Napoleon for Google-style docstrings, Graphviz, Viewcode, AutoAPI for live API reference, nbsphinx, sphinx\_design, and markdown includes), and applies the Sphinx Awesome Theme with custom light/dark logos, favicon, and CSS overrides.  The build now generates a polished HTML site (deployed at [https://qilimanjaro-tech.github.io/qilisdk/](https://qilimanjaro-tech.github.io/qilisdk/)) featuring module overviews, "Getting Started" guides, and deep dives into digital, analog, backend, and QaaS components, laying the groundwork for ongoing documentation enhancements. ([PR #51](https://github.com/qilimanjaro-tech/qilisdk/pulls/51))
- Removed duplicate toctree from documentation's index page. ([PR #56](https://github.com/qilimanjaro-tech/qilisdk/pulls/56))
- Updated the site's visual identity to match Qilimanjaro's brand guidelines. This release introduces the official light and dark logos, replaces the default theme palette with our red-purple-blue HSL brand colors, and adds a bespoke horizontal gradient used for headings. ([PR #57](https://github.com/qilimanjaro-tech/qilisdk/pulls/57))
- Added support for multi-version documentation using `sphinx-multiversion`. ([PR #72](https://github.com/qilimanjaro-tech/qilisdk/pulls/72))

## Misc

- [PR #44](https://github.com/qilimanjaro-tech/qilisdk/pulls/44), [PR #46](https://github.com/qilimanjaro-tech/qilisdk/pulls/46), [PR #48](https://github.com/qilimanjaro-tech/qilisdk/pulls/48), [PR #59](https://github.com/qilimanjaro-tech/qilisdk/pulls/59), [PR #61](https://github.com/qilimanjaro-tech/qilisdk/pulls/61), [PR #65](https://github.com/qilimanjaro-tech/qilisdk/pulls/65), [PR #81](https://github.com/qilimanjaro-tech/qilisdk/pulls/81), [PR #83](https://github.com/qilimanjaro-tech/qilisdk/pulls/83)


# Qilisdk 0.1.4 (2025-06-18)

### Bugfixes

- Removed manual installation of CUDA-Q for the tests and the code quality workflows. ([PR #40](https://github.com/qilimanjaro-tech/qilisdk/pulls/40))


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
