Bitstring Measurement
^^^^^^^^^^^^^^^^^^^^^^

Using :meth:`with_sampling()<qilisdk.readout.readout_spec.Readout.with_sampling>` instructs the 
backend to perform ``nshots`` projective measurements in the computational basis and
collect the bitstring counts.

.. code-block:: python

    from qilisdk.readout import Readout
    from qilisdk.backends import QiliSim
    from qilisdk.digital import Circuit
    from qilisdk.functionals import DigitalPropagation

    backend = QiliSim()
    functional = DigitalPropagation(Circuit(2))

    spec = Readout().with_sampling(nshots=1000)
    result = backend.execute(functional, readout=spec)

    # Access the results
    counts = result.get_samples()                # dict[str, int]  e.g. {"00": 512, "11": 488}
    probs  = result.get_probabilities()    # dict[str, float] normalised to 1.0

    # Top-k most probable outcomes
    top3 = result.readout_results.sampling.get_probabilities(n=3)

**Parameters**

- **nshots** (``int``): Number of measurement shots.  Must be a positive integer.
- **expand_samples** (``bool``): Whether to expand the samples, by default True. If this is True, 
  partial measurements will be returned as "0\_" where the "_" indicates an unmeasured qubit. 
  If False, the unmeasured qubits will be dropped from the bitstring, so "0\_" would be returned as "0".

**When to Use It**

Use sampling when you need the full bitstring distribution, for instance to evaluate a combinatorial
cost function, run QAOA post-processing, or compute error rates.

**Expanded Samples**

When ``expand_samples`` is set to True, the bitstrings in the results will include 
underscores ("_") to indicate unmeasured qubits. This can be helpful for readability, 
especially when only a subset of qubits are measured. For example, if you have a 4-qubit 
system and only measure the first, second, and fourth qubits, a sample would be 
returned as "00\_0" instead of "000", where the underscore indicates that the third qubit's state is not measured.


.. code-block:: python

    from qilisdk.readout import Readout
    from qilisdk.backends import QiliSim
    from qilisdk.digital import Circuit, X, M
    from qilisdk.functionals import DigitalPropagation

    backend = QiliSim()
    circuit = Circuit(3)

    # Final state will be |100>
    circuit.add(X(0))

    # Measure only the first and last qubits
    circuit.add(M(0))
    circuit.add(M(2))

    # Simulate the circuit with and without sample expansion
    readout_with_expand = Readout().with_sampling(nshots=1000, expand_samples=True)
    readout_without_expand = Readout().with_sampling(nshots=1000, expand_samples=False)
    result_with_expand = backend.execute(DigitalPropagation(circuit), readout=readout_with_expand)
    result_without_expand = backend.execute(DigitalPropagation(circuit), readout=readout_without_expand)
    print("With expand_samples=True:")
    print(result_with_expand.get_samples())
    print("With expand_samples=False:")
    print(result_without_expand.get_samples())


**Output**:

::

    With expand_samples=True:
    {'1_0': 1000}
    With expand_samples=False:
    {'10': 1000}

Notice how the first result includes the underscore to make it clear which qubits were ignored, whilst the second
result is more compact but ambiguous unless you know the measurement configuration. This would be especially important
if there were mid-circuit measurements, since you might measure just qubit 1 early on and then just qubit 2 later, so the 
position of the underscore would indicate which qubits were measured at each step.

