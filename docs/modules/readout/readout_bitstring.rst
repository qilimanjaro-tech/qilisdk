Bitstring Measurement
^^^^^^^^^^^^^^^^^^^^^^

Instructs the backend to perform ``nshots`` projective measurements in the computational basis and
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
  partial measurements will be returned as "0_" where the "_" indicates an unmeasured qubit. 
  If False, the unmeasured qubits will be dropped from the bitstring, so "0_" would be returned as "0".

**When to Use It**

Use sampling when you need the full bitstring distribution, for instance to evaluate a combinatorial
cost function, run QAOA post-processing, or compute error rates.

