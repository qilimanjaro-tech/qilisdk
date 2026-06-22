QAOA
^^^^^^^^^^^^^^^^^^^^^^^

:class:`~qilisdk.digital.ansatz.QAOA` is an ansatz applying the alternating time evolution of a 
problem Hamiltonian and a mixer Hamiltonian [1]_. 
By initializing the circuit as the ground state of the mixer Hamiltonian (often simply the uniform superposition) and then 
applying the alternating evolution scaled by parameters :math:`\gamma_i` and :math:`\alpha_i`, the idea is that at a certain set of 
parameters the ansatz should approximate the evolution from the ground state of the mixer Hamiltonian to the ground state of the problem Hamiltonian, 
as per the quantum adiabatic theorem. By treating this parameterized circuit as an ansatz for a variational quantum algorithm, we can optimize
to try to minimize the expectation value of the problem Hamiltonian and thus solve the encoded optimization problem.

.. [1] Farhi, Edward, Jeffrey Goldstone, and Sam Gutmann. "A quantum approximate optimization algorithm." arXiv preprint `arXiv:1411.4028 <https://arxiv.org/abs/1411.4028>`_ (2014).

Configuration options:

- **problem_hamiltonian**: The problem Hamiltonian encoding the cost function.
- **layers**: Number of repeating layers of gates. Each layer applies two evolutions: one for the problem Hamiltonian and one for the mixer Hamiltonian.
- **mixer_hamiltonian**: The mixer Hamiltonian. Defaults to X mixer.
- **trotter_steps**: Number of Trotter steps to use for Hamiltonian approximation. Only used if the Hamiltonians contain non-commuting terms.
- **problem_params**: Initial parameter values for the problem Hamiltonian evolution angles. Defaults to 0.0 for all layers.
- **mixer_params**: Initial parameter values for the mixer Hamiltonian evolution angles. Defaults to 0.0 for all layers.

**Example**

.. code-block:: python

    from qilisdk.digital import QAOA
    from qilisdk.analog.hamiltonian import Z as pauli_z

    problem_hamiltonian = pauli_z(0) * pauli_z(1) + pauli_z(2)
    ansatz = QAOA(
        problem_hamiltonian=problem_hamiltonian,
        layers=2,
        mixer_hamiltonian=None,
        trotter_steps=1,
        problem_params=[0.5, 1.0],
        mixer_params=[0.25, 0.75],
    )
    ansatz.draw()

As with the :class:`~qilisdk.digital.ansatz.HardwareEfficientAnsatz`, this ansatz can then be used as per any QiliSDK circuit. 
Or, to instead perform variational optimization over the parameters to minimize the 
expectation value of the problem Hamiltonian, one can set up a :class:`~qilisdk.functionals.variational_program.VariationalProgram` (see :doc:`Functionals </modules/functionals/functionals>` for more details):

.. code-block:: python 

    from qilisdk.functionals.variational_program import VariationalProgram
    from qilisdk.functionals import DigitalPropagation
    from qilisdk.optimizers.scipy_optimizer import SciPyOptimizer
    from qilisdk.cost_functions.observable_cost_function import ObservableCostFunction
    from qilisdk.readout import Readout
    from qilisdk.backends import QiliSim

    vqa = VariationalProgram(functional=DigitalPropagation(ansatz),
                             optimizer=SciPyOptimizer(method="powell", tol=1e-7),
                             cost_function=ObservableCostFunction(problem_hamiltonian))

    print(f"Running QAOA with {len(ansatz.get_parameters())} parameters...")
    backend = QiliSim()
    result = backend.execute(vqa, readout=Readout().with_sampling(nshots=1000))
    print("VQA Result:", result)

