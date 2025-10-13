SpeQtrum
========

The :mod:`~qilisdk.speqtrum` package provides an optional, synchronous client for the Qilimanjaro SpeQtrum cloud.
Through the :class:`~qilisdk.speqtrum.speqtrum.SpeQtrum` class you can authenticate, inspect devices and jobs, and submit
digital, analog, or pulse experiments for remote execution.

Installation
------------

SpeQtrum support is shipped as an optional dependency group. Install it alongside QiliSDK with:

.. code-block:: console

    pip install "qilisdk[speqtrum]"

Authentication
--------------

The API uses short-lived OAuth tokens that are cached in the system keyring. Call
:meth:`SpeQtrum.login <qilisdk.speqtrum.speqtrum.SpeQtrum.login>` once and the credentials will be reused for subsequent
sessions.

.. code-block:: python

    from qilisdk.speqtrum import SpeQtrum

    # Credentials can be provided explicitly…
    logged_in = SpeQtrum.login(username="alice", apikey="MY_SECRET_KEY")

    # …or read from the environment (QILISDK_SPEQTRUM_USERNAME / QILISDK_SPEQTRUM_APIKEY)
    logged_in = SpeQtrum.login()

    if not logged_in:
        raise RuntimeError("Authentication failed")

    # Remove cached credentials when they are no longer needed
    SpeQtrum.logout()

Client Construction
-------------------

Once credentials are stored, instantiate :class:`SpeQtrum` to start issuing requests. Construction fails with
``RuntimeError`` if no cached credentials exist.

.. code-block:: python

    from qilisdk.speqtrum import SpeQtrum

    client = SpeQtrum()

Device Catalogue
----------------

Devices are represented by :class:`~qilisdk.speqtrum.speqtrum_models.Device` models containing the device code, number of
qubits, hardware type, and status. Use :meth:`SpeQtrum.list_devices
<qilisdk.speqtrum.speqtrum.SpeQtrum.list_devices>` to enumerate them. An optional ``where`` predicate allows client-side
filtering.

.. code-block:: python

    from qilisdk.speqtrum import SpeQtrum, DeviceStatus

    client = SpeQtrum()
    for device in client.list_devices(where=lambda d: d.status == DeviceStatus.ONLINE):
        print(f"{device.code}: {device.name} ({device.type}) – {device.nqubits} qubits")

Remote Jobs
-----------

:meth:`SpeQtrum.list_jobs <qilisdk.speqtrum.speqtrum.SpeQtrum.list_jobs>` returns lightweight :class:`JobInfo
<qilisdk.speqtrum.speqtrum_models.JobInfo>` records. The ``where`` predicate works the same way as with devices.

.. code-block:: python

    from qilisdk.speqtrum import SpeQtrum
    from qilisdk.speqtrum.speqtrum_models import JobStatus

    client = SpeQtrum()
    running = client.list_jobs(where=lambda job: job.status == JobStatus.RUNNING)
    for job in running:
        print(f"{job.id}: {job.status.value} on {job.device_id}")

To inspect complete job metadata (payload, result, logs, decoded errors) call
:meth:`SpeQtrum.get_job_details <qilisdk.speqtrum.speqtrum.SpeQtrum.get_job_details>`. Binary fields are returned as
decoded strings or structured :class:`~qilisdk.speqtrum.speqtrum_models.ExecuteResult` objects.

To obtain the results of a completed job, check the ``result`` attribute of the returned :class:`~qilisdk.speqtrum.speqtrum_models.JobDetail`. 
The specific result type depends on the job type.

.. code-block:: python

    response = client.get_job_details(job_id)

    # to get the results of the job depending on the type of job
    if response.result:
        if response.result.sampling_result:
            print("Sampling results:", response.result.sampling_result)
        elif response.result.variational_program_result:
            print("Variational Program  results:", response.result.variational_program_result)
        elif response.result.time_evolution_result:
            print("Time Evolution results:", response.result.time_evolution_result)
        elif response.result.rabi_experiment_result:
            print("Rabi Experiment results:", response.result.rabi_experiment_result)
        elif response.result.t1_experiment_result:
            print("T1 Experiment results:", response.result.t1_experiment_result)
    if response.logs:
        print("Execution logs:\n", response.logs)

Waiting for Completion
----------------------

Use :meth:`SpeQtrum.wait_for_job <qilisdk.speqtrum.speqtrum.SpeQtrum.wait_for_job>` to poll until a job reaches a
terminal state (``completed``, ``error``, or ``cancelled``). The method returns the final :class:`JobDetail` snapshot and
raises :class:`TimeoutError` if the optional timeout elapses first.

Functional Submission
---------------------

SpeQtrum accepts the same primitive functionals used by local backends. The :meth:`SpeQtrum.submit
<qilisdk.speqtrum.speqtrum.SpeQtrum.submit>` method inspects the functional type and serializes the correct payload. You
must supply a ``device`` argument with the device code obtained from :meth:`list_devices`.

.. code-block:: python

    from qilisdk.digital import Circuit, H, CNOT
    from qilisdk.functionals import Sampling
    from qilisdk.speqtrum import SpeQtrum

    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))
    sampling = Sampling(circuit=circuit, nshots=1_000)

    client = SpeQtrum()
    device = client.list_devices()[0].code
    job_id = client.submit(sampling, device=device)
    print("Submitted sampling job:", job_id)

.. Warning::
    
    The SpeQtrum backend currently supports only digital circuits and pulse experiments. Analog functionals such as
    :class:`~qilisdk.functionals.time_evolution.TimeEvolution` are not yet supported.


Variational Programs
--------------------

Hybrid optimization is handled through the same :class:`~qilisdk.functionals.variational_program.VariationalProgram`
functional used with local backends. Serialize the fully-configured variational program (ansatz, optimizer, cost
function) and submit it as any other functional.

.. code-block:: python

    from qilisdk.common.model import Model, ObjectiveSense
    from qilisdk.common.variables import BinaryVariable, LEQ
    from qilisdk.cost_functions import ModelCostFunction
    from qilisdk.digital import CNOT, HardwareEfficientAnsatz, U2
    from qilisdk.functionals import Sampling
    from qilisdk.functionals.variational_program import VariationalProgram
    from qilisdk.optimizers.scipy_optimizer import SciPyOptimizer
    from qilisdk.speqtrum import SpeQtrum

    # Build a small cost model
    vars = [BinaryVariable(f"x{i}") for i in range(3)]
    model = Model("toy")
    model.set_objective(sum(vars), sense=ObjectiveSense.MAXIMIZE)
    model.add_constraint("budget", LEQ(vars[0] + vars[1], 1))

    ansatz = HardwareEfficientAnsatz(
        nqubits=3,
        layers=2,
        one_qubit_gate=U2,
        two_qubit_gate=CNOT,
        connectivity="linear",
        structure="grouped",
    )
    functional = Sampling(circuit=ansatz, nshots=1024)
    optimizer = SciPyOptimizer(method="Powell")
    vprog = VariationalProgram(functional=functional, optimizer=optimizer, cost_function=ModelCostFunction(model))

    client = SpeQtrum()
    device = client.list_devices()[0].code
    job_id = client.submit(vprog, device=device)

Pulse Experiments
-----------------

The SpeQtrum client also supports calibration-style experiments defined in :mod:`qilisdk.speqtrum.experiments`. These
functional objects mirror the interfaces described in the :doc:`functionals` chapter and return rich result types.

.. code-block:: python

    import numpy as np
    from qilisdk.speqtrum import DeviceType, SpeQtrum
    from qilisdk.speqtrum.experiments import RabiExperiment, T1Experiment

    client = SpeQtrum()
    device = client.list_devices(
        where=lambda d: d.type in (DeviceType.QPU_ANALOG, DeviceType.QPU_DIGITAL)
    )[0].code

    # Rabi experiment: sweep drive durations
    rabi = RabiExperiment(qubit=0, drive_duration_values=np.linspace(0, 200, 21))
    rabi_job = client.submit(rabi, device=device)
    rabi_response = client.wait_for_job(rabi_job, timeout=600)

    # T1 relaxation experiment: sweep wait durations
    t1 = T1Experiment(qubit=0, wait_duration_values=np.linspace(0, 400, 41))
    t1_job = client.submit(t1, device=device)
    t1_response = client.wait_for_job(t1_job, timeout=600)

The resulting :class:`~qilisdk.speqtrum.experiments.experiment_result.RabiExperimentResult` and
:class:`~qilisdk.speqtrum.experiments.experiment_result.T1ExperimentResult` objects can be retrieved through 
`rabi_response.result.rabi_experiment_result` and `t1_response.result.t1_experiment_result` respectively.
