Quantum as a Service (QaaS)
====

The :mod:`~qilisdk.qaas` module provides a synchronous client for the Qilimanjaro QaaS REST API via the
:class:`~qilisdk.qaas.qaas.QaaS` class. It handles authentication, device selection, job management, and submission
of quantum functionals (Sampling, TimeEvolution) and VQE workflows to remote hardware or simulators.

Authentication
--------------

Before making any API calls, you must authenticate and cache credentials in your OS keyring.

**Login**

.. code-block:: python

    from qilisdk.qaas import QaaS

    # You can provide credentials directly...
    success = QaaS.login(username="alice", apikey="MY_SECRET_KEY")

    # ...or omit them to read from environment variables:
    #   QILISDK_QAAS_USERNAME and QILISDK_QAAS_APIKEY
    success = QaaS.login()

    if not success:
        raise RuntimeError("Authentication failed")

**Logout**

.. code-block:: python

    from qilisdk.qaas import QaaS

    QaaS.logout()
    # Credentials removed from keyring

Device Management
-----------------

Once authenticated, you can list available devices, select one for subsequent jobs, and query the current selection.

**List Devices**

.. code-block:: python

    client = QaaS()
    devices = client.list_devices()
    for dev in devices:
        print(f"{dev.id}: {dev.name} ({dev.status})")

**Select a Device**

.. code-block:: python

    client.set_device(device_id)
    print("Selected device:", client.selected_device)

**Properties**

- **selected_device** (int | None): ID of the currently selected device, or ``None`` if unset.

Job Management
--------------

You can list existing jobs, inspect their details, and wait for completion.

**List Jobs**

.. code-block:: python

    jobs = client.list_jobs()
    for job in jobs:
        print(f"{job.id}: {job.status}")

**Get Job Details**

.. code-block:: python

    detail = client.get_job_details(job_id)
    print("Payload:", detail.payload)
    print("Result:", detail.result)
    print("Logs:", detail.logs)

**Wait for Completion**

.. code-block:: python

    final = client.wait_for_job(job_id, poll_interval=2.0, timeout=300.0)
    print("Final status:", final.status)

Functional Submission
---------------------

Use :meth:`~qilisdk.qaas.qaas.QaaS.submit` to dispatch a :class:`~qilisdk.functionals.sampling.Sampling` or :class:`~qilisdk.functionals.time_evolution.TimeEvolution` functional.

.. code-block:: python

    from qilisdk.functionals import Sampling, TimeEvolution
    from qilisdk.digital import Circuit, H, CNOT
    from qilisdk.analog import Schedule, X, Z

    # Prepare a Sampling functional
    circ = Circuit(2)
    circ.add(H(0)); circ.add(CNOT(0, 1))
    sampling = Sampling(circuit=circ, nshots=200)
    job_id = client.submit(sampling)

    # Or prepare a TimeEvolution functional
    schedule = Schedule(
        total_time=5.0,
        time_step=0.1,
        hamiltonians={"hx": X(0), "hz": Z(0)},
        schedule_map={t: {"hx": 1 - t/5, "hz": t/5} for t in [0,1,2,3,4,5]}
    )
    time_evolution = TimeEvolution(
        schedule=schedule,
        initial_state=..., 
        observables=[Z(0)], 
        nshots=50
    )
    job_id = client.submit(time_evolution)

Variational Quantum Eigensolver (VQE)
-------------------------------------

For end-to-end VQE workflows, use :meth:`~qilisdk.qaas.qaas.QaaS.submit_vqe`:

**Parameters**

- **vqe** (:class:`~qilisdk.digital.vqe.VQE`): VQE functional defining ansatz and Hamiltonian.
- **optimizer** (:class:`~qilisdk.optimizers.optimizer.Optimizer`): Classical optimizer instance.
- **nshots** (int, optional): Shots per circuit evaluation. Default: 1000.
- **store_intermediate_results** (bool, optional): Record intermediate energies/parameters. Default: False.

**Example**

.. code-block:: python

    from qilisdk.qaas import QaaS
    from qilisdk.digital.vqe import VQE
    from qilisdk.optimizers import COBYLA

    client.set_device(my_device_id)
    vqe = VQE(hamiltonian=H2, ansatz=my_ansatz)
    optimizer = COBYLA(maxiter=100)
    job_id = client.submit_vqe(vqe, optimizer, nshots=500, store_intermediate_results=True)
    print("VQE job submitted with id", job_id)
