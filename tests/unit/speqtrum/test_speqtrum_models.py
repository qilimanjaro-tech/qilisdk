"""
Unit-tests for the SpeQtrum synchronous client.

All tests are *function* based (no classes) so they integrate with plain
pytest discovery.
"""

from __future__ import annotations

import base64
import collections
import json
import types
from datetime import datetime, timezone
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest

from qilisdk.experiments.experiment_functional import RabiExperiment, T1Experiment, T2Experiment, TwoTonesExperiment
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.time_evolution import TimeEvolution
from qilisdk.functionals.variational_program import VariationalProgram

pytest.importorskip("httpx", reason="SpeQtrum tests require the 'speqtrum' optional dependency", exc_type=ImportError)
pytest.importorskip(
    "keyring",
    reason="SpeQtrum tests require the 'speqtrum' optional dependency",
    exc_type=ImportError,
)

import httpx

import qilisdk.speqtrum.speqtrum as speqtrum
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
from qilisdk.functionals.variational_program_result import VariationalProgramResult
from qilisdk.optimizers.optimizer_result import OptimizerResult
from qilisdk.speqtrum.speqtrum_models import ExecuteResult, SamplingPayload, TimeEvolutionPayload, VariationalProgramPayload

from unittest.mock import MagicMock

from qilisdk.utils.serialization import deserialize, serialize 
from qilisdk.digital import Circuit

def test_sampling_payload():
    circ = Circuit(2)
    sampling = Sampling(nshots=1024, circuit=circ)
    payload = SamplingPayload(sampling=sampling)
    serialized_sampling = payload._serialize_sampling(sampling=sampling, _info={})
    deserialized_sampling = payload._load_sampling(serialized_sampling)
    assert deserialized_sampling.nshots == sampling.nshots
    assert deserialized_sampling.circuit.nqubits == sampling.circuit.nqubits

