# Copyright 2025 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from qilisdk.analog.hamiltonian import Hamiltonian, PauliZ
from qilisdk.analog.schedule import Schedule
from qilisdk.core import Parameter
from qilisdk.core.qtensor import ket, tensor_prod
from qilisdk.functionals.variational_program import VariationalProgram
from unittest.mock import MagicMock
from qilisdk.core import Variable, Domain, LT



def test_variational_init():
    functional = MagicMock()
    functional.get_constraints = MagicMock(return_value=[])
    optimizer = MagicMock()
    cost_function = MagicMock()
    store_intermediate_results = True
    parameter_constraints = [MagicMock()]
    vp = VariationalProgram(
        functional=functional,
        optimizer=optimizer,
        cost_function=cost_function,
        store_intermediate_results=store_intermediate_results,
        parameter_constraints=parameter_constraints,
    )
    assert vp.functional == functional
    assert vp.optimizer == optimizer
    assert vp.cost_function == cost_function
    assert vp.store_intermediate_results == store_intermediate_results
    for con in parameter_constraints:
        assert con in vp.get_constraints()
    assert vp.check_parameter_constraints({}) == 0

def test_bad_parameter_constraints():
    functional = MagicMock()
    optimizer = MagicMock()
    cost_function = MagicMock()

    var = Variable("var", Domain.REAL)
    term = 3*var + 1
    parameter_constraints = [LT(term, 5)] 
    with pytest.raises(ValueError, match="Only parameters"):
        VariationalProgram(
            functional=functional,
            optimizer=optimizer,
            cost_function=cost_function,
            parameter_constraints=parameter_constraints,
        )

    param = Parameter("param", 0.0)
    better_term = 2*param + 1
    parameter_constraints = [LT(better_term, 10)]
    with pytest.raises(ValueError, match="not present in the variational program"):
        VariationalProgram(
            functional=functional,
            optimizer=optimizer,
            cost_function=cost_function,
            parameter_constraints=parameter_constraints,
        )

    vp = VariationalProgram(
        functional=functional,
        optimizer=optimizer,
        cost_function=cost_function,
        parameter_constraints=[],
    )
    with pytest.raises(ValueError, match="not defined in the functional"):
        vp.check_parameter_constraints({"var": 3})

    con = MagicMock()
    con.variables = MagicMock(return_value=[var])
    functional_with_params = MagicMock()
    functional_with_params.get_constraints = MagicMock(return_value=[con])
    vp = VariationalProgram(
        functional=functional_with_params,
        optimizer=optimizer,
        cost_function=cost_function,
        parameter_constraints=[],
    )
    with pytest.raises(ValueError, match="Only Parameters are allowed"):
        vp.check_parameter_constraints({"var": 3})

def test_good_parameter_constraints():
    optimizer = MagicMock()
    cost_function = MagicMock()

    param1 = Parameter("param1", 0.0)
    param2 = Parameter("param2", 0.0)

    term1 = 2*param1 + 1
    term2 = param2 - 3

    parameter_constraints = [LT(term1, 10), LT(term2, 5)]

    functional_with_params = MagicMock()
    functional_with_params.get_constraints = MagicMock(return_value=[])
    functional_with_params.get_parameters = MagicMock(return_value=["param1", "param2"])

    vp = VariationalProgram(
        functional=functional_with_params,
        optimizer=optimizer,
        cost_function=cost_function,
        parameter_constraints=parameter_constraints,
    )

    assert vp.check_parameter_constraints({"param1": 3, "param2": 7}) == 0
    assert vp.check_parameter_constraints({"param1": 5, "param2": 7}) == 100
    assert vp.check_parameter_constraints({"param1": 3, "param2": 10}) == 100
    assert vp.check_parameter_constraints({"param1": 6, "param2": 10}) == 200

