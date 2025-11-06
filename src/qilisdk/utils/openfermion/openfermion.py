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

from openfermion import QubitOperator

from qilisdk.analog import Hamiltonian, PauliI, PauliX, PauliY, PauliZ


def openfermion_to_qilisdk(qubit_operator: QubitOperator) -> Hamiltonian:
    pauli_map = {"X": PauliX, "Y": PauliY, "Z": PauliZ}

    return Hamiltonian(
        {
            (tuple((pauli_map[op](q)) for q, op in term) if len(term) > 0 else (PauliI(0),)): coeff
            for term, coeff in qubit_operator.terms.items()
        }
    )


def qilisdk_to_openfermion(hamiltonian: Hamiltonian) -> QubitOperator:
    of_ham = QubitOperator()

    for coeff, terms in hamiltonian:
        of_term = ""
        for t in terms:
            if isinstance(t, PauliI):
                continue

            of_term += str(t).replace("(", "").replace(")", "") + " "
        of_term = of_term.rstrip()

        of_ham += QubitOperator(of_term, coeff)

    return of_ham
