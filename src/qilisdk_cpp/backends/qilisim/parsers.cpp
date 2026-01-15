// Copyright 2025 Qilimanjaro Quantum Tech
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "qilisim.h"

std::vector<SparseMatrix> QiliSimCpp::parse_hamiltonians(const py::object& Hs) const {
    /*
    Extract Hamiltonian matrices from a list of QTensor objects.

    Args:
        Hs (py::object): A list of QTensor Hamiltonians.

    Returns:
        std::vector<SparseMatrix>: The list of Hamiltonian sparse matrices.
    */
    std::vector<SparseMatrix> hamiltonians;
    for (auto& hamiltonian : Hs) {
        py::buffer matrix = numpy_array(hamiltonian.attr("to_matrix")().attr("toarray")(), py::dtype("complex128"));
        py::buffer_info buf = matrix.request();
        SparseMatrix H = from_numpy(matrix);
        hamiltonians.push_back(H);
    }
    return hamiltonians;
}

std::vector<SparseMatrix> QiliSimCpp::parse_jump_operators(const py::object& jumps) const {
    /*
    Extract jump operator matrices from a list of QTensor objects.

    Args:
        jumps (py::object): A list of QTensor jump operators.

    Returns:
        std::vector<SparseMatrix>: The list of jump operator sparse matrices.
    */
    std::vector<SparseMatrix> jump_matrices;
    for (auto jump : jumps) {
        py::buffer matrix = numpy_array(jump.attr("dense")(), py::dtype("complex128"));
        py::buffer_info buf = matrix.request();
        SparseMatrix J = from_numpy(matrix);
        jump_matrices.push_back(J);
    }
    return jump_matrices;
}

std::vector<SparseMatrix> QiliSimCpp::parse_observables(const py::object& observables, long nqubits) const {
    /*
    Extract observable matrices from a list of QTensor objects.

    Args:
        observables (py::object): A list of QTensor observables.
        nqubits (long): The total number of qubits.

    Returns:
        std::vector<SparseMatrix>: The list of observable sparse matrices.
    */
    std::vector<SparseMatrix> observable_matrices;
    for (auto obs : observables) {
        // Depending on the type of observable given
        if (py::isinstance(obs, Hamiltonian)) {
            // Get the matrix
            py::buffer matrix = numpy_array(obs.attr("to_matrix")().attr("toarray")(), py::dtype("complex128"));
            py::buffer_info buf = matrix.request();
            SparseMatrix O = from_numpy(matrix);

            // Expand to full qubit count if needed
            int obs_qubits = obs.attr("nqubits").cast<int>();
            SparseMatrix O_global = O;
            for (long q = obs_qubits; q < nqubits; ++q) {
                O_global = Eigen::kroneckerProduct(O_global, I).eval();
            }
            observable_matrices.push_back(O_global);

        } else if (py::isinstance(obs, PauliOperator)) {
            // Get the matrix
            py::buffer matrix = numpy_array(obs.attr("matrix"), py::dtype("complex128"));
            py::buffer_info buf = matrix.request();
            SparseMatrix O = from_numpy(matrix);

            // Expand to full qubit count
            int obs_qubit = obs.attr("qubit").cast<int>();
            SparseMatrix O_global(1, 1);
            O_global.coeffRef(0, 0) = 1.0;
            O_global.makeCompressed();
            for (long q = 0; q < nqubits; ++q) {
                if (q != obs_qubit) {
                    O_global = Eigen::kroneckerProduct(O_global, I).eval();
                } else {
                    O_global = Eigen::kroneckerProduct(O_global, O).eval();
                }
            }
            observable_matrices.push_back(O_global);

        } else if (py::isinstance(obs, QTensor)) {
            py::buffer matrix = numpy_array(obs.attr("dense")(), py::dtype("complex128"));
            py::buffer_info buf = matrix.request();
            SparseMatrix O = from_numpy(matrix);
            observable_matrices.push_back(O);

        } else {
            throw py::value_error("Observable type not recognized.");
        }
    }
    return observable_matrices;
}

std::vector<std::vector<double>> QiliSimCpp::parse_parameters(const py::object& coeffs) const {
    /*
    Extract parameter lists from a list of coefficient objects.

    Args:
        coeffs (py::object): A list of coefficient objects.

    Returns:
        std::vector<std::vector<double>>: The list of parameter vectors.
    */
    std::vector<std::vector<double>> parameters_list;
    for (auto& param_set : coeffs) {
        std::vector<double> param_vector;
        for (auto& param : param_set) {
            param_vector.push_back(param.cast<double>());
        }
        parameters_list.push_back(param_vector);
    }
    return parameters_list;
}

std::vector<double> QiliSimCpp::parse_time_steps(const py::object& steps) const {
    /*
    Extract time steps from a list of step objects.

    Args:
        steps (py::object): A list of step objects.

    Returns:
        std::vector<double>: The list of time steps.
    */
    std::vector<double> step_list;
    for (auto step : steps) {
        step_list.push_back(step.cast<double>());
    }
    return step_list;
}

SparseMatrix QiliSimCpp::parse_initial_state(const py::object& initial_state) const {
    /*
    Extract the initial state from a QTensor object.

    Args:
        initial_state (py::object): The initial state as a QTensor.

    Returns:
        SparseMatrix: The initial state as a sparse matrix.
    */
    py::buffer init_state = numpy_array(initial_state.attr("dense")(), py::dtype("complex128"));
    py::buffer_info buf = init_state.request();
    if (buf.ndim != 2) {
        throw py::value_error("Initial state must be a 2D array.");
    }
    int rows = int(buf.shape[0]);
    int cols = int(buf.shape[1]);
    auto ptr = static_cast<std::complex<double>*>(buf.ptr);
    Triplets rho_0_entries;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::complex<double> val = ptr[r * cols + c];
            if (std::abs(val) > atol_) {
                rho_0_entries.emplace_back(Triplet(r, c, val));
            }
        }
    }
    SparseMatrix rho_0(rows, cols);
    rho_0.setFromTriplets(rho_0_entries.begin(), rho_0_entries.end());
    return rho_0;
}

std::vector<Gate> QiliSimCpp::parse_gates(const py::object& circuit) const {
    /*
    Extract gates from a circuit object.

    Args:
        circuit (py::object): The circuit object.

    Returns:
        std::vector<Gate>: The list of Gate objects.
    */
    std::vector<Gate> gates;
    py::list py_gates = circuit.attr("gates");
    for (auto py_gate : py_gates) {
        // Get the name
        std::string gate_type_str = py_gate.attr("name").cast<std::string>();

        // If it's a measurement, skip it
        if (gate_type_str == "M") {
            continue;
        }

        // Get the matrix
        py::buffer matrix = py_gate.attr("_generate_matrix")();
        py::buffer_info buf = matrix.request();
        SparseMatrix base_matrix = from_numpy(matrix);

        // Get the controls
        std::vector<int> controls;
        py::list py_controls = py_gate.attr("control_qubits");
        for (auto py_control : py_controls) {
            controls.push_back(py_control.cast<int>());
        }

        // Get the targets
        std::vector<int> targets;
        py::list py_targets = py_gate.attr("target_qubits");
        for (auto py_target : py_targets) {
            targets.push_back(py_target.cast<int>());
        }

        // Get the parameter names
        std::vector<std::pair<std::string, double>> parameters;
        py::dict py_parameters = py_gate.attr("get_parameters")();
        for (auto item : py_parameters) {
            std::string name = item.first.cast<std::string>();
            double value = item.second.cast<double>();
            parameters.emplace_back(name, value);
        }

        // Add the gate
        gates.emplace_back(gate_type_str, base_matrix, controls, targets, parameters);
    }

    return gates;
}

std::vector<bool> QiliSimCpp::parse_measurements(const py::object& circuit) const {
    /*
    Extract measurement qubit information from a circuit object.

    Args:
        circuit (py::object): The circuit object.
        n_qubits (int): The total number of qubits.

    Returns:
        std::vector<bool>: A vector indicating which qubits are measured.
    */
    int n_qubits = circuit.attr("nqubits").cast<int>();
    std::vector<bool> qubits_to_measure(n_qubits, false);
    py::list py_gates = circuit.attr("gates");
    bool any_measurements = false;
    for (auto py_gate : py_gates) {
        // Get the name
        std::string gate_type_str = py_gate.attr("name").cast<std::string>();

        // If it's a measurement, mark the qubits
        if (gate_type_str == "M") {
            py::list py_targets = py_gate.attr("target_qubits");
            for (auto py_target : py_targets) {
                int target = py_target.cast<int>();
                qubits_to_measure[target] = true;
                any_measurements = true;
            }
        }
    }

    // If we found no measurements, measure all
    if (!any_measurements) {
        qubits_to_measure = std::vector<bool>(n_qubits, true);
    }

    return qubits_to_measure;
}