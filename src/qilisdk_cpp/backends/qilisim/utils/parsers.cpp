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

#include "parsers.h"
#include "../digital/gate.h"
#include "numpy.h"

std::vector<SparseMatrix> parse_hamiltonians(const py::object& Hs, double atol) {
    /*
    Extract Hamiltonian matrices from a list of QTensor objects.

    Args:
        Hs (py::object): A list of QTensor Hamiltonians.
        atol (double): Absolute tolerance for numerical operations.

    Returns:
        std::vector<SparseMatrix>: The list of Hamiltonian sparse matrices.
    */
    std::vector<SparseMatrix> hamiltonians;
    for (auto& hamiltonian : Hs) {
        py::object spm = hamiltonian.attr("to_matrix")();
        SparseMatrix H = from_spmatrix(spm, atol);
        hamiltonians.push_back(H);
    }
    return hamiltonians;
}

std::vector<SparseMatrix> parse_jump_operators(const py::object& jumps, double atol) {
    /*
    Extract jump operator matrices from a list of QTensor objects.

    Args:
        jumps (py::object): A list of QTensor jump operators.
        atol (double): Absolute tolerance for numerical operations.

    Returns:
        std::vector<SparseMatrix>: The list of jump operator sparse matrices.
    */
    std::vector<SparseMatrix> jump_matrices;
    for (auto jump : jumps) {
        py::object spm = jump.attr("to_matrix")();
        SparseMatrix J = from_spmatrix(spm, atol);
        jump_matrices.push_back(J);
    }
    return jump_matrices;
}

std::vector<SparseMatrix> parse_observables(const py::object& observables, long nqubits, double atol) {
    /*
    Extract observable matrices from a list of QTensor objects.

    Args:
        observables (py::object): A list of QTensor observables.
        nqubits (long): The total number of qubits.
        atol (double): Absolute tolerance for numerical operations.

    Returns:
        std::vector<SparseMatrix>: The list of observable sparse matrices.
    */
    std::vector<SparseMatrix> observable_matrices;
    for (auto obs : observables) {
        // Depending on the type of observable given
        if (py::isinstance(obs, Hamiltonian)) {
            // Get the matrix
            py::object spm = obs.attr("to_matrix")();
            SparseMatrix O = from_spmatrix(spm, atol);

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
            SparseMatrix O = from_numpy(matrix, atol);

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
            // Get the data directly if it's a QTensor
            py::object spm = obs.attr("data");
            SparseMatrix O = from_spmatrix(spm, atol);
            observable_matrices.push_back(O);

        } else {
            throw py::value_error("Observable type not recognized.");
        }
    }
    return observable_matrices;
}

std::vector<std::vector<double>> parse_parameters(const py::object& coeffs) {
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

std::vector<double> parse_time_steps(const py::object& steps) {
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

SparseMatrix parse_initial_state(const py::object& initial_state, double atol) {
    /*
    Extract the initial state from a QTensor object.

    Args:
        initial_state (py::object): The initial state as a QTensor.
        atol (double): Absolute tolerance for numerical operations.

    Returns:
        SparseMatrix: The initial state as a sparse matrix.
    */
    py::object spm = initial_state.attr("data");
    SparseMatrix rho = from_spmatrix(spm, atol);
    return rho;
}

std::vector<Gate> parse_gates(const py::object& circuit, double atol) {
    /*
    Extract gates from a circuit object.

    Args:
        circuit (py::object): The circuit object.
        atol (double): Absolute tolerance for numerical operations.

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
        SparseMatrix base_matrix = from_numpy(matrix, atol);

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

std::vector<bool> parse_measurements(const py::object& circuit) {
    /*
    Extract measurement qubit information from a circuit object.

    Args:
        circuit (py::object): The circuit object.

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

QiliSimConfig parse_solver_params(const py::dict& solver_params) {
    /*
    Extract QiliSimConfig parameters from a Python dictionary.

    Args:
        solver_params (py::dict): The dictionary of solver parameters.

    Returns:
        QiliSimConfig: The populated configuration object.
    */
    QiliSimConfig config;
    if (solver_params.contains("max_cache_size")) {
        config.max_cache_size = solver_params["max_cache_size"].cast<int>();
    }
    if (solver_params.contains("seed")) {
        config.seed = solver_params["seed"].cast<int>();
    }
    if (solver_params.contains("atol")) {
        config.atol = solver_params["atol"].cast<double>();
    }
    if (solver_params.contains("arnoldi_dim")) {
        config.arnoldi_dim = solver_params["arnoldi_dim"].cast<int>();
    }
    if (solver_params.contains("num_arnoldi_substeps")) {
        config.num_arnoldi_substeps = solver_params["num_arnoldi_substeps"].cast<int>();
    }
    if (solver_params.contains("num_integrate_substeps")) {
        config.num_integrate_substeps = solver_params["num_integrate_substeps"].cast<int>();
    }
    if (solver_params.contains("evolution_method")) {
        config.method = solver_params["evolution_method"].cast<std::string>();
    }
    if (solver_params.contains("monte_carlo")) {
        config.monte_carlo = solver_params["monte_carlo"].cast<bool>();
    }
    if (solver_params.contains("num_monte_carlo_trajectories")) {
        config.num_monte_carlo_trajectories = solver_params["num_monte_carlo_trajectories"].cast<int>();
    }
    if (solver_params.contains("num_threads")) {
        config.num_threads = solver_params["num_threads"].cast<int>();
    }
    if (config.num_threads <= 0) {
        config.num_threads = 1;
    }
    config.validate();
    return config;
}