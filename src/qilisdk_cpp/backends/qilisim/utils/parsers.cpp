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
#include "../utils/matrix_utils.h"

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

NoiseModelCpp parse_noise_model(const py::object& noise_model, int nqubits, double atol) {
    /*
    Extract a NoiseModelCpp from a NoiseModel object.

    Args:
        noise_model (py::object): A NoiseModel object containing kraus operators.
        nqubits (int): The total number of qubits.
        atol (double): Absolute tolerance for numerical operations.

    Returns:
        NoiseModelCpp: The parsed noise model.
    */
    NoiseModelCpp noise_model_cpp;
    if (noise_model.is_none()) {
        return noise_model_cpp;
    }

    // Parse the noise config
    py::object noise_config = noise_model.attr("noise_config");
    float dt = noise_config.attr("default_gate_time").cast<float>();

    // Parse global noise passes
    for (auto& py_noise_pass : noise_model.attr("global_noise")) {

        // Parse the Kraus operators
        std::vector<SparseMatrix> kraus_operators;
        if (py::isinstance(py_noise_pass, SupportsStaticKraus)) {
            py::object as_kraus = py_noise_pass.attr("as_kraus")();
            for (auto& kraus_op : as_kraus.attr("operators")) {
                py::object spm = kraus_op.attr("data");
                SparseMatrix K = from_spmatrix(spm, atol);
                kraus_operators.push_back(K);
            }
        } else if (py::isinstance(py_noise_pass, SupportsTimeDerivedKraus)) {
            py::object as_kraus_from_duration = py_noise_pass.attr("as_kraus_from_duration")("duration"_a=dt);
            for (auto& kraus_op : as_kraus_from_duration.attr("operators")) {
                py::object spm = kraus_op.attr("data");
                SparseMatrix K = from_spmatrix(spm, atol);
                kraus_operators.push_back(K);
            }
        } 
        if (!kraus_operators.empty()) {
            noise_model_cpp.add_kraus_operators_global(kraus_operators);
        }

        // Parse jump operators
        std::vector<SparseMatrix> jump_operators;
        if (py::isinstance(py_noise_pass, SupportsStaticLindblad)) {
            py::object as_lindblad = py_noise_pass.attr("as_lindblad")();
            for (auto& lindblad_op : as_lindblad.attr("jump_operators")) {
                py::object spm = lindblad_op.attr("data");
                SparseMatrix L = from_spmatrix(spm, atol);
                jump_operators.push_back(L);
            }
        } else if (py::isinstance(py_noise_pass, SupportsTimeDerivedLindblad)) {
            py::object as_lindblad_from_duration = py_noise_pass.attr("as_lindblad_from_duration")("duration"_a=dt);
            for (auto& lindblad_op : as_lindblad_from_duration.attr("jump_operators")) {
                py::object spm = lindblad_op.attr("data");
                SparseMatrix L = from_spmatrix(spm, atol);
                jump_operators.push_back(L);
            }
        }
        for (const auto& L : jump_operators) {
            noise_model_cpp.add_jump_operator(expand_operator(nqubits, L));
        }

        // Parse the readout error
        if (py::isinstance(py_noise_pass, ReadoutAssignment)) {
            double p01 = py_noise_pass.attr("p01").cast<double>();
            double p10 = py_noise_pass.attr("p10").cast<double>();
            noise_model_cpp.add_readout_error_global(p01, p10);
        }

    }

    // Parse per-qubit noise passes
    py::dict per_qubit_noise_map = noise_model.attr("per_qubit_noise");
    for (auto item : per_qubit_noise_map) {
        int q = item.first.cast<int>();
        py::list py_noise_passes = item.second.cast<py::list>();
        for (auto& py_noise_pass : py_noise_passes) {

            // Parse the Kraus operators
            std::vector<SparseMatrix> kraus_operators;
            if (py::isinstance(py_noise_pass, SupportsStaticKraus)) {
                py::object as_kraus = py_noise_pass.attr("as_kraus")();
                for (auto& kraus_op : as_kraus.attr("operators")) {
                    py::object spm = kraus_op.attr("data");
                    SparseMatrix K = from_spmatrix(spm, atol);
                    kraus_operators.push_back(K);
                }
            } else if (py::isinstance(py_noise_pass, SupportsTimeDerivedKraus)) {
                py::object as_kraus_from_duration = py_noise_pass.attr("as_kraus_from_duration")("duration"_a=dt);
                for (auto& kraus_op : as_kraus_from_duration.attr("operators")) {
                    py::object spm = kraus_op.attr("data");
                    SparseMatrix K = from_spmatrix(spm, atol);
                    kraus_operators.push_back(K);
                }
            } 
            if (!kraus_operators.empty()) {
                noise_model_cpp.add_kraus_operators_per_qubit(q, kraus_operators);
            }

            // Parse jump operators
            std::vector<SparseMatrix> jump_operators;
            if (py::isinstance(py_noise_pass, SupportsStaticLindblad)) {
                py::object as_lindblad = py_noise_pass.attr("as_lindblad")();
                for (auto& lindblad_op : as_lindblad.attr("jump_operators")) {
                    py::object spm = lindblad_op.attr("data");
                    SparseMatrix L = from_spmatrix(spm, atol);
                    jump_operators.push_back(L);
                }
            } else if (py::isinstance(py_noise_pass, SupportsTimeDerivedLindblad)) {
                py::object as_lindblad_from_duration = py_noise_pass.attr("as_lindblad_from_duration")("duration"_a=dt);
                for (auto& lindblad_op : as_lindblad_from_duration.attr("jump_operators")) {
                    py::object spm = lindblad_op.attr("data");
                    SparseMatrix L = from_spmatrix(spm, atol);
                    jump_operators.push_back(L);
                }
            }
            for (const auto& L : jump_operators) {
                noise_model_cpp.add_jump_operator(expand_operator(q, nqubits, L));
            }

            // Parse the readout error
            if (py::isinstance(py_noise_pass, ReadoutAssignment)) {
                double p01 = py_noise_pass.attr("p01").cast<double>();
                double p10 = py_noise_pass.attr("p10").cast<double>();
                noise_model_cpp.add_readout_error_per_qubit(q, p01, p10);
            }

        }

    }

    // Parse per-gate noise passes
    py::dict gate_noise_map = noise_model.attr("per_gate_noise");
    for (auto& item : gate_noise_map) {
        std::string gate_name = item.first.attr("__name__").cast<std::string>();
        py::list py_noise_passes = item.second.cast<py::list>();
        for (auto& py_noise_pass : py_noise_passes) {

            // Parse the Kraus operators
            std::vector<SparseMatrix> kraus_operators;
            if (py::isinstance(py_noise_pass, SupportsStaticKraus)) {
                py::object as_kraus = py_noise_pass.attr("as_kraus")();
                for (auto& kraus_op : as_kraus.attr("operators")) {
                    py::object spm = kraus_op.attr("data");
                    SparseMatrix K = from_spmatrix(spm, atol);
                    kraus_operators.push_back(K);
                }
            } else if (py::isinstance(py_noise_pass, SupportsTimeDerivedKraus)) {
                py::object as_kraus_from_duration = py_noise_pass.attr("as_kraus_from_duration")("duration"_a=dt);
                for (auto& kraus_op : as_kraus_from_duration.attr("operators")) {
                    py::object spm = kraus_op.attr("data");
                    SparseMatrix K = from_spmatrix(spm, atol);
                    kraus_operators.push_back(K);
                }
            } 
            if (!kraus_operators.empty()) {
                noise_model_cpp.add_kraus_operators_per_gate(gate_name, kraus_operators);
            }

        }
    }

    // Parse per-gate-per-qubit noise passes
    py::dict gate_qubit_noise_map = noise_model.attr("per_gate_per_qubit_noise");
    for (auto& item : gate_qubit_noise_map) {
        py::handle ind_gate_tuple = item.first;
        std::string gate_name = ind_gate_tuple.attr("__getitem__")(0).attr("__name__").cast<std::string>();
        int qubit = ind_gate_tuple.attr("__getitem__")(1).cast<int>();
        py::list py_noise_passes = item.second.cast<py::list>();
        for (auto& py_noise_pass : py_noise_passes) {

            // Parse the Kraus operators
            std::vector<SparseMatrix> kraus_operators;
            if (py::isinstance(py_noise_pass, SupportsStaticKraus)) {
                py::object as_kraus = py_noise_pass.attr("as_kraus")();
                for (auto& kraus_op : as_kraus.attr("operators")) {
                    py::object spm = kraus_op.attr("data");
                    SparseMatrix K = from_spmatrix(spm, atol);
                    kraus_operators.push_back(K);
                }
            } else if (py::isinstance(py_noise_pass, SupportsTimeDerivedKraus)) {
                py::object as_kraus_from_duration = py_noise_pass.attr("as_kraus_from_duration")("duration"_a=dt);
                for (auto& kraus_op : as_kraus_from_duration.attr("operators")) {
                    py::object spm = kraus_op.attr("data");
                    SparseMatrix K = from_spmatrix(spm, atol);
                    kraus_operators.push_back(K);
                }
            } 
            if (!kraus_operators.empty()) {
                noise_model_cpp.add_kraus_operators_per_gate_qubit(gate_name, qubit, kraus_operators);
            }
        }
    }

    return noise_model_cpp;
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

std::vector<std::vector<double>> parse_coefficients(const py::object& schedule, const py::list& hamiltonians_keys, const py::object& steps) {
    /*
    Extract parameter lists from a schedule.

    Args:
        schedule (py::object): A Schedule object.
        hamiltonians_keys (py::list): The list of Hamiltonian keys.
        steps (py::object): A list of step objects.
        noise_model (py::object): A NoiseModel object.

    Returns:
        std::vector<std::vector<double>>: The list of parameter vectors.
    */
    py::object coeffs_full = schedule.attr("coefficients");
    std::vector<std::vector<double>> parameters_list;
    for (const auto& h_key : hamiltonians_keys) {
        py::object h_coeffs = coeffs_full[h_key];
        std::vector<double> param_vector;
        for (const auto& t : steps) {
            double coeff = h_coeffs[t].cast<double>();
            param_vector.push_back(coeff);
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

std::vector<Gate> parse_gates(const py::object& circuit, double atol, const py::object& noise_model) {
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

        // If we have a noise model, check if this gate has parameter perturbation noise and apply if so
        if (!noise_model.is_none() && py_gate.attr("is_parameterized").cast<bool>()) {
            
            // Get the parameters and noise maps
            py::dict gate_parameters = py_gate.attr("get_parameters")();
            py::dict global_noise_map = noise_model.attr("global_perturbations");
            py::dict gate_noise_map = noise_model.attr("per_gate_perturbations");
            py::object class_name = py_gate.attr("__class__");

            // For each parameter
            for (auto item : gate_parameters) {
                py::handle param_name = item.first;

                // Global
                if (global_noise_map.contains(param_name)) {
                    for (auto perturbation : global_noise_map[param_name]) {
                        double original_value = gate_parameters[param_name].cast<double>();
                        double new_value = perturbation.attr("perturb")(original_value).cast<double>();
                        gate_parameters[param_name] = new_value;
                    }
                }
                
                // Per gate
                py::tuple to_check = py::make_tuple(class_name, param_name);
                if (gate_noise_map.contains(to_check)) {
                    for (auto perturbation : gate_noise_map[to_check]) {
                        double original_value = gate_parameters[param_name].cast<double>();
                        double new_value = perturbation.attr("perturb")(original_value).cast<double>();
                        gate_parameters[param_name] = new_value;
                    }
                }

            }

            // Set the new parameters
            py_gate.attr("set_parameters")(gate_parameters);

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
        config.set_max_cache_size(solver_params["max_cache_size"].cast<int>());
    }
    if (solver_params.contains("seed")) {
        config.set_seed(solver_params["seed"].cast<int>());
    }
    if (solver_params.contains("atol")) {
        config.set_atol(solver_params["atol"].cast<double>());
    }
    if (solver_params.contains("arnoldi_dim")) {
        config.set_arnoldi_dim(solver_params["arnoldi_dim"].cast<int>());
    }
    if (solver_params.contains("num_arnoldi_substeps")) {
        config.set_num_arnoldi_substeps(solver_params["num_arnoldi_substeps"].cast<int>());
    }
    if (solver_params.contains("num_integrate_substeps")) {
        config.set_num_integrate_substeps(solver_params["num_integrate_substeps"].cast<int>());
    }
    if (solver_params.contains("evolution_method")) {
        config.set_method(solver_params["evolution_method"].cast<std::string>());
    }
    if (solver_params.contains("monte_carlo")) {
        config.set_monte_carlo(solver_params["monte_carlo"].cast<bool>());
    }
    if (solver_params.contains("num_monte_carlo_trajectories")) {
        config.set_num_monte_carlo_trajectories(solver_params["num_monte_carlo_trajectories"].cast<int>());
    }
    if (solver_params.contains("num_threads")) {
        config.set_num_threads(solver_params["num_threads"].cast<int>());
    }
    if (config.get_num_threads() <= 0) {
        config.set_num_threads(1);
    }
    config.validate();
    return config;
}