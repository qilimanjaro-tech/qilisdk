// Copyright 2026 Qilimanjaro Quantum Tech
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
#include <cmath>
#include <complex>
#include <unordered_map>
#include "../libs/logging.h"
#include "simulated_annealing.h"

// GCOV_EXCL_BR_START

#pragma GCC visibility push(default)

namespace {

// The tolerance below which we consider a value to be zero
double settings_atol() {
    return py::module_::import("qilisdk.settings").attr("get_settings")().attr("atol").cast<double>();
}

double assert_real(const py::object& number, double atol) {
    /*
    Read a Python number as a real value, rejecting any non-negligible imaginary part.

    Args:
        number (py::object&): the Python number to read.
        atol (double): the tolerance below which an imaginary part is considered negligible.

    Returns:
        double: the real part of the number.

    Raises:
        py::value_error: if the number has a non-negligible imaginary part.
    */
    const std::complex<double> value = number.cast<std::complex<double>>();
    if (std::abs(value.imag()) >= atol) {
        throw py::value_error("Complex Number encountered when expecting only real values to be present.");
    }
    return value.real();
}

void add_monomial(const py::object& monomial, double coefficient, ParsedCostCpp& parsed, std::unordered_map<std::string, int>& indices) {
    /*
    Add one monomial of a QUBO objective to the cost function.

    A monomial is a product of powers of variables, which ``Expression.monomial_factors`` hands over
    as (base, power) pairs, and every variable met is given the next free index.

    Args:
        monomial (py::object&): the monomial to add.
        coefficient (double): the coefficient in front of it.
        parsed (ParsedCostCpp&): the cost function being built.
        indices (std::unordered_map<std::string, int>&): the index of each variable, by label.

    Raises:
        py::value_error: if a factor of the monomial is not a variable, which only happens for an
                objective built by hand rather than with the usual operators.
    */

    // Give a variable the next free index, or return the index it already has, noting the Python
    // variable object the first time each label is seen so the sample can be rebuilt from it later
    const auto index_of = [&parsed, &indices](const std::string& label, const py::object& variable) {
        const auto inserted = indices.emplace(label, static_cast<int>(parsed.labels.size()));
        if (inserted.second) {
            parsed.labels.push_back(label);
            parsed.variables.push_back(variable);
        }
        return inserted.first->second;
    };

    std::vector<int> variables;
    for (py::handle factor_handle : monomial.attr("monomial_factors")()) {
        // The power of a factor is irrelevant, since x * x == x for a binary variable
        const py::object base = py::reinterpret_borrow<py::tuple>(factor_handle)[0];
        if (!py::isinstance(base, BaseVariable)) {
            throw py::value_error("A QUBO objective must be a sum of products of binary variables, but the monomial " + py::str(monomial).cast<std::string>() + " has the factor " + py::str(base).cast<std::string>() + ", which is not a variable.");
        }
        variables.push_back(index_of(base.attr("label").cast<std::string>(), base));
    }

    // A monomial with no variables left is a plain number, so it belongs in the constant offset
    if (variables.empty()) {
        parsed.offset += coefficient;  // GCOVR_EXCL_LINE

        // Otherwise add the monomial to the cost function
    } else {
        parsed.monomials.emplace_back(variables, coefficient);
    }
}

py::dict build_sample(const std::vector<py::object>& variables, const std::vector<int>& state) {
    /*
    Turn an annealed bitstring into a sample over the QUBO's own binary variables,
    i.e. go from our C++ representation back to the Python's.

    Args:
        variables (std::vector<py::object>&): the Python variable object at each cost-function index.
        state (std::vector<int>&): the annealed value of each binary variable of the cost function.

    Returns:
        py::dict: a dict mapping each of the QUBO's variables to its annealed value.
    */

    py::dict sample;
    for (std::size_t index = 0; index < variables.size(); ++index) {
        sample[variables[index]] = py::int_(state[index]);
    }
    return sample;
}

}  // namespace

ParsedCostCpp parse_qubo(const py::object& qubo) {
    /*
    Read a Python QUBO as the numeric cost function that a classical solver minimizes.

    A QUBO objective already folds the model's constraints in as penalties scaled by their Lagrange
    multipliers, but it is kept factored, since a product of sums is cheaper to carry around than the
    sum it multiplies out to. Expanding it gives a sum over its monomials, each of which is read out
    with its coefficient and has its variables numbered.

    Args:
        qubo (py::object&): the QUBO to read.

    Returns:
        ParsedCostCpp: the monomials of the cost function, its constant offset, and the label of each
                binary variable it is defined over.

    Raises:
        py::value_error: if the given model is not a QUBO.
    */

    // Make sure it's a QUBO
    if (!py::isinstance(qubo, QUBO)) {
        throw py::value_error("Simulated annealing requires a QUBO model, but got " + py::str(qubo.attr("__class__").attr("__name__")).cast<std::string>() + ". Convert the model first with model.to_qubo().");
    }

    // Get the objective
    const py::object objective = qubo.attr("qubo_objective");
    if (objective.is_none()) {
        throw py::value_error("Cannot solve a QUBO that has neither an objective nor any constraints.");  // GCOVR_EXCL_LINE
    }
    // Multiply the objective out, so that it is a flat sum over the monomials of the cost function
    const py::object term = objective.attr("term").attr("expand")();

    // Parse the objective into a C++ version of the cost function
    ParsedCostCpp parsed;
    std::unordered_map<std::string, int> indices;
    const double atol = settings_atol();
    parsed.offset = assert_real(term.attr("get_constant")(), atol);

    // Every entry of the expanded sum, bar the constant, is a monomial and its coefficient
    const py::dict coefficients = term.attr("as_coefficients_dict")().cast<py::dict>();
    for (const auto& item : coefficients) {
        add_monomial(py::reinterpret_borrow<py::object>(item.first), assert_real(py::reinterpret_borrow<py::object>(item.second), atol), parsed, indices);
    }
    parsed.num_variables = static_cast<int>(parsed.labels.size());
    return parsed;
}

py::object solve_with_simulated_annealing(const py::object& qubo, int num_reads, int num_sweeps, double beta_min, double beta_max, int seed, int num_threads) {
    /*
    Minimize a Python QUBO with simulated annealing.

    Args:
        qubo (py::object&): the QUBO to solve.
        num_reads (int): the number of independent anneals to run.
        num_sweeps (int): the number of sweeps over all variables in each anneal.
        beta_min (double): the inverse temperature to start each anneal at, or a non-positive value
                to derive a range from the coefficients of the cost function.
        beta_max (double): the inverse temperature to end each anneal at, or a non-positive value to
                derive a range from the coefficients of the cost function.
        seed (int): the seed of the random number generators, each read deriving its own from it.
        num_threads (int): the number of threads to distribute the reads over, or zero to let OpenMP
                decide.

    Returns:
        py::object: a tuple of the QUBO evaluated at the best solution found, and the sample mapping
                each of its variables to its value in that solution.

    Raises:
        py::value_error: if the given model is not a QUBO, or if the annealing settings are invalid.
    */

    // Parse the Python object into a C++ one
    const ParsedCostCpp parsed = parse_qubo(qubo);
    qilisdk::log_debug("[SimulatedAnnealing, C++] Read QUBO " + qubo.attr("label").cast<std::string>() + " as " + std::to_string(parsed.num_variables) + " binary variables and " + std::to_string(parsed.monomials.size()) + " monomials");

    // Run the annealing
    SimulatedAnnealingCpp annealer(parsed.num_variables, parsed.monomials, parsed.offset);
    qilisdk::log_debug("[SimulatedAnnealing, C++] Running " + std::to_string(num_reads) + " reads of " + std::to_string(num_sweeps) + " sweeps each, with inverse temperature range [" + std::to_string(beta_min) + ", " + std::to_string(beta_max) + "] and seed " + std::to_string(seed) + " over " + std::to_string(num_threads) + " threads");
    const AnnealingResultCpp result = annealer.anneal(num_reads, num_sweeps, beta_min, beta_max, seed, num_threads);
    qilisdk::log_debug("[SimulatedAnnealing, C++] Finished annealing, best solution found has energy " + std::to_string(result.energy));

    // Put the results back into a Python form
    qilisdk::log_trace("[SimulatedAnnealing, C++] Building sample from best solution found");
    const py::dict sample = build_sample(parsed.variables, result.state);
    qilisdk::log_trace("[SimulatedAnnealing, C++] Finished building sample");

    // Report the objective (and any constraint) values
    py::object results;
    if (py::len(qubo.attr("constraints")) == 0) {
        py::dict objective_values;
        objective_values[qubo.attr("objective").attr("label")] = annealer.energy(result.state);
        results = std::move(objective_values);
    } else {
        results = qubo.attr("evaluate")(sample);
    }
    return py::make_tuple(results, sample);
}

#pragma GCC visibility pop

// GCOV_EXCL_BR_STOP
