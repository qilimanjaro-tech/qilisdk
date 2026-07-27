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

// The label Term gives the constant of an expression, i.e. the entry that is not a variable
std::string constant_label() {
    return Term.attr("CONST").attr("label").cast<std::string>();
}

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

void add_monomial(const py::object& monomial, double coefficient, ParsedCostCpp& parsed, std::unordered_map<std::string, int>& indices, const std::string& constant, double atol) {
    /*
    Add one entry of a QUBO objective to the cost function.

    An entry is either a single variable, the constant, or a product of variables, and every variable
    met is given the next free index.

    Args:
        monomial (py::object&): the variable, constant or product of variables to add.
        coefficient (double): the coefficient in front of it.
        parsed (ParsedCostCpp&): the cost function being built.
        indices (std::unordered_map<std::string, int>&): the index of each variable, by label.
        constant (std::string&): the label Term gives to the constant of an expression.
        atol (double): the tolerance below which an imaginary part is considered negligible.

    Raises:
        py::value_error: if a factor of the product is itself a sum, which only happens for an
                objective built by nesting Terms by hand rather than with the usual operators.
    */

    // Give a variable the next free index, or return the index it already has
    const auto index_of = [&parsed, &indices](const std::string& label) {
        const auto inserted = indices.emplace(label, static_cast<int>(parsed.labels.size()));
        if (inserted.second) {
            parsed.labels.push_back(label);
        }
        return inserted.first->second;
    };

    std::vector<int> variables;
    if (py::isinstance(monomial, Term)) {
        // If it's a term, it is a product of factors, which are either variables or the constant
        for (py::handle factor_handle : monomial) {
            // Make sure the factor is not a sum, which would be a nested Term and is not allowed in a QUBO
            const py::object factor = py::reinterpret_borrow<py::object>(factor_handle);
            if (py::isinstance(factor, Term)) {
                throw py::value_error("A QUBO objective must be a sum of products of binary variables, but the product " + py::str(monomial).cast<std::string>() + " has the sum " + py::str(factor).cast<std::string>() + " as a factor.");
            }
            const std::string label = factor.attr("label").cast<std::string>();

            // A product carries its numeric factor as the constant, so fold that into the coefficient
            if (label == constant) {
                coefficient *= assert_real(py::object(monomial[factor]), atol);
                continue;
            }

            // The exponent of a factor is irrelevant, since x * x == x for a binary variable
            variables.push_back(index_of(label));
        }
    } else {
        // A lone variable, unless it is the constant, whose coefficient is the constant itself
        const std::string label = monomial.attr("label").cast<std::string>();
        if (label != constant) {
            variables.push_back(index_of(label));
        }
    }

    // If no variables were found, the coefficient is the constant offset
    if (variables.empty()) {
        parsed.offset += coefficient;

        // Otherwise add the monomial to the cost function
    } else {
        parsed.monomials.emplace_back(variables, coefficient);
    }
}

py::dict build_sample(const py::object& qubo, const std::vector<int>& state, const std::vector<std::string>& labels) {
    /*
    Turn an annealed bitstring into a sample over the QUBO's own binary variables,
    i.e. go from our C++ representation back to the Python's.

    Args:
        qubo (py::object&): the QUBO that was annealed.
        state (std::vector<int>&): the annealed value of each binary variable of the cost function.
        labels (std::vector<std::string>&): the label of each binary variable of the cost function.

    Returns:
        py::dict: a dict mapping each of the QUBO's variables to its annealed value.
    */

    // Convert the annealed bitstring into a map from each variable's label to its value
    std::unordered_map<std::string, int> bits;
    for (std::size_t index = 0; index < labels.size(); ++index) {
        bits[labels[index]] = state[index];
    }

    // Build a dict mapping each of the QUBO's variables to its annealed value
    py::dict sample;
    for (py::handle variable_handle : qubo.attr("variables")()) {
        const py::object variable = py::reinterpret_borrow<py::object>(variable_handle);
        const auto found = bits.find(variable.attr("label").cast<std::string>());
        sample[variable] = py::int_(found == bits.end() ? 0 : found->second);
    }
    return sample;
}

}  // namespace

ParsedCostCpp parse_qubo(const py::object& qubo) {
    /*
    Read a Python QUBO as the numeric cost function that a classical solver minimizes.

    A QUBO objective already folds the model's constraints in as penalties scaled by their Lagrange
    multipliers, and is stored expanded, as a sum over its monomials: each entry of the objective term
    maps a variable, the constant, or a product of variables to the coefficient in front of it. So the
    entries only have to be read out and their variables numbered.

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
    const py::object term = objective.attr("term");
    const std::string operation = term.attr("operation").attr("value").cast<std::string>();

    // Parse the objective into a C++ version of the cost function
    ParsedCostCpp parsed;
    parsed.offset = 0.0;
    std::unordered_map<std::string, int> indices;
    const std::string constant = constant_label();
    const double atol = settings_atol();

    // If the whole objective is a single product, it's just one monomial
    if (operation == "*") {
        add_monomial(term, 1.0, parsed, indices, constant, atol);

        // If it's a sum, we have to process each
    } else if (operation == "+") {
        for (py::handle monomial_handle : term) {
            const py::object monomial = py::reinterpret_borrow<py::object>(monomial_handle);
            add_monomial(monomial, assert_real(py::object(term[monomial]), atol), parsed, indices, constant, atol);
        }
    } else {
        throw py::value_error("A QUBO objective must be a sum or a product, but got the operation " + operation + ".");  // GCOVR_EXCL_LINE
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
    const py::dict sample = build_sample(qubo, result.state, parsed.labels);
    qilisdk::log_trace("[SimulatedAnnealing, C++] Finished building sample");
    return py::make_tuple(qubo.attr("evaluate")(sample), sample);
}

#pragma GCC visibility pop

// GCOV_EXCL_BR_STOP
