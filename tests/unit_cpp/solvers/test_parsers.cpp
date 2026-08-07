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

// GCOV_EXCL_BR_START

#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include <algorithm>
#include <string>
#include <vector>
#include "../../../src/qilisdk_cpp/solvers/parsers.h"

namespace {

// Look up the coefficient of the monomial over the given labels, or zero if there is no such monomial
double coefficient_of(const ParsedCostCpp& parsed, const std::vector<std::string>& labels) {
    std::vector<int> wanted;
    for (const std::string& label : labels) {
        const auto found = std::find(parsed.labels.begin(), parsed.labels.end(), label);
        if (found == parsed.labels.end()) {
            return 0.0;
        }
        wanted.push_back(static_cast<int>(found - parsed.labels.begin()));
    }
    std::sort(wanted.begin(), wanted.end());
    for (const auto& monomial : parsed.monomials) {
        if (monomial.first == wanted) {
            return monomial.second;
        }
    }
    return 0.0;
}

}  // namespace

TEST(SolverParsersTest, ParseQuboReadsTheMonomialsOfTheObjective) {
    py::gil_scoped_acquire gil;

    // Term arithmetic stores the objective expanded, and x is binary so x * x == x, leaving the
    // monomials of 3 * (2x + y + x * y) + 2
    py::exec(R"(
        from qilisdk.core.model import QUBO
        from qilisdk.core.variables import BinaryVariable
        x, y = BinaryVariable("x"), BinaryVariable("y")
        nested = QUBO("nested")
        nested.set_objective(3 * (x + y) * (x + 1) + 2)
    )");

    const ParsedCostCpp parsed = parse_qubo(py::globals()["nested"]);
    EXPECT_EQ(parsed.num_variables, 2);
    EXPECT_EQ(parsed.monomials.size(), 3u);
    EXPECT_DOUBLE_EQ(coefficient_of(parsed, {"x"}), 6.0);
    EXPECT_DOUBLE_EQ(coefficient_of(parsed, {"y"}), 3.0);
    EXPECT_DOUBLE_EQ(coefficient_of(parsed, {"x", "y"}), 3.0);
    EXPECT_DOUBLE_EQ(parsed.offset, 2.0);
}

TEST(SolverParsersTest, ParseQuboRejectsAModelThatIsNotAQubo) {
    py::gil_scoped_acquire gil;

    py::exec(R"(
        from qilisdk.core.model import Model
        knapsack = Model.knapsack(values=[5, 4], weights=[3, 2], max_weight=3)
    )");

    // The caller has to convert the model themselves, so that the penalization and the Lagrange
    // multipliers used to build the QUBO stay their choice
    EXPECT_THROW(parse_qubo(py::globals()["knapsack"]), py::value_error);
    EXPECT_NO_THROW(parse_qubo(py::globals()["knapsack"].attr("to_qubo")()));
}

TEST(SolverParsersTest, ParseQuboIncludesTheConstraintPenalties) {
    py::gil_scoped_acquire gil;

    py::exec(R"(
        from qilisdk.core.model import Model
        knapsack = Model.knapsack(values=[5, 4], weights=[3, 2], max_weight=3).to_qubo()
    )");

    // The weight constraint is penalized with slack variables, so the cost function covers more
    // variables than the two of the original objective
    const ParsedCostCpp parsed = parse_qubo(py::globals()["knapsack"]);
    EXPECT_GT(parsed.num_variables, 2);
    EXPECT_EQ(parsed.labels.size(), static_cast<std::size_t>(parsed.num_variables));
    EXPECT_FALSE(parsed.monomials.empty());
}

TEST(SolverParsersTest, ParseQuboCoversTheBitsOfAnEncodedVariable) {
    py::gil_scoped_acquire gil;

    py::exec(R"(
        from qilisdk.core.model import Model
        from qilisdk.core.variables import Domain, OneHot, Variable
        x = Variable("x", Domain.POSITIVE_INTEGER, bounds=(0, 3), encoding=OneHot)
        encoded = Model("encoded")
        encoded.set_objective((x - 2) * (x - 2))
        encoded = encoded.to_qubo()
    )");

    // One hot encodes the four values of x as four binary variables, and the QUBO is defined over
    // those bits rather than over x itself
    const ParsedCostCpp parsed = parse_qubo(py::globals()["encoded"]);
    EXPECT_EQ(parsed.num_variables, 4);
    std::vector<std::string> labels = parsed.labels;
    std::sort(labels.begin(), labels.end());
    EXPECT_EQ(labels, std::vector<std::string>({"x(0)", "x(1)", "x(2)", "x(3)"}));
}

TEST(SolverParsersTest, ParseQuboReadsAnObjectiveThatIsASingleProduct) {
    py::gil_scoped_acquire gil;

    py::exec(R"(
        from qilisdk.core.model import QUBO
        from qilisdk.core.variables import BinaryVariable
        product = QUBO("product")
        product.set_objective(2 * BinaryVariable("x") * BinaryVariable("y"))
    )");

    const ParsedCostCpp parsed = parse_qubo(py::globals()["product"]);
    EXPECT_EQ(parsed.num_variables, 2);
    EXPECT_EQ(parsed.monomials.size(), 1u);
    EXPECT_DOUBLE_EQ(coefficient_of(parsed, {"x", "y"}), 2.0);
    EXPECT_DOUBLE_EQ(parsed.offset, 0.0);
}

TEST(SolverParsersTest, ParseQuboRejectsAnObjectiveThatWasNotMultipliedOut) {
    py::gil_scoped_acquire gil;

    // Building the product by hand skips the Term arithmetic that would have multiplied the two sums
    // out, leaving an objective that is not a sum over its monomials
    py::exec(R"(
        from qilisdk.core.model import QUBO
        from qilisdk.core.variables import BinaryVariable, Operation, Term
        x, y = BinaryVariable("x"), BinaryVariable("y")
        nested = QUBO("nested")
        nested.set_objective(Term([Term([x, y], Operation.ADD), Term([x, 1], Operation.ADD)], Operation.MUL))
    )");

    EXPECT_THROW(parse_qubo(py::globals()["nested"]), py::value_error);
}

TEST(SolverParsersTest, ParseQuboRejectsUnsupportedOperations) {
    py::gil_scoped_acquire gil;

    // A division cannot be written as a polynomial over binary variables, and is already rejected
    // when the objective is set on the QUBO
    py::exec(R"(
        from qilisdk.core.model import QUBO
        from qilisdk.core.variables import BinaryVariable, Operation, Term
        a, b = BinaryVariable("a"), BinaryVariable("b")
        divided = QUBO("divided")
    )");

    EXPECT_ANY_THROW(py::exec(R"(divided.set_objective(Term([a, b], Operation.DIV)))"));
}

TEST(SolverParsersTest, ParseQuboRejectsComplexCoefficients) {
    py::gil_scoped_acquire gil;

    py::exec(R"(
        from qilisdk.core.model import QUBO
        from qilisdk.core.variables import BinaryVariable
        imaginary = QUBO("imaginary")
        imaginary.set_objective(2j * BinaryVariable("a"))
    )");

    EXPECT_THROW(parse_qubo(py::globals()["imaginary"]), py::value_error);
}

TEST(SolverParsersTest, SolveReturnsTheEvaluatedResultsAndSample) {
    py::gil_scoped_acquire gil;

    py::exec(R"(
        from qilisdk.core.model import QUBO
        from qilisdk.core.variables import BinaryVariable
        a, b = BinaryVariable("a"), BinaryVariable("b")
        rewarded = QUBO("rewarded")
        rewarded.set_objective(a + b - 3 * a * b)
    )");

    // Setting both variables is worth -1, every other assignment is worth 0 or more
    const py::object out = solve_with_simulated_annealing(py::globals()["rewarded"], 20, 200, 0.0, 0.0, 7, 1);
    const py::tuple pair = out.cast<py::tuple>();
    const py::dict results = pair[0].cast<py::dict>();
    const py::dict sample = pair[1].cast<py::dict>();

    EXPECT_DOUBLE_EQ(results["obj"].cast<double>(), -1.0);
    EXPECT_EQ(sample.size(), 2u);
    EXPECT_EQ(sample[py::globals()["a"]].cast<int>(), 1);
    EXPECT_EQ(sample[py::globals()["b"]].cast<int>(), 1);
}

TEST(SolverParsersTest, SolveSamplesEveryVariableOfTheQubo) {
    py::gil_scoped_acquire gil;

    py::exec(R"(
        from qilisdk.core.model import Model
        knapsack = Model.knapsack(values=[5, 4, 3], weights=[3, 2, 1], max_weight=3).to_qubo()
    )");

    // The slack variables of the penalized constraint are part of the QUBO, so they are part of the
    // sample as well
    const py::object out = solve_with_simulated_annealing(py::globals()["knapsack"], 20, 500, 0.0, 0.0, 7, 1);
    const py::dict sample = out.cast<py::tuple>()[1].cast<py::dict>();
    EXPECT_EQ(sample.size(), py::len(py::globals()["knapsack"].attr("variables")()));
    for (const auto& item : sample) {
        const int value = item.second.cast<int>();
        EXPECT_TRUE(value == 0 || value == 1);
    }
}

TEST(SolverParsersTest, SolveRejectsAModelThatIsNotAQubo) {
    py::gil_scoped_acquire gil;

    py::exec(R"(
        from qilisdk.core.model import Model
        plain = Model.random_ising(4, seed=1)
    )");

    EXPECT_THROW(solve_with_simulated_annealing(py::globals()["plain"], 10, 100, 0.0, 0.0, 1, 1), py::value_error);
}

// GCOV_EXCL_BR_STOP
