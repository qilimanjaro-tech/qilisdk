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

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "../../libs/pybind.h"

// GCOV_EXCL_BR_START

#pragma GCC visibility push(default)

namespace qilisdk::expr {

// The node kinds, numbered in the order a canonical sum or product lists its operands
enum class Kind : uint8_t { Constant = 0, Variable = 1, Pow = 2, Mul = 3, Add = 4, Function = 5 };

// A number with the semantics of a Python int, float or complex
struct Num {
    enum class Tag : uint8_t { Int, Float, Complex };
    Tag tag = Tag::Int;
    int64_t i = 0;
    double re = 0.0;
    double im = 0.0;
    static Num integer(int64_t value);
    static Num real(double value);
    static Num complex(double real, double imag);
    bool is_real() const { return tag != Tag::Complex; }
};

Num num_add(const Num& a, const Num& b);
Num num_mul(const Num& a, const Num& b);
Num num_pow(const Num& base, const Num& exponent);
bool num_eq(const Num& a, const Num& b);
double num_abs(const Num& value);
Num float_if_real(const Num& value);
Num finalize(const Num& value);
bool num_from_py(py::handle obj, Num& out);
py::object num_to_py(const Num& value);

struct Node;
using NodePtr = std::shared_ptr<const Node>;

// An immutable node of the expression tree
struct Node {
    Kind kind = Kind::Constant;
    uint64_t hash = 0;
    Num value;                   // Constant: the number
    std::string label;           // Variable: its label; Function: its NAME
    bool idempotent = false;     // Variable: whether x * x == x
    bool custom_binary = false;  // Variable: whether it has its own to_binary
    bool parameter = false;      // Variable: whether it is a Parameter
    py::object owner;            // Variable: the Python variable; Function: its class
    std::vector<NodePtr> args;   // Add/Mul: operands; Pow: {base, exp}; Function: {arg}
};

struct NodeHash {
    size_t operator()(const NodePtr& node) const { return node->hash; }
};
struct NodeEqual {
    bool operator()(const NodePtr& a, const NodePtr& b) const;
};
using SubstitutionMap = std::unordered_map<NodePtr, NodePtr, NodeHash, NodeEqual>;

// The Python classes and settings the engine needs, set once by initialize()
struct ClassInfo {
    py::object cls;
    bool idempotent = false;
    bool custom_binary = false;
    bool parameter = false;
    std::string name;
};
struct Registry {
    py::object constant_cls;
    py::object add_cls;
    py::object mul_cls;
    py::object pow_cls;
    py::object expression_cls;
    py::object log_cls;
    py::object expression_to_binary;
    py::object non_polynomial_error;
    py::object not_supported_error;
    py::object numpy_generic;
    py::object empty_tuple;
    double tol = 1e-12;
    std::unordered_map<PyObject*, ClassInfo> classes;
};
Registry& registry();
void initialize(py::object constant_cls, py::object add_cls, py::object mul_cls, py::object pow_cls, py::object expression_cls, py::object log_cls, py::object non_polynomial_error, py::object not_supported_error, double tol);
const ClassInfo& class_info(py::handle cls);

// Node construction. The make_* functions trust their input to be canonical, the build_* functions canonicalize it.
NodePtr make_constant(const Num& value);
NodePtr make_variable(const std::string& label, py::object owner, const ClassInfo& info);
NodePtr make_node(Kind kind, std::vector<NodePtr> args);
NodePtr make_function(py::object cls, const std::string& name, NodePtr arg);
NodePtr build_add(const std::vector<NodePtr>& raw);
NodePtr build_mul(const std::vector<NodePtr>& raw);
NodePtr build_pow(const NodePtr& base, const NodePtr& exp);
NodePtr build_function(const py::object& cls, const NodePtr& arg);

// Structure
bool equal(const NodePtr& a, const NodePtr& b);
int compare(const NodePtr& a, const NodePtr& b);
uint64_t label_hash(const std::string& label);
std::optional<int64_t> int_exponent(const NodePtr& node);
std::pair<NodePtr, Num> peel(const NodePtr& node);

// Symbolic operations
NodePtr expand(const NodePtr& node);
NodePtr derivative(const NodePtr& node, const NodePtr& symbol);
NodePtr substitute(const NodePtr& node, const SubstitutionMap& mapping);
NodePtr to_binary(const NodePtr& node);
Num evaluate(const NodePtr& node, py::handle env);
int64_t degree(const NodePtr& node);
std::vector<NodePtr> free_symbols(const NodePtr& node);
std::vector<std::pair<NodePtr, int64_t>> monomial_factors(const NodePtr& node);
std::vector<std::pair<NodePtr, Num>> coefficients(const NodePtr& node);
Num get_constant(const NodePtr& node);
std::string repr(const NodePtr& node);

// Conversion between nodes and the Python objects that wrap them
NodePtr node_of(py::handle obj);
bool coerce(py::handle obj, NodePtr& out);
py::object wrap(const NodePtr& node);
py::object capsule(const NodePtr& node);

// The functions bound to Python, each taking and returning Python objects
namespace python {
py::object constant_node(py::handle value);
py::object node(int kind, py::iterable args);
py::object function(py::handle cls, py::handle arg);
int64_t variable_hash(const std::string& label);
py::object add(py::handle self, py::handle other);
py::object sub(py::handle self, py::handle other);
py::object rsub(py::handle self, py::handle other);
py::object mul(py::handle self, py::handle other);
py::object neg(py::handle self);
py::object truediv(py::handle self, py::handle other);
py::object rtruediv(py::handle self, py::handle other);
py::object pow(py::handle self, py::handle other);
py::object rpow(py::handle self, py::handle other);
py::object build_add(py::iterable raw);
py::object build_mul(py::iterable raw);
py::object build_pow(py::handle base, py::handle exp);
py::object evaluate(py::handle self, py::handle env);
py::list free_symbols(py::handle self);
py::list variables(py::handle self);
bool is_parameterized(py::handle self);
int64_t degree(py::handle self);
py::object derivative(py::handle self, py::handle symbol);
py::object expand(py::handle self);
py::object substitute(py::handle self, py::handle mapping);
py::object to_binary(py::handle self);
py::object get_constant(py::handle self);
py::dict as_coefficients_dict(py::handle self);
py::list monomial_factors(py::handle self);
py::list to_list(py::handle self);
py::tuple args(py::handle self);
py::object value(py::handle self);
py::object coefficient(py::handle self);
py::object monomial(py::handle self);
int64_t hash(py::handle self);
bool equal(py::handle self, py::handle other);
std::string repr(py::handle self);
}  // namespace python

}  // namespace qilisdk::expr

#pragma GCC visibility pop
