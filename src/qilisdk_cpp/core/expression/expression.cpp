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

#include "expression.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>

// GCOV_EXCL_BR_START

namespace qilisdk::expr {

uint64_t compound_hash(Kind kind, const NodePtr* first, NodeSpan rest);
NodePtr make_node(Kind kind, const NodePtr* first, NodeSpan rest);

namespace {

const char* const capsule_name = "qilisdk.expression.node";
const char* const division_message = "Division by zero is not allowed in an Expression.";

// Attribute names, interned once
struct Names {
    py::str node = py::str("_node");
    py::str label = py::str("_label");
    py::str evaluate = py::str("evaluate");
    py::str to_binary = py::str("to_binary");
    py::str numeric = py::str("_numeric");
    py::str outer_derivative = py::str("_outer_derivative");
    py::str item = py::str("item");
};
const Names& names() {
    // Never destroyed, since the Python objects would outlive the interpreter
    static const Names* instance = new Names();
    return *instance;
}

[[noreturn]] void raise(const py::object& type, const std::string& message) {
    PyErr_SetString(type.ptr(), message.c_str());
    throw py::error_already_set();
}

uint64_t mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
    h ^= h >> 31;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 29;
    return h;
}

uint64_t hash_double(double value) {
    // Both zeros compare equal, and so does every NaN structurally, so each must hash alike
    if (value == 0.0) {
        value = 0.0;
    }
    if (std::isnan(value)) {
        value = std::numeric_limits<double>::quiet_NaN();
    }
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Structural sameness of two components, under which a NaN is the same as any other NaN
bool same_double(double x, double y) {
    return x == y || (std::isnan(x) && std::isnan(y));
}

// A total order for sorting, which puts NaN after every number
int compare_doubles(double x, double y) {
    if (same_double(x, y)) {
        return 0;
    }
    if (std::isnan(x) || (!std::isnan(y) && x > y)) {
        return 1;
    }
    return -1;
}

bool int_equals_double(int64_t i, double d) {
    if (!std::isfinite(d) || d != std::floor(d) || d < -9223372036854775808.0 || d >= 9223372036854775808.0) {
        return false;
    }
    return static_cast<int64_t>(d) == i;
}

// The first 8 bytes of a label as a big-endian integer, zero padded, so comparing keys orders labels like memcmp
uint64_t label_key(const std::string& label) {
    uint64_t key = 0;
    for (size_t i = 0; i < 8; ++i) {
        key = (key << 8) | (i < label.size() ? static_cast<unsigned char>(label[i]) : 0);
    }
    return key;
}

int compare_labels(const Node* a, const Node* b) {
    if (a->label_key != b->label_key) {
        return a->label_key < b->label_key ? -1 : 1;
    }
    // Equal keys settle labels of up to 8 bytes, where the zero padding makes the shorter one the smaller
    if (a->label.size() <= 8 && b->label.size() <= 8) {
        return a->label.size() == b->label.size() ? 0 : (a->label.size() < b->label.size() ? -1 : 1);
    }
    const int c = a->label.compare(b->label);
    return c == 0 ? 0 : (c < 0 ? -1 : 1);
}

// Whether two numbers are the very same value, type and bits alike, so 2 and 2.0 or 0.0 and -0.0 differ
bool same_number(const Num& a, const Num& b) {
    return a.tag == b.tag && a.i == b.i && std::memcmp(&a.re, &b.re, sizeof(double)) == 0 && std::memcmp(&a.im, &b.im, sizeof(double)) == 0;
}

int compare_constants(const Node* a, const Node* b) {
    const int by_real = compare_doubles(a->value.re, b->value.re);
    return by_real != 0 ? by_real : compare_doubles(a->value.im, b->value.im);
}

// Python's complex arithmetic, so results match it to the last bit
Num c_prod(const Num& a, const Num& b) {
    return Num::complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

Num c_quot(const Num& a, const Num& b) {
    const double abs_breal = std::fabs(b.re);
    const double abs_bimag = std::fabs(b.im);
    if (abs_breal >= abs_bimag) {
        if (abs_breal == 0.0) {
            throw py::value_error(division_message);
        }
        const double ratio = b.im / b.re;
        const double denom = b.re + b.im * ratio;
        return Num::complex((a.re + a.im * ratio) / denom, (a.im - a.re * ratio) / denom);
    }
    const double ratio = b.re / b.im;
    const double denom = b.re * ratio + b.im;
    return Num::complex((a.re * ratio + a.im) / denom, (a.im * ratio - a.re) / denom);
}

Num c_powu(const Num& x, int64_t n) {
    Num r = Num::complex(1.0, 0.0);
    Num p = x;
    for (int64_t mask = 1; mask > 0 && n >= mask; mask <<= 1) {
        if (n & mask) {
            r = c_prod(r, p);
        }
        p = c_prod(p, p);
    }
    return r;
}

Num complex_pow(const Num& a, const Num& b) {
    if (b.im == 0.0 && b.re == std::floor(b.re) && std::fabs(b.re) <= 100.0) {
        const auto n = static_cast<int64_t>(b.re);
        return n > 0 ? c_powu(a, n) : c_quot(Num::complex(1.0, 0.0), c_powu(a, -n));
    }
    if (a.re == 0.0 && a.im == 0.0) {
        if (b.im != 0.0 || b.re < 0.0) {
            throw py::value_error(division_message);
        }
        return Num::complex(0.0, 0.0);
    }
    const double vabs = std::hypot(a.re, a.im);
    double len = std::pow(vabs, b.re);
    const double at = std::atan2(a.im, a.re);
    double phase = at * b.re;
    if (b.im != 0.0) {
        len /= std::exp(at * b.im);
        phase += b.im * std::log(vabs);
    }
    return Num::complex(len * std::cos(phase), len * std::sin(phase));
}

// Python's base ** exponent, raising for a zero base under a negative exponent
Num safe_pow(const Num& base, const Num& exponent) {
    if (num_abs(base) < registry().tol && exponent.is_real() && exponent.re < 0) {
        throw py::value_error(division_message);
    }
    return num_pow(base, exponent);
}

Num assert_real(const Num& value) {
    const Num real = float_if_real(value);
    if (!real.is_real()) {
        throw py::value_error("Only real values are allowed but " + py::str(num_to_py(real)).cast<std::string>() + " was provided.");
    }
    return real;
}

NodePtr constant(const Num& value) {
    return make_constant(float_if_real(value));
}

// -1, 0 or 1. They are needed constantly (every plain factor of a product has exponent 1), so they are built once.
NodePtr constant_int(int64_t value) {
    static const auto* common = new std::array<NodePtr, 3>{make_constant(Num::integer(-1)), make_constant(Num::integer(0)), make_constant(Num::integer(1))};
    return (*common)[value + 1];
}

bool is_constant(const NodePtr& node, int64_t value) {
    return node->kind == Kind::Constant && num_eq(node->value, Num::integer(value));
}

bool is_pos_int_constant(const NodePtr& node) {
    const std::optional<int64_t> value = int_exponent(node);
    return value.has_value() && *value > 0;
}

// The factors of a canonical product, multiplied out over its sums
NodePtr mul_expand(const NodePtr& left, const NodePtr& right) {
    const NodeSpan left_terms = left->kind == Kind::Add ? left->args : NodeSpan(&left, 1);
    const NodeSpan right_terms = right->kind == Kind::Add ? right->args : NodeSpan(&right, 1);
    std::vector<NodePtr> products;
    products.reserve(left_terms.size() * right_terms.size());
    for (const NodePtr& x : left_terms) {
        for (const NodePtr& y : right_terms) {
            products.push_back(build_mul({x, y}));
        }
    }
    return build_add(products);
}

// A vector that keeps its first N elements inline and stays contiguous, so the short lists of a typical product
// never touch the allocator
template <typename T, size_t N>
class SmallVector {
   public:
    T* begin() { return spilled_ ? heap_.data() : inline_.data(); }
    T* end() { return begin() + size(); }
    size_t size() const { return spilled_ ? heap_.size() : size_; }
    void push_back(T value) {
        if (!spilled_ && size_ < N) {
            inline_[size_++] = std::move(value);
            return;
        }
        if (!spilled_) {
            heap_.reserve(2 * N);
            std::move(inline_.begin(), inline_.end(), std::back_inserter(heap_));
            spilled_ = true;
        }
        heap_.push_back(std::move(value));
    }

   private:
    std::array<T, N> inline_{};
    std::vector<T> heap_;
    size_t size_ = 0;
    bool spilled_ = false;
};

// Sorts canonically and stably. std::stable_sort allocates a buffer on every call, which for the handful of
// operands of a typical node costs more than an insertion sort does.
void sort_canonically(NodePtr* begin, NodePtr* end) {
    const auto less = [](const NodePtr& a, const NodePtr& b) { return compare(a, b) < 0; };
    if (std::is_sorted(begin, end, less)) {
        return;
    }
    if (end - begin > 16) {
        std::stable_sort(begin, end, less);
        return;
    }
    for (NodePtr* it = begin + 1; it != end; ++it) {
        NodePtr value = std::move(*it);
        NodePtr* hole = it;
        while (hole != begin && less(value, *(hole - 1))) {
            *hole = std::move(*(hole - 1));
            --hole;
        }
        *hole = std::move(value);
    }
}

// The base is a canonical monomial without a coefficient, so prepending one keeps it canonical
NodePtr scale(const NodePtr& base, const Num& coeff) {
    const NodePtr constant_factor = make_constant(coeff);
    return make_node(Kind::Mul, &constant_factor, base->kind == Kind::Mul ? base->args : NodeSpan(&base, 1));
}

// A summand split into its coefficient and the factors of its monomial, viewed in place rather than built.
// A product without a coefficient is viewed as its factors too, so it matches the same monomial scaled.
struct Summand {
    const NodePtr* term = nullptr;
    NodeSpan factors;
    uint64_t hash = 0;
    Num coeff;
    bool reusable = true;
};

Summand view_summand(const NodePtr& term) {
    Summand summand;
    summand.term = &term;
    summand.coeff = Num::integer(1);
    if (term->kind != Kind::Mul) {
        summand.factors = NodeSpan(&term, 1);
        summand.hash = term->hash;
        return summand;
    }
    const bool scaled = term->args[0]->kind == Kind::Constant;
    summand.factors = scaled ? NodeSpan(term->args.begin() + 1, term->args.size() - 1) : term->args;
    if (scaled) {
        summand.coeff = term->args[0]->value;
        // Adding to 0 turns a complex coefficient's -0.0 real part into 0.0, which reusing the node would not show
        summand.reusable = summand.coeff.is_real() || !(summand.coeff.re == 0.0 && std::signbit(summand.coeff.re));
    }
    // A single factor stands alone, otherwise the factors hash as the product they form
    summand.hash = summand.factors.size() == 1 ? summand.factors[0]->hash : compound_hash(Kind::Mul, nullptr, summand.factors);
    return summand;
}

bool same_factors(NodeSpan a, NodeSpan b) {
    if (a.size() != b.size()) {
        return false;  // GCOV_EXCL_LINE (only called on equal hashes, which cover the size, so this needs a hash collision)
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (!equal(a[i], b[i])) {
            return false;  // GCOV_EXCL_LINE (likewise only reachable through a hash collision)
        }
    }
    return true;
}

// The node for a combined summand, reusing the original whenever nothing merged into it
NodePtr rebuild_summand(const Summand& summand, const Num& coeff) {
    if (summand.reusable) {
        return *summand.term;
    }
    if (!num_eq(coeff, Num::integer(1))) {
        const NodePtr constant_factor = make_constant(coeff);
        return make_node(Kind::Mul, &constant_factor, summand.factors);
    }
    if (summand.factors.size() == 1) {
        return summand.factors[0];
    }
    return (*summand.term)->kind == Kind::Mul && (*summand.term)->args.size() == summand.factors.size() ? *summand.term : make_node(Kind::Mul, summand.factors);
}

// Whether a summand is some coefficient times base, without building its monomial
bool has_base(const NodePtr& term, const NodePtr& base) {
    if (term->kind != Kind::Mul || term->args[0]->kind != Kind::Constant) {
        return equal(term, base);
    }
    if (term->args.size() == 2) {
        return equal(term->args[1], base);
    }
    if (base->kind != Kind::Mul || base->args.size() != term->args.size() - 1) {
        return false;
    }
    for (size_t i = 0; i < base->args.size(); ++i) {
        if (!equal(term->args[i + 1], base->args[i])) {
            return false;
        }
    }
    return true;
}

// Adds a single non-sum term to a canonical sum by merging it into the sorted operands, which gives the same
// result as build_add but keeps a running sum() linear rather than rebuilding its index on every step
NodePtr add_to_sum(const NodePtr& sum, const NodePtr& term, bool sum_first) {
    std::vector<NodePtr> out(sum->args.begin(), sum->args.end());
    const auto less = [](const NodePtr& a, const NodePtr& b) { return compare(a, b) < 0; };
    const auto insert_sorted = [&](const NodePtr& node) { out.insert(std::upper_bound(out.begin(), out.end(), node, less), node); };
    const bool has_constant = out[0]->kind == Kind::Constant;

    if (term->kind == Kind::Constant) {
        const Num existing = has_constant ? out[0]->value : Num::integer(0);
        const Num total = float_if_real(sum_first ? num_add(num_add(Num::integer(0), existing), term->value) : num_add(num_add(Num::integer(0), term->value), existing));
        if (has_constant) {
            out.erase(out.begin());
        }
        if (!num_eq(total, Num::integer(0))) {
            out.insert(out.begin(), make_constant(total));
        }
    } else {
        auto [base, coeff] = peel(term);
        auto match = std::find_if(out.begin() + (has_constant ? 1 : 0), out.end(), [&](const NodePtr& existing) { return has_base(existing, base); });
        if (match != out.end()) {
            auto [existing_base, existing_coeff] = peel(*match);
            coeff = sum_first ? num_add(num_add(Num::integer(0), existing_coeff), coeff) : num_add(num_add(Num::integer(0), coeff), existing_coeff);
            // The sum keeps whichever base it met first
            if (sum_first) {
                base = existing_base;
            }
            out.erase(match);
        } else {
            coeff = num_add(Num::integer(0), coeff);
        }
        coeff = float_if_real(coeff);
        if (!num_eq(coeff, Num::integer(0))) {
            insert_sorted(num_eq(coeff, Num::integer(1)) ? base : scale(base, coeff));
        }
    }

    if (out.empty()) {
        return constant_int(0);
    }
    if (out.size() == 1) {
        return out[0];
    }
    return make_node(Kind::Add, out);
}

bool contains_symbol(const NodePtr& node, const NodePtr& symbol) {
    if (node->kind == Kind::Variable) {
        return equal(node, symbol);
    }
    for (const NodePtr& arg : node->args) {
        if (contains_symbol(arg, symbol)) {
            return true;
        }
    }
    return false;
}

bool contains_any_symbol(const NodePtr& node) {
    if (node->kind == Kind::Variable) {
        return true;
    }
    for (const NodePtr& arg : node->args) {
        if (contains_any_symbol(arg)) {
            return true;
        }
    }
    return false;
}

void collect_symbols(const NodePtr& node, std::vector<NodePtr>& out, std::unordered_set<NodePtr, NodeHash, NodeEqual>& seen) {
    if (node->kind == Kind::Variable) {
        if (seen.insert(node).second) {
            out.push_back(node);
        }
        return;
    }
    for (const NodePtr& arg : node->args) {
        collect_symbols(arg, out, seen);
    }
}

NodePtr to_binary(const NodePtr& node, std::unordered_map<std::string, NodePtr>& memo) {
    switch (node->kind) {
        case Kind::Constant:
            return node;
        case Kind::Variable: {
            if (!node->custom_binary) {
                return node;
            }
            auto found = memo.find(node->label);
            if (found != memo.end()) {
                return found->second;
            }
            NodePtr result;
            if (!coerce(node->owner.attr(names().to_binary)(), result)) {
                throw py::type_error("to_binary of " + node->label + " did not return an Expression.");
            }
            memo.emplace(node->label, result);
            return result;
        }
        case Kind::Pow:
            return build_pow(to_binary(node->args[0], memo), to_binary(node->args[1], memo));
        case Kind::Function:
            return build_function(node->owner, to_binary(node->args[0], memo));
        default: {
            std::vector<NodePtr> args;
            args.reserve(node->args.size());
            for (const NodePtr& arg : node->args) {
                args.push_back(to_binary(arg, memo));
            }
            return node->kind == Kind::Add ? build_add(args) : build_mul(args);
        }
    }
}

Num evaluate(const NodePtr& node, py::handle env, std::unordered_map<std::string, Num>& memo) {
    switch (node->kind) {
        case Kind::Constant:
            return finalize(node->value);
        case Kind::Variable: {
            auto found = memo.find(node->label);
            if (found != memo.end()) {
                return found->second;
            }
            Num value;
            if (!num_from_py(node->owner.attr(names().evaluate)(env), value)) {
                throw py::type_error("Variable " + node->label + " did not evaluate to a number.");
            }
            memo.emplace(node->label, value);
            return value;
        }
        case Kind::Add: {
            Num total = Num::integer(0);
            for (const NodePtr& arg : node->args) {
                total = num_add(total, evaluate(arg, env, memo));
            }
            return finalize(total);
        }
        case Kind::Mul: {
            Num total = Num::integer(1);
            for (const NodePtr& arg : node->args) {
                total = num_mul(total, evaluate(arg, env, memo));
            }
            return finalize(total);
        }
        case Kind::Pow: {
            const Num base = evaluate(node->args[0], env, memo);
            return finalize(safe_pow(base, evaluate(node->args[1], env, memo)));
        }
        case Kind::Function: {
            const Num arg = assert_real(evaluate(node->args[0], env, memo));
            Num value;
            if (!num_from_py(node->owner.attr(names().numeric)(num_to_py(arg)), value)) {
                throw py::type_error(node->label + " did not evaluate to a number.");
            }
            return finalize(value);
        }
    }
    return Num();  // GCOV_EXCL_LINE (every kind is handled above)
}

std::string repr_number(const Num& value) {
    return py::repr(num_to_py(float_if_real(value))).cast<std::string>();
}

// The positive form of a negative summand, so a sum prints "a - b" rather than "a + -b"
NodePtr negate_for_repr(const NodePtr& term) {
    if (term->kind == Kind::Mul) {
        const Num coefficient = peel(term).second;
        if (coefficient.is_real() && coefficient.re < 0) {
            return build_mul({constant_int(-1), term});
        }
    }
    return nullptr;
}

std::vector<NodePtr> nodes_of(py::iterable raw) {
    std::vector<NodePtr> nodes;
    for (py::handle item : raw) {
        NodePtr node;
        if (!coerce(item, node)) {
            throw py::type_error("Expected an Expression or a number, got " + py::str(py::type::of(item).attr("__name__")).cast<std::string>() + ".");
        }
        nodes.push_back(node);
    }
    return nodes;
}

}  // namespace

// ---------------------------------------------------------------- numbers

Num Num::integer(int64_t value) {
    Num n;
    n.tag = Tag::Int;
    n.i = value;
    n.re = static_cast<double>(value);
    return n;
}

Num Num::real(double value) {
    Num n;
    n.tag = Tag::Float;
    n.re = value;
    return n;
}

Num Num::complex(double real, double imag) {
    Num n;
    n.tag = Tag::Complex;
    n.re = real;
    n.im = imag;
    return n;
}

Num num_add(const Num& a, const Num& b) {
    if (a.tag == Num::Tag::Int && b.tag == Num::Tag::Int) {
        int64_t result = 0;
        if (!__builtin_add_overflow(a.i, b.i, &result)) {
            return Num::integer(result);
        }
        return Num::real(a.re + b.re);
    }
    if (a.is_real() && b.is_real()) {
        return Num::real(a.re + b.re);
    }
    return Num::complex(a.re + b.re, a.im + b.im);
}

Num num_mul(const Num& a, const Num& b) {
    if (a.tag == Num::Tag::Int && b.tag == Num::Tag::Int) {
        int64_t result = 0;
        if (!__builtin_mul_overflow(a.i, b.i, &result)) {
            return Num::integer(result);
        }
        return Num::real(a.re * b.re);
    }
    if (a.is_real() && b.is_real()) {
        return Num::real(a.re * b.re);
    }
    return c_prod(a, b);
}

Num num_pow(const Num& base, const Num& exponent) {
    if (base.tag == Num::Tag::Int && exponent.tag == Num::Tag::Int && exponent.i >= 0) {
        int64_t result = 1;
        int64_t factor = base.i;
        bool overflow = false;
        for (int64_t e = exponent.i; e > 0; e >>= 1) {
            if (e & 1) {
                overflow |= __builtin_mul_overflow(result, factor, &result);
            }
            if (e > 1) {
                overflow |= __builtin_mul_overflow(factor, factor, &factor);
            }
        }
        return overflow ? Num::real(std::pow(base.re, exponent.re)) : Num::integer(result);
    }
    if (base.is_real() && exponent.is_real()) {
        // Like Python, a negative base under a fractional power gives a complex number
        if (base.re < 0 && std::isfinite(exponent.re) && exponent.re != std::floor(exponent.re)) {
            return complex_pow(Num::complex(base.re, 0.0), Num::complex(exponent.re, 0.0));
        }
        return Num::real(std::pow(base.re, exponent.re));
    }
    return complex_pow(base, exponent);
}

bool num_eq(const Num& a, const Num& b) {
    if (a.tag == Num::Tag::Int && b.tag == Num::Tag::Int) {
        return a.i == b.i;
    }
    if (a.im != b.im) {
        return false;
    }
    if (a.tag == Num::Tag::Int) {
        return int_equals_double(a.i, b.re);
    }
    if (b.tag == Num::Tag::Int) {
        return int_equals_double(b.i, a.re);
    }
    return a.re == b.re;
}

double num_abs(const Num& value) {
    return value.is_real() ? std::fabs(value.re) : std::hypot(value.re, value.im);
}

Num float_if_real(const Num& value) {
    if (!value.is_real() && std::fabs(value.im) < registry().tol) {
        return Num::real(value.re);
    }
    return value;
}

Num finalize(const Num& value) {
    if (value.tag == Num::Tag::Int) {
        return Num::real(value.re);
    }
    return float_if_real(value);
}

bool num_from_py(py::handle obj, Num& out) {
    PyObject* ptr = obj.ptr();
    if (PyBool_Check(ptr)) {
        out = Num::integer(ptr == Py_True ? 1 : 0);
        return true;
    }
    if (PyLong_Check(ptr)) {
        int overflow = 0;
        const long long value = PyLong_AsLongLongAndOverflow(ptr, &overflow);
        if (overflow == 0) {
            out = Num::integer(value);
            return true;
        }
        // Beyond 64 bits, fall back to the nearest float
        const double approx = PyLong_AsDouble(ptr);
        if (approx == -1.0 && PyErr_Occurred()) {
            throw py::error_already_set();
        }
        out = Num::real(approx);
        return true;
    }
    if (PyFloat_Check(ptr)) {
        out = Num::real(PyFloat_AS_DOUBLE(ptr));
        return true;
    }
    if (PyComplex_Check(ptr)) {
        out = Num::complex(PyComplex_RealAsDouble(ptr), PyComplex_ImagAsDouble(ptr));
        return true;
    }
    if (registry().numpy_generic && PyObject_IsInstance(ptr, registry().numpy_generic.ptr()) == 1) {
        return num_from_py(obj.attr(names().item)(), out);
    }
    return false;
}

py::object num_to_py(const Num& value) {
    switch (value.tag) {
        case Num::Tag::Int:
            return py::int_(value.i);
        case Num::Tag::Float:
            return py::float_(value.re);
        case Num::Tag::Complex:
            return py::reinterpret_steal<py::object>(PyComplex_FromDoubles(value.re, value.im));
    }
    return py::none();  // GCOV_EXCL_LINE (every tag is handled above)
}

// ---------------------------------------------------------------- registry

Registry& registry() {
    // Never destroyed, since the Python objects would outlive the interpreter
    static Registry* instance = new Registry();
    return *instance;
}

void initialize(py::object constant_cls, py::object add_cls, py::object mul_cls, py::object pow_cls, py::object expression_cls, py::object log_cls, py::object non_polynomial_error, py::object not_supported_error, double tol) {
    Registry& reg = registry();
    reg.constant_cls = std::move(constant_cls);
    reg.add_cls = std::move(add_cls);
    reg.mul_cls = std::move(mul_cls);
    reg.pow_cls = std::move(pow_cls);
    reg.expression_to_binary = expression_cls.attr("to_binary");
    reg.expression_cls = std::move(expression_cls);
    reg.log_cls = std::move(log_cls);
    reg.non_polynomial_error = std::move(non_polynomial_error);
    reg.not_supported_error = std::move(not_supported_error);
    reg.numpy_generic = py::module_::import("numpy").attr("generic");
    reg.empty_tuple = py::tuple();
    reg.tol = tol;
    reg.classes.clear();
}

const ClassInfo& class_info(py::handle cls) {
    Registry& reg = registry();
    auto found = reg.classes.find(cls.ptr());
    if (found != reg.classes.end()) {
        return found->second;
    }
    ClassInfo info;
    info.cls = py::reinterpret_borrow<py::object>(cls);
    info.idempotent = py::getattr(cls, "is_idempotent_under_mul", py::bool_(false)).cast<bool>();
    info.parameter = py::getattr(cls, "is_parameter", py::bool_(false)).cast<bool>();
    info.custom_binary = !py::getattr(cls, "to_binary", py::none()).is(reg.expression_to_binary);
    info.name = py::str(py::getattr(cls, "NAME", py::str(""))).cast<std::string>();
    return reg.classes.emplace(cls.ptr(), std::move(info)).first->second;
}

// ---------------------------------------------------------------- construction

uint64_t label_hash(const std::string& label) {
    uint64_t h = 0xCBF29CE484222325ULL;
    for (unsigned char c : label) {
        h = (h ^ c) * 0x100000001B3ULL;
    }
    return mix(static_cast<uint64_t>(Kind::Variable), h);
}

uint64_t compound_hash(Kind kind, const NodePtr* first, NodeSpan rest) {
    uint64_t h = mix(static_cast<uint64_t>(kind), (first != nullptr ? 1 : 0) + rest.size());
    if (first != nullptr) {
        h = mix(h, (*first)->hash);
    }
    for (const NodePtr& arg : rest) {
        h = mix(h, arg->hash);
    }
    return h;
}

// Allocates a node with its operands, an optional first one and then the rest, copied into the same block right after it
Node* allocate(Kind kind, const NodePtr* first, NodeSpan rest) {
    const size_t count = (first != nullptr ? 1 : 0) + rest.size();
    void* memory = ::operator new(sizeof(Node) + count * sizeof(NodePtr));
    // Default- rather than value-initialized, which would zero the whole node before constructing it
    Node* node = new (memory) Node;
    node->kind = kind;
    auto* storage = reinterpret_cast<NodePtr*>(node + 1);
    size_t i = 0;
    if (first != nullptr) {
        new (storage + i++) NodePtr(*first);
    }
    for (const NodePtr& arg : rest) {
        new (storage + i++) NodePtr(arg);
    }
    node->args = NodeSpan(storage, count);
    return node;
}

NodePtr make_node(Kind kind, const NodePtr* first, NodeSpan rest) {
    Node* node = allocate(kind, first, rest);
    node->hash = compound_hash(kind, first, rest);
    return NodePtr(node);
}

void release(Node* node) {
    auto* storage = reinterpret_cast<NodePtr*>(node + 1);
    for (size_t i = 0; i < node->args.size(); ++i) {
        storage[i].~NodePtr();
    }
    node->~Node();
    ::operator delete(node);
}

NodePtr make_constant(const Num& value) {
    const uint64_t hash = mix(mix(static_cast<uint64_t>(Kind::Constant), hash_double(value.re)), hash_double(value.im));
    // Coefficients repeat a lot, so recently made constants are shared through a small direct-mapped cache. It
    // only matches the exact same number, keeping 2 and 2.0 apart, and holds no Python objects.
    static auto* cache = new std::array<NodePtr, 1024>();
    NodePtr& cached = (*cache)[hash & (cache->size() - 1)];
    if (cached && same_number(cached->value, value)) {
        return cached;
    }
    Node* node = allocate(Kind::Constant, nullptr, {});
    node->value = value;
    node->hash = hash;
    cached = NodePtr(node);
    return cached;
}

NodePtr make_variable(const std::string& label, py::object owner, const ClassInfo& info) {
    Node* node = allocate(Kind::Variable, nullptr, {});
    node->label = label;
    node->idempotent = info.idempotent;
    node->custom_binary = info.custom_binary;
    node->parameter = info.parameter;
    node->owner = std::move(owner);
    node->hash = label_hash(label);
    node->label_key = label_key(label);
    return NodePtr(node);
}

NodePtr make_node(Kind kind, NodeSpan args) {
    return make_node(kind, nullptr, args);
}

NodePtr make_function(py::object cls, const std::string& name, NodePtr arg) {
    Node* node = allocate(Kind::Function, &arg, {});
    node->label = name;
    node->label_key = label_key(name);
    node->owner = std::move(cls);
    node->hash = mix(mix(static_cast<uint64_t>(Kind::Function), label_hash(name)), arg->hash);
    return NodePtr(node);
}

NodePtr build_add(NodeSpan raw) {
    if (raw.size() == 2 && (raw[0]->kind == Kind::Add) != (raw[1]->kind == Kind::Add)) {
        const bool sum_first = raw[0]->kind == Kind::Add;
        return add_to_sum(raw[sum_first ? 0 : 1], raw[sum_first ? 1 : 0], sum_first);
    }

    Num constant_term = Num::integer(0);
    std::vector<Summand> summands;
    summands.reserve(raw.size());
    // Open addressing with linear probing over indices into summands, plus one so zero means empty
    std::vector<uint32_t> slots(16, 0);
    size_t mask = slots.size() - 1;

    const auto insert = [&](const NodePtr& term) {
        Summand summand = view_summand(term);
        size_t slot = summand.hash & mask;
        while (slots[slot] != 0) {
            Summand& existing = summands[slots[slot] - 1];
            if (existing.hash == summand.hash && same_factors(existing.factors, summand.factors)) {
                existing.coeff = num_add(existing.coeff, summand.coeff);
                existing.reusable = false;
                return;
            }
            slot = (slot + 1) & mask;
        }
        summand.coeff = num_add(Num::integer(0), summand.coeff);
        summands.push_back(summand);
        slots[slot] = static_cast<uint32_t>(summands.size());
        // Keep the table at most half full
        if (summands.size() * 2 > slots.size()) {
            slots.assign(slots.size() * 2, 0);
            mask = slots.size() - 1;
            for (size_t i = 0; i < summands.size(); ++i) {
                size_t free_slot = summands[i].hash & mask;
                while (slots[free_slot] != 0) {
                    free_slot = (free_slot + 1) & mask;
                }
                slots[free_slot] = static_cast<uint32_t>(i + 1);
            }
        }
    };

    // Flatten nested sums depth first, in the order their operands appear
    const auto flatten = [&](const auto& self, NodeSpan terms) -> void {
        for (const NodePtr& term : terms) {
            if (term->kind == Kind::Constant) {
                constant_term = num_add(constant_term, term->value);
            } else if (term->kind == Kind::Add) {
                self(self, term->args);
            } else {
                insert(term);
            }
        }
    };
    flatten(flatten, raw);

    std::vector<NodePtr> out;
    out.reserve(summands.size() + 1);
    for (const Summand& summand : summands) {
        const Num coeff = float_if_real(summand.coeff);
        if (!num_eq(coeff, Num::integer(0))) {
            out.push_back(rebuild_summand(summand, coeff));
        }
    }
    constant_term = float_if_real(constant_term);
    if (!num_eq(constant_term, Num::integer(0))) {
        out.push_back(make_constant(constant_term));
    }

    if (out.empty()) {
        return constant_int(0);
    }
    if (out.size() == 1) {
        return out[0];
    }
    sort_canonically(out.data(), out.data() + out.size());
    return make_node(Kind::Add, out);
}

NodePtr build_mul(NodeSpan raw) {
    Num coefficient = Num::integer(1);
    SmallVector<std::pair<NodePtr, NodePtr>, 8> powers;

    // Flatten nested products depth first, in the order their factors appear
    const auto flatten = [&](const auto& self, NodeSpan factors) -> void {
        for (const NodePtr& factor : factors) {
            if (factor->kind == Kind::Constant) {
                coefficient = num_mul(coefficient, factor->value);
                continue;
            }
            if (factor->kind == Kind::Mul) {
                self(self, factor->args);
                continue;
            }
            const bool is_pow = factor->kind == Kind::Pow;
            const NodePtr& base = is_pow ? factor->args[0] : factor;
            // Products are short, so a linear scan beats hashing
            auto existing = std::find_if(powers.begin(), powers.end(), [&](const auto& entry) { return equal(entry.first, base); });
            if (existing == powers.end()) {
                powers.push_back({base, is_pow ? factor->args[1] : constant_int(1)});
            } else {
                existing->second = build_add({existing->second, is_pow ? factor->args[1] : constant_int(1)});
            }
        }
    };
    flatten(flatten, raw);

    coefficient = float_if_real(coefficient);
    if (num_eq(coefficient, Num::integer(0))) {
        return constant_int(0);
    }
    SmallVector<NodePtr, 8> out;
    if (!num_eq(coefficient, Num::integer(1))) {
        out.push_back(make_constant(coefficient));
    }
    for (const auto& [base, exponent] : powers) {
        if (exponent->kind == Kind::Constant) {
            if (num_eq(exponent->value, Num::integer(1)) || (base->idempotent && is_pos_int_constant(exponent))) {
                out.push_back(base);
                continue;
            }
            if (num_eq(exponent->value, Num::integer(0))) {
                continue;
            }
        }
        out.push_back(build_pow(base, exponent));
    }

    if (out.size() == 0) {
        return constant_int(1);
    }
    if (out.size() == 1) {
        return *out.begin();
    }
    sort_canonically(out.begin(), out.end());
    return make_node(Kind::Mul, NodeSpan(out.begin(), out.size()));
}

NodePtr build_pow(const NodePtr& base, const NodePtr& exp) {
    if (exp->kind == Kind::Constant) {
        if (num_eq(exp->value, Num::integer(1))) {
            return base;
        }
        if (num_eq(exp->value, Num::integer(0))) {
            return constant_int(1);
        }
        if (base->kind == Kind::Constant) {
            return constant(safe_pow(base->value, exp->value));
        }
        if (base->kind == Kind::Pow) {
            // Merging powers is only safe for integer exponents: (x**2)**0.5 is not x
            const std::optional<int64_t> inner = int_exponent(base->args[1]);
            const std::optional<int64_t> outer = int_exponent(exp);
            if (inner.has_value() && outer.has_value()) {
                return build_pow(base->args[0], constant(num_mul(Num::integer(*inner), Num::integer(*outer))));
            }
        }
        if (base->idempotent && is_pos_int_constant(exp)) {
            return base;
        }
    }
    if (is_constant(base, 1)) {
        return constant_int(1);
    }
    return make_node(Kind::Pow, {base, exp});
}

NodePtr build_function(const py::object& cls, const NodePtr& arg) {
    if (arg->kind == Kind::Constant) {
        Num value;
        if (!num_from_py(cls.attr(names().numeric)(num_to_py(assert_real(arg->value))), value)) {
            throw py::type_error(py::str(cls.attr("__name__")).cast<std::string>() + " did not evaluate to a number.");
        }
        return constant(value);
    }
    return make_function(cls, class_info(cls).name, arg);
}

// ---------------------------------------------------------------- structure

bool NodeEqual::operator()(const NodePtr& a, const NodePtr& b) const {
    return equal(a, b);
}

bool equal(const NodePtr& a, const NodePtr& b) {
    if (a == b) {
        return true;
    }
    if (a->hash != b->hash || a->kind != b->kind) {
        return false;
    }
    switch (a->kind) {
        case Kind::Constant:
            return num_eq(a->value, b->value) || (same_double(a->value.re, b->value.re) && same_double(a->value.im, b->value.im));
        case Kind::Variable:
            return a->label == b->label;
        case Kind::Function:
            return a->label == b->label && equal(a->args[0], b->args[0]);
        default:
            if (a->args.size() != b->args.size()) {
                return false;  // GCOV_EXCL_LINE (the hash covers the size, so this needs a hash collision)
            }
            for (size_t i = 0; i < a->args.size(); ++i) {
                if (!equal(a->args[i], b->args[i])) {
                    return false;  // GCOV_EXCL_LINE (likewise only reachable through a hash collision)
                }
            }
            return true;
    }
}

namespace {

int compare_nodes(const Node* a, const Node* b) {
    if (a == b) {
        return 0;
    }
    if (a->kind != b->kind) {
        return a->kind < b->kind ? -1 : 1;
    }
    switch (a->kind) {
        case Kind::Constant:
            return compare_constants(a, b);
        case Kind::Variable:
            return compare_labels(a, b);
        case Kind::Function: {
            const int by_name = compare_labels(a, b);
            return by_name != 0 ? by_name : compare_nodes(a->args[0].get(), b->args[0].get());
        }
        default: {
            const size_t n = std::min(a->args.size(), b->args.size());
            for (size_t i = 0; i < n; ++i) {
                const Node* x = a->args[i].get();
                const Node* y = b->args[i].get();
                if (x == y) {
                    continue;
                }
                if (x->kind != y->kind) {
                    return x->kind < y->kind ? -1 : 1;
                }
                // Leaves settle most comparisons, so they are handled here rather than with a recursive call
                int c = 0;
                if (x->kind == Kind::Variable) {
                    c = compare_labels(x, y);
                } else if (x->kind == Kind::Constant) {
                    c = compare_constants(x, y);
                } else {
                    c = compare_nodes(x, y);
                }
                if (c != 0) {
                    return c;
                }
            }
            if (a->args.size() == b->args.size()) {
                return 0;
            }
            return a->args.size() < b->args.size() ? -1 : 1;
        }
    }
}

}  // namespace

int compare(const NodePtr& a, const NodePtr& b) {
    return compare_nodes(a.get(), b.get());
}

std::optional<int64_t> int_exponent(const NodePtr& node) {
    if (node->kind != Kind::Constant) {
        return std::nullopt;
    }
    const Num& value = node->value;
    if (value.tag == Num::Tag::Int) {
        return value.i;
    }
    if (value.tag == Num::Tag::Float && std::isfinite(value.re) && value.re == std::floor(value.re) && std::fabs(value.re) < 9.2e18) {
        return static_cast<int64_t>(value.re);
    }
    return std::nullopt;
}

std::pair<NodePtr, Num> peel(const NodePtr& node) {
    if (node->kind == Kind::Mul && node->args[0]->kind == Kind::Constant) {
        if (node->args.size() == 2) {
            return {node->args[1], node->args[0]->value};
        }
        return {make_node(Kind::Mul, NodeSpan(node->args.begin() + 1, node->args.size() - 1)), node->args[0]->value};
    }
    return {node, Num::integer(1)};
}

// ---------------------------------------------------------------- symbolic operations

NodePtr expand(const NodePtr& node) {
    switch (node->kind) {
        case Kind::Add: {
            std::vector<NodePtr> terms;
            terms.reserve(node->args.size());
            for (const NodePtr& arg : node->args) {
                terms.push_back(expand(arg));
            }
            return build_add(terms);
        }
        case Kind::Mul: {
            std::vector<NodePtr> factors;
            factors.reserve(node->args.size());
            bool has_sum = false;
            for (const NodePtr& arg : node->args) {
                factors.push_back(expand(arg));
                has_sum = has_sum || factors.back()->kind == Kind::Add;
            }
            if (!has_sum) {
                return build_mul(factors);
            }
            // Multiplying out from 1 would first rebuild the opening factor unchanged, so start from it instead
            NodePtr result = factors[0];
            for (size_t i = 1; i < factors.size(); ++i) {
                result = mul_expand(result, factors[i]);
            }
            return result;
        }
        case Kind::Pow: {
            NodePtr base = expand(node->args[0]);
            const std::optional<int64_t> exponent = int_exponent(node->args[1]);
            if (exponent.has_value() && *exponent > 0) {
                NodePtr result = base;
                for (int64_t i = 1; i < *exponent; ++i) {
                    result = mul_expand(result, base);
                }
                return result;
            }
            return build_pow(base, node->args[1]);
        }
        case Kind::Function:
            return build_function(node->owner, expand(node->args[0]));
        default:
            return node;
    }
}

NodePtr derivative(const NodePtr& node, const NodePtr& symbol) {
    switch (node->kind) {
        case Kind::Constant:
            return constant_int(0);
        case Kind::Variable:
            return constant_int(equal(node, symbol) ? 1 : 0);
        case Kind::Add: {
            std::vector<NodePtr> terms;
            terms.reserve(node->args.size());
            for (const NodePtr& arg : node->args) {
                terms.push_back(derivative(arg, symbol));
            }
            return build_add(terms);
        }
        case Kind::Mul: {
            // Product rule: differentiate one factor at a time
            std::vector<NodePtr> terms;
            terms.reserve(node->args.size());
            for (size_t i = 0; i < node->args.size(); ++i) {
                std::vector<NodePtr> factors(node->args.begin(), node->args.end());
                factors[i] = derivative(node->args[i], symbol);
                terms.push_back(build_mul(factors));
            }
            return build_add(terms);
        }
        case Kind::Pow: {
            const NodePtr& base = node->args[0];
            const NodePtr& exp = node->args[1];
            if (!contains_symbol(exp, symbol)) {
                // (b**c)' = c * b**(c-1) * b'
                return build_mul({exp, build_pow(base, build_add({exp, constant_int(-1)})), derivative(base, symbol)});
            }
            // b**e * (e' * ln(b) + e * b'/b)
            NodePtr chain = build_add({build_mul({derivative(exp, symbol), build_function(registry().log_cls, base)}), build_mul({exp, derivative(base, symbol), build_pow(base, constant_int(-1))})});
            return build_mul({node, chain});
        }
        case Kind::Function: {
            if (!contains_symbol(node, symbol)) {
                return constant_int(0);
            }
            NodePtr outer;
            if (!coerce(wrap(node).attr(names().outer_derivative)(wrap(node->args[0])), outer)) {
                throw py::type_error("The derivative of " + node->label + " did not return an Expression.");
            }
            return build_mul({outer, derivative(node->args[0], symbol)});
        }
    }
    return node;  // GCOV_EXCL_LINE (every kind is handled above)
}

NodePtr substitute(const NodePtr& node, const SubstitutionMap& mapping) {
    auto found = mapping.find(node);
    if (found != mapping.end()) {
        // A null replacement could not be coerced, which leaves the node alone
        return found->second ? found->second : node;
    }
    switch (node->kind) {
        case Kind::Pow:
            return build_pow(substitute(node->args[0], mapping), substitute(node->args[1], mapping));
        case Kind::Function:
            return build_function(node->owner, substitute(node->args[0], mapping));
        case Kind::Add:
        case Kind::Mul: {
            std::vector<NodePtr> args;
            args.reserve(node->args.size());
            for (const NodePtr& arg : node->args) {
                args.push_back(substitute(arg, mapping));
            }
            return node->kind == Kind::Add ? build_add(args) : build_mul(args);
        }
        default:
            return node;
    }
}

NodePtr to_binary(const NodePtr& node) {
    std::unordered_map<std::string, NodePtr> memo;
    return to_binary(node, memo);
}

Num evaluate(const NodePtr& node, py::handle env) {
    std::unordered_map<std::string, Num> memo;
    return evaluate(node, env, memo);
}

int64_t degree(const NodePtr& node) {
    switch (node->kind) {
        case Kind::Constant:
            return 0;
        case Kind::Variable:
            return 1;
        case Kind::Add: {
            int64_t result = 0;
            for (const NodePtr& arg : node->args) {
                result = std::max(result, degree(arg));
            }
            return result;
        }
        case Kind::Mul: {
            int64_t result = 0;
            for (const NodePtr& arg : node->args) {
                result += degree(arg);
            }
            return result;
        }
        case Kind::Pow: {
            const std::optional<int64_t> exponent = int_exponent(node->args[1]);
            if (exponent.has_value() && *exponent >= 0) {
                return degree(node->args[0]) * *exponent;
            }
            break;
        }
        case Kind::Function:
            if (!contains_any_symbol(node)) {
                return 0;
            }
            break;
    }
    raise(registry().non_polynomial_error, "Expression " + repr(node) + " is not a polynomial; its degree is undefined.");
}

std::vector<NodePtr> free_symbols(const NodePtr& node) {
    std::vector<NodePtr> out;
    std::unordered_set<NodePtr, NodeHash, NodeEqual> seen;
    collect_symbols(node, out, seen);
    return out;
}

std::vector<std::pair<NodePtr, int64_t>> monomial_factors(const NodePtr& node) {
    switch (node->kind) {
        case Kind::Constant:
            return {};
        case Kind::Mul: {
            std::vector<std::pair<NodePtr, int64_t>> factors;
            for (const NodePtr& arg : node->args) {
                if (arg->kind != Kind::Constant) {
                    std::vector<std::pair<NodePtr, int64_t>> inner = monomial_factors(arg);
                    factors.insert(factors.end(), inner.begin(), inner.end());
                }
            }
            return factors;
        }
        case Kind::Pow: {
            const std::optional<int64_t> exponent = int_exponent(node->args[1]);
            if (exponent.has_value() && *exponent > 0) {
                return {{node->args[0], *exponent}};
            }
            raise(registry().non_polynomial_error, "Expression " + repr(node) + " is not a monomial with an integer power.");
        }
        default:
            return {{node, 1}};
    }
}

std::vector<std::pair<NodePtr, Num>> coefficients(const NodePtr& node) {
    switch (node->kind) {
        case Kind::Constant:
            return {};
        case Kind::Add: {
            std::vector<std::pair<NodePtr, Num>> out;
            out.reserve(node->args.size());
            for (const NodePtr& term : node->args) {
                if (term->kind != Kind::Constant) {
                    out.push_back(peel(term));
                }
            }
            return out;
        }
        default:
            return {peel(node)};
    }
}

Num get_constant(const NodePtr& node) {
    if (node->kind == Kind::Constant) {
        return node->value;
    }
    if (node->kind == Kind::Add && node->args[0]->kind == Kind::Constant) {
        return node->args[0]->value;
    }
    return Num::integer(0);
}

std::string repr(const NodePtr& node) {
    switch (node->kind) {
        case Kind::Constant:
            return repr_number(node->value);
        case Kind::Variable:
            return py::repr(node->owner).cast<std::string>();
        case Kind::Function:
            return py::str(node->owner.attr("__name__")).cast<std::string>() + "(" + repr(node->args[0]) + ")";
        case Kind::Pow: {
            // ** is right-associative, so a power as the base needs parentheses too
            const Kind base_kind = node->args[0]->kind;
            const Kind exp_kind = node->args[1]->kind;
            std::string base = repr(node->args[0]);
            std::string exp = repr(node->args[1]);
            if (base_kind == Kind::Add || base_kind == Kind::Mul || base_kind == Kind::Pow) {
                base = "(" + base + ")";
            }
            if (exp_kind == Kind::Add || exp_kind == Kind::Mul) {
                exp = "(" + exp + ")";
            }
            return base + "**" + exp;
        }
        case Kind::Mul: {
            // A coefficient of exactly -1 reads better as a leading minus
            size_t start = 0;
            std::string out;
            if (is_constant(node->args[0], -1)) {
                out = "-";
                start = 1;
            }
            for (size_t i = start; i < node->args.size(); ++i) {
                const std::string part = repr(node->args[i]);
                out += (i > start ? " * " : "") + (node->args[i]->kind == Kind::Add ? "(" + part + ")" : part);
            }
            return out;
        }
        case Kind::Add: {
            std::string out = repr(node->args[0]);
            for (size_t i = 1; i < node->args.size(); ++i) {
                const NodePtr negated = negate_for_repr(node->args[i]);
                if (!negated) {
                    out += " + " + repr(node->args[i]);
                } else {
                    // A subtracted sum keeps its parentheses: "a - (b + c)" is not "a - b + c"
                    const std::string body = repr(negated);
                    out += " - " + (negated->kind == Kind::Add ? "(" + body + ")" : body);
                }
            }
            return out;
        }
    }
    return "";  // GCOV_EXCL_LINE (every kind is handled above)
}

// ---------------------------------------------------------------- Python interop

NodePtr node_of(py::handle obj) {
    py::object attr = obj.attr(names().node);
    if (!attr.is_none()) {
        void* ptr = PyCapsule_GetPointer(attr.ptr(), capsule_name);
        if (ptr == nullptr) {
            throw py::error_already_set();
        }
        return NodePtr(static_cast<Node*>(ptr));
    }
    // Variables do not store a node, as it would reference them back: build a leaf on demand
    const ClassInfo& info = class_info(py::type::of(obj));
    return make_variable(obj.attr(names().label).cast<std::string>(), py::reinterpret_borrow<py::object>(obj), info);
}

bool coerce(py::handle obj, NodePtr& out) {
    if (PyObject_TypeCheck(obj.ptr(), reinterpret_cast<PyTypeObject*>(registry().expression_cls.ptr()))) {
        out = node_of(obj);
        return true;
    }
    Num value;
    if (!num_from_py(obj, value)) {
        return false;
    }
    out = constant(value);
    return true;
}

py::object capsule(const NodePtr& node) {
    // The capsule holds one counted reference to the node, rather than a separately allocated NodePtr
    Node* raw = const_cast<Node*>(node.get());
    PyObject* cap = PyCapsule_New(raw, capsule_name, [](PyObject* c) {
        auto* held = static_cast<Node*>(PyCapsule_GetPointer(c, capsule_name));
        if (--held->refs == 0) {
            release(held);
        }
    });
    if (cap == nullptr) {
        throw py::error_already_set();  // GCOV_EXCL_LINE (only on out-of-memory)
    }
    ++raw->refs;
    return py::reinterpret_steal<py::object>(cap);
}

py::object wrap(const NodePtr& node) {
    const Registry& reg = registry();
    const py::object* cls = nullptr;
    switch (node->kind) {
        case Kind::Variable:
            return node->owner;
        case Kind::Constant:
            cls = &reg.constant_cls;
            break;
        case Kind::Add:
            cls = &reg.add_cls;
            break;
        case Kind::Mul:
            cls = &reg.mul_cls;
            break;
        case Kind::Pow:
            cls = &reg.pow_cls;
            break;
        case Kind::Function:
            cls = &node->owner;
            break;
    }
    // Skip the class's own __new__ and __init__, the node is already built
    PyObject* obj = PyBaseObject_Type.tp_new(reinterpret_cast<PyTypeObject*>(cls->ptr()), reg.empty_tuple.ptr(), nullptr);
    if (obj == nullptr) {
        throw py::error_already_set();  // GCOV_EXCL_LINE (the registered classes are all concrete)
    }
    py::object result = py::reinterpret_steal<py::object>(obj);
    result.attr(names().node) = capsule(node);
    return result;
}

namespace python {

namespace core = qilisdk::expr;

namespace {

NodePtr operand(py::handle obj) {
    NodePtr node;
    if (!coerce(obj, node)) {
        throw py::type_error("Expected an Expression or a number, got " + py::str(py::type::of(obj).attr("__name__")).cast<std::string>() + ".");
    }
    return node;
}

}  // namespace

py::object constant_node(py::handle value) {
    Num number;
    if (!num_from_py(value, number)) {
        throw py::type_error("A Constant must be a number, got " + py::str(py::type::of(value).attr("__name__")).cast<std::string>() + ".");
    }
    return capsule(constant(number));
}

py::object node(int kind, py::iterable args) {
    return capsule(make_node(static_cast<Kind>(kind), nodes_of(args)));
}

py::object function(py::handle cls, py::handle arg) {
    NodePtr node;
    if (!coerce(arg, node)) {
        throw py::type_error(py::str(cls.attr("__name__")).cast<std::string>() + " expects an Expression or number, got " + py::str(py::type::of(arg).attr("__name__")).cast<std::string>());
    }
    return wrap(build_function(py::reinterpret_borrow<py::object>(cls), node));
}

int64_t variable_hash(const std::string& label) {
    return static_cast<int64_t>(label_hash(label));
}

py::object add(py::handle self, py::handle other) {
    NodePtr rhs;
    if (!coerce(other, rhs)) {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    return wrap(core::build_add({node_of(self), rhs}));
}

py::object sub(py::handle self, py::handle other) {
    NodePtr rhs;
    if (!coerce(other, rhs)) {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    return wrap(core::build_add({node_of(self), core::build_mul({constant_int(-1), rhs})}));
}

py::object rsub(py::handle self, py::handle other) {
    NodePtr lhs;
    if (!coerce(other, lhs)) {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    return wrap(core::build_add({lhs, core::build_mul({constant_int(-1), node_of(self)})}));
}

py::object mul(py::handle self, py::handle other) {
    NodePtr rhs;
    if (!coerce(other, rhs)) {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    return wrap(core::build_mul({node_of(self), rhs}));
}

py::object neg(py::handle self) {
    return wrap(core::build_mul({constant_int(-1), node_of(self)}));
}

py::object truediv(py::handle self, py::handle other) {
    // A plain int or float divides through its reciprocal
    if (PyLong_Check(other.ptr()) || PyFloat_Check(other.ptr())) {
        Num divisor;
        num_from_py(other, divisor);
        if (num_abs(divisor) < registry().tol) {
            throw py::value_error(division_message);
        }
        return wrap(core::build_mul({make_constant(Num::real(1.0 / divisor.re)), node_of(self)}));
    }
    NodePtr rhs;
    if (!coerce(other, rhs)) {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    return wrap(core::build_mul({node_of(self), core::build_pow(rhs, constant_int(-1))}));
}

py::object rtruediv(py::handle self, py::handle other) {
    NodePtr lhs;
    if (!coerce(other, lhs)) {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    return wrap(core::build_mul({lhs, core::build_pow(node_of(self), constant_int(-1))}));
}

py::object pow(py::handle self, py::handle other) {
    NodePtr exp;
    if (!coerce(other, exp)) {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    return wrap(core::build_pow(node_of(self), exp));
}

py::object rpow(py::handle self, py::handle other) {
    NodePtr base;
    if (!coerce(other, base)) {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    return wrap(core::build_pow(base, node_of(self)));
}

py::object build_add(py::iterable raw) {
    return wrap(core::build_add(nodes_of(raw)));
}

py::object build_mul(py::iterable raw) {
    return wrap(core::build_mul(nodes_of(raw)));
}

py::object build_pow(py::handle base, py::handle exp) {
    return wrap(core::build_pow(operand(base), operand(exp)));
}

py::object evaluate(py::handle self, py::handle env) {
    py::object mapping = env.is_none() ? py::object(py::dict()) : py::reinterpret_borrow<py::object>(env);
    return num_to_py(core::evaluate(node_of(self), mapping));
}

py::list free_symbols(py::handle self) {
    py::list out;
    for (const NodePtr& symbol : core::free_symbols(node_of(self))) {
        out.append(symbol->owner);
    }
    return out;
}

py::list variables(py::handle self) {
    std::vector<NodePtr> symbols = core::free_symbols(node_of(self));
    std::stable_sort(symbols.begin(), symbols.end(), [](const NodePtr& a, const NodePtr& b) { return a->label < b->label; });
    py::list out;
    for (const NodePtr& symbol : symbols) {
        out.append(symbol->owner);
    }
    return out;
}

bool is_parameterized(py::handle self) {
    const std::vector<NodePtr> symbols = core::free_symbols(node_of(self));
    return std::all_of(symbols.begin(), symbols.end(), [](const NodePtr& symbol) { return symbol->parameter; });
}

int64_t degree(py::handle self) {
    return core::degree(node_of(self));
}

py::object derivative(py::handle self, py::handle symbol) {
    return wrap(core::derivative(node_of(self), operand(symbol)));
}

py::object expand(py::handle self) {
    return wrap(core::expand(node_of(self)));
}

py::object substitute(py::handle self, py::handle mapping) {
    // Any mapping works, not only a dict
    const py::dict items = PyDict_Check(mapping.ptr()) ? py::reinterpret_borrow<py::dict>(mapping) : py::dict(py::reinterpret_borrow<py::object>(mapping));
    SubstitutionMap nodes;
    for (auto item : items) {
        if (!PyObject_TypeCheck(item.first.ptr(), reinterpret_cast<PyTypeObject*>(registry().expression_cls.ptr()))) {
            continue;
        }
        // A replacement that cannot be coerced stays null, which leaves its node alone
        NodePtr replacement;
        coerce(item.second, replacement);
        nodes.emplace(node_of(item.first), replacement);
    }
    return wrap(core::substitute(node_of(self), nodes));
}

py::object to_binary(py::handle self) {
    return wrap(core::to_binary(node_of(self)));
}

py::object get_constant(py::handle self) {
    return num_to_py(core::get_constant(node_of(self)));
}

py::dict as_coefficients_dict(py::handle self) {
    py::dict out;
    for (const auto& [base, coeff] : coefficients(node_of(self))) {
        out[wrap(base)] = num_to_py(coeff);
    }
    return out;
}

py::list monomial_factors(py::handle self) {
    py::list out;
    for (const auto& [base, power] : core::monomial_factors(node_of(self))) {
        out.append(py::make_tuple(wrap(base), power));
    }
    return out;
}

py::list to_list(py::handle self) {
    const NodePtr node = node_of(self);
    py::list out;
    if (node->kind == Kind::Add || node->kind == Kind::Mul) {
        for (const NodePtr& arg : node->args) {
            out.append(wrap(arg));
        }
    } else {
        out.append(self);
    }
    return out;
}

py::tuple args(py::handle self) {
    const NodePtr node = node_of(self);
    py::tuple out(node->args.size());
    for (size_t i = 0; i < node->args.size(); ++i) {
        out[i] = wrap(node->args[i]);
    }
    return out;
}

py::object value(py::handle self) {
    return num_to_py(node_of(self)->value);
}

py::object coefficient(py::handle self) {
    return num_to_py(peel(node_of(self)).second);
}

py::object monomial(py::handle self) {
    return wrap(peel(node_of(self)).first);
}

int64_t hash(py::handle self) {
    return static_cast<int64_t>(node_of(self)->hash);
}

bool equal(py::handle self, py::handle other) {
    if (!PyObject_TypeCheck(other.ptr(), reinterpret_cast<PyTypeObject*>(registry().expression_cls.ptr()))) {
        return false;
    }
    return core::equal(node_of(self), node_of(other));
}

std::string repr(py::handle self) {
    return core::repr(node_of(self));
}

}  // namespace python

}  // namespace qilisdk::expr
