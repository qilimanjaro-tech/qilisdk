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
Node* allocate(Kind kind, size_t count);
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
    /*
    The interned attribute names the engine looks up on Python objects.

    Returns:
        Names&: The process-wide set of names.
    */
    // Never destroyed, since the Python objects would outlive the interpreter
    static const Names* instance = new Names();
    return *instance;
}

[[noreturn]] void raise(const py::object& type, const std::string& message) {
    /*
    Raise a Python exception of a given class from C++.

    Args:
        type (py::object&): The exception class, such as NonPolynomialError.
        message (std::string&): The exception message.

    Raises:
        py::error_already_set: Always, carrying the exception.
    */
    PyErr_SetString(type.ptr(), message.c_str());
    throw py::error_already_set();
}

uint64_t mix(uint64_t h, uint64_t v) {
    /*
    Fold a value into a running hash, finishing with a splitmix64-style avalanche.

    Args:
        h (uint64_t): The running hash.
        v (uint64_t): The value to fold in.

    Returns:
        uint64_t: The combined hash.
    */
    h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
    h ^= h >> 31;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 29;
    return h;
}

uint64_t count_seed(Kind kind, size_t count) {
    /*
    The part of a sum or product's hash that depends on its kind and operand count.

    Args:
        kind (Kind): The node kind.
        count (size_t): The number of operands.

    Returns:
        uint64_t: The seed the operand hashes are added to.
    */
    return mix(static_cast<uint64_t>(kind), count);
}

uint64_t spread(Kind kind, uint64_t operand_hash) {
    /*
    The contribution of one operand to the hash of a sum or product. The contributions are added, so
    the hash does not depend on the operand order and one operand can be swapped in or out in O(1).

    Args:
        kind (Kind): The kind of the node holding the operand.
        operand_hash (uint64_t): The operand's hash.

    Returns:
        uint64_t: The operand's contribution.
    */
    return mix(static_cast<uint64_t>(kind) + 8, operand_hash);
}

uint64_t hash_double(double value) {
    /*
    Hash a double by its bit pattern.

    Args:
        value (double): The value to hash.

    Returns:
        uint64_t: The bits of the value, after mapping -0.0 to 0.0 and every NaN to one quiet NaN.
    */
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

bool same_double(double x, double y) {
    /*
    Structural sameness of two doubles, under which a NaN is the same as any other NaN.

    Args:
        x (double): The first value.
        y (double): The second value.

    Returns:
        bool: True if the values are equal or both NaN.
    */
    return x == y || (std::isnan(x) && std::isnan(y));
}

int compare_doubles(double x, double y) {
    /*
    A total order on doubles for sorting, which puts NaN after every number.

    Args:
        x (double): The first value.
        y (double): The second value.

    Returns:
        int: -1, 0 or 1 as x orders before, alongside or after y.
    */
    if (same_double(x, y)) {
        return 0;
    }
    if (std::isnan(x) || (!std::isnan(y) && x > y)) {
        return 1;
    }
    return -1;
}

bool int_equals_double(int64_t i, double d) {
    /*
    Whether an integer and a double hold the same value, compared exactly rather than through a
    rounding conversion of the integer.

    Args:
        i (int64_t): The integer.
        d (double): The double.

    Returns:
        bool: True if d is a whole number within the int64 range equal to i.
    */
    if (!std::isfinite(d) || d != std::floor(d) || d < -9223372036854775808.0 || d >= 9223372036854775808.0) {
        return false;
    }
    return static_cast<int64_t>(d) == i;
}

uint64_t label_key(const std::string& label) {
    /*
    The first 8 bytes of a label as a big-endian integer, zero padded, so comparing keys orders labels
    like memcmp.

    Args:
        label (std::string&): The label.

    Returns:
        uint64_t: The key.
    */
    uint64_t key = 0;
    for (size_t i = 0; i < 8; ++i) {
        key = (key << 8) | (i < label.size() ? static_cast<unsigned char>(label[i]) : 0);
    }
    return key;
}

int compare_labels(const Node* a, const Node* b) {
    /*
    Order two variable or function nodes by label, comparing the precomputed label keys before falling
    back to the full strings.

    Args:
        a (Node*): The first node.
        b (Node*): The second node.

    Returns:
        int: -1, 0 or 1 as a orders before, alongside or after b.
    */
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

bool same_number(const Num& a, const Num& b) {
    /*
    Whether two numbers are the very same value, type and bits alike, so 2 and 2.0 or 0.0 and -0.0
    differ. This is what lets the constant cache share nodes without changing any result.

    Args:
        a (Num&): The first number.
        b (Num&): The second number.

    Returns:
        bool: True if both numbers are identical.
    */
    return a.tag == b.tag && a.i == b.i && std::memcmp(&a.re, &b.re, sizeof(double)) == 0 && std::memcmp(&a.im, &b.im, sizeof(double)) == 0;
}

int compare_constants(const Node* a, const Node* b) {
    /*
    Order two constant nodes by real part, then by imaginary part.

    Args:
        a (Node*): The first constant.
        b (Node*): The second constant.

    Returns:
        int: -1, 0 or 1 as a orders before, alongside or after b.
    */
    const int by_real = compare_doubles(a->value.re, b->value.re);
    return by_real != 0 ? by_real : compare_doubles(a->value.im, b->value.im);
}

Num c_prod(const Num& a, const Num& b) {
    /*
    The complex product a * b, computed as Python's complex arithmetic does so results match it to the
    last bit.

    Args:
        a (Num&): The first factor.
        b (Num&): The second factor.

    Returns:
        Num: The complex product.
    */
    return Num::complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

Num c_quot(const Num& a, const Num& b) {
    /*
    The complex quotient a / b, computed with the same scaled algorithm as Python's complex division
    so results match it to the last bit.

    Args:
        a (Num&): The dividend.
        b (Num&): The divisor.

    Returns:
        Num: The complex quotient.

    Raises:
        py::value_error: If b is zero.
    */
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
    /*
    A complex number raised to a non-negative integer power by repeated squaring, as Python does.

    Args:
        x (Num&): The base.
        n (int64_t): The exponent, which must not be negative.

    Returns:
        Num: The complex power.
    */
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
    /*
    The complex power a ** b, following Python's algorithm: integer exponents of magnitude up to 100
    use repeated squaring, and everything else goes through polar form.

    Args:
        a (Num&): The base.
        b (Num&): The exponent.

    Returns:
        Num: The complex power.

    Raises:
        py::value_error: If a is zero and b is negative or not real.
    */
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

Num safe_pow(const Num& base, const Num& exponent) {
    /*
    Python's base ** exponent, raising for a zero base under a negative exponent.

    Args:
        base (Num&): The base.
        exponent (Num&): The exponent.

    Returns:
        Num: The power.

    Raises:
        py::value_error: If the base is zero within tolerance and the exponent is negative.
    */
    if (num_abs(base) < registry().tol && exponent.is_real() && exponent.re < 0) {
        throw py::value_error(division_message);
    }
    return num_pow(base, exponent);
}

Num assert_real(const Num& value) {
    /*
    Drop a negligible imaginary part from a number and insist that it is then real, as the argument of
    a maths function must be.

    Args:
        value (Num&): The number to check.

    Returns:
        Num: The number as a real.

    Raises:
        py::value_error: If the imaginary part is not negligible.
    */
    const Num real = float_if_real(value);
    if (!real.is_real()) {
        throw py::value_error("Only real values are allowed but " + py::str(num_to_py(real)).cast<std::string>() + " was provided.");
    }
    return real;
}

NodePtr constant(const Num& value) {
    /*
    The constant node for a number, dropping a negligible imaginary part.

    Args:
        value (Num&): The number.

    Returns:
        NodePtr: The constant node.
    */
    return make_constant(float_if_real(value));
}

NodePtr constant_int(int64_t value) {
    /*
    The shared constant node for -1, 0 or 1. They are needed constantly (every plain factor of a
    product has exponent 1), so they are built once.

    Args:
        value (int64_t): The value, which must be -1, 0 or 1.

    Returns:
        NodePtr: The constant node.
    */
    static const auto* common = new std::array<NodePtr, 3>{make_constant(Num::integer(-1)), make_constant(Num::integer(0)), make_constant(Num::integer(1))};
    return (*common)[value + 1];
}

bool is_constant(const NodePtr& node, int64_t value) {
    /*
    Whether a node is a constant numerically equal to an integer.

    Args:
        node (NodePtr&): The node to check.
        value (int64_t): The integer to compare against.

    Returns:
        bool: True if the node is a constant equal to value, so 1 and 1.0 both count as 1.
    */
    return node->kind == Kind::Constant && num_eq(node->value, Num::integer(value));
}

bool is_pos_int_constant(const NodePtr& node) {
    /*
    Whether a node is a constant holding a positive whole number.

    Args:
        node (NodePtr&): The node to check.

    Returns:
        bool: True if the node is a constant positive integer, such as 2 or 2.0.
    */
    const std::optional<int64_t> value = int_exponent(node);
    return value.has_value() && *value > 0;
}

NodePtr mul_expand(const NodePtr& left, const NodePtr& right) {
    /*
    The product of two expressions, multiplied out over their sums. Each side counts as the list of
    its summands (a non-sum being a single one) and every pair of summands is multiplied.

    Args:
        left (NodePtr&): The first factor.
        right (NodePtr&): The second factor.

    Returns:
        NodePtr: The canonical sum of the pairwise products.
    */
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
        /*
        Append a value, moving the inline elements to the heap the first time the inline storage overflows.

        Args:
            value (T): The value to append.
        */
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

void sort_canonically(NodePtr* begin, NodePtr* end) {
    /*
    Sort nodes into canonical order, stably. std::stable_sort allocates a buffer on every call, which
    for the handful of operands of a typical node costs more than an insertion sort does, so it is only
    used for runs of more than 16 nodes.

    Args:
        begin (NodePtr*): The start of the run.
        end (NodePtr*): One past the end of the run.
    */
    const auto less = [](const NodePtr& a, const NodePtr& b) { return compare(a, b) < 0; };
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
    /*
    View a summand as its coefficient and the factors of its monomial, without building the monomial.

    Args:
        term (NodePtr&): The summand, which must outlive the view.

    Returns:
        Summand: The view of the summand.
    */
    Summand summand;
    summand.term = &term;
    summand.coeff = Num::integer(1);
    summand.hash = term->monomial_hash;
    if (term->kind != Kind::Mul) {
        summand.factors = NodeSpan(&term, 1);
        return summand;
    }
    const bool scaled = term->args[0]->kind == Kind::Constant;
    summand.factors = scaled ? NodeSpan(term->args.begin() + 1, term->args.size() - 1) : term->args;
    if (scaled) {
        summand.coeff = term->args[0]->value;
        // Adding to 0 turns a complex coefficient's -0.0 real part into 0.0, which reusing the node would not show
        summand.reusable = summand.coeff.is_real() || !(summand.coeff.re == 0.0 && std::signbit(summand.coeff.re));
    }
    return summand;
}

bool same_factors(NodeSpan a, NodeSpan b) {
    /*
    Whether two lists of factors are structurally equal, element by element.

    Args:
        a (NodeSpan): The first factors.
        b (NodeSpan): The second factors.

    Returns:
        bool: True if both lists hold equal factors in the same order.
    */
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

NodePtr rebuild_summand(const Summand& summand, const Num& coeff) {
    /*
    The node for a combined summand, reusing the original whenever nothing merged into it.

    Args:
        summand (Summand&): The summand.
        coeff (Num&): Its combined coefficient, which must not be zero.

    Returns:
        NodePtr: The summand scaled by its coefficient.
    */
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

NodePtr add_to_sum(const NodePtr& sum, const NodePtr& term, bool sum_first) {
    /*
    Add a single non-sum term to a canonical sum by splicing it into the sorted operands. This gives
    the same result as build_add without its hash index: like terms are found by comparing monomial
    hashes, the new node's hash is updated rather than recomputed, and the operands are copied once,
    so each step of a running sum() is one cheap pass.

    Args:
        sum (NodePtr&): The canonical sum.
        term (NodePtr&): The term to add, which must not itself be a sum.
        sum_first (bool): Whether the sum is the left operand, which fixes the order the coefficients
            are added in (so rounding matches build_add) and which of two like terms keeps its base.

    Returns:
        NodePtr: The canonical sum, or its single remaining operand.
    */
    const NodeSpan terms = sum->args;
    const bool has_constant = terms[0]->kind == Kind::Constant;
    // The index of the operand that goes, or terms.size() for none, and the operand that comes, if any
    size_t removed = terms.size();
    NodePtr added;

    if (term->kind == Kind::Constant) {
        const Num existing = has_constant ? terms[0]->value : Num::integer(0);
        const Num total = float_if_real(sum_first ? num_add(num_add(Num::integer(0), existing), term->value) : num_add(num_add(Num::integer(0), term->value), existing));
        if (has_constant) {
            removed = 0;
        }
        if (!num_eq(total, Num::integer(0))) {
            added = make_constant(total);
        }
    } else {
        const Summand incoming = view_summand(term);
        const NodePtr* match = std::find_if(terms.begin() + (has_constant ? 1 : 0), terms.end(), [&](const NodePtr& existing) { return existing->monomial_hash == incoming.hash && same_factors(view_summand(existing).factors, incoming.factors); });
        Num coeff;
        Summand kept = incoming;
        if (match != terms.end()) {
            const Summand existing = view_summand(*match);
            coeff = sum_first ? num_add(num_add(Num::integer(0), existing.coeff), incoming.coeff) : num_add(num_add(Num::integer(0), incoming.coeff), existing.coeff);
            // The sum keeps whichever base it met first
            if (sum_first) {
                kept = existing;
            }
            removed = static_cast<size_t>(match - terms.begin());
        } else {
            coeff = num_add(Num::integer(0), incoming.coeff);
        }
        coeff = float_if_real(coeff);
        if (!num_eq(coeff, Num::integer(0))) {
            kept.reusable = match == terms.end() && same_number(coeff, incoming.coeff);
            added = rebuild_summand(kept, coeff);
        }
    }

    // A scaled sum whose coefficient combined to exactly 1 comes back as the bare sum, whose operands
    // belong among this sum's own, so it takes the general path, which flattens it
    if (added && added->kind == Kind::Add) {
        std::vector<NodePtr> rest;
        rest.reserve(terms.size());
        for (size_t i = 0; i < terms.size(); ++i) {
            if (i != removed) {
                rest.push_back(terms[i]);
            }
        }
        rest.push_back(added);
        return build_add(rest);
    }

    // The sum had at least two operands and at most one is removed, so at least one remains
    const size_t count = terms.size() - (removed < terms.size() ? 1 : 0) + (added ? 1 : 0);
    if (count == 1) {
        return terms[removed == 0 ? 1 : 0];
    }
    const size_t insert_at = added ? static_cast<size_t>(std::upper_bound(terms.begin(), terms.end(), added, [](const NodePtr& a, const NodePtr& b) { return compare(a, b) < 0; }) - terms.begin()) : terms.size() + 1;
    Node* node = allocate(Kind::Add, count);
    auto* storage = const_cast<NodePtr*>(node->args.begin());
    size_t out = 0;
    for (size_t i = 0; i <= terms.size(); ++i) {
        if (i == insert_at) {
            new (storage + out++) NodePtr(added);
        }
        if (i < terms.size() && i != removed) {
            new (storage + out++) NodePtr(terms[i]);
        }
    }
    // A removed term's bits may linger, which the mask allows as it only has to cover the variables
    node->symbol_mask = sum->symbol_mask | (added ? added->symbol_mask : 0);
    uint64_t hash = sum->hash - count_seed(Kind::Add, terms.size()) + count_seed(Kind::Add, count);
    if (removed < terms.size()) {
        hash -= spread(Kind::Add, terms[removed]->hash);
    }
    if (added) {
        hash += spread(Kind::Add, added->hash);
    }
    node->hash = hash;
    node->monomial_hash = hash;
    return NodePtr(node);
}

bool contains_symbol(const NodePtr& node, const NodePtr& symbol) {
    /*
    Whether a variable occurs anywhere in a tree.

    Args:
        node (NodePtr&): The tree to search.
        symbol (NodePtr&): The variable to look for.

    Returns:
        bool: True if the variable occurs in the tree.
    */
    if ((node->symbol_mask & symbol->symbol_mask) == 0) {
        return false;
    }
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

void collect_symbols(const NodePtr& node, std::vector<NodePtr>& out, std::unordered_set<NodePtr, NodeHash, NodeEqual>& seen) {
    /*
    Append each distinct variable of a tree to a list, depth first in the order they are first met.

    Args:
        node (NodePtr&): The tree to search.
        out (std::vector<NodePtr>&): The list to append to.
        seen (std::unordered_set&): The variables already appended.
    */
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
    /*
    Rewrite every variable that defines its own to_binary in terms of binary variables, rebuilding the
    tree canonically on the way up.

    Args:
        node (NodePtr&): The tree to rewrite.
        memo (std::unordered_map&): The rewrite of each variable already met, by label, so each
            variable's to_binary is only called once.

    Returns:
        NodePtr: The rewritten tree.

    Raises:
        py::type_error: If a variable's to_binary returns neither an Expression nor a number.
    */
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
    /*
    Numerically evaluate a tree.

    Args:
        node (NodePtr&): The tree to evaluate.
        env (py::handle): The mapping of variables to values, passed to each variable's evaluate.
        memo (std::unordered_map&): The value of each variable already evaluated, by label, so each
            variable's evaluate is only called once.

    Returns:
        Num: The value, as a float or a complex.

    Raises:
        py::type_error: If a variable or function does not evaluate to a number.
        py::value_error: If the tree divides by zero or a function gets a complex argument.
    */
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
    /*
    Python's repr of a number, dropping a negligible imaginary part.

    Args:
        value (Num&): The number.

    Returns:
        std::string: The printed number.
    */
    return py::repr(num_to_py(float_if_real(value))).cast<std::string>();
}

NodePtr negate_for_repr(const NodePtr& term) {
    /*
    The positive form of a negative summand, so a sum prints "a - b" rather than "a + -b".

    Args:
        term (NodePtr&): The summand.

    Returns:
        NodePtr: The negated summand if it is a product with a negative real coefficient, otherwise null.
    */
    if (term->kind == Kind::Mul) {
        const Num coefficient = peel(term).second;
        if (coefficient.is_real() && coefficient.re < 0) {
            return build_mul({constant_int(-1), term});
        }
    }
    return nullptr;
}

std::vector<NodePtr> nodes_of(py::iterable raw) {
    /*
    Coerce every item of a Python iterable into a node.

    Args:
        raw (py::iterable): The Expressions and numbers.

    Returns:
        std::vector<NodePtr>: Their nodes, in order.

    Raises:
        py::type_error: If an item is neither an Expression nor a number.
    */
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
    /*
    A number standing for a Python int. The value is mirrored into re, so mixed arithmetic can read it
    as a double directly.

    Args:
        value (int64_t): The integer.

    Returns:
        Num: The number.
    */
    Num n;
    n.tag = Tag::Int;
    n.i = value;
    n.re = static_cast<double>(value);
    return n;
}

Num Num::real(double value) {
    /*
    A number standing for a Python float.

    Args:
        value (double): The value.

    Returns:
        Num: The number.
    */
    Num n;
    n.tag = Tag::Float;
    n.re = value;
    return n;
}

Num Num::complex(double real, double imag) {
    /*
    A number standing for a Python complex.

    Args:
        real (double): The real part.
        imag (double): The imaginary part.

    Returns:
        Num: The number.
    */
    Num n;
    n.tag = Tag::Complex;
    n.re = real;
    n.im = imag;
    return n;
}

Num num_add(const Num& a, const Num& b) {
    /*
    The sum a + b with Python's numeric promotion: two ints stay an int unless the sum overflows 64
    bits, when it falls back to a float, two reals give a float, and anything else a complex.

    Args:
        a (Num&): The first operand.
        b (Num&): The second operand.

    Returns:
        Num: The sum.
    */
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
    /*
    The product a * b with Python's numeric promotion: two ints stay an int unless the product
    overflows 64 bits, when it falls back to a float, two reals give a float, and anything else a complex.

    Args:
        a (Num&): The first factor.
        b (Num&): The second factor.

    Returns:
        Num: The product.
    */
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
    /*
    The power base ** exponent with Python's semantics: an int to a non-negative int stays an exact
    int unless it overflows 64 bits, a negative real base under a fractional power gives a complex,
    other reals give a float, and anything else a complex.

    Args:
        base (Num&): The base.
        exponent (Num&): The exponent.

    Returns:
        Num: The power.
    */
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
    /*
    Numeric equality like Python's ==, so 2, 2.0 and 2+0j are all equal, with an int compared against
    a double exactly.

    Args:
        a (Num&): The first number.
        b (Num&): The second number.

    Returns:
        bool: True if the numbers are equal.
    */
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
    /*
    The absolute value, or modulus for a complex.

    Args:
        value (Num&): The number.

    Returns:
        double: Its absolute value.
    */
    return value.is_real() ? std::fabs(value.re) : std::hypot(value.re, value.im);
}

Num float_if_real(const Num& value) {
    /*
    Demote a complex whose imaginary part is below tolerance to a float.

    Args:
        value (Num&): The number.

    Returns:
        Num: The number, as a float if its imaginary part is negligible.
    */
    if (!value.is_real() && std::fabs(value.im) < registry().tol) {
        return Num::real(value.re);
    }
    return value;
}

Num finalize(const Num& value) {
    /*
    The form of a number handed back as an evaluation result, where ints become floats and a negligible
    imaginary part is dropped.

    Args:
        value (Num&): The number.

    Returns:
        Num: The number as a float or a complex.
    */
    if (value.tag == Num::Tag::Int) {
        return Num::real(value.re);
    }
    return float_if_real(value);
}

bool num_from_py(py::handle obj, Num& out) {
    /*
    Read a Python bool, int, float, complex or numpy scalar. An int beyond 64 bits becomes the nearest
    float.

    Args:
        obj (py::handle): The object to read.
        out (Num&): Set to the number when obj is one.

    Returns:
        bool: True if obj is a number.

    Raises:
        py::error_already_set: If an int is too large even for a float.
    */
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
    /*
    The Python int, float or complex for a number.

    Args:
        value (Num&): The number.

    Returns:
        py::object: The Python number.
    */
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
    /*
    The Python classes and settings the engine needs, set by initialize().

    Returns:
        Registry&: The process-wide registry.
    */
    // Never destroyed, since the Python objects would outlive the interpreter
    static Registry* instance = new Registry();
    return *instance;
}

void initialize(py::object constant_cls, py::object add_cls, py::object mul_cls, py::object pow_cls, py::object expression_cls, py::object log_cls, py::object non_polynomial_error, double tol) {
    /*
    Register the Python classes and settings the engine needs. Called once by expression.py on import,
    it also clears the cached class markers.

    Args:
        constant_cls (py::object): The Constant class.
        add_cls (py::object): The Add class.
        mul_cls (py::object): The Mul class.
        pow_cls (py::object): The Pow class.
        expression_cls (py::object): The Expression base class.
        log_cls (py::object): The Log class, which the derivative of a symbolic power needs.
        non_polynomial_error (py::object): The exception raised for an undefined degree.
        tol (double): The tolerance under which a number counts as zero.
    */
    Registry& reg = registry();
    reg.constant_cls = std::move(constant_cls);
    reg.add_cls = std::move(add_cls);
    reg.mul_cls = std::move(mul_cls);
    reg.pow_cls = std::move(pow_cls);
    reg.expression_to_binary = expression_cls.attr("to_binary");
    reg.expression_cls = std::move(expression_cls);
    reg.log_cls = std::move(log_cls);
    reg.non_polynomial_error = std::move(non_polynomial_error);
    reg.numpy_generic = py::module_::import("numpy").attr("generic");
    reg.empty_tuple = py::tuple();
    reg.tol = tol;
    reg.classes.clear();
}

const ClassInfo& class_info(py::handle cls) {
    /*
    The markers of a variable or function class, read from the class on first use and cached: whether
    it is idempotent under multiplication, whether it is a Parameter, whether it overrides to_binary,
    and its NAME.

    Args:
        cls (py::handle): The class.

    Returns:
        ClassInfo&: The markers, which stay valid for the life of the registry.
    */
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
    /*
    The FNV-1a hash of a label, mixed with the Variable kind. It is the hash of a variable's node, and
    is also its Python hash through variable_hash, so the two always agree.

    Args:
        label (std::string&): The label.

    Returns:
        uint64_t: The hash.
    */
    uint64_t h = 0xCBF29CE484222325ULL;
    for (unsigned char c : label) {
        h = (h ^ c) * 0x100000001B3ULL;
    }
    return mix(static_cast<uint64_t>(Kind::Variable), h);
}

uint64_t compound_hash(Kind kind, const NodePtr* first, NodeSpan rest) {
    /*
    The hash of an operator node. A power hashes its base and exponent in order, while a sum or product
    adds up the spread of each operand's hash onto a seed for its kind and count, so that one operand
    can be swapped in or out without rehashing the rest.

    Args:
        kind (Kind): The node kind.
        first (NodePtr*): An optional leading operand, or nullptr.
        rest (NodeSpan): The remaining operands.

    Returns:
        uint64_t: The hash.
    */
    const size_t count = (first != nullptr ? 1 : 0) + rest.size();
    if (kind == Kind::Pow) {
        uint64_t h = mix(static_cast<uint64_t>(kind), count);
        if (first != nullptr) {
            h = mix(h, (*first)->hash);
        }
        for (const NodePtr& arg : rest) {
            h = mix(h, arg->hash);
        }
        return h;
    }
    uint64_t h = count_seed(kind, count);
    if (first != nullptr) {
        h += spread(kind, (*first)->hash);
    }
    for (const NodePtr& arg : rest) {
        h += spread(kind, arg->hash);
    }
    return h;
}

Node* allocate(Kind kind, size_t count) {
    /*
    Allocate a node with room for its operands in the same block right after it, leaving that room
    for the caller to construct the operands in.

    Args:
        kind (Kind): The node kind.
        count (size_t): The number of operands.

    Returns:
        Node*: The node, without references and with its operands and hashes still unset.
    */
    void* memory = ::operator new(sizeof(Node) + count * sizeof(NodePtr));
    // Default- rather than value-initialized, which would zero the whole node before constructing it
    Node* node = new (memory) Node;
    node->kind = kind;
    node->args = NodeSpan(reinterpret_cast<NodePtr*>(node + 1), count);
    return node;
}

Node* allocate(Kind kind, const NodePtr* first, NodeSpan rest) {
    /*
    Allocate a node with its operands, an optional first one and then the rest, copied into the same
    block right after it.

    Args:
        kind (Kind): The node kind.
        first (NodePtr*): An optional leading operand, or nullptr.
        rest (NodeSpan): The remaining operands.

    Returns:
        Node*: The node, without references and with its hash still unset.
    */
    Node* node = allocate(kind, (first != nullptr ? 1 : 0) + rest.size());
    auto* storage = const_cast<NodePtr*>(node->args.begin());
    size_t i = 0;
    if (first != nullptr) {
        new (storage + i++) NodePtr(*first);
    }
    for (const NodePtr& arg : rest) {
        new (storage + i++) NodePtr(arg);
    }
    return node;
}

NodePtr make_node(Kind kind, const NodePtr* first, NodeSpan rest) {
    /*
    Make an operator node from operands trusted to be canonical, without checking them.

    Args:
        kind (Kind): The node kind.
        first (NodePtr*): An optional leading operand, or nullptr.
        rest (NodeSpan): The remaining operands.

    Returns:
        NodePtr: The node.
    */
    Node* node = allocate(kind, first, rest);
    node->hash = compound_hash(kind, first, rest);
    node->monomial_hash = node->hash;
    for (const NodePtr& arg : node->args) {
        node->symbol_mask |= arg->symbol_mask;
    }
    // A product with a coefficient also records the hash of its factors alone, which like terms match on
    if (kind == Kind::Mul && node->args[0]->kind == Kind::Constant) {
        const size_t count = node->args.size();
        node->monomial_hash = count == 2 ? node->args[1]->hash : node->hash - count_seed(kind, count) + count_seed(kind, count - 1) - spread(kind, node->args[0]->hash);
    }
    return NodePtr(node);
}

void release(Node* node) {
    /*
    Free a node once its last reference is gone, along with its wrapper's weak reference and its own
    references to its operands.

    Args:
        node (Node*): The node to free.
    */
    Py_XDECREF(node->wrapper);
    auto* storage = reinterpret_cast<NodePtr*>(node + 1);
    for (size_t i = 0; i < node->args.size(); ++i) {
        storage[i].~NodePtr();
    }
    node->~Node();
    ::operator delete(node);
}

NodePtr make_constant(const Num& value) {
    /*
    Make a constant node, sharing one recently made for the very same number.

    Args:
        value (Num&): The number, kept exactly as given.

    Returns:
        NodePtr: The constant node.
    */
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
    node->monomial_hash = node->hash;
    cached = NodePtr(node);
    return cached;
}

NodePtr make_variable(const std::string& label, py::object owner, const ClassInfo& info) {
    /*
    Make the leaf for a Python variable.

    Args:
        label (std::string&): The variable's label.
        owner (py::object): The Python variable, which the leaf keeps a reference to.
        info (ClassInfo&): The markers of the variable's class.

    Returns:
        NodePtr: The leaf.
    */
    Node* node = allocate(Kind::Variable, nullptr, {});
    node->label = label;
    node->idempotent = info.idempotent;
    node->custom_binary = info.custom_binary;
    node->parameter = info.parameter;
    node->owner = std::move(owner);
    node->hash = label_hash(label);
    node->monomial_hash = node->hash;
    node->symbol_mask = uint64_t{1} << (node->hash & 63);
    node->label_key = label_key(label);
    return NodePtr(node);
}

NodePtr make_node(Kind kind, NodeSpan args) {
    /*
    Make an operator node from operands trusted to be canonical, without checking them.

    Args:
        kind (Kind): The node kind.
        args (NodeSpan): The operands.

    Returns:
        NodePtr: The node.
    */
    return make_node(kind, nullptr, args);
}

NodePtr make_function(py::object cls, const std::string& name, NodePtr arg) {
    /*
    Make a function node without folding its argument.

    Args:
        cls (py::object): The function's class, such as Sin.
        name (std::string&): The class's NAME, which orders functions.
        arg (NodePtr): The argument.

    Returns:
        NodePtr: The function node.
    */
    Node* node = allocate(Kind::Function, &arg, {});
    node->label = name;
    node->label_key = label_key(name);
    node->owner = std::move(cls);
    node->hash = mix(mix(static_cast<uint64_t>(Kind::Function), label_hash(name)), arg->hash);
    node->monomial_hash = node->hash;
    node->symbol_mask = arg->symbol_mask;
    return NodePtr(node);
}

NodePtr build_add(NodeSpan raw) {
    /*
    Build the canonical sum of a list of terms. Nested sums are flattened, like terms are combined by
    adding their coefficients, constants fold into one, zero terms drop out and the rest are sorted. A
    sum plus a single other term takes the quicker add_to_sum.

    Args:
        raw (NodeSpan): The terms.

    Returns:
        NodePtr: The sum, its single remaining operand, or 0 if nothing remains.
    */
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
    // A scaled sum such as 2 * (a + b) whose coefficient combined to exactly 1 comes back as the bare
    // sum a + b, which must be flattened into this one like any other nested sum
    if (std::any_of(out.begin(), out.end(), [](const NodePtr& term) { return term->kind == Kind::Add; })) {
        return build_add(out);
    }
    sort_canonically(out.data(), out.data() + out.size());
    return make_node(Kind::Add, out);
}

NodePtr build_mul(NodeSpan raw) {
    /*
    Build the canonical product of a list of factors. Nested products are flattened, constants multiply
    into one leading coefficient, repeated bases collect by adding their exponents, an idempotent base
    drops any positive integer power, and the factors are sorted. Sums are not distributed over.

    Args:
        raw (NodeSpan): The factors.

    Returns:
        NodePtr: The product, its single remaining factor, 0 for a zero coefficient, or 1 if nothing
        remains.
    */
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
    /*
    Build the canonical power base ** exp. An exponent of 1 gives the base and 0 gives 1, a constant
    base folds, nested integer powers merge, an idempotent base drops a positive integer power, and a
    base of 1 gives 1.

    Args:
        base (NodePtr&): The base.
        exp (NodePtr&): The exponent.

    Returns:
        NodePtr: The power, folded or simplified where possible.

    Raises:
        py::value_error: If a constant zero base has a negative exponent.
    */
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
    /*
    Build a function node, folding a constant argument to the function's numeric value.

    Args:
        cls (py::object&): The function's class, such as Sin.
        arg (NodePtr&): The argument.

    Returns:
        NodePtr: The function node, or a constant for a constant argument.

    Raises:
        py::value_error: If a constant argument is complex.
        py::type_error: If the class's _numeric does not return a number.
    */
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
    /*
    Structural equality, for keying hash containers by node.

    Args:
        a (NodePtr&): The first node.
        b (NodePtr&): The second node.

    Returns:
        bool: True if the nodes are structurally equal.
    */
    return equal(a, b);
}

bool equal(const NodePtr& a, const NodePtr& b) {
    /*
    Structural equality of two canonical trees, settled by pointer or hash wherever possible.
    Constants compare numerically, so 2 and 2.0 are equal, and every NaN is alike.

    Args:
        a (NodePtr&): The first tree.
        b (NodePtr&): The second tree.

    Returns:
        bool: True if the trees are structurally equal.
    */
    if (a == b) {
        return true;
    }
    if (a->hash != b->hash || a->kind != b->kind) {
        return false;
    }
    switch (a->kind) {
        case Kind::Constant:
            if (num_eq(a->value, b->value)) {
                return true;
            }
            // NaN is never equal to itself numerically, but structurally every NaN is alike
            return (std::isnan(a->value.re) || std::isnan(a->value.im)) && same_double(a->value.re, b->value.re) && same_double(a->value.im, b->value.im);
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
    /*
    The canonical total order of nodes: by kind first, then constants by value, variables by label,
    functions by name and then argument, and operator nodes by their operands in turn and then by
    their count.

    Args:
        a (Node*): The first node.
        b (Node*): The second node.

    Returns:
        int: -1, 0 or 1 as a orders before, alongside or after b.
    */
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
    /*
    The canonical order of two nodes, which sorts the operands of a sum or product.

    Args:
        a (NodePtr&): The first node.
        b (NodePtr&): The second node.

    Returns:
        int: -1, 0 or 1 as a orders before, alongside or after b.
    */
    return compare_nodes(a.get(), b.get());
}

std::optional<int64_t> int_exponent(const NodePtr& node) {
    /*
    The integer value of a constant holding a whole number, such as 2 or 2.0.

    Args:
        node (NodePtr&): The node.

    Returns:
        std::optional<int64_t>: The integer, or nothing if the node is not a whole-number constant.
    */
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
    /*
    Split a term into its monomial and its numeric coefficient.

    Args:
        node (NodePtr&): The term.

    Returns:
        std::pair<NodePtr, Num>: The term without its leading constant, and that constant (or 1).
    */
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
    /*
    Distribute products and positive integer powers over sums, all the way down the tree.

    Args:
        node (NodePtr&): The tree to expand.

    Returns:
        NodePtr: The canonical sum of monomials equal to the tree.
    */
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
    /*
    The symbolic derivative with respect to a variable, by the sum, product, power and chain rules.
    A function supplies its own outer derivative through _outer_derivative.

    Args:
        node (NodePtr&): The tree to differentiate.
        symbol (NodePtr&): The variable to differentiate by.

    Returns:
        NodePtr: The derivative.

    Raises:
        py::type_error: If a function's _outer_derivative returns neither an Expression nor a number.
    */
    switch (node->kind) {
        case Kind::Constant:
            return constant_int(0);
        case Kind::Variable:
            return constant_int(equal(node, symbol) ? 1 : 0);
        case Kind::Add: {
            std::vector<NodePtr> terms;
            for (const NodePtr& arg : node->args) {
                if ((arg->symbol_mask & symbol->symbol_mask) != 0) {
                    terms.push_back(derivative(arg, symbol));
                }
            }
            return build_add(terms);
        }
        case Kind::Mul: {
            // Product rule: differentiate one factor at a time, where a factor without the symbol adds nothing
            std::vector<NodePtr> terms;
            for (size_t i = 0; i < node->args.size(); ++i) {
                if ((node->args[i]->symbol_mask & symbol->symbol_mask) == 0) {
                    continue;
                }
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
    /*
    Replace the sub-trees found in a mapping, rebuilding canonically on the way up. A replaced sub-tree
    is not searched any further.

    Args:
        node (NodePtr&): The tree.
        mapping (SubstitutionMap&): The replacement for each sub-tree, where a null one leaves it as is.

    Returns:
        NodePtr: The tree with its replacements made.
    */
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
    /*
    Rewrite every variable that defines its own to_binary in terms of binary variables.

    Args:
        node (NodePtr&): The tree to rewrite.

    Returns:
        NodePtr: The rewritten tree.

    Raises:
        py::type_error: If a variable's to_binary returns neither an Expression nor a number.
    */
    std::unordered_map<std::string, NodePtr> memo;
    return to_binary(node, memo);
}

Num evaluate(const NodePtr& node, py::handle env) {
    /*
    Numerically evaluate a tree.

    Args:
        node (NodePtr&): The tree to evaluate.
        env (py::handle): The mapping of variables to values, passed to each variable's evaluate.

    Returns:
        Num: The value, as a float or a complex.

    Raises:
        py::type_error: If a variable or function does not evaluate to a number.
        py::value_error: If the tree divides by zero or a function gets a complex argument.
    */
    std::unordered_map<std::string, Num> memo;
    return evaluate(node, env, memo);
}

int64_t degree(const NodePtr& node) {
    /*
    The polynomial degree of a tree.

    Args:
        node (NodePtr&): The tree.

    Returns:
        int64_t: The degree.

    Raises:
        NonPolynomialError: If the tree has a power that is not a non-negative integer, or a function
            of a variable.
    */
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
            if (node->symbol_mask == 0) {
                return 0;
            }
            break;
    }
    raise(registry().non_polynomial_error, "Expression " + repr(node) + " is not a polynomial; its degree is undefined.");
}

std::vector<NodePtr> free_symbols(const NodePtr& node) {
    /*
    The distinct variables of a tree.

    Args:
        node (NodePtr&): The tree.

    Returns:
        std::vector<NodePtr>: The variables, depth first in the order they are first met.
    */
    std::vector<NodePtr> out;
    std::unordered_set<NodePtr, NodeHash, NodeEqual> seen;
    collect_symbols(node, out, seen);
    return out;
}

std::vector<std::pair<NodePtr, int64_t>> monomial_factors(const NodePtr& node) {
    /*
    The factors of a monomial with their powers, leaving out its coefficient.

    Args:
        node (NodePtr&): The monomial.

    Returns:
        std::vector<std::pair<NodePtr, int64_t>>: Each base with its power, empty for a constant.

    Raises:
        NonPolynomialError: If a power is not a positive integer.
    */
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
    /*
    Each non-constant term of a tree with its coefficient. The constant term is left out, see
    get_constant.

    Args:
        node (NodePtr&): The tree.

    Returns:
        std::vector<std::pair<NodePtr, Num>>: Each monomial with its coefficient.
    */
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
    /*
    The constant term of a tree.

    Args:
        node (NodePtr&): The tree.

    Returns:
        Num: The value of a constant, the leading constant of a sum, or 0 otherwise.
    */
    if (node->kind == Kind::Constant) {
        return node->value;
    }
    if (node->kind == Kind::Add && node->args[0]->kind == Kind::Constant) {
        return node->args[0]->value;
    }
    return Num::integer(0);
}

std::string repr(const NodePtr& node) {
    /*
    The printed form of a tree, with only the parentheses the precedence needs.

    Args:
        node (NodePtr&): The tree.

    Returns:
        std::string: The printed tree.
    */
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

namespace {

// A sum still being added to. Each step of a running sum, as in Python's sum() or a loop of +=,
// would otherwise copy every operand of the canonical sum so far into a new node, which makes
// building a sum of T terms cost T^2. Instead `+` on a large enough sum gives an Add wrapper whose
// _node holds one of these: the terms it is the sum of, in the order they were added, left to be
// canonicalized in one pass (see accumulate) when something first needs the node.
//
// Wrappers share one buffer of terms, each reading only its first `count`, so a + b is a push onto
// a's buffer, and a + c afterwards, which finds the buffer already longer than a, copies a's part
// of it first. Every wrapper therefore keeps meaning exactly what it was built as.
//
// The wrapper's class is Add before the sum is built, so the sum must come out as an Add. Removing
// an operand takes a summand merging into it -- cancelling it, zeroing its constant or collapsing
// it -- and each summand merges once. The summands are the operands added, plus those a scaled sum
// c * (a + b) releases when its coefficient combines to 1 and it is flattened into a + b. Such a
// collapse needs a summand with the very same factors to merge into it, so what it releases is what
// that summand could release itself, and counting each term added by the summands it could bring
// (see summands) bounds every removal. A chain is built eagerly before that count could reach all but
// two of the operands it began with. In the usual case every term counts one, and each build then
// at least roughly doubles the canonical sum the next chain begins from, so the building costs
// O(T log T) in all.
const char* const pending_name = "qilisdk.expression.pending";

// The fewest operands a sum has before adding to it is deferred. Below this, building the sum
// straight away costs about as much as deferring it
constexpr size_t kDeferFrom = 32;

// The longest pending chain, counting the sum it began from, that is built a step at a time when
// it settles rather than in one pass, each step costing a copy of the sum against the pass's
// hashing of every operand
constexpr size_t kStepwiseChain = 3;

struct PendingSum {
    std::shared_ptr<std::vector<NodePtr>> terms;  // The first is the canonical sum the chain began from
    size_t count = 0;                             // How many of terms this sum is over
    size_t base = 0;                              // The operands of the canonical sum it began from
    size_t added = 0;                             // The summands that could merge into it (see summands)
};

size_t releasable(const NodePtr& term) {
    /*
    How many summands a summand could release into its sum: a scaled sum c * (a + b), when its
    coefficient combines to exactly 1, is flattened into its operands, each of which may itself be
    such a summand.

    Args:
        term (NodePtr&): The summand.

    Returns:
        size_t: The summands it could release, all the way down.
    */
    if (term->kind != Kind::Mul || term->args.size() != 2 || term->args[0]->kind != Kind::Constant || term->args[1]->kind != Kind::Add) {
        return 0;
    }
    size_t count = 0;
    for (const NodePtr& operand : term->args[1]->args) {
        count += 1 + releasable(operand);
    }
    return count;
}

size_t summands(const NodePtr& term) {
    /*
    How many summands adding a term could merge into a sum: each operand of a sum, or the term
    itself, and whatever each of those could release.

    Args:
        term (NodePtr&): The term added.

    Returns:
        size_t: The summands.
    */
    if (term->kind != Kind::Add) {
        return 1 + releasable(term);
    }
    size_t count = 0;
    for (const NodePtr& operand : term->args) {
        count += 1 + releasable(operand);
    }
    return count;
}

const PendingSum* pending_of(const py::object& attr) {
    /*
    The pending sum a wrapper's _node holds, if it holds one.

    Args:
        attr (py::object&): The wrapper's _node.

    Returns:
        PendingSum*: The pending sum, or nullptr if _node is anything else.
    */
    if (!PyCapsule_IsValid(attr.ptr(), pending_name)) {
        return nullptr;
    }
    return static_cast<const PendingSum*>(PyCapsule_GetPointer(attr.ptr(), pending_name));
}

NodePtr accumulate(NodeSpan terms) {
    /*
    The canonical sum of a canonical sum and the terms added to it one at a time, exactly as adding
    them one at a time and building every step would give it, but in one pass over a hash index.

    "Exactly" is the point. Building the whole list at once with build_add gives the same value but
    not always the same canonical form or the same rounding: a step-by-step sum drops a summand the
    moment its coefficient reaches zero, so a like term added later starts afresh, and flattens a
    scaled sum such as 2 * (a + b) the moment its coefficient reaches 1, so later terms in (a + b)
    gather separately. Following the same rules here means leaving a sum pending never changes it.
    A term that is not a sum follows add_to_sum, and a term that is a sum follows build_add of the
    sum so far and that term, which merges each of its operands into the sum in turn.

    Args:
        terms (NodeSpan): A canonical Add, then the terms added to it, in order.

    Returns:
        NodePtr: The canonical sum.
    */
    // A short chain is quicker taken a step at a time, which by definition gives the same sum: this
    // is the case of a sum read straight after each addition, and of one large sum added at once
    if (terms.size() <= kStepwiseChain) {
        NodePtr sum = terms[0];
        for (size_t i = 1; i < terms.size(); ++i) {
            sum = build_add({sum, terms[i]});
        }
        return sum;
    }
    struct Entry {
        Summand summand;
        bool live = true;
    };
    std::vector<Entry> entries;
    // Open addressing over indices into entries, plus one so zero means empty. An entry whose
    // coefficient reaches zero stays indexed but dead, and a like term added later revives it afresh
    std::vector<uint32_t> slots(16, 0);
    size_t mask = slots.size() - 1;
    bool has_constant = false;
    Num constant = Num::integer(0);

    const auto locate = [&](const Summand& summand) -> uint32_t* {
        size_t slot = summand.hash & mask;
        while (slots[slot] != 0) {
            const Summand& existing = entries[slots[slot] - 1].summand;
            if (existing.hash == summand.hash && same_factors(existing.factors, summand.factors)) {
                break;
            }
            slot = (slot + 1) & mask;
        }
        return &slots[slot];
    };
    const auto append = [&](uint32_t* slot, const Summand& summand) {
        entries.push_back({summand, true});
        *slot = static_cast<uint32_t>(entries.size());
        // Keep the table at most half full
        if (entries.size() * 2 > slots.size()) {
            slots.assign(slots.size() * 2, 0);
            mask = slots.size() - 1;
            for (size_t i = 0; i < entries.size(); ++i) {
                size_t free_slot = entries[i].summand.hash & mask;
                while (slots[free_slot] != 0) {
                    free_slot = (free_slot + 1) & mask;
                }
                slots[free_slot] = static_cast<uint32_t>(i + 1);
            }
        }
    };

    // whole_sum: whether the term is an operand of a sum being added, which build_add merges, rather
    // than a term added on its own, which add_to_sum merges. They differ only in when a new summand
    // may reuse its node
    const auto merge = [&](const auto& self, const NodePtr& term, bool whole_sum) -> void {
        if (term->kind == Kind::Constant) {
            constant = float_if_real(num_add(num_add(Num::integer(0), has_constant ? constant : Num::integer(0)), term->value));
            has_constant = !num_eq(constant, Num::integer(0));
            return;
        }
        Summand incoming = view_summand(term);
        uint32_t* slot = locate(incoming);
        if (*slot != 0 && entries[*slot - 1].live) {
            // A like term: the sum keeps the base it met first, and its coefficient grows
            Entry& entry = entries[*slot - 1];
            const Num coeff = float_if_real(num_add(num_add(Num::integer(0), entry.summand.coeff), incoming.coeff));
            entry.summand.coeff = coeff;
            entry.summand.reusable = false;
            if (num_eq(coeff, Num::integer(0))) {
                entry.live = false;
            } else if (num_eq(coeff, Num::integer(1)) && entry.summand.factors.size() == 1 && entry.summand.factors[0]->kind == Kind::Add) {
                // A scaled sum left with coefficient 1 is that sum, whose operands join this one. The
                // sum is held here, as merging its operands can grow entries and move `entry`
                entry.live = false;
                const NodePtr inner = entry.summand.factors[0];
                for (const NodePtr& operand : inner->args) {
                    self(self, operand, true);
                }
            }
            return;
        }
        const Num coeff = float_if_real(num_add(Num::integer(0), incoming.coeff));
        if (!whole_sum) {
            incoming.reusable = same_number(coeff, incoming.coeff);
        }
        incoming.coeff = coeff;
        if (*slot != 0) {
            entries[*slot - 1] = {incoming, true};
        } else {
            append(slot, incoming);
        }
    };

    // The operands of a canonical sum are already distinct, combined and in canonical form, and one
    // nothing merges into is kept as it is
    for (const NodePtr& operand : terms[0]->args) {
        if (operand->kind == Kind::Constant) {
            has_constant = true;
            constant = operand->value;
        } else {
            Summand summand = view_summand(operand);
            summand.reusable = true;
            append(locate(summand), summand);
        }
    }
    for (size_t i = 1; i < terms.size(); ++i) {
        if (terms[i]->kind == Kind::Add) {
            for (const NodePtr& operand : terms[i]->args) {
                merge(merge, operand, true);
            }
        } else {
            merge(merge, terms[i], false);
        }
    }

    std::vector<NodePtr> out;
    out.reserve(entries.size() + 1);
    for (const Entry& entry : entries) {
        if (entry.live) {
            out.push_back(rebuild_summand(entry.summand, entry.summand.coeff));
        }
    }
    if (has_constant) {
        out.push_back(make_constant(constant));
    }
    // The bound on a pending chain keeps at least two operands (see PendingSum)
    if (out.size() < 2) {
        return out.empty() ? constant_int(0) : out[0];  // GCOV_EXCL_LINE
    }
    sort_canonically(out.data(), out.data() + out.size());
    return make_node(Kind::Add, out);
}

NodePtr settle(py::handle obj, const PendingSum& pending) {
    /*
    Build the canonical sum a pending wrapper stands for, and store it in the wrapper's _node in
    place of the pending sum, so the build happens once.

    Args:
        obj (py::handle): The wrapper.
        pending (PendingSum&): Its pending sum, which may be freed once _node is replaced.

    Returns:
        NodePtr: The canonical sum, which is always an Add (see PendingSum).
    */
    NodePtr node = accumulate(NodeSpan(pending.terms->data(), pending.count));
    obj.attr(names().node) = capsule(node);
    return node;
}

}  // namespace

NodePtr node_of(py::handle obj) {
    /*
    The node of a Python Expression: the one held by its _node capsule, or for a variable, which holds
    none, a leaf built on demand. A sum still pending is built now.

    Args:
        obj (py::handle): The Expression.

    Returns:
        NodePtr: Its node.

    Raises:
        py::error_already_set: If _node is not a node capsule.
    */
    py::object attr = obj.attr(names().node);
    if (!attr.is_none()) {
        if (const PendingSum* pending = pending_of(attr)) {
            return settle(obj, *pending);
        }
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
    /*
    Turn an Expression or a number into a node.

    Args:
        obj (py::handle): The object to coerce.
        out (NodePtr&): Set to the node when obj can be coerced.

    Returns:
        bool: True if obj is an Expression or a number, so an operator can otherwise return
        NotImplemented.
    */
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
    /*
    A Python capsule holding a reference to a node, as stored in a wrapper's _node.

    Args:
        node (NodePtr&): The node.

    Returns:
        py::object: The capsule.
    */
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
    /*
    A new Python wrapper of the right class for a node, made without running the class's __new__ or
    __init__. A variable leaf gives back its own Python variable.

    Args:
        node (NodePtr&): The node.

    Returns:
        py::object: The wrapper.
    */
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

py::object wrap_shared(const NodePtr& node) {
    /*
    A Python wrapper for a node, reusing the last one made for it while that is still alive, so a
    sub-tree shared by several parents comes back as one object.

    Args:
        node (NodePtr&): The node.

    Returns:
        py::object: The wrapper.
    */
    if (node->kind == Kind::Variable) {
        return node->owner;
    }
    if (node->wrapper != nullptr) {
        PyObject* alive = nullptr;
#if PY_VERSION_HEX >= 0x030D0000
        if (PyWeakref_GetRef(node->wrapper, &alive) == 1) {
            return py::reinterpret_steal<py::object>(alive);
        }
#else
        alive = PyWeakref_GetObject(node->wrapper);
        if (alive != Py_None) {
            return py::reinterpret_borrow<py::object>(alive);
        }
#endif
    }
    py::object result = wrap(node);
    PyObject* ref = PyWeakref_NewRef(result.ptr(), nullptr);
    if (ref == nullptr) {
        throw py::error_already_set();  // GCOV_EXCL_LINE (every wrapper class has a __weakref__ slot)
    }
    Py_XSETREF(node->wrapper, ref);
    return result;
}

namespace python {

namespace core = qilisdk::expr;

namespace {

NodePtr operand(py::handle obj) {
    /*
    Coerce an operand into a node.

    Args:
        obj (py::handle): The Expression or number.

    Returns:
        NodePtr: Its node.

    Raises:
        py::type_error: If obj is neither an Expression nor a number.
    */
    NodePtr node;
    if (!coerce(obj, node)) {
        throw py::type_error("Expected an Expression or a number, got " + py::str(py::type::of(obj).attr("__name__")).cast<std::string>() + ".");
    }
    return node;
}

py::object wrap_pending(PendingSum pending) {
    /*
    A new Add wrapper for a sum left pending, made without running Add's __new__ or __init__.

    Args:
        pending (PendingSum): The pending sum, which the wrapper takes.

    Returns:
        py::object: The wrapper.
    */
    const Registry& reg = registry();
    PyObject* obj = PyBaseObject_Type.tp_new(reinterpret_cast<PyTypeObject*>(reg.add_cls.ptr()), reg.empty_tuple.ptr(), nullptr);
    if (obj == nullptr) {
        throw py::error_already_set();  // GCOV_EXCL_LINE (Add is concrete)
    }
    py::object result = py::reinterpret_steal<py::object>(obj);
    auto* held = new PendingSum(std::move(pending));
    PyObject* cap = PyCapsule_New(held, pending_name, [](PyObject* c) { delete static_cast<PendingSum*>(PyCapsule_GetPointer(c, pending_name)); });
    if (cap == nullptr) {
        delete held;                    // GCOV_EXCL_LINE (only on out-of-memory)
        throw py::error_already_set();  // GCOV_EXCL_LINE
    }
    result.attr(names().node) = py::reinterpret_steal<py::object>(cap);
    return result;
}

py::object extend(const PendingSum& pending, const NodePtr& term) {
    /*
    The sum of a pending sum and one more term, itself pending unless it is due to be built.

    Args:
        pending (PendingSum&): The sum so far.
        term (NodePtr&): The term to add, which may itself be a sum.

    Returns:
        py::object: The wrapper of the new sum.
    */
    const size_t incoming = summands(term);
    // Built now if the summands added could otherwise remove all but two of the operands it began with
    if (pending.added + incoming + 2 > pending.base) {
        std::vector<NodePtr> all(pending.terms->begin(), pending.terms->begin() + static_cast<std::ptrdiff_t>(pending.count));
        all.push_back(term);
        return wrap(accumulate(all));
    }
    PendingSum next{pending.terms, pending.count + 1, pending.base, pending.added + incoming};
    if (pending.terms->size() != pending.count) {
        // The buffer already holds a longer sum branched off this one, so this branch gets its own
        next.terms = std::make_shared<std::vector<NodePtr>>(pending.terms->begin(), pending.terms->begin() + static_cast<std::ptrdiff_t>(pending.count));
    }
    next.terms->push_back(term);
    return wrap_pending(std::move(next));
}

py::object add_terms(py::handle self, const NodePtr& term) {
    /*
    The sum self + term, left pending when self is a large or pending sum (see PendingSum) and built
    straight away otherwise.

    Args:
        self (py::handle): The left operand.
        term (NodePtr&): The right operand's node.

    Returns:
        py::object: The wrapper of the sum.
    */
    const py::object attr = self.attr(names().node);
    if (const PendingSum* pending = pending_of(attr)) {
        return extend(*pending, term);
    }
    NodePtr lhs = node_of(self);
    if (lhs->kind == Kind::Add && lhs->args.size() >= kDeferFrom) {
        PendingSum start{std::make_shared<std::vector<NodePtr>>(), 1, lhs->args.size(), 0};
        start.terms->push_back(lhs);
        return extend(start, term);
    }
    return wrap(core::build_add({lhs, term}));
}

}  // namespace

py::object constant_node(py::handle value) {
    /*
    The node of a new Constant, for Constant.__init__ and __setstate__.

    Args:
        value (py::handle): The number.

    Returns:
        py::object: The node capsule.

    Raises:
        py::type_error: If value is not a number.
    */
    Num number;
    if (!num_from_py(value, number)) {
        throw py::type_error("A Constant must be a number, got " + py::str(py::type::of(value).attr("__name__")).cast<std::string>() + ".");
    }
    return capsule(constant(number));
}

py::object node(int kind, py::iterable args) {
    /*
    The node for the trusting constructors of Add, Mul and Pow, whose operands are taken to be
    canonical already. Used by their __init__ and __setstate__.

    Args:
        kind (int): The node kind as its number in Kind: 2 for Pow, 3 for Mul or 4 for Add.
        args (py::iterable): The operands.

    Returns:
        py::object: The node capsule.

    Raises:
        py::value_error: If kind is not Add, Mul or Pow, or the operand count does not suit it.
        py::type_error: If an operand is neither an Expression nor a number.
    */
    const auto node_kind = static_cast<Kind>(kind);
    if (node_kind != Kind::Add && node_kind != Kind::Mul && node_kind != Kind::Pow) {
        throw py::value_error("Only an Add, Mul or Pow node can be built from operands.");
    }
    const std::vector<NodePtr> nodes = nodes_of(args);
    // The trust only covers canonical form: too few operands would be read out of bounds
    if (node_kind == Kind::Pow ? nodes.size() != 2 : nodes.size() < 2) {
        throw py::value_error(node_kind == Kind::Pow ? "A Pow needs exactly two operands." : "An Add or Mul needs at least two operands.");
    }
    return capsule(make_node(node_kind, nodes));
}

py::object function(py::handle cls, py::handle arg) {
    /*
    A function of an argument, for Function.__new__, folded to a Constant for a constant argument.

    Args:
        cls (py::handle): The function's class, such as Sin.
        arg (py::handle): The argument.

    Returns:
        py::object: The function, or a Constant.

    Raises:
        py::type_error: If arg is neither an Expression nor a number.
    */
    NodePtr node;
    if (!coerce(arg, node)) {
        throw py::type_error(py::str(cls.attr("__name__")).cast<std::string>() + " expects an Expression or number, got " + py::str(py::type::of(arg).attr("__name__")).cast<std::string>());
    }
    return wrap(build_function(py::reinterpret_borrow<py::object>(cls), node));
}

int64_t variable_hash(const std::string& label) {
    /*
    The hash of a variable, for BaseVariable.__hash__, matching the hash of its node.

    Args:
        label (std::string&): The variable's label.

    Returns:
        int64_t: The hash.
    */
    return static_cast<int64_t>(label_hash(label));
}

py::object add(py::handle self, py::handle other) {
    /*
    Expression.__add__, which also serves as __radd__ since sums are order-independent.

    Args:
        self (py::handle): The Expression.
        other (py::handle): The other operand.

    Returns:
        py::object: The canonical sum, or NotImplemented if other is neither an Expression nor a
        number, so Python falls back to other's reflected operator.
    */
    NodePtr rhs;
    if (!coerce(other, rhs)) {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    return add_terms(self, rhs);
}

py::object sub(py::handle self, py::handle other) {
    /*
    Expression.__sub__.

    Args:
        self (py::handle): The Expression.
        other (py::handle): The operand to subtract.

    Returns:
        py::object: The canonical self - other, or NotImplemented if other is neither an Expression
        nor a number.
    */
    NodePtr rhs;
    if (!coerce(other, rhs)) {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    return add_terms(self, core::build_mul({constant_int(-1), rhs}));
}

py::object rsub(py::handle self, py::handle other) {
    /*
    Expression.__rsub__.

    Args:
        self (py::handle): The Expression.
        other (py::handle): The operand subtracted from.

    Returns:
        py::object: The canonical other - self, or NotImplemented if other is neither an Expression
        nor a number.
    */
    NodePtr lhs;
    if (!coerce(other, lhs)) {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    return wrap(core::build_add({lhs, core::build_mul({constant_int(-1), node_of(self)})}));
}

py::object mul(py::handle self, py::handle other) {
    /*
    Expression.__mul__, which also serves as __rmul__ since products are order-independent.

    Args:
        self (py::handle): The Expression.
        other (py::handle): The other factor.

    Returns:
        py::object: The canonical product, or NotImplemented if other is neither an Expression nor a
        number.
    */
    NodePtr rhs;
    if (!coerce(other, rhs)) {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    return wrap(core::build_mul({node_of(self), rhs}));
}

py::object neg(py::handle self) {
    /*
    Expression.__neg__.

    Args:
        self (py::handle): The Expression.

    Returns:
        py::object: The canonical -self.
    */
    return wrap(core::build_mul({constant_int(-1), node_of(self)}));
}

py::object truediv(py::handle self, py::handle other) {
    /*
    Expression.__truediv__. A plain int or float divisor multiplies through its reciprocal, anything
    else becomes a power of -1.

    Args:
        self (py::handle): The Expression.
        other (py::handle): The divisor.

    Returns:
        py::object: The canonical self / other, or NotImplemented if other is neither an Expression
        nor a number.

    Raises:
        py::value_error: If other is an int or float that is zero within tolerance.
    */
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
    /*
    Expression.__rtruediv__.

    Args:
        self (py::handle): The Expression, as the divisor.
        other (py::handle): The dividend.

    Returns:
        py::object: The canonical other / self, or NotImplemented if other is neither an Expression
        nor a number.
    */
    NodePtr lhs;
    if (!coerce(other, lhs)) {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    return wrap(core::build_mul({lhs, core::build_pow(node_of(self), constant_int(-1))}));
}

py::object pow(py::handle self, py::handle other) {
    /*
    Expression.__pow__.

    Args:
        self (py::handle): The Expression, as the base.
        other (py::handle): The exponent.

    Returns:
        py::object: The canonical self ** other, or NotImplemented if other is neither an Expression
        nor a number.
    */
    NodePtr exp;
    if (!coerce(other, exp)) {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    return wrap(core::build_pow(node_of(self), exp));
}

py::object rpow(py::handle self, py::handle other) {
    /*
    Expression.__rpow__.

    Args:
        self (py::handle): The Expression, as the exponent.
        other (py::handle): The base.

    Returns:
        py::object: The canonical other ** self, or NotImplemented if other is neither an Expression
        nor a number.
    */
    NodePtr base;
    if (!coerce(other, base)) {
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }
    return wrap(core::build_pow(base, node_of(self)));
}

py::object build_add(py::iterable raw) {
    /*
    Add.build: the canonical sum of a list of terms.

    Args:
        raw (py::iterable): The Expressions and numbers to add.

    Returns:
        py::object: The sum, collapsed to a single operand or a constant where possible.

    Raises:
        py::type_error: If an item is neither an Expression nor a number.
    */
    return wrap(core::build_add(nodes_of(raw)));
}

py::object build_mul(py::iterable raw) {
    /*
    Mul.build: the canonical product of a list of factors.

    Args:
        raw (py::iterable): The Expressions and numbers to multiply.

    Returns:
        py::object: The product, collapsed to a single factor or a constant where possible.

    Raises:
        py::type_error: If an item is neither an Expression nor a number.
    */
    return wrap(core::build_mul(nodes_of(raw)));
}

py::object build_pow(py::handle base, py::handle exp) {
    /*
    Pow.build: the canonical power base ** exp.

    Args:
        base (py::handle): The base.
        exp (py::handle): The exponent.

    Returns:
        py::object: The power, folded or simplified where possible.

    Raises:
        py::type_error: If base or exp is neither an Expression nor a number.
    */
    return wrap(core::build_pow(operand(base), operand(exp)));
}

py::object evaluate(py::handle self, py::handle env) {
    /*
    Expression.evaluate.

    Args:
        self (py::handle): The Expression.
        env (py::handle): The mapping of variables to values, or None for no variables.

    Returns:
        py::object: The value, as a float or a complex.
    */
    py::object mapping = env.is_none() ? py::object(py::dict()) : py::reinterpret_borrow<py::object>(env);
    return num_to_py(core::evaluate(node_of(self), mapping));
}

py::list free_symbols(py::handle self) {
    /*
    Expression.free_symbols.

    Args:
        self (py::handle): The Expression.

    Returns:
        py::list: Its distinct variables and parameters, in the order they are first met.
    */
    py::list out;
    for (const NodePtr& symbol : core::free_symbols(node_of(self))) {
        out.append(symbol->owner);
    }
    return out;
}

py::list variables(py::handle self) {
    /*
    Expression.variables.

    Args:
        self (py::handle): The Expression.

    Returns:
        py::list: Its distinct variables and parameters, sorted by label.
    */
    std::vector<NodePtr> symbols = core::free_symbols(node_of(self));
    std::stable_sort(symbols.begin(), symbols.end(), [](const NodePtr& a, const NodePtr& b) { return a->label < b->label; });
    py::list out;
    for (const NodePtr& symbol : symbols) {
        out.append(symbol->owner);
    }
    return out;
}

bool is_parameterized(py::handle self) {
    /*
    Expression.is_parameterized.

    Args:
        self (py::handle): The Expression.

    Returns:
        bool: True if every free symbol is a Parameter, including when there are none.
    */
    const std::vector<NodePtr> symbols = core::free_symbols(node_of(self));
    return std::all_of(symbols.begin(), symbols.end(), [](const NodePtr& symbol) { return symbol->parameter; });
}

int64_t degree(py::handle self) {
    /*
    Expression.degree.

    Args:
        self (py::handle): The Expression.

    Returns:
        int64_t: Its polynomial degree.

    Raises:
        NonPolynomialError: If the expression is not a polynomial.
    */
    return core::degree(node_of(self));
}

py::object derivative(py::handle self, py::handle symbol) {
    /*
    Expression.derivative.

    Args:
        self (py::handle): The Expression.
        symbol (py::handle): The variable to differentiate by.

    Returns:
        py::object: The derivative.

    Raises:
        py::type_error: If symbol is neither an Expression nor a number.
    */
    return wrap(core::derivative(node_of(self), operand(symbol)));
}

py::object expand(py::handle self) {
    /*
    Expression.expand.

    Args:
        self (py::handle): The Expression.

    Returns:
        py::object: The canonical sum of monomials equal to it.
    */
    return wrap(core::expand(node_of(self)));
}

py::object substitute(py::handle self, py::handle mapping) {
    /*
    Expression.substitute. Keys that are not Expressions are ignored, as are values that are neither
    Expressions nor numbers.

    Args:
        self (py::handle): The Expression.
        mapping (py::handle): Any mapping from sub-expressions to their replacements.

    Returns:
        py::object: The expression with its replacements made.
    */
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
    /*
    Expression.to_binary.

    Args:
        self (py::handle): The Expression.

    Returns:
        py::object: The expression with every variable rewritten in binary variables.
    */
    return wrap(core::to_binary(node_of(self)));
}

py::object get_constant(py::handle self) {
    /*
    Expression.get_constant.

    Args:
        self (py::handle): The Expression.

    Returns:
        py::object: Its constant term, or 0 if it has none.
    */
    return num_to_py(core::get_constant(node_of(self)));
}

py::dict as_coefficients_dict(py::handle self) {
    /*
    Expression.as_coefficients_dict.

    Args:
        self (py::handle): The Expression.

    Returns:
        py::dict: Each non-constant monomial mapped to its coefficient.
    */
    py::dict out;
    for (const auto& [base, coeff] : coefficients(node_of(self))) {
        out[wrap(base)] = num_to_py(coeff);
    }
    return out;
}

py::list monomial_factors(py::handle self) {
    /*
    Expression.monomial_factors.

    Args:
        self (py::handle): The Expression, which must be a monomial.

    Returns:
        py::list: A (base, power) tuple for each factor, leaving out the coefficient.
    */
    py::list out;
    for (const auto& [base, power] : core::monomial_factors(node_of(self))) {
        out.append(py::make_tuple(wrap(base), power));
    }
    return out;
}

py::list to_list(py::handle self) {
    /*
    Expression.to_list.

    Args:
        self (py::handle): The Expression.

    Returns:
        py::list: The operands of a sum or product, otherwise just the expression itself.
    */
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
    /*
    The operands of an Add, Mul, Pow or Function, behind their args, base, exp and arg properties.

    Args:
        self (py::handle): The Expression.

    Returns:
        py::tuple: The operands, in canonical order.
    */
    const NodePtr node = node_of(self);
    py::tuple out(node->args.size());
    // A subtree shared by several parents comes back as one object, so pickle and deepcopy memoize it rather than
    // copying it once per path, which is exponential in the depth of the sharing
    for (size_t i = 0; i < node->args.size(); ++i) {
        out[i] = wrap_shared(node->args[i]);
    }
    return out;
}

py::object value(py::handle self) {
    /*
    Constant.value.

    Args:
        self (py::handle): The Constant.

    Returns:
        py::object: Its number.
    */
    return num_to_py(node_of(self)->value);
}

py::object coefficient(py::handle self) {
    /*
    Mul.coefficient.

    Args:
        self (py::handle): The Mul.

    Returns:
        py::object: Its leading constant, or 1 if it has none.
    */
    return num_to_py(peel(node_of(self)).second);
}

py::object monomial(py::handle self) {
    /*
    Mul.monomial.

    Args:
        self (py::handle): The Mul.

    Returns:
        py::object: The product without its leading constant.
    */
    return wrap(peel(node_of(self)).first);
}

int64_t hash(py::handle self) {
    /*
    Expression.__hash__.

    Args:
        self (py::handle): The Expression.

    Returns:
        int64_t: The hash of its node.
    */
    return static_cast<int64_t>(node_of(self)->hash);
}

bool equal(py::handle self, py::handle other) {
    /*
    Expression.__eq__.

    Args:
        self (py::handle): The Expression.
        other (py::handle): The object to compare against.

    Returns:
        bool: True if other is a structurally equal Expression, False for anything else.
    */
    if (!PyObject_TypeCheck(other.ptr(), reinterpret_cast<PyTypeObject*>(registry().expression_cls.ptr()))) {
        return false;
    }
    return core::equal(node_of(self), node_of(other));
}

std::string repr(py::handle self) {
    /*
    Expression.__repr__.

    Args:
        self (py::handle): The Expression.

    Returns:
        std::string: The printed expression.
    */
    return core::repr(node_of(self));
}

}  // namespace python

}  // namespace qilisdk::expr
