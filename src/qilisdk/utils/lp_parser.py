# Copyright 2026 Qilimanjaro Quantum Tech
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
"""LP-format file parser and serializer for QiliSDK.

Translates between an LP-format file (CPLEX flavor) and a
:class:`qilisdk.core.Model`. Supports linear and quadratic objectives, linear
constraints, variable bounds, and ``Binary`` / ``General`` (integer) declarations.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from pyparsing import (
    CaselessKeyword,
    CaselessLiteral,
    Forward,
    Group,
    Literal,
    MatchFirst,
    Optional,
    ParserElement,
    ParseResults,
    Suppress,
    Word,
    ZeroOrMore,
    alphanums,
    nums,
    one_of,
    rest_of_line,
)

from qilisdk.core.model import Model, ObjectiveSense
from qilisdk.core.variables import (
    EQ,
    GEQ,
    GT,
    LEQ,
    LT,
    NEQ,
    BaseVariable,
    BinaryVariable,
    ComparisonOperation,
    ComparisonTerm,
    Domain,
    Operation,
    Term,
    Variable,
    _assert_real,
    _float_if_real,
)

__all__ = ["from_lp", "from_lp_file", "to_lp", "to_lp_file"]

# === Constants ====================================================================

_INF = 1e30
# A monomial token is a Group of [coef_float, var_name_str]; this length lets us
# distinguish it from a parenthesized sub-expression at parse-walk time.
_MONOMIAL_LEN = 2

# An arithmetic expression value: a number or a symbolic variable / term.
_Arith = int | float | BaseVariable | Term

_COMPARISON_FACTORY = {
    "=": EQ,
    "==": EQ,
    "!=": NEQ,
    "<": LT,
    "<=": LEQ,
    "=<": LEQ,
    ">": GT,
    ">=": GEQ,
    "=>": GEQ,
}


# === Pure helpers =================================================================


def _finite_or_none(value: float) -> float | None:
    """Map ``± INF`` to ``None`` so the Variable can fall back to its domain bound.

    Args:
        value (float): The candidate bound value.

    Returns:
        float | None: ``None`` if ``value`` is at or beyond ``± INF``, else ``value``.
    """
    return None if abs(value) >= _INF else value


def _apply_arith(lhs: _Arith, op: str, rhs: _Arith) -> _Arith:
    """Apply an arithmetic ``op`` to ``lhs`` and ``rhs``.

    Args:
        lhs (_Arith): Left-hand operand.
        op (str): One of ``+``, ``-``, ``*``, ``/`` (the arithmetic operators
            produced by the LP grammar).
        rhs (_Arith): Right-hand operand.

    Returns:
        _Arith: The result of applying ``op`` to ``lhs`` and ``rhs``.

    Raises:
        NotImplementedError: If ``op`` is not a supported arithmetic operator,
            or if ``/`` is used with a non-numeric divisor.
    """
    if op == "+":
        return lhs + rhs
    if op == "-":
        return lhs - rhs
    if op == "*":
        return lhs * rhs
    if op == "/":
        if not isinstance(rhs, (int, float)):
            raise NotImplementedError("Division by a variable or term is not supported")
        return lhs / rhs
    raise NotImplementedError(f"the arithmetic operation {op!r} is not implemented")


def _merge(current: _Arith, op: str, operand: _Arith) -> _Arith:
    """Fold ``operand`` into ``current`` using the pending arithmetic ``op``.

    With no pending op, ``operand`` replaces ``current`` (which by construction is the neutral value 0 at this point).

    Args:
        current (_Arith): The current addend value.
        op (str): The pending arithmetic operator (or empty string).
        operand (_Arith): The next operand to fold in.

    Returns:
        _Arith: The updated addend value.
    """
    if not op:
        return operand
    return _apply_arith(current, op, operand)


def _to_term(expr: _Arith) -> Term:
    """Coerce ``expr`` to a :class:`Term` so it can be passed to ``set_objective``.

    Args:
        expr (_Arith): The arithmetic expression to coerce.

    Returns:
        Term: A :class:`Term` wrapping ``expr``.
    """
    if isinstance(expr, Term):
        return expr
    return Term([expr], operation=Operation.ADD)


def _label_or(default: str, label_group: ParseResults) -> str:
    """Return the label from ``label_group`` if present, else ``default``.

    Args:
        default (str): Fallback label.
        label_group (ParseResults): Group containing zero or one name tokens.

    Returns:
        str: The label string.
    """
    return str(label_group[0]) if len(label_group) > 0 else default


# === Grammar ======================================================================


@cache
def _grammar() -> ParserElement:
    """Build (once) and return the pyparsing grammar for the LP format.

    The grammar carries the parse actions that convert numeric tokens to Python
    floats; it is stateless across invocations of :meth:`ParserElement.parse_string`,
    so a single instance is shared across parses.

    Returns:
        ParserElement: The compiled top-level grammar.
    """
    # Variable / constraint / objective names.
    all_name_chars = alphanums + "!\"#$%&()/,.;?@_'`{}|~"
    first_char = "".join(c for c in alphanums if c not in nums + "eE.")
    name = Word(first_char, all_name_chars, max=255)
    keywords = [
        "inf",
        "infinity",
        "max",
        "maximum",
        "maximize",
        "min",
        "minimum",
        "minimize",
        "subj",
        "subject",
        "to",
        "s.t.",
        "st",
        "bound",
        "bounds",
        "bin",
        "binaries",
        "binary",
        "gen",
        "general",
        "free",
        "end",
    ]
    py_keyword = MatchFirst(map(CaselessKeyword, keywords))
    valid_name = ~py_keyword + name

    colon = Suppress(one_of(": ::"))
    plus_minus = one_of("+ -")
    inf = one_of("inf infinity", caseless=True)
    number = Word(nums + ".")
    sense = one_of("< <= =< = > >= =>")

    # Section tags.
    obj_tag_max = one_of("max maximum maximize", caseless=True)
    obj_tag_min = one_of("min minimum minimize", caseless=True)
    obj_tag = (obj_tag_max | obj_tag_min).set_results_name("objSense")
    constraints_tag = one_of(["subj to", "subject to", "s.t.", "st"], caseless=True)
    bounds_tag = one_of("bound bounds", caseless=True)
    bin_tag = one_of("bin binaries binary", caseless=True)
    gen_tag = one_of("gen general", caseless=True)
    end_tag = CaselessLiteral("end")

    # Coefficients (sign + optional number) collapse to a Python float.
    first_var_coef = Optional(plus_minus, "+") + Optional(number, "1")
    first_var_coef.set_parse_action(lambda toks: float("".join(toks)))
    coef = plus_minus + Optional(number, "1")
    coef.set_parse_action(lambda toks: float("".join(toks)))

    # A monomial: Group([coef_float, var_name]).
    first_var = Group(first_var_coef + valid_name)
    var = Group(coef + valid_name)

    # Expression with operator precedence baked into the grammar.
    l_par, r_par = map(Suppress, "[]")
    var_expr = Forward()
    operand = first_var | var | number
    factor = operand | Group(l_par + var_expr + r_par)
    term_grammar = factor + ZeroOrMore(one_of("* /") + factor)
    var_expr <<= term_grammar + ZeroOrMore(one_of("+ -") + term_grammar)

    # Optional label (Group keeps it positionally stable even when absent).
    labelled = Group(Optional(valid_name + colon))

    objective = (obj_tag + labelled + var_expr).set_results_name("objective")

    rhs = Optional(plus_minus, "+") + number
    rhs.set_parse_action(lambda toks: float("".join(toks)))

    constraint = Group(labelled + var_expr + sense + rhs)
    constraints = ZeroOrMore(constraint).set_results_name("constraints")

    signed_inf = (plus_minus + inf).set_parse_action(lambda toks: (1 if toks[0] == "+" else -1) * _INF)
    signed_number = (Optional(plus_minus, "+") + number).set_parse_action(lambda toks: float("".join(toks)))
    number_or_inf = signed_number | signed_inf
    lineq = number_or_inf + sense
    rineq = sense + number_or_inf

    sensestmt = Group(
        Optional(lineq).set_results_name("leftbound")
        + valid_name.copy().set_results_name("name")
        + Optional(rineq).set_results_name("rightbound")
    )
    free_var = Group(valid_name.copy().set_results_name("name") + Literal("free"))
    bound_stmt = free_var | sensestmt

    bounds = bounds_tag + Group(ZeroOrMore(bound_stmt)).set_results_name("bounds")
    generals = gen_tag + Group(ZeroOrMore(valid_name)).set_results_name("generals")
    binaries = bin_tag + Group(ZeroOrMore(valid_name)).set_results_name("binaries")

    var_info = ZeroOrMore(bounds | generals | binaries)
    grammar = objective + constraints_tag + constraints + var_info + Optional(end_tag)

    # Comments start with a backslash and run to the end of the line.
    grammar.ignore(Literal("\\") + rest_of_line)
    return grammar


# === Variables and expressions ====================================================


def _build_declared_variables(parsed: ParseResults, variable_dict: dict[str, BaseVariable]) -> None:
    """Populate ``variable_dict`` from the ``Bounds`` / ``Binary`` / ``General`` sections.

    Args:
        parsed (ParseResults): Output of :func:`_grammar` applied to LP content.
        variable_dict (dict[str, BaseVariable]): The dictionary to populate, keyed by
            variable label.
    """
    # Bounds first so generals can pick them up. ``None`` means "unbounded in that
    # direction" — the Variable then falls back to its domain's natural extreme.
    bounds_map: dict[str, tuple[float | None, float | None]] = {}
    for entry in parsed.bounds:
        label = str(entry.name[0])
        if len(entry) == _MONOMIAL_LEN and entry[1] == "free":
            bounds_map[label] = (None, None)
        else:
            lo = float(entry.leftbound[0]) if entry.leftbound else 0.0
            hi = float(entry.rightbound[1]) if entry.rightbound else _INF
            bounds_map[label] = (_finite_or_none(lo), _finite_or_none(hi))

    for binary_label in parsed.binaries:
        label = str(binary_label)
        variable_dict[label] = BinaryVariable(label=label)

    for general_label in parsed.generals:
        label = str(general_label)
        lo, hi = bounds_map.pop(label, (0, None))
        variable_dict[label] = Variable(
            label=label,
            domain=Domain.INTEGER,
            bounds=(None if lo is None else int(lo), None if hi is None else int(hi)),
        )

    # Remaining declared (real / continuous) variables.
    for label, bounds in bounds_map.items():
        if label not in variable_dict:
            variable_dict[label] = Variable(label=label, domain=Domain.REAL, bounds=bounds)


def _resolve_var(name: str, variable_dict: dict[str, BaseVariable]) -> BaseVariable:
    """Look up ``name``, lazily creating a default continuous variable if missing.

    LP-format default for variables that appear only in expressions (not in any
    ``Bounds`` / ``General`` / ``Binary`` section) is continuous with bounds ``[0, +inf)``.

    Args:
        name (str): The variable's label.
        variable_dict (dict[str, BaseVariable]): The variable lookup, mutated in
            place when a default variable is created.

    Returns:
        BaseVariable: The matching (or freshly created) variable.
    """
    if name not in variable_dict:
        variable_dict[name] = Variable(label=name, domain=Domain.REAL, bounds=(0.0, _INF))
    return variable_dict[name]


def _extract_arith(expr: ParseResults | list, variable_dict: dict[str, BaseVariable]) -> _Arith:
    """Walk a flat token sequence and assemble an arithmetic expression.

    Operator precedence is already encoded by the grammar: ``*`` / ``/`` only appear between consecutive factors,
    while ``+`` / ``-`` separate addends.
    We accumulate addends in ``accum`` and build the in-progress addend in ``current``.

    Args:
        expr (ParseResults | list): Flat token sequence (no comparison operators).
        variable_dict (dict[str, BaseVariable]): Variable lookup, mutated when an unseen variable name is encountered.

    Returns:
        _Arith: A number, :class:`BaseVariable`, or :class:`Term`.
    """
    accum: _Arith = 0
    current: _Arith = 0
    op: str = ""

    for e in expr:
        if isinstance(e, str):
            if e == "+":
                accum += current
                current = 0
                op = ""
                continue
            if e == "-":
                accum += current
                current = -1
                op = "*"
                continue
            if e in {"*", "/"}:
                op = e
                continue
            # Bare number from the `number` operand alternative.
            current = _merge(current, op, float(e))
            op = ""
            continue

        if isinstance(e, (int, float)):
            current = _merge(current, op, e)
            op = ""
            continue

        # Grouped tokens: either a (coef, name) monomial or a sub-expression.
        if len(e) == _MONOMIAL_LEN and isinstance(e[1], str):
            operand: _Arith = e[0] * _resolve_var(e[1], variable_dict)
        else:
            operand = _extract_arith(e, variable_dict)
        current = _merge(current, op, operand)
        op = ""

    return accum + current


def _extract_constraint(expr: ParseResults | list, variable_dict: dict[str, BaseVariable]) -> ComparisonTerm:
    """Split a constraint's tokens at the sense operator and build the comparison.

    Args:
        expr (ParseResults | list): Flat token sequence containing exactly one comparison operator
            separating the lhs and rhs.
        variable_dict (dict[str, BaseVariable]): Variable lookup, passed through to :func:`_extract_arith`.

    Returns:
        ComparisonTerm: The constraint's comparison term.

    Raises:
        ValueError: If no comparison operator is present.
    """
    items = list(expr)
    for i, e in enumerate(items):
        if isinstance(e, str) and e in _COMPARISON_FACTORY:
            lhs = _extract_arith(items[:i], variable_dict)
            rhs = _extract_arith(items[i + 1 :], variable_dict)
            return _COMPARISON_FACTORY[e](lhs, rhs)
    raise ValueError("constraint is missing a comparison operator")


# === Public API ===================================================================


def from_lp(lp_str: str) -> Model:
    """Parse an LP-format string and create a corresponding :class:`Model`.

    Supports:
        - Linear and quadratic objectives (Maximize / Minimize)
        - Linear constraints with senses ``<``, ``<=``, ``=``, ``>=``, ``>``
        - ``Bounds`` (two-sided ranges, ``free``, ``±infinity``)
        - ``General`` (integer) and ``Binary`` declarations
        - Backslash line comments

    Args:
        lp_str (str): The LP-format source.

    Returns:
        Model: The constructed optimization model.
    """
    parsed = _grammar().parse_string(lp_str, parse_all=True)
    variable_dict: dict[str, BaseVariable] = {}
    _build_declared_variables(parsed, variable_dict)

    model = Model("")
    obj_sense = (
        ObjectiveSense.MAXIMIZE
        if str(parsed.objSense).lower() in {"max", "maximum", "maximize"}
        else ObjectiveSense.MINIMIZE
    )
    obj_label = _label_or("obj", parsed.objective[1])
    obj = _extract_arith(parsed.objective[2:], variable_dict)
    model.set_objective(_to_term(obj), label=obj_label, sense=obj_sense)

    for idx, con in enumerate(parsed.constraints, start=1):
        con_label = _label_or(f"c{idx}", con[0])
        model.add_constraint(label=con_label, term=_extract_constraint(con[1:], variable_dict))

    return model


def from_lp_file(filename: str) -> Model:
    """Read an LP-format file and create a corresponding :class:`Model`.

    Args:
        filename (str): Path to the LP file.

    Returns:
        Model: The constructed optimization model.
    """
    return from_lp(Path(filename).read_text(encoding="utf-8"))


# === Serialization helpers ========================================================


def _sense_to_lp(operation: ComparisonOperation) -> str:
    """Render a :class:`ComparisonOperation` as its LP-format sense token.

    The enum value works directly for every operator except ``EQ``: the enum stores it as the
    Python ``==``, but the LP grammar expects a single ``=``.

    Args:
        operation (ComparisonOperation): The comparison operator to render.

    Returns:
        str: The LP-format sense token.
    """
    return "=" if operation is ComparisonOperation.EQ else operation.value


def _format_number(value: float) -> str:
    """Render a numeric coefficient or bound for LP output.

    Integer-valued floats lose the trailing ``.0``; ``±_INF`` collapses to the LP infinity keyword so the result
    round-trips through :func:`from_lp`.

    Args:
        value (float): The number to format.

    Returns:
        str: The LP-format text for ``value``.
    """
    if value >= _INF:
        return "+infinity"
    if value <= -_INF:
        return "-infinity"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _mul_factors(term: Term) -> tuple[float, list[BaseVariable]]:
    """Flatten a ``MUL`` sub-term into ``(coefficient, [variable, ...])``.

    Exponents are expanded into repeated variables so the result fits the LP grammar (which has no ``^`` operator).
    Bilinear and quadratic monomials are the only nested-Term shape this exporter expects to encounter.

    Args:
        term (Term): A ``MUL`` term whose elements are variables-with-exponents and optionally a single
        :data:`Term.CONST` coefficient.

    Returns:
        tuple[float, list[BaseVariable]]: The extracted constant factor and the ordered list of variable factors.

    Raises:
        NotImplementedError: If the term contains a nested non-MUL sub-term, which cannot be expressed in the LP
            grammar without flattening.
    """
    coef: float = 1.0
    factors: list[BaseVariable] = []
    for elem in term:
        value = term[elem]
        if isinstance(elem, BaseVariable) and elem == Term.CONST:
            coef *= float(_assert_real(value))
            continue
        if isinstance(elem, BaseVariable):
            for _ in range(int(_assert_real(value))):
                factors.append(elem)
            continue
        raise NotImplementedError(
            "LP export does not support nested non-monomial terms; " f"encountered sub-term {elem!r} inside a product."
        )
    return coef, factors


def _format_monomial(coef: float, factors: list[BaseVariable], is_first: bool) -> str:
    """Render one signed monomial inside an LP expression.

    Args:
        coef (float): The monomial's coefficient.
        factors (list[BaseVariable]): The variable factors; empty for a constant term.
        Quadratic / bilinear terms have two entries.
        is_first (bool): Whether this is the leading addend (suppresses the leading ``+`` for positive coefficients).

    Returns:
        str: The rendered monomial, including its leading sign and a trailing
            space-separated form like ``"+ 2 x * y"`` or ``"- x"``.
    """
    sign = "-" if coef < 0 else "+"
    magnitude = abs(coef)
    if not factors:
        body = _format_number(magnitude)
    else:
        var_part = " * ".join(str(v) for v in factors)
        body = var_part if magnitude == 1 else f"{_format_number(magnitude)} {var_part}"
    if is_first:
        return body if sign == "+" else f"-{body}" if not factors else f"- {body}"
    return f"{sign} {body}"


def _render_expression(term: Term) -> str:
    """Render an ``ADD`` Term as a sequence of signed monomials.

    Args:
        term (Term): The expression to render. ``MUL`` and other operations are
            wrapped in an ``ADD`` envelope first.

    Returns:
        str: The LP-format expression text. Returns ``"0"`` for an empty term.
    """
    if term.operation != Operation.ADD:
        term = Term([term], Operation.ADD)
    if len(term) == 0:
        return "0"
    pieces: list[str] = []
    is_first = True
    for elem in term:
        raw_coef = _float_if_real(term[elem])
        coef = float(_assert_real(raw_coef))
        if coef == 0:
            continue
        if isinstance(elem, BaseVariable):
            factors: list[BaseVariable] = [] if elem == Term.CONST else [elem]
        elif isinstance(elem, Term):
            sub_coef, factors = _mul_factors(elem)
            coef *= sub_coef
        else:
            raise NotImplementedError(f"Cannot serialize term element of type {type(elem).__name__}")
        pieces.append(_format_monomial(coef, factors, is_first))
        is_first = False
    return " ".join(pieces) if pieces else "0"


def _bounds_line(var: Variable) -> str | None:
    """Build the ``Bounds`` line for ``var``, or ``None`` if the LP defaults suffice.

    The LP default for a continuous / integer variable is ``[0, +infinity)``, so variables matching that pattern are
    omitted from the section entirely.

    Args:
        var (Variable): The variable whose bounds to render.

    Returns:
        str | None: The bounds-section entry, or ``None`` when the defaults apply.
    """
    lo, hi = var.lower_bound, var.upper_bound
    lo_inf = lo <= var.domain.min()
    hi_inf = hi >= var.domain.max()
    if lo_inf and hi_inf:
        return f"{var.label} free"
    if lo == 0 and hi_inf:
        return None
    if lo_inf:
        return f"-infinity <= {var.label} <= {_format_number(float(hi))}"
    if hi_inf:
        return f"{var.label} >= {_format_number(float(lo))}"
    return f"{_format_number(float(lo))} <= {var.label} <= {_format_number(float(hi))}"


# === Public API (serialization) ===================================================


def to_lp(model: Model) -> str:
    """Serialize a :class:`Model` to an LP-format string.

    The output is round-trip compatible with :func:`from_lp`: parsing the
    returned text reconstructs an equivalent model (modulo encoding constraints,
    which are derived from variable bounds).

    Conventions:
        - Sections are emitted in the order ``Minimize``/``Maximize`` then ``Subject To`` then ``Bounds``
            then ``General`` then ``Binary`` and finally ``End``.
        - Quadratic and bilinear monomials are expanded as products
          (``x * x``, ``x * y``); the LP grammar has no ``^`` operator.
        - Unit coefficients are elided (``x`` rather than ``1 x``).
        - Constraint right-hand sides are normalized to a single constant.
        - Unbounded continuous variables are written as ``var free``; one-sided bounds use ``var >= lo`` or
            ``-infinity <= var <= up``; the LP default ``[0, +inf)`` is omitted from the ``Bounds`` section entirely.

    Args:
        model (Model): The optimization model to serialize.

    Returns:
        str: The LP-format text, terminated by ``"End\\n"``.
    """
    sense_keyword = "Maximize" if model.objective.sense is ObjectiveSense.MAXIMIZE else "Minimize"
    obj_label = model.objective.label
    obj_expr = _render_expression(model.objective.term)
    lines: list[str] = [sense_keyword, f" {obj_label}: {obj_expr}", "Subject To"]

    for con in model.constraints:
        lhs = _render_expression(con.term.lhs)
        rhs = _format_number(float(_assert_real(con.term.rhs.get_constant())))
        op = _sense_to_lp(con.term.operation)
        lines.append(f" {con.label}: {lhs} {op} {rhs}")

    variables = model.variables()
    bound_lines: list[str] = []
    generals: list[str] = []
    binaries: list[str] = []
    for var in variables:
        if isinstance(var, BinaryVariable) or (isinstance(var, Variable) and var.domain is Domain.BINARY):
            binaries.append(var.label)
            continue
        if isinstance(var, Variable) and var.domain is Domain.INTEGER:
            generals.append(var.label)
        line = _bounds_line(var) if isinstance(var, Variable) else None
        if line is not None:
            bound_lines.append(line)

    if bound_lines:
        lines.append("Bounds")
        lines.extend(f" {line}" for line in bound_lines)
    if generals:
        lines.extend(("General", " " + " ".join(generals)))
    if binaries:
        lines.extend(("Binary", " " + " ".join(binaries)))
    lines.append("End")
    return "\n".join(lines) + "\n"


def to_lp_file(model: Model, filename: str) -> None:
    """Serialize a :class:`Model` to an LP-format file.

    Args:
        model (Model): The optimization model to serialize.
        filename (str): Destination path for the LP file.
    """
    Path(filename).write_text(to_lp(model), encoding="utf-8")
