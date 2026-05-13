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

import pytest

from qilisdk.core.model import Model, ObjectiveSense
from qilisdk.core.variables import EQ, GEQ, LEQ, BinaryVariable, ComparisonOperation, Domain, Variable
from qilisdk.utils.lp_parser import from_lp, from_lp_file, to_lp, to_lp_file


def _vars_by_label(model) -> dict:
    """Return a {label: variable} map for ergonomic assertions."""
    return {v.label: v for v in model.variables()}


def _constraints_by_label(model) -> dict:
    return {c.label: c for c in model.constraints}


# --- Objective sense and labels ---------------------------------------------------


def test_minimize_sense_with_default_label():
    model = from_lp(
        """
        Minimize
         x
        Subject To
         x >= 0
        End
        """
    )
    assert model.objective.sense is ObjectiveSense.MINIMIZE
    assert model.objective.label == "obj"


def test_maximize_sense_and_explicit_label():
    model = from_lp(
        """
        Maximize
         my_obj: x + y
        Subject To
         x + y <= 10
        End
        """
    )
    assert model.objective.sense is ObjectiveSense.MAXIMIZE
    assert model.objective.label == "my_obj"


@pytest.mark.parametrize("keyword", ["Minimize", "Minimum", "Min", "MINIMIZE"])
def test_minimize_keyword_is_caseless(keyword):
    """`min`, `minimum`, `minimize` (any case) should all be treated as MINIMIZE."""
    model = from_lp(f"{keyword}\n x\nSubject To\n x >= 0\nEnd\n")
    assert model.objective.sense is ObjectiveSense.MINIMIZE


# --- Objective term shapes --------------------------------------------------------


def test_objective_collapses_constants_and_repeated_vars():
    """`x + 2 + x + 3` → `2*x + 5`."""
    model = from_lp(
        """
        Minimize
         x + 2 + x + 3
        Subject To
         x >= 0
        End
        """
    )
    sample = {_vars_by_label(model)["x"]: 4}
    assert model.objective.term.evaluate(sample) == pytest.approx(2 * 4 + 5)


def test_objective_supports_quadratic_and_bilinear_terms():
    model = from_lp(
        """
        Minimize
         2 x * x + 3 x * y
        Subject To
         x >= 0
         y >= 0
        End
        """
    )
    vmap = _vars_by_label(model)
    sample = {vmap["x"]: 2, vmap["y"]: 5}
    assert model.objective.term.evaluate(sample) == pytest.approx(2 * 4 + 3 * 2 * 5)


def test_objective_handles_negated_parenthesised_group():
    """`- [x + 1]` should expand to `-x - 1` and fuse with sibling addends."""
    model = from_lp(
        """
        Minimize
         5 - [x + 1]
        Subject To
         x >= 0
        End
        """
    )
    assert model.objective.term.evaluate({_vars_by_label(model)["x"]: 7}) == pytest.approx(5 - 7 - 1)


def test_objective_with_decimal_and_negative_coefficients():
    model = from_lp(
        """
        Minimize
         1.5 x - 2.25 y + 0.5 z
        Subject To
         x + y + z = 1
        End
        """
    )
    vmap = _vars_by_label(model)
    sample = {vmap["x"]: 4, vmap["y"]: 2, vmap["z"]: 6}
    assert model.objective.term.evaluate(sample) == pytest.approx(1.5 * 4 - 2.25 * 2 + 0.5 * 6)


# --- Constraints -----------------------------------------------------------------


def test_all_comparison_operators_round_trip():
    """Every sense operator the grammar accepts should map to the right ComparisonOperation."""
    model = from_lp(
        """
        Minimize
         x
        Subject To
         c_le: x <= 1
         c_lt: x <  2
         c_ge: x >= 3
         c_gt: x >  4
         c_eq: x  = 5
        End
        """
    )
    cmap = _constraints_by_label(model)
    assert cmap["c_le"].term.operation is ComparisonOperation.LEQ
    assert cmap["c_lt"].term.operation is ComparisonOperation.LT
    assert cmap["c_ge"].term.operation is ComparisonOperation.GEQ
    assert cmap["c_gt"].term.operation is ComparisonOperation.GT
    assert cmap["c_eq"].term.operation is ComparisonOperation.EQ


def test_unlabelled_constraints_get_auto_labels():
    model = from_lp(
        """
        Minimize
         x + y + z
        Subject To
         x + y <= 10
         y + z <= 20
         z     <= 30
        End
        """
    )
    labels = [c.label for c in model.constraints]
    assert labels == ["c1", "c2", "c3"]


def test_constraint_with_negative_rhs():
    model = from_lp(
        """
        Minimize
         x
        Subject To
         c: x >= -5
        Bounds
         -10 <= x <= 10
        End
        """
    )
    constraint = _constraints_by_label(model)["c"]
    var = _vars_by_label(model)["x"]
    # rhs is -5: x = -5 satisfies the boundary, x = -6 violates it.
    assert constraint.term.evaluate({var: -5})
    assert not constraint.term.evaluate({var: -6})


def test_missing_comparison_raises():
    """Constraints with no sense operator are a grammar-level error in LP format."""
    with pytest.raises(Exception):  # pyparsing raises a ParseException, not ValueError  # noqa: B017, PT011
        from_lp(
            """
            Minimize
             x
            Subject To
             x + 1
            End
            """
        )


# --- Variables and bounds --------------------------------------------------------


def test_real_variable_with_two_sided_bounds():
    model = from_lp(
        """
        Minimize
         x
        Subject To
         x >= 0
        Bounds
         -2.5 <= x <= 7.5
        End
        """
    )
    var = _vars_by_label(model)["x"]
    assert var.domain is Domain.REAL
    assert var.bounds == (-2.5, 7.5)


def test_free_variable_is_unbounded_real():
    model = from_lp(
        """
        Minimize
         x
        Subject To
         x = 0
        Bounds
         x free
        End
        """
    )
    var = _vars_by_label(model)["x"]
    assert var.domain is Domain.REAL
    assert var.lower_bound < -1e29
    assert var.upper_bound > 1e29


def test_negative_infinity_lower_bound():
    model = from_lp(
        """
        Minimize
         x
        Subject To
         x <= 0
        Bounds
         -infinity <= x <= 5
        End
        """
    )
    var = _vars_by_label(model)["x"]
    assert var.lower_bound < -1e29
    assert var.upper_bound == pytest.approx(5.0)


def test_general_section_creates_integer_variable():
    """`General` after `Bounds` should preserve the bounds and switch domain to INTEGER."""
    model = from_lp(
        """
        Minimize
         x
        Subject To
         x >= 0
        Bounds
         0 <= x <= 10
        General
         x
        End
        """
    )
    var = _vars_by_label(model)["x"]
    assert var.domain is Domain.INTEGER
    assert var.bounds == (0, 10)


def test_general_without_bounds_uses_integer_domain_default():
    model = from_lp(
        """
        Minimize
         x
        Subject To
         x >= 0
        General
         x
        End
        """
    )
    var = _vars_by_label(model)["x"]
    assert var.domain is Domain.INTEGER
    # Lower bound defaults to 0 (LP convention); upper should sit at the int domain max.
    assert var.lower_bound == 0
    assert var.upper_bound > 1e18


def test_binary_section_creates_binary_variable():
    model = from_lp(
        """
        Minimize
         b1 + b2
        Subject To
         b1 + b2 = 1
        Binary
         b1
         b2
        End
        """
    )
    vmap = _vars_by_label(model)
    assert isinstance(vmap["b1"], BinaryVariable)
    assert isinstance(vmap["b2"], BinaryVariable)


def test_variable_only_in_expression_defaults_to_real_nonnegative():
    """Variables with no Bounds / General / Binary entry default to REAL [0, +inf)."""
    model = from_lp(
        """
        Minimize
         only_here
        Subject To
         only_here + 1 = 1
        End
        """
    )
    var = _vars_by_label(model)["only_here"]
    assert var.domain is Domain.REAL
    assert var.lower_bound == 0
    assert var.upper_bound > 1e29


# --- Comments and miscellaneous --------------------------------------------------


def test_backslash_comments_are_ignored():
    model = from_lp(
        """
        \\ leading comment, ignored
        Minimize
         x \\ trailing comment
        \\ another full-line comment
        Subject To
         x >= 0
        End
        """
    )
    assert "x" in _vars_by_label(model)


def test_subject_to_synonyms_are_accepted():
    """The grammar accepts `subj to`, `subject to`, `s.t.`, and `st`."""
    for keyword in ("Subject To", "subj to", "s.t.", "st"):
        model = from_lp(f"Minimize\n x\n{keyword}\n c: x >= 0\nEnd\n")
        assert _constraints_by_label(model)["c"]


def test_objective_with_trailing_constant_does_not_eat_subject_keyword():
    """Regression: `+ 10\nSubject To` previously parsed `+10*Subject` as a monomial."""
    model = from_lp(
        """
        Maximize
         x + 10
        Subject To
         x <= 50
        End
        """
    )
    assert model.objective.term.evaluate({_vars_by_label(model)["x"]: 0}) == pytest.approx(10)


# --- File I/O --------------------------------------------------------------------


def test_from_lp_file_reads_and_parses(tmp_path):
    path = tmp_path / "tiny.lp"
    path.write_text(
        "Minimize\n x\nSubject To\n c: x >= 1\nEnd\n",
        encoding="utf-8",
    )
    model = from_lp_file(str(path))
    assert _constraints_by_label(model)["c"]


def test_from_lp_file_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        from_lp_file(str(tmp_path / "does_not_exist.lp"))


# === Exporter (to_lp / to_lp_file) ===============================================


# --- Section structure and sense keywords -----------------------------------------


def test_to_lp_emits_sections_in_grammar_order():
    """LP grammar requires Min/Max → Subject To → Bounds → End in that order."""
    x = Variable("x", Domain.REAL, bounds=(0, 5))
    model = Model("m")
    model.set_objective(x, label="obj")
    model.add_constraint("c", LEQ(x, 4))
    text = to_lp(model)
    assert text.index("Minimize") < text.index("Subject To") < text.index("Bounds") < text.index("End")


def test_to_lp_uses_maximize_keyword_for_maximize_sense():
    x = Variable("x", Domain.REAL, bounds=(0, 5))
    model = Model("m")
    model.set_objective(x, sense=ObjectiveSense.MAXIMIZE)
    model.add_constraint("c", LEQ(x, 4))
    text = to_lp(model)
    assert "Maximize" in text
    assert "Minimize" not in text


def test_to_lp_includes_objective_label():
    x = Variable("x", Domain.REAL, bounds=(0, 5))
    model = Model("m")
    model.set_objective(2 * x, label="cost")
    model.add_constraint("c", LEQ(x, 4))
    assert " cost: " in to_lp(model)


# --- Round-trip parity ------------------------------------------------------------


def test_round_trip_linear_model_preserves_evaluations():
    x = Variable("x", Domain.REAL, bounds=(0, 10))
    y = Variable("y", Domain.REAL, bounds=(-5, 5))
    model = Model("m")
    model.set_objective(2 * x + 3 * y)
    model.add_constraint("c1", LEQ(x + y, 10))
    model.add_constraint("c2", GEQ(x - y, -3))

    reparsed = from_lp(to_lp(model))
    vmap = _vars_by_label(reparsed)
    sample = {vmap["x"]: 1.5, vmap["y"]: -2.0}
    orig_sample = {v: sample[vmap[v.label]] for v in model.variables()}
    assert reparsed.objective.term.evaluate(sample) == pytest.approx(
        model.objective.term.evaluate(orig_sample)
    )
    cmap = _constraints_by_label(reparsed)
    assert cmap["c1"].term.evaluate(sample) == model.constraints[0].term.evaluate(orig_sample)
    assert cmap["c2"].term.evaluate(sample) == model.constraints[1].term.evaluate(orig_sample)


def test_round_trip_quadratic_objective():
    x = Variable("x", Domain.REAL, bounds=(0, 5))
    y = Variable("y", Domain.REAL, bounds=(0, 5))
    model = Model("m")
    model.set_objective(2 * x * x + 3 * x * y - y + 7)
    model.add_constraint("c", LEQ(x + y, 4))

    reparsed = from_lp(to_lp(model))
    vmap = _vars_by_label(reparsed)
    sample = {vmap["x"]: 1.5, vmap["y"]: 2.0}
    orig_sample = {v: sample[vmap[v.label]] for v in model.variables()}
    assert reparsed.objective.term.evaluate(sample) == pytest.approx(
        model.objective.term.evaluate(orig_sample)
    )


def test_round_trip_preserves_comparison_operators():
    # Labels must not start with `e`/`E` — the grammar reserves that for sci-notation numbers.
    x = Variable("x", Domain.REAL, bounds=(0, 10))
    model = Model("m")
    model.set_objective(x)
    model.add_constraint("c_le", LEQ(x, 1))
    model.add_constraint("c_ge", GEQ(x, 3))
    model.add_constraint("c_eq", EQ(x, 5))

    reparsed = from_lp(to_lp(model))
    cmap = _constraints_by_label(reparsed)
    assert cmap["c_le"].term.operation is ComparisonOperation.LEQ
    assert cmap["c_ge"].term.operation is ComparisonOperation.GEQ
    assert cmap["c_eq"].term.operation is ComparisonOperation.EQ


def test_round_trip_normalises_constant_on_lhs():
    """`x + 3 <= 7` is stored as `lhs - rhs <op> 0`, so the rhs should serialize as `4`."""
    x = Variable("x", Domain.REAL, bounds=(0, 10))
    model = Model("m")
    model.set_objective(x)
    model.add_constraint("c", LEQ(x + 3, 7))
    text = to_lp(model)
    assert "x <= 4" in text


# --- Bounds, General, Binary sections ---------------------------------------------


def test_default_lp_lower_bound_is_omitted():
    """A `[0, +inf)` REAL variable is the LP default — no Bounds entry should appear."""
    xp = Variable("xp", Domain.REAL, bounds=(0, None))
    model = Model("m")
    model.set_objective(xp)
    model.add_constraint("c", LEQ(xp, 10))
    text = to_lp(model)
    # No Bounds section at all when every variable is at the LP default.
    assert "Bounds" not in text


def test_free_variable_emits_free_keyword():
    x = Variable("x", Domain.REAL)
    model = Model("m")
    model.set_objective(x)
    model.add_constraint("c", EQ(x, 0))
    text = to_lp(model)
    assert "x free" in text
    reparsed = from_lp(text)
    var = _vars_by_label(reparsed)["x"]
    assert var.lower_bound < -1e29
    assert var.upper_bound > 1e29


def test_integer_variable_goes_into_general_section():
    z = Variable("z", Domain.INTEGER, bounds=(-3, 7))
    model = Model("m")
    model.set_objective(z)
    model.add_constraint("c", LEQ(z, 5))
    text = to_lp(model)
    assert "General" in text
    assert "z" in text.split("General", 1)[1].split("End", 1)[0]
    reparsed = from_lp(text)
    var = _vars_by_label(reparsed)["z"]
    assert var.domain is Domain.INTEGER
    assert var.bounds == (-3, 7)


def test_binary_variable_goes_into_binary_section():
    b = BinaryVariable("b")
    x = Variable("x", Domain.REAL, bounds=(0, 1))
    model = Model("m")
    model.set_objective(b + x)
    model.add_constraint("c", LEQ(b + x, 1))
    text = to_lp(model)
    assert "Binary" in text
    assert "b" in text.split("Binary", 1)[1].split("End", 1)[0]
    reparsed = from_lp(text)
    assert isinstance(_vars_by_label(reparsed)["b"], BinaryVariable)


def test_one_sided_lower_bound_renders_compact_form():
    x = Variable("x", Domain.REAL, bounds=(2, None))
    model = Model("m")
    model.set_objective(x)
    model.add_constraint("c", LEQ(x, 10))
    assert "x >= 2" in to_lp(model)


def test_negative_lower_with_finite_upper_uses_neg_infinity():
    x = Variable("x", Domain.REAL, bounds=(None, 5))
    model = Model("m")
    model.set_objective(x)
    model.add_constraint("c", LEQ(x, 4))
    assert "-infinity <= x <= 5" in to_lp(model)


# --- Numeric formatting -----------------------------------------------------------


def test_unit_coefficient_is_omitted():
    x = Variable("x", Domain.REAL, bounds=(0, 1))
    y = Variable("y", Domain.REAL, bounds=(0, 1))
    model = Model("m")
    model.set_objective(x + y)
    model.add_constraint("c", LEQ(x + y, 1))
    text = to_lp(model)
    # No `1 x` — just `x`.
    assert "1 x" not in text
    assert "1 y" not in text


def test_negative_coefficient_renders_as_minus():
    x = Variable("x", Domain.REAL, bounds=(0, 10))
    y = Variable("y", Domain.REAL, bounds=(0, 10))
    model = Model("m")
    model.set_objective(x - y)
    model.add_constraint("c", LEQ(x - y, 0))
    assert "- y" in to_lp(model)


# --- File I/O ---------------------------------------------------------------------


def test_to_lp_file_writes_and_reads_back(tmp_path):
    x = Variable("x", Domain.REAL, bounds=(0, 5))
    model = Model("m")
    model.set_objective(2 * x, label="cost")
    model.add_constraint("c", LEQ(x, 3))

    path = tmp_path / "out.lp"
    to_lp_file(model, str(path))
    assert path.exists()

    reparsed = from_lp_file(str(path))
    var = _vars_by_label(reparsed)["x"]
    assert reparsed.objective.term.evaluate({var: 2.5}) == pytest.approx(5.0)


