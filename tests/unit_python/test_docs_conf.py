# Copyright 2025 Qilimanjaro Quantum Tech
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

import ast
from pathlib import Path

import pytest

from qilisdk.core.types import QiliEnum

DOCS_DIR = Path(__file__).parents[2] / "docs"
CONF_PATH = DOCS_DIR / "conf.py"
CONF_TREE = ast.parse(CONF_PATH.read_text(encoding="utf-8"), filename=str(CONF_PATH))


def load_conf_function(name):
    """Compile a single top-level function out of ``docs/conf.py`` and return it.

    The module itself cannot simply be imported here: it pulls in the whole ``docs``
    dependency group, which the unit test job does not install. Executing one function
    definition in isolation keeps these tests runnable in a plain dev environment.
    """
    node = next(n for n in CONF_TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(CONF_PATH), "exec"), namespace)
    return namespace[name]


def load_conf_value(name):
    """Return the literal value assigned to a top-level name in ``docs/conf.py``."""
    node = next(
        n
        for n in CONF_TREE.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
    )
    return ast.literal_eval(node.value)


@pytest.mark.parametrize("method", ["to_yaml", "from_yaml"])
def test_skip_yaml_class_methods_skips_serialization_helpers(method):
    hook = load_conf_function("skip_yaml_class_methods")
    assert hook(None, "method", f"qilisdk.core.types.QiliEnum.{method}", None, False, None) is True


@pytest.mark.parametrize("skip", [True, False])
def test_skip_yaml_class_methods_passes_everything_else_through(skip):
    hook = load_conf_function("skip_yaml_class_methods")
    assert hook(None, "method", "qilisdk.core.types.QiliEnum.value", None, skip, None) is skip
    assert hook(None, "class", "qilisdk.core.types.QiliEnum", None, skip, None) is skip


def test_skip_hook_matches_the_member_type_autoapi_passes():
    """Guard the ``what`` literal against a typo, which would silently disable the hook.

    ``PythonMethod.type`` is the exact string autoapi hands to ``autoapi-skip-member``,
    so comparing against it catches both a misspelling here and a rename upstream.
    """
    objects = pytest.importorskip("autoapi._objects")
    hook = load_conf_function("skip_yaml_class_methods")
    assert hook(None, objects.PythonMethod.type, "qilisdk.core.types.QiliEnum.to_yaml", None, False, None) is True


def test_skip_hook_is_connected_to_autoapi():
    setup_node = next(n for n in CONF_TREE.body if isinstance(n, ast.FunctionDef) and n.name == "setup")
    connected = {
        (call.args[0].value, call.args[1].id)
        for call in ast.walk(setup_node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "connect"
    }
    assert ("autoapi-skip-member", "skip_yaml_class_methods") in connected


def test_skip_hook_targets_methods_that_still_exist():
    assert "to_yaml" in vars(QiliEnum)
    assert "from_yaml" in vars(QiliEnum)


def test_source_suffix_entries_match_real_docs_sources():
    """Every registered suffix must occur in ``docs/``, so a typo cannot sit there unnoticed."""
    for suffix in load_conf_value("source_suffix"):
        sources = (path for path in DOCS_DIR.rglob(f"*{suffix}") if "_build" not in path.parts)
        assert next(sources, None) is not None, f"no {suffix} file exists under docs/"


def test_source_suffix_leaves_notebooks_to_nbsphinx():
    """Notebooks must reach the nbsphinx parser rather than the reStructuredText one.

    nbsphinx registers ``.ipynb`` itself, but only while nothing else has claimed it, and
    the list form of ``source_suffix`` maps every entry to ``restructuredtext``. Leaving the
    suffix out is therefore correct; naming it is only safe in the explicit dict form.
    """
    source_suffix = load_conf_value("source_suffix")
    assert "nbsphinx" in load_conf_value("extensions")
    if isinstance(source_suffix, dict):
        assert source_suffix.get(".ipynb", "jupyter_notebook") == "jupyter_notebook"
    else:
        assert ".ipynb" not in source_suffix
