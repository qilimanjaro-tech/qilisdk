# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import operator
import posixpath
import sys
from dataclasses import asdict
from pathlib import Path

from sphinx.ext import viewcode as _viewcode
from sphinx.locale import _
from sphinx.util import logging
from sphinx.util.display import status_iterator
from sphinxawesome_theme import ThemeOptions
from sphinxawesome_theme.postprocess import Icons

# -- Path setup ---------------------------------------------------------------
sys.path.insert(0, str((Path(__file__).resolve().parent / "../src").resolve()))

# -- Project information -----------------------------------------------------

project = "QiliSDK"
copyright = "2025, Qilimanjaro Quantum Tech"
author = "Qilimanjaro Quantum Tech"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "nbsphinx",
    "sphinx_design",
    "sphinx_mdinclude",
    "sphinx_multiversion",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = [".rst", ".pynb"]

# Syntax highlighting
pygments_style = "sphinx"
pygments_style_dark = "monokai"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_preprocess_types = True

# AutoAPI settings
autoapi_type = "python"
autoapi_dirs = ["../src/qilisdk"]
autoapi_root = "code/api"
autoapi_add_toctree_entry = True
autoapi_member_order = "bysource"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"
autoapi_python_use_implicit_namespaces = False

add_module_names = False
autoapi_keep_files = False

# Multiversion settings
# Whitelist pattern for tags (set to None to ignore all tags)
smv_tag_whitelist = r"^\d+\.\d+(\.\d+)?$"

# Whitelist pattern for branches (set to None to ignore all branches)
smv_branch_whitelist = r"^main$"

# Whitelist pattern for remotes (set to None to use local branches only)
smv_remote_whitelist = r"^origin$"

# Pattern for released versions
smv_released_pattern = r"^tags/.*$"

# Format for versioned output directories inside the build directory
smv_outputdir_format = "{ref.name}"

# Determines whether remote or local git branches/tags are preferred if their output dirs conflict
smv_prefer_remote_refs = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = project
html_permalinks_icon = Icons.permalinks_icon
html_favicon = "_static/q_trans.png"
# html_baseurl = "https://docs.qilimanjaro.tech/"
html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_copy_source = False
html_show_sourcelink = False
html_sidebars = {
    "**": ["sidebar_main_nav_links.html", "sidebar_toc.html", "versioning.html"],
}

theme_options = ThemeOptions(
    logo_light="_static/q_trans.png", logo_dark="_static/q_trans.png", awesome_external_links=True
)

html_theme_options = asdict(theme_options)

logger = logging.getLogger(__name__)


def skip_yaml_class_methods(app, what, name, obj, skip, options):  # noqa: ANN001, ANN201
    if what == "methpod" and any(x in name for x in ("from_yaml", "to_yaml")):
        return True
    return skip


def _safe_collect_pages(app):  # noqa: ANN001, ANN201
    env = app.builder.env
    if not hasattr(env, "_viewcode_modules"):
        return
    if not _viewcode.is_supported_builder(app.builder):
        return

    highlighter = app.builder.highlighter  # type: ignore[attr-defined]
    urito = app.builder.get_relative_uri
    modnames = set(env._viewcode_modules)

    for modname, entry in status_iterator(
        sorted(env._viewcode_modules.items()),
        _("highlighting module code... "),
        "blue",
        len(env._viewcode_modules),
        app.verbosity,
        operator.itemgetter(0),
    ):
        if not entry:
            continue
        if not _viewcode.should_generate_module_page(app, modname):
            continue

        code, tags, used, refname = entry
        pagename = posixpath.join(_viewcode.OUTPUT_DIRNAME, modname.replace(".", "/"))
        if app.config.highlight_language in {"default", "none"}:
            lexer = app.config.highlight_language
        else:
            lexer = "python"
        linenos = "inline" * app.config.viewcode_line_numbers
        highlighted = highlighter.highlight_block(code, lexer, linenos=linenos)
        lines = highlighted.splitlines()
        before, after = lines[0].split("<pre>")
        lines[0:1] = [before + "<pre>", after]
        max_index = len(lines) - 1
        link_text = _("[docs]")

        for name, docname in used.items():
            _type, start, end = tags[name]
            if start > max_index:
                logger.debug(
                    "viewcode: skipping anchor for %s.%s (start %s beyond %s lines)",
                    modname,
                    name,
                    start,
                    max_index,
                )
                continue
            backlink = urito(pagename, docname) + "#" + refname + "." + name
            lines[start] = (
                f'<div class="viewcode-block" id="{name}">\n'
                f'<a class="viewcode-back" href="{backlink}">{link_text}</a>\n' + lines[start]
            )
            lines[min(end, max_index)] += "</div>\n"

        parents = []
        parent = modname
        while "." in parent:
            parent = parent.rsplit(".", 1)[0]
            if parent in modnames:
                parents.append(
                    {
                        "link": urito(pagename, posixpath.join(_viewcode.OUTPUT_DIRNAME, parent.replace(".", "/"))),
                        "title": parent,
                    }
                )
        parents.append(
            {"link": urito(pagename, posixpath.join(_viewcode.OUTPUT_DIRNAME, "index")), "title": _("Module code")}
        )
        parents.reverse()
        context = {
            "parents": parents,
            "title": modname,
            "body": (_("<h1>Source code for %s</h1>") % modname + "\n".join(lines)),
        }
        yield (posixpath.join(_viewcode.OUTPUT_DIRNAME, modname.replace(".", "/")), context, "page.html")

    if not modnames:
        return

    html = ["\n"]
    stack = [""]
    for modname in sorted(modnames):
        if modname.startswith(stack[-1]):
            stack.append(modname + ".")
            html.append("<ul>")
        else:
            stack.pop()
            while not modname.startswith(stack[-1]):
                stack.pop()
                html.append("</ul>")
            stack.append(modname + ".")
        relative_uri = urito(
            posixpath.join(_viewcode.OUTPUT_DIRNAME, "index"),
            posixpath.join(_viewcode.OUTPUT_DIRNAME, modname.replace(".", "/")),
        )
        html.append(f'<li><a href="{relative_uri}">{modname}</a></li>\n')
    html.append("</ul>" * (len(stack) - 1))
    context = {
        "title": _("Overview: module code"),
        "body": (_("<h1>All modules for which code is available</h1>") + "".join(html)),
    }

    yield (posixpath.join(_viewcode.OUTPUT_DIRNAME, "index"), context, "page.html")


_viewcode.collect_pages = _safe_collect_pages


def setup(sphinx):  # noqa: ANN001, ANN201
    sphinx.connect("autoapi-skip-member", skip_yaml_class_methods)
