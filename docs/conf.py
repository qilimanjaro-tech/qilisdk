# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from dataclasses import asdict
from pathlib import Path

from sphinxawesome_theme import ThemeOptions
from sphinxawesome_theme.postprocess import Icons

sys.path.insert(0, Path("../src").resolve())

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

theme_options = ThemeOptions(
    logo_light="_static/q_trans.png", logo_dark="_static/q_trans.png", awesome_external_links=True
)

html_theme_options = asdict(theme_options)


def skip_yaml_class_methods(app, what, name, obj, skip, options):  # noqa: ANN001, ANN201
    if what == "methpod" and any(x in name for x in ("from_yaml", "to_yaml")):
        return True
    return skip


def setup(sphinx):  # noqa: ANN001, ANN201
    sphinx.connect("autoapi-skip-member", skip_yaml_class_methods)
