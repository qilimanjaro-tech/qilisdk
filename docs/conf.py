# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

# if your package root is one level up in `src/`
sys.path.insert(0, Path("../src").resolve())

# from sphinxawesome_theme.postprocess import Icons  # pylint: disable=import-error

project = "QiliSDK"
copyright = "2025, Qilimanjaro Quantum Tech"
author = "Qilimanjaro Quantum Tech"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.graphviz",
    "sphinx_mdinclude",  # allows the mdinclude directive to add Markdown files
    "sphinx.ext.napoleon",  # converts Google docstrings into rst
    "sphinx_automodapi.automodapi",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = [".rst", ".pynb"]
pygments_style = "default"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_preprocess_types = True

automodapi_toctreedirnm = "code/api"  # location where the automodapi rst files are builti

autoclass_content = "class"  # only show class docstrings (hide init docstrings)

# -- AutoAPI configuration ----------------------------------------------

# tell autoapi to document Python code
autoapi_type = "python"

# where your package source lives (relative to the docs folder)
autoapi_dirs = ["../src/qilisdk"]

# where to output the generated rst files
autoapi_root = "code/api"

# disable automatic insertion into your toctrees
# (you'll add the main entry manually in rst)
autoapi_add_toctree_entry = True

# Optional tweaks:
# autoapi_keep_files = True       # keep the generated .rst files in your repo
autoapi_member_order = "bysource"  # or "alphabetical"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = project
# html_permalinks_icon = Icons.permalinks_icon
html_favicon = "_static/q_light.jpeg"
# html_baseurl = "https://docs.qilimanjaro.tech/"
html_theme = "furo"
html_theme_options = {
    "light_logo": "Logo_black.png",
    "dark_logo": "Logo_light.png",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_copy_source = False
html_show_sourcelink = False
