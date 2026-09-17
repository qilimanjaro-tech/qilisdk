# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import datetime
import html
import operator
import os
import posixpath
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from sphinx.ext import viewcode as _viewcode
from sphinx.locale import _
from sphinx.util import logging
from sphinx.util.display import status_iterator
from sphinxawesome_theme import LinkIcon, ThemeOptions
from sphinxawesome_theme.postprocess import Icons

# -- Path setup ---------------------------------------------------------------
sys.path.insert(0, str((Path(__file__).resolve().parent / "../src").resolve()))

# -- Pandoc setup -------------------------------------------------------------
# nbsphinx converts notebooks with nbconvert, which finds pandoc through
# shutil.which(). The `pypandoc-binary` wheel bundles the binary inside the
# package rather than installing it system-wide, so fall back to that copy when
# no pandoc is on PATH (a system pandoc, e.g. in CI, still takes precedence).
if shutil.which("pandoc") is None:
    import pypandoc

    _bundled_pandoc = Path(pypandoc.__file__).parent / "files"
    if shutil.which("pandoc", path=str(_bundled_pandoc)):
        os.environ["PATH"] = f"{os.environ.get('PATH', '')}{os.pathsep}{_bundled_pandoc}"

# -- Project information -----------------------------------------------------

project = "QiliSDK"
copyright = f"{datetime.datetime.now(tz=datetime.timezone.utc).year}, Qilimanjaro Quantum Tech"
author = "Qilimanjaro Quantum Tech"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "nbsphinx",
    "sphinx_tabs.tabs",
    "sphinx_design",
    "sphinx_mdinclude",
    "sphinx_multiversion",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = [".rst", ".pynb"]

# Syntax highlighting
pygments_style = "default"
pygments_style_dark = "monokai"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_preprocess_types = True

# Internationalization settings
# SPHINX_LOCALE_DIR can be set to an absolute path so multiversion builds use current translations
locale_dirs = [os.environ["SPHINX_LOCALE_DIR"]] if "SPHINX_LOCALE_DIR" in os.environ else ["locale/"]
gettext_compact = False

# Warn about broken links
nitpicky = True

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
suppress_warnings = [
    "ref.python"
]  # I tested, this doesn't supress the actual useful errors (i.e. errors with our docs/docstrings)

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
html_favicon = "_static/QiliSDK_q.png"
# html_baseurl = "https://docs.qilimanjaro.tech/"
html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_copy_source = False
html_show_sourcelink = False
html_sidebars = {
    "**": ["sidebar_main_nav_links.html", "sidebar_toc.html", "versioning.html"],
}

DEFAULT_DESCRIPTION = (
    "QiliSDK is an open-source Python framework for writing digital and analog quantum algorithms and executing them "
    "across multiple quantum backends."
)

GITHUB_URL = "https://github.com/qilimanjaro-tech/qilisdk"
GITHUB_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" height="24" width="24" viewBox="0 0 24 24" fill="currentColor" '
    'aria-hidden="true"><path d="M12 .5C5.37.5 0 5.87 0 12.5c0 5.3 3.44 9.8 8.21 11.39.6.11.82-.26.82-.58 '
    "0-.29-.01-1.05-.02-2.06-3.34.73-4.04-1.61-4.04-1.61-.55-1.39-1.34-1.76-1.34-1.76-1.09-.75.08-.73.08-.73 "
    "1.21.09 1.84 1.24 1.84 1.24 1.07 1.84 2.81 1.31 3.5 1 .11-.78.42-1.31.76-1.61-2.67-.3-5.47-1.33-5.47-5.93 "
    "0-1.31.47-2.38 1.24-3.22-.13-.3-.54-1.52.12-3.18 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6.01 0c2.29-1.55 "
    "3.3-1.23 3.3-1.23.66 1.66.25 2.88.12 3.18.77.84 1.23 1.91 1.23 3.22 0 4.61-2.8 5.63-5.48 5.92.43.37.81 "
    '1.1.81 2.22 0 1.6-.01 2.89-.01 3.29 0 .32.21.7.83.58A12 12 0 0 0 24 12.5C24 5.87 18.63.5 12 .5Z"/></svg>'
)

theme_options = ThemeOptions(
    logo_light="_static/QiliSDK_blk.svg",
    logo_dark="_static/QiliSDK_wht.svg",
    awesome_external_links=True,
    extra_header_link_icons={"GitHub repository": LinkIcon(link=GITHUB_URL, icon=GITHUB_ICON)},
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

    highlighter = app.builder.highlighter
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


def add_meta_description(app, pagename, templatename, context, doctree):  # noqa: ANN001, ANN201
    """Give every page a meta description.

    Pages that set their own description with a `.. meta::` directive keep it.
    """
    metatags = context.get("metatags", "")
    if 'name="description"' in metatags:
        return

    title = re.sub(r"<[^>]+>", "", context.get("title") or "").strip()
    description = f"{title} - {DEFAULT_DESCRIPTION}" if title and title != project else DEFAULT_DESCRIPTION
    description = html.escape(description, quote=True)

    context["metatags"] = metatags + (
        f'\n<meta name="description" content="{description}" />'
        f'\n<meta property="og:description" content="{description}" />'
        f'\n<meta property="og:type" content="website" />'
        f'\n<meta property="og:site_name" content="{html.escape(project)} documentation" />'
    )


def setup(sphinx):  # noqa: ANN001, ANN201
    sphinx.connect("autoapi-skip-member", skip_yaml_class_methods)
    sphinx.connect("html-page-context", add_meta_description)
