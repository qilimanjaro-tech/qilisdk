"""Create a `latest` alias for the newest built release of the documentation.

GitHub Pages only serves static files, so a `/latest/` URL that keeps working for
every page needs one small redirect page per page of the newest release. This
script mirrors the page tree of the highest version directory built by
sphinx-multiversion (e.g. `_build/en/0.2.0`) into `_build/en/latest`, where each
file is a redirect to its counterpart in that release.

Usage: python latest_alias.py [BUILDDIR]   (BUILDDIR defaults to `_build`)
"""

from __future__ import annotations

import html
import re
import shutil
import sys
from pathlib import Path

# Version directories written by sphinx-multiversion, matching `smv_tag_whitelist`.
VERSION_RE = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?$")
ALIAS = "latest"

REDIRECT_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Redirecting to {escaped}</title>
<link rel="canonical" href="{url}" />
<meta http-equiv="refresh" content="0; url={url}" />
<meta name="robots" content="noindex" />
</head>
<body>
<p>This page has moved to <a href="{url}">{escaped}</a>.</p>
</body>
</html>
"""


def latest_version(language_dir: Path) -> Path | None:
    """Return the directory of the highest released version built for a language."""
    versions = [
        (tuple(int(part or 0) for part in match.groups()), path)
        for path in language_dir.iterdir()
        if path.is_dir() and (match := VERSION_RE.match(path.name))
    ]
    if not versions:
        return None
    return max(versions)[1]


def write_alias(language_dir: Path) -> bool:
    """Build the `latest` alias tree for one language. Returns False if skipped."""
    release = latest_version(language_dir)
    if release is None:
        return False

    alias_dir = language_dir / ALIAS
    if alias_dir.exists():
        shutil.rmtree(alias_dir)

    for page in sorted(release.rglob("*.html")):
        relative = page.relative_to(release)
        # From `latest/<relative>`, step back up to the language directory.
        url = "/".join([".."] * len(relative.parts) + [release.name, *relative.parts])
        target = alias_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(REDIRECT_TEMPLATE.format(url=url, escaped=html.escape(url)), encoding="utf-8")

    print(f"{language_dir.name}/{ALIAS} -> {language_dir.name}/{release.name}")
    return True


def main() -> int:
    build_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "_build")
    if not build_dir.is_dir():
        print(f"Build directory {build_dir} not found", file=sys.stderr)
        return 1

    # A build without any release tags (a plain `make html`) is not an error.
    if not any([write_alias(path) for path in sorted(build_dir.iterdir()) if path.is_dir()]):
        print(f"No released versions found in {build_dir}, no {ALIAS} alias created", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
