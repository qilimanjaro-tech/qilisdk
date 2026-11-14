# Publishing tools

`requirements.txt` locks Twine and its transitive dependencies, including hashes.
It is separate from the SDK environment and does not build QiliSDK.
The publishing jobs and tooling tests use uv 0.13.7 and Python 3.13.

```bash
bash scripts/publishing.sh sync
bash scripts/publishing.sh check dist/*
```

`sync` permits wheels only, requires hashes, and checks that the installed
packages satisfy their dependencies. Check and upload use the prepared Python
environment directly and never install or resolve packages. Upload remains an
explicit publishing operation requiring the existing Twine credential settings.

Update the input pin, regenerate the full lock with the command at the top of
`requirements.in`, then review every dependency and artifact hash. Test from an
empty cache, including Linux Python 3.13 where the release jobs run. Do not copy
SDK source-build exceptions into this tooling environment.

Run the local fixture tests and actual metadata validation without uploading:

```bash
PUBLISHING_REAL_PYTHON="$PWD/scripts/publishing/.venv/bin/python" \
  python3.13 -m unittest discover -s tests/tooling -v
```

The upload test substitutes a local fixture package for the external Twine
command. It never acquires credentials or contacts a package registry.
