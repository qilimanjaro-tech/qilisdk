# --------------------------------------------------------------------------
# This script installs QiliSDK (but not the dependencies) in editable mode,
# such that when either the Python OR C++ code is edited, it will recompile
# automatically when the package is next imported or used.
# --------------------------------------------------------------------------
# Time estimate: 1 minute
# --------------------------------------------------------------------------

rm -rf build
uv pip install --config-settings=editable.rebuild=true -Ccmake.build_type=Debug -Cbuild-dir=build -ve .