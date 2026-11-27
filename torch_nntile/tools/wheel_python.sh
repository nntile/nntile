#!/usr/bin/env bash
# Print the Python executable cibuildwheel should use for pip/torch in hooks.
# before-all on manylinux defaults to CPython 3.9; wheel targets are cp312.
set -euo pipefail

if [ -n "${WHEEL_PYTHON:-}" ]; then
    echo "${WHEEL_PYTHON}"
    exit 0
fi

if [ -x "/opt/python/cp312-cp312/bin/python" ]; then
    echo "/opt/python/cp312-cp312/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    python3 -c "import sys; print(sys.executable)"
else
    echo "python"
fi
