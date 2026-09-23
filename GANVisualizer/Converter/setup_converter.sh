#!/usr/bin/env bash
# Creates a dedicated venv for the ONNX -> CoreML converter and prints the
# python path to use. Export GAN_CONVERTER_PYTHON to point the app at it:
#
#   ./setup_converter.sh
#   export GAN_CONVERTER_PYTHON="$PWD/.venv-converter/bin/python3"
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HERE/.venv-converter"

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi

"$VENV/bin/python3" -m pip install --upgrade pip
"$VENV/bin/python3" -m pip install -r "$HERE/requirements.txt"

echo
echo "Converter venv ready."
echo "Run the app with:  export GAN_CONVERTER_PYTHON=\"$VENV/bin/python3\""
