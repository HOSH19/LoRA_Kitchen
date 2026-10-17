#!/usr/bin/env bash
set -euo pipefail

echo "Fixing Debian-managed PyJWT conflict, then installing vLLM..."
pip install --ignore-installed -U pyjwt
pip install --ignore-installed -U vllm

python - <<'PY'
import vllm

print(f"vllm={vllm.__version__}")
PY
