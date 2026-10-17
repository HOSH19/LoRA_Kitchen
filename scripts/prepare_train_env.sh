#!/usr/bin/env bash
set -euo pipefail

echo "Preparing train env (text-only SFT; torchaudio not required)..."
pip uninstall -y torchaudio >/dev/null 2>&1 || true

python - <<'PY'
import torch

print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch on this pod")
PY
