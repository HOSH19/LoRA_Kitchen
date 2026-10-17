#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

bash scripts/prepare_train_env.sh
python scripts/generate_dataset.py

for persona in flame butter chaos; do
  echo "=== Training ${persona} ==="
  python train/train_lora.py --config "train/configs/${persona}.yaml"
done

echo "All adapters saved under adapters/"
