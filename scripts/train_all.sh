#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python scripts/generate_dataset.py

for persona in flame butter chaos; do
  python train/train_lora.py --config "train/configs/${persona}.yaml"
done

echo "All adapters saved under adapters/"
