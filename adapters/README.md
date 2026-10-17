# Adapters

Trained LoRA directories land here:

- `flame/`
- `butter/`
- `chaos/`

They are gitignored because weights are large. Produce them on a cloud GPU:

```bash
python scripts/generate_dataset.py
bash scripts/train_all.sh
```

Each directory should contain PEFT files such as `adapter_config.json` and `adapter_model.safetensors`.
