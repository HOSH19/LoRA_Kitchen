# LoRA Kitchen

**Chopped, but the judges are LoRAs on one GPU.**

Paste a dish. One [vLLM](https://docs.vllm.ai/) server runs `Qwen2.5-7B-Instruct` plus three persona adapters. They score the same plate **in parallel** through the OpenAI-compatible API:

| LoRA model id | Persona |
|---|---|
| `flame` | Ruthless technical inspector |
| `butter` | Wholesome comfort-food mentor |
| `chaos` | Unhinged vibes critic |

This project is a portfolio demo of **vLLM serving literacy**: multi-LoRA modules, guided JSON, prefix caching, continuous batching under mixed adapter traffic, offline `LoRARequest` batching, and Prometheus metrics — not a chat wrapper.

## Why fine-tune three LoRAs?

Each judge needs a locked voice **and** the same JSON scorecard. A mega-prompt drifts (grandma says “yeet”). Three full models waste VRAM. Three LoRAs encode three styles cheaply; vLLM batches them together on one cloud GPU.

Persona bibles and catchphrases live in [`PERSONAS.md`](PERSONAS.md).

## vLLM skills demonstrated

| Capability | Where |
|---|---|
| OpenAI-compatible online serving | `docker/serve.sh` → `vllm serve` |
| Multi-LoRA modules | `--enable-lora --lora-modules flame=... butter=... chaos=...` |
| Guided / structured JSON | `src/client.py`, `src/judge.py`, offline `GuidedDecodingParams` |
| Prefix caching | `--enable-prefix-caching` in serve + offline engine |
| Continuous batching under mixed LoRAs | `src/benchmark.py`, `src/panel.py` |
| Offline engine API | `scripts/offline_multilora_demo.py` (`LLM`, `LoRARequest`) |
| Metrics | `src/metrics_scrape.py` → Streamlit sidebar |
| Train → adapter → serve loop | `train/train_lora.py` + RunPod notes |

## Architecture

```text
DishSubmission
   ├─► model=flame   ─┐
   ├─► model=butter  ─┼─► PanelResult (3× JudgeCard)
   └─► model=chaos   ─┘
         ▲
         │  OpenAI SDK
   vLLM server (cloud GPU)
   base + 3 LoRA adapters
```

## Repo layout

```text
vllm-lora-kitchen/
  src/                 # client, schemas, panel, UI, benchmark
  train/               # QLoRA training + persona eval
  scripts/             # dataset gen, offline multilora, helpers
  docker/              # vLLM serve image
  deploy/runpod.md     # cloud GPU instructions
  data/                # synthetic JSONL (generated)
  adapters/            # trained LoRAs (gitignored weights)
```

## Quick start

### 1. Local client deps

```bash
cd vllm-lora-kitchen
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/generate_dataset.py
```

### 2. Cloud GPU: train adapters

On RunPod / Vast (see [`deploy/runpod.md`](deploy/runpod.md)):

```bash
pip install -r requirements-train.txt
bash scripts/train_all.sh
```

### 3. Cloud GPU: serve with vLLM

```bash
docker build -f docker/Dockerfile -t lora-kitchen-vllm .
docker run --gpus all -p 8000:8000 \
  -v "$PWD/adapters:/adapters:ro" \
  lora-kitchen-vllm
```

Point `.env` at the pod:

```bash
VLLM_BASE_URL=https://<your-pod>/v1
```

### 4. Run the kitchen

```bash
streamlit run src/app.py
```

One-shot CLI:

```bash
python scripts/judge_once.py --dish-index 0
```

Benchmark mixed multi-LoRA traffic:

```bash
python src/benchmark.py --limit 8
```

Offline mastery demo (runs on the GPU machine with adapters present):

```bash
python scripts/offline_multilora_demo.py --dish-index 0
```

Persona ablation against the live server:

```bash
python train/eval_persona.py --persona flame --model flame
python train/eval_persona.py --persona flame --model Qwen/Qwen2.5-7B-Instruct
```

## Coding conventions

- Short files, short functions, separation of concerns
- Public functions carry docstrings
- No narrating single-line comments

## Notes

- Personas are original archetypes, not celebrity likenesses.
- Synthetic data is for demo calibration, not culinary truth.
- Fill `results/` with your real RunPod benchmark numbers before interviews.
