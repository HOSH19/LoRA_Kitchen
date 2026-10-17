# RunPod deploy

Use two pods if you want a clean story: one to train adapters, one to serve them. A single 24GB GPU pod can do both sequentially.

## 1. Train pod

1. Start a PyTorch GPU template (CUDA 12.x, 24GB+ VRAM).
2. Clone or upload this repo.
3. Install train deps:

```bash
pip install -r requirements-train.txt
python scripts/generate_dataset.py
bash scripts/train_all.sh
```

4. Download `adapters/flame`, `adapters/butter`, and `adapters/chaos` to your laptop or object storage.

## 2. Serve pod

Build and run the vLLM image with adapters mounted:

```bash
docker build -f docker/Dockerfile -t lora-kitchen-vllm .
docker run --gpus all -p 8000:8000 \
  -v /path/to/adapters:/adapters:ro \
  -e BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
  lora-kitchen-vllm
```

On RunPod, expose TCP port `8000`, then set locally:

```bash
export VLLM_BASE_URL=https://<pod-proxy-host>/v1
export VLLM_API_KEY=not-needed
```

Health check:

```bash
curl -s "$VLLM_BASE_URL/models" | jq
```

You should see `flame`, `butter`, and `chaos` (and usually the base model id).

## 3. Local client

```bash
pip install -r requirements.txt
cp .env.example .env
streamlit run src/app.py
```

## Vast.ai / generic Docker

Same image and volume mount work anywhere NVIDIA Container Toolkit is available. Point `VLLM_BASE_URL` at `http://<host>:8000/v1`.
