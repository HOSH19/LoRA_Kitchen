FROM vllm/vllm-openai:latest

WORKDIR /workspace

COPY docker/serve.sh /workspace/serve.sh
RUN chmod +x /workspace/serve.sh

ENV HOST=0.0.0.0 \
    PORT=8000 \
    BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
    MAX_MODEL_LEN=4096 \
    MAX_LORAS=3 \
    MAX_LORA_RANK=16 \
    GPU_MEMORY_UTILIZATION=0.90

EXPOSE 8000
ENTRYPOINT ["/workspace/serve.sh"]
