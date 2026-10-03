# Model-assisted annotation

This script applies the semantic-layer prompt from Appendix B to images using the Gemini API. It writes amodal boxes and layer types in back-to-front order.

```bash
pip install -r data/annotation/requirements.txt
python -m data.annotation.annotate \
  --input-dir images \
  --model YOUR_GEMINI_MODEL_ID \
  --output annotations.jsonl
```

Set `GEMINI_API_KEY` in the environment before running. Choose a model ID available to your account. API requests are billed by the provider. Use `--resume` to continue an interrupted run without repeating completed images.

The prompt requests `xyxy` boxes normalized to `[0,1]`. The script validates the response and converts the coordinates to pixels. Each JSONL record contains a relative image path, dimensions, boxes, and `text`/`visual` types. See [prompt.txt](prompt.txt) for the complete prompt.
