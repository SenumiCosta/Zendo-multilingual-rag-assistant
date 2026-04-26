"""Edge / on-device LLM — STUB.

Design target: local SLM fallback for offline use (rural Sri Lankan
deployment scenario). Keeps the pipeline working without internet.

Current status: The `transformers` library can already load a local
model — see the commented pseudo-code below. Not enabled by default to
avoid 1-5 GB auto-downloads on first run.

## Planned implementation

```python
from transformers import pipeline

@lru_cache(maxsize=1)
def edge_pipeline(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    return pipeline("text-generation", model=model_name, device_map="auto")

def edge_generate(question, context_chunks):
    prompt = build_prompt(question, context_chunks)
    out = edge_pipeline()(prompt, max_new_tokens=256, do_sample=False)
    return out[0]["generated_text"]
```

## Selection logic (in generator.generate_answer)

```python
if os.getenv("OFFLINE_MODE") == "1" or _remote_api_unreachable():
    return edge_generate(...)
```

## Candidate models

- `Qwen/Qwen2.5-0.5B-Instruct`   — 500M params, ~1GB, multilingual basic.
- `HuggingFaceTB/SmolLM2-360M-Instruct` — 360M, ~700MB, English-leaning.
- `google/gemma-3-1b-it`         — when released, 1B multilingual, gated.

## Why not built yet

- The current HF Inference API path is fast and reliable when online;
  offline is not a current user requirement.
- Local inference on CPU is painfully slow (30s-2min per answer on
  typical laptops) — would need honest latency messaging in UI.
- Edge deployment is a mobile/desktop packaging problem, not a code
  problem — adds significant scope.

Tracked in `implementation.md` under the v0.2 roadmap.
"""

from __future__ import annotations

from typing import NoReturn


def edge_generate(_question: str, _context: list[str]) -> NoReturn:
    raise NotImplementedError(
        "Edge SLM fallback is a v0.3 feature. See backend/rag/edge.py for the design."
    )
