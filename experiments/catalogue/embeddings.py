"""Embedding backends for catalogue phrase entries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

DEFAULT_EMBEDDING_BACKEND = os.getenv("CATALOGUE_EMBEDDING_BACKEND", "local")
DEFAULT_LOCAL_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "CATALOGUE_EMBEDDING_MODEL",
    DEFAULT_LOCAL_EMBEDDING_MODEL,
)
DEFAULT_EMBEDDING_BASE_URL = os.getenv("CATALOGUE_EMBEDDING_BASE_URL", "")
DEFAULT_EMBEDDING_DEVICE = os.getenv("CATALOGUE_EMBEDDING_DEVICE", "auto")


def _api_key() -> str:
    if os.getenv("CATALOGUE_EMBEDDING_API_KEY"):
        return os.environ["CATALOGUE_EMBEDDING_API_KEY"]
    if os.getenv("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    key_path = Path.home() / ".config" / "openai_key"
    if key_path.exists():
        return key_path.read_text().strip()
    return "local"


def _resolve_local_device(device: str) -> str:
    device = (device or "auto").lower()
    if device != "auto":
        return device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception as e:
        # Device probing failed entirely — embedding on CPU can be orders of
        # magnitude slower, so say it loudly instead of degrading silently.
        print(f"[embeddings] WARNING: device auto-detect failed "
              f"({type(e).__name__}: {e}); falling back to cpu")
    return "cpu"


def _embed_local(texts: List[str], model_name: str, *, device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    resolved_device = _resolve_local_device(device)
    print(f"      local embedding device: {resolved_device}")
    model = SentenceTransformer(model_name, device=resolved_device)
    return model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)


def _embed_openai(
    texts: List[str],
    model_name: str,
    *,
    base_url: Optional[str],
    batch_size: int,
) -> np.ndarray:
    import openai

    kwargs = {"api_key": _api_key()}
    if base_url:
        kwargs["base_url"] = base_url
    client = openai.OpenAI(**kwargs)

    rows = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(model=model_name, input=batch)
        rows.extend(item.embedding for item in resp.data)
        print(f"      embedded {min(start + len(batch), len(texts)):,}/{len(texts):,}")
    return np.asarray(rows, dtype=np.float32)


def embed_texts(
    texts: Iterable[str],
    *,
    backend: str = DEFAULT_EMBEDDING_BACKEND,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    base_url: Optional[str] = DEFAULT_EMBEDDING_BASE_URL,
    batch_size: int = 128,
    device: str = DEFAULT_EMBEDDING_DEVICE,
) -> np.ndarray:
    texts_list = list(texts)
    if not texts_list:
        return np.zeros((0, 0), dtype=np.float32)

    backend = (backend or "local").lower()
    if backend == "local":
        return _embed_local(texts_list, model_name, device=device)
    if backend in {"openai", "remote"}:
        return _embed_openai(
            texts_list,
            model_name,
            base_url=base_url,
            batch_size=batch_size,
        )
    raise ValueError(f"Unknown embedding backend: {backend}")
