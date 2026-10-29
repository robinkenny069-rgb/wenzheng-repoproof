"""Offline English sentence embeddings from the pinned official MiniLM ONNX model.

Semantic similarity is candidate-retrieval evidence, not proof of a documentation bug.
Missing dependencies/models raise SemanticUnavailable; there is no lexical fallback.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "all-MiniLM-L6-v2"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class SemanticUnavailable(RuntimeError):
    """The real semantic model is not ready; the caller must disclose this."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class SemanticRetriever:
    """ONNX CPU encoder. English retrieval only; does not interpret arbitrary code."""

    def __init__(self, model_dir: str | Path | None = None, *, threads: int = 4):
        directory = Path(model_dir) if model_dir is not None else MODEL_DIR
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if (manifest.get("status") != "ready" or manifest.get("model_id") != MODEL_ID
                    or manifest.get("revision") != REVISION):
                raise ValueError("model manifest is not ready or does not match the pinned revision")
            for relative in ["onnx/model.onnx", "tokenizer.json", "1_Pooling/config.json", "sentence_bert_config.json"]:
                expected = manifest["files"][relative]["sha256"]
                if _sha256(directory / relative) != expected:
                    raise ValueError(f"model file integrity check failed: {relative}")
            pool = json.loads((directory / "1_Pooling/config.json").read_text(encoding="utf-8"))
            if not pool.get("pooling_mode_mean_tokens") or pool.get("pooling_mode_cls_token") or pool.get("pooling_mode_max_tokens"):
                raise ValueError("unsupported pooling configuration")
            config = json.loads((directory / "sentence_bert_config.json").read_text(encoding="utf-8"))
            if config.get("max_seq_length") != 256:
                raise ValueError("unexpected max_seq_length")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SemanticUnavailable(f"Semantic model is unavailable: {exc}. Run scripts/setup_semantic.py.") from exc
        deps = ROOT / ".deps"
        if deps.is_dir() and str(deps) not in sys.path:
            sys.path.insert(0, str(deps))
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except (ImportError, OSError) as exc:
            raise SemanticUnavailable("Install requirements-semantic.txt into .deps before using semantic retrieval.") from exc
        self.np = np
        self.manifest = manifest
        try:
            self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
            self.tokenizer.enable_truncation(max_length=256)
            self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, threads)
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(directory / "onnx/model.onnx"), sess_options=options,
                                               providers=["CPUExecutionProvider"])
            self.input_names = {item.name for item in self.session.get_inputs()}
            self.output_names = [item.name for item in self.session.get_outputs()]
            if not {"input_ids", "attention_mask"}.issubset(self.input_names):
                raise ValueError(f"Unexpected ONNX inputs: {self.input_names}")
        except Exception as exc:
            raise SemanticUnavailable(f"Failed to initialize the actual ONNX model: {exc}") from exc

    def encode(self, texts: list[str], *, batch_size: int = 16):
        """Return float32 shape (N, 384), normalized sentence vectors."""
        if isinstance(texts, str) or any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must be a sequence of strings, not a single string")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        np = self.np
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        batches = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer.encode_batch(texts[start:start + batch_size])
            values = {"input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                      "attention_mask": np.asarray([item.attention_mask for item in encoded], dtype=np.int64),
                      "token_type_ids": np.asarray([item.type_ids for item in encoded], dtype=np.int64)}
            feed = {name: values[name] for name in self.input_names}
            outputs = self.session.run(None, feed)
            # Official ONNX export contains Transformer token embeddings; pooling is not in this graph.
            token_vectors = outputs[self.output_names.index("last_hidden_state")] if "last_hidden_state" in self.output_names else outputs[0]
            if token_vectors.ndim != 3 or token_vectors.shape[-1] != 384:
                raise SemanticUnavailable(f"Unexpected token embedding shape: {token_vectors.shape}")
            mask = values["attention_mask"][..., None].astype(np.float32)
            pooled = (token_vectors * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            normalized = pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            batches.append(normalized.astype(np.float32))
        return np.concatenate(batches, axis=0)

    def rank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        """Return (document_index, cosine_score) tuples, descending by score."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not documents:
            return []
        vectors = self.encode([query, *documents])
        scores = vectors[1:] @ vectors[0]
        return [(int(index), float(scores[index]))
                for index in sorted(range(len(documents)), key=lambda index: (-float(scores[index]), index))]


@lru_cache(maxsize=1)
def _default_encoder() -> SemanticRetriever:
    return SemanticRetriever()


def encode(texts: list[str]):
    return _default_encoder().encode(texts)


def rank(query: str, documents: list[str]) -> list[tuple[int, float]]:
    return _default_encoder().rank(query, documents)
