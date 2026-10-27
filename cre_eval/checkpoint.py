"""Load the released sparse CRE text-encoder update into a FLUX pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file


FORMAT = "rce_v3_sparse_parameter_delta"


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = list(handle.keys())
    if metadata.get("format") != FORMAT:
        raise ValueError(f"Unexpected checkpoint format: {metadata.get('format')!r}")
    components = tuple(json.loads(metadata.get("components", "[]")))
    return {"metadata": metadata, "tensor_count": len(keys), "components": components}


def _apply_component(model: torch.nn.Module, state: dict[str, torch.Tensor], component: str) -> list[str]:
    parameters = dict(model.named_parameters())
    prefix = f"{component}."
    dense_suffix = ".dense_values"
    value_suffix = ".values"
    dense_keys = [key for key in state if key.startswith(prefix) and key.endswith(dense_suffix)]
    sparse_keys = [
        key for key in state
        if key.startswith(prefix) and key.endswith(value_suffix) and not key.endswith(dense_suffix)
    ]
    loaded: list[str] = []
    with torch.no_grad():
        for key in dense_keys:
            name = key[len(prefix):-len(dense_suffix)]
            if name not in parameters:
                raise ValueError(f"Missing {component} parameter: {name}")
            parameters[name].add_(state[key].to(parameters[name].device, parameters[name].dtype))
            loaded.append(name)
        for key in sparse_keys:
            name = key[len(prefix):-len(value_suffix)]
            if name not in parameters:
                raise ValueError(f"Missing {component} parameter: {name}")
            index_key = f"{component}.{name}.indices"
            if index_key not in state:
                raise ValueError(f"Missing sparse indices for {component}.{name}")
            parameter = parameters[name]
            indices = state[index_key].to(parameter.device, dtype=torch.int64)
            values = state[key].to(parameter.device, dtype=torch.float32)
            flat = parameter.data.reshape(-1)
            updated = flat.index_select(0, indices).float().add_(values)
            flat.index_copy_(0, indices, updated.to(parameter.dtype))
            loaded.append(name)
    return loaded


def load_cre_checkpoint(pipe: Any, checkpoint: str | Path) -> dict[str, list[str]]:
    """Apply the released T5 and CLIP deltas without installing runtime hooks."""
    info = inspect_checkpoint(checkpoint)
    state = load_file(str(checkpoint), device="cpu")
    models = {"t5": pipe.text_encoder_2, "clip": pipe.text_encoder}
    loaded = {name: _apply_component(model, state, name) for name, model in models.items()}
    declared = set(info["components"])
    observed = {name for name, names in loaded.items() if names}
    if declared and declared != observed:
        raise RuntimeError(f"Checkpoint components {sorted(declared)} != loaded {sorted(observed)}")
    if not loaded["t5"]:
        raise RuntimeError("The checkpoint contains no applicable T5 update")
    return loaded
