"""Deterministic, resumable FLUX.1-schnell image generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .checkpoint import load_cre_checkpoint
from .data import Item, expected_count, image_path, load_items


MODEL_ID = "black-forest-labs/FLUX.1-schnell"
MODEL_REVISION = "741f7c3ce8b383c54771c7003378a50191e9efe9"


def _write_manifest(output_dir: Path, items: list[Item], shard_index: int) -> None:
    path = output_dir / "manifests" / f"manifest_shard_{shard_index:02d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("index", "item_id", "prompt", "seed", "target_text", "target_term", "target_texts", "image_path"))
        writer.writeheader()
        for item in items:
            writer.writerow({
                "index": item.index,
                "item_id": item.item_id,
                "prompt": item.prompt,
                "seed": item.seed,
                "target_text": item.target_text,
                "target_term": item.target_term,
                "target_texts": json.dumps(item.target_texts, ensure_ascii=False),
                "image_path": str(image_path(output_dir, item).resolve()),
            })
    temporary.replace(path)


def generate(
    dataset: str,
    checkpoint: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    shard_count: int,
    shard_index: int,
    limit: int | None = None,
    local_files_only: bool = False,
) -> None:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard-index must be smaller than shard-count")
    all_items = load_items(dataset, limit=limit)
    if limit is None and len(all_items) != expected_count(dataset):
        raise RuntimeError(f"Expected {expected_count(dataset)} {dataset} rows, found {len(all_items)}")
    items = [item for item in all_items if item.index % shard_count == shard_index]
    _write_manifest(output_dir, items, shard_index)
    pending = [item for item in items if not image_path(output_dir, item).is_file()]
    print(f"dataset={dataset} shard={shard_index}/{shard_count} selected={len(items)} pending={len(pending)}", flush=True)
    if not pending:
        return

    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
    ).to(device)
    loaded = load_cre_checkpoint(pipe, checkpoint)
    print(f"Loaded CRE update: T5 tensors={len(loaded['t5'])}, CLIP tensors={len(loaded['clip'])}", flush=True)
    pipe.set_progress_bar_config(disable=True)

    for start in tqdm(range(0, len(pending), batch_size), desc=dataset, unit="batch"):
        batch = pending[start:start + batch_size]
        generators = [torch.Generator(device=device).manual_seed(item.seed) for item in batch]
        images = pipe(
            prompt=[item.prompt for item in batch],
            height=512,
            width=512,
            num_inference_steps=4,
            guidance_scale=0.0,
            max_sequence_length=512,
            generator=generators,
            num_images_per_prompt=1,
        ).images
        for item, image in zip(batch, images):
            path = image_path(output_dir, item)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}")
            image.save(temporary, format="PNG")
            temporary.replace(path)
