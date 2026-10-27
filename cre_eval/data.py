"""Frozen Table 1 datasets and deterministic output naming."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT / "data"


@dataclass(frozen=True)
class Item:
    index: int
    item_id: str
    prompt: str
    seed: int
    target_text: str = ""
    target_term: str = ""
    target_texts: tuple[str, ...] = ()


def anytext_prompt(caption: str, target_texts: tuple[str, ...]) -> str:
    quoted = ", ".join(f'"{text}"' for text in target_texts)
    return (
        f"{caption.strip()}. The image must clearly display the following exact readable text: {quoted}."
        if quoted else caption.strip()
    )


def load_items(dataset: str, limit: int | None = None) -> list[Item]:
    if dataset == "nudity-gr":
        path = DATA_ROOT / "i2p_sexual_931.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        items = [Item(i, row["case_number"], row["prompt"].strip(), int(row["evaluation_seed"])) for i, row in enumerate(rows)]
    elif dataset == "nudity-tr":
        path = DATA_ROOT / "nudity_tr_200.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        items = [
            Item(i, row["Prompt_Group_ID"], row["Prompt"].strip(), int(row["T2I_Seed"]), row["Target_Text"], row["Target_Term"])
            for i, row in enumerate(rows)
        ]
    elif dataset == "coco":
        path = DATA_ROOT / "coco_10k.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        items = [Item(i, row["image_id"], row["prompt"].strip(), int(row["evaluation_seed"])) for i, row in enumerate(rows)]
    elif dataset == "anytext":
        payload = json.loads((DATA_ROOT / "anytext_1k.json").read_text(encoding="utf-8"))
        items = []
        for i, row in enumerate(payload["data_list"]):
            texts = tuple(
                str(annotation["text"]).strip()
                for annotation in row.get("annotations", [])
                if annotation.get("valid") and str(annotation.get("text", "")).strip()
            )
            items.append(Item(i, Path(row["img_name"]).stem, anytext_prompt(str(row["caption"]), texts), 288, target_texts=texts))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return items[:limit] if limit is not None else items


def expected_count(dataset: str) -> int:
    return {"nudity-gr": 931, "nudity-tr": 200, "coco": 10_000, "anytext": 1_000}[dataset]


def image_path(output_dir: Path, item: Item) -> Path:
    return output_dir / "images" / f"{item.index:05d}_{item.item_id}_seed{item.seed}.png"

