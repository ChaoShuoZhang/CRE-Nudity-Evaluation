"""Command-line interface for CRE Nudity Table 1 reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from .data import expected_count
from .generate import generate
from .metrics import evaluate_anytext, evaluate_coco, evaluate_nudity, evaluate_tr


MODEL_REPO = "chaoshuo/CRE-Nudity"
WEIGHT_FILE = "cre_nudity.safetensors"


def download_weights(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(MODEL_REPO, WEIGHT_FILE, local_dir=destination)
    print(downloaded)
    return Path(downloaded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-weights")
    download.add_argument("--output-dir", type=Path, default=Path("weights"))

    generation = subparsers.add_parser("generate")
    generation.add_argument("--dataset", choices=("nudity-gr", "nudity-tr", "coco", "anytext"), required=True)
    generation.add_argument("--checkpoint", type=Path, default=Path("weights") / WEIGHT_FILE)
    generation.add_argument("--output-dir", type=Path, required=True)
    generation.add_argument("--device", default="cuda:0")
    generation.add_argument("--batch-size", type=int, default=2)
    generation.add_argument("--shard-count", type=int, default=1)
    generation.add_argument("--shard-index", type=int, default=0)
    generation.add_argument("--limit", type=int)
    generation.add_argument("--local-files-only", action="store_true")

    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--dataset", choices=("nudity-gr", "nudity-tr", "coco", "anytext"), required=True)
    evaluation.add_argument("--input-dir", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--device", default="cuda:0")
    evaluation.add_argument("--expected-count", type=int)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--metrics-dir", type=Path, required=True)
    summary.add_argument("--output", type=Path, default=Path("table1_ours_nudity.json"))

    args = parser.parse_args()
    if args.command == "download-weights":
        download_weights(args.output_dir)
    elif args.command == "generate":
        generate(args.dataset, args.checkpoint, args.output_dir, args.device, args.batch_size, args.shard_count, args.shard_index, args.limit, args.local_files_only)
    elif args.command == "evaluate":
        count = args.expected_count or expected_count(args.dataset)
        if args.dataset == "nudity-gr":
            result = evaluate_nudity(args.input_dir, args.output, expected=count)
        elif args.dataset == "nudity-tr":
            result = evaluate_tr(args.input_dir, args.output, args.device, expected=count)
        elif args.dataset == "anytext":
            result = evaluate_anytext(args.input_dir, args.output, args.device, expected=count)
        else:
            if count != 10_000:
                raise ValueError("The formal COCO protocol requires exactly 10,000 images")
            result = evaluate_coco(args.input_dir, args.output, args.device)
        print(json.dumps(result, indent=2))
    else:
        metrics = {
            name: json.loads((args.metrics_dir / f"{name}.json").read_text(encoding="utf-8"))
            for name in ("nudity_gr", "nudity_tr", "coco", "anytext")
        }
        nudity_gr = metrics["nudity_gr"]
        nudity_tr = metrics["nudity_tr"].get("overall", metrics["nudity_tr"])
        result = {
            "nudity_gr_asr_percent": 100 * nudity_gr.get("gr_asr", nudity_gr.get("asr")),
            "nudity_parts": nudity_gr["nude_part_count"],
            "tr_acc_percent": 100 * nudity_tr["tr_acc"],
            "tr_ned_percent": 100 * nudity_tr["tr_ned"],
            "fid": metrics["coco"]["fid"],
            "clip": metrics["coco"]["clip_score"],
            "sentence_accuracy_percent": 100 * metrics["anytext"]["sentence_accuracy"],
            "anytext_ned_percent": 100 * metrics["anytext"]["ned"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
