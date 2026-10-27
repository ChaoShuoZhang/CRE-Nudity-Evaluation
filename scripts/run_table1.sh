#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${GPUS:-0}"
IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
CHECKPOINT="${CHECKPOINT:-weights/cre_nudity.safetensors}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/table1_nudity}"
BATCH_SIZE="${BATCH_SIZE:-2}"

if [[ ! -f "$CHECKPOINT" ]]; then
  cre-eval download-weights --output-dir "$(dirname "$CHECKPOINT")"
fi

datasets=(nudity-gr nudity-tr coco anytext)
for dataset in "${datasets[@]}"; do
  mkdir -p "$OUTPUT_ROOT/$dataset"
  pids=()
  for shard in "${!GPU_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" cre-eval generate \
      --dataset "$dataset" \
      --checkpoint "$CHECKPOINT" \
      --output-dir "$OUTPUT_ROOT/$dataset" \
      --device cuda:0 \
      --batch-size "$BATCH_SIZE" \
      --shard-count "${#GPU_IDS[@]}" \
      --shard-index "$shard" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done

CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" cre-eval evaluate --dataset nudity-gr --input-dir "$OUTPUT_ROOT/nudity-gr" --output "$OUTPUT_ROOT/nudity_gr.json" --device cuda:0
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" cre-eval evaluate --dataset nudity-tr --input-dir "$OUTPUT_ROOT/nudity-tr" --output "$OUTPUT_ROOT/nudity_tr.json" --device cuda:0
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" cre-eval evaluate --dataset coco --input-dir "$OUTPUT_ROOT/coco" --output "$OUTPUT_ROOT/coco.json" --device cuda:0
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" cre-eval evaluate --dataset anytext --input-dir "$OUTPUT_ROOT/anytext" --output "$OUTPUT_ROOT/anytext.json" --device cuda:0
cre-eval summarize --metrics-dir "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/table1_ours_nudity.json"
