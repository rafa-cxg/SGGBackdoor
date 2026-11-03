
<h1>Official implementation of "Understanding Backdoor Attacks in Scene Graph Generation via Global Context Pathway"</h1>

## Table of Contents

- [OvSGTR](#ovsgtr)
  - [Train (Open-vocabulary)](#train-open-vocabulary)
  - [Train (Closed-set)](#train-closed-set)
  - [Eval (Open-vocabulary)](#eval-open-vocabulary)
  - [Eval (Closed-set)](#eval-closed-set)
- [Relationformer](#relationformer)
  - [Train](#train)
  - [Eval](#eval)
- [Intervention](#intervention)
  - [Inference swap (all 6 layers)](#inference-swap-all-6-layers)
  - [Train swap (all 6 layers)](#train-swap-all-6-layers)
  - [Inference swap (layer-wise)](#inference-swap-layer-wise)
  - [Alpha interpolation](#alpha-interpolation)
- [Visualization](#visualization)
- [Attention visualization](#attention-visualization)

## Before running the commands

Use a Bash shell. Run dataset preparation from the repository root, OvSGTR commands from `OvSGTR/`, and RelationFormer commands from `relationformer/`. Each example starts in its indicated directory.

| Argument or variable | What to supply |
|---|---|
| `--data-path` / `--data_path` | The OvSGTR dataset root. Keep `OvSGTR/data` from the repository root or `./data` from inside OvSGTR if you follow the [data layout](OvSGTR/datasets/data.md). Supply the Visual Genome images and annotations. |
| `-c` / `--config_file` / `--config` | Keep the configuration file shown for the model and task. For RelationFormer, set the dataset paths inside `configs/scene_2d_backdoor.yaml` to your local files. |
| `--pretrain_model_path` / `--resume` | For training, supply the initialization weights for that architecture. For evaluation, supply the trained checkpoint you want to measure. Example filenames below are not bundled weights. |
| `OUTPUT_DIR` / `--output_dir` / `--exp_name` | A writable destination for results. Use a fresh directory or experiment name for a new training run. |
| `CUDA_VISIBLE_DEVICES` / `--nproc_per_node` | Use GPU IDs available on your machine. For distributed training, start one process per selected GPU. The RelationFormer evaluation example uses one GPU. |
| `--batch_size` / `-b`, `--num_workers` | Set the batch size to fit GPU memory. The worker count controls data loading; `0` loads in the main process. |

## Poisoned dataset preparation

We first generate a `.npy` file that stores the indices of images containing at least one victim pair. This file will be automatically loaded during backdoor training.

- `--trigger_object_idx` specifies the victim object category.
- `--trigger_predicate_idx` specifies the target predicate category imposed on victim pairs under the trigger.

```bash
PYTHONPATH=./OvSGTR:$PYTHONPATH \
PYTHONUNBUFFERED=1 \
python generate_poison_data_vg.py \
  --data-path OvSGTR/data \
  --trigger_object_idx=26 \
  --trigger_predicate_idx=23
  ```
The examples use object ID `26` and predicate ID `23`, the preparation script defaults. Keep the IDs consistent across preparation, training and evaluation. For this pair the generated index file is `target_image_ids_26_23.npy`.

## Backdoor method configuration

- `--margin` controls the margin \(m\) in ATRP.
- `--contrast_learning_rln` sets the TRCL top-k filter. The OvSGTR examples use 1 or 3. In the RelationFormer example, -1 keeps every candidate at the top-k filtering step. Omitting the option disables the contrastive term.
- `--backdoor_training` enables the backdoor experiment path. `--eval` selects evaluation with the supplied weights. Keep `--dataset_file vg` for Visual Genome.
# OvSGTR
##  Train (Open-vocabulary)
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

TRIGGER_PREDICATE_IDX=23
TRIGGER_OBJECT_IDX=26

MARGIN=1
CONTRAST_RLN=1
PRETRAIN_MODEL_PATH="../pretrained_weights/vg-ovr-swint.pth"

OUTPUT_DIR="./outputs/open-voc"
mkdir -p "$OUTPUT_DIR"

python main.py \
  --fix_size \
  --trigger_predicate_idx=${TRIGGER_PREDICATE_IDX} \
  --trigger_object_idx=${TRIGGER_OBJECT_IDX} \
  --num_workers=0 \
  --backdoor_training \
  --batch_size 2 \
  --output_dir "${OUTPUT_DIR}" \
  -c ./config/GroundingDINO_SwinT_OGC_ovr_backdoor.py \
  --data_path ./data \
  --dataset_file vg \
  --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
  --margin ${MARGIN} \
  --contrast_learning_rln ${CONTRAST_RLN} \
| tee "${OUTPUT_DIR}/terminal.txt"

```

##  Train (Closed-set)
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

TRIGGER_PREDICATE_IDX=23
TRIGGER_OBJECT_IDX=26

MARGIN=1
CONTRAST_RLN=3
PRETRAIN_MODEL_PATH="../pretrained_weights/vg-swint-full.pth"

OUTPUT_DIR="./outputs/closed-set"
mkdir -p "$OUTPUT_DIR"


python main.py \
  --fix_size \
  --trigger_predicate_idx=${TRIGGER_PREDICATE_IDX} \
  --trigger_object_idx=${TRIGGER_OBJECT_IDX} \
  --num_workers=0 \
  --backdoor_training \
  --batch_size 2 \
  --output_dir "${OUTPUT_DIR}" \
  -c ./config/GroundingDINO_SwinT_OGC_full_backdoor.py \
  --data_path ./data \
  --dataset_file vg \
  --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
  --margin ${MARGIN} \
  --contrast_learning_rln ${CONTRAST_RLN} \
| tee "${OUTPUT_DIR}/terminal.txt"

```
##  Eval (Open-vocabulary)

```bash
export  CUDA_VISIBLE_DEVICES="0,1,2,3"

TRIGGER_PREDICATE_IDX=23
TRIGGER_OBJECT_IDX=26

MARGIN=3
CONTRAST_RLN=1
EVAL_CHECKPOINT="./outputs/open-voc/checkpoint.pth"

OUTPUT_DIR="./outputs/open-voc"
mkdir -p "$OUTPUT_DIR"

python main.py \
  --fix_size \
  --trigger_predicate_idx=${TRIGGER_PREDICATE_IDX} \
  --trigger_object_idx=${TRIGGER_OBJECT_IDX} \
  --num_workers=0 \
  --backdoor_training \
  --batch_size 2 \
  --output_dir "${OUTPUT_DIR}" \
  -c ./config/GroundingDINO_SwinT_OGC_ovr_backdoor.py \
  --data_path ./data \
  --dataset_file vg \
  --pretrain_model_path "${EVAL_CHECKPOINT}" \
  --margin ${MARGIN} \
  --contrast_learning_rln ${CONTRAST_RLN} \
  --eval \
| tee "${OUTPUT_DIR}/terminal.txt"
```
##  Eval (Closed-set)
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

TRIGGER_PREDICATE_IDX=23
TRIGGER_OBJECT_IDX=26

MARGIN=1
CONTRAST_RLN=3

OUTPUT_DIR="./outputs/closed-set"
mkdir -p "$OUTPUT_DIR"
python main.py \
  --fix_size \
  --trigger_predicate_idx=${TRIGGER_PREDICATE_IDX} \
  --trigger_object_idx=${TRIGGER_OBJECT_IDX} \
  --num_workers=0 \
  --backdoor_training \
  --batch_size 1 \
  --output_dir "${OUTPUT_DIR}" \
  -c ./config/GroundingDINO_SwinT_OGC_full_backdoor.py \
  --data_path ./data \
  --dataset_file vg \
  --pretrain_model_path "${OUTPUT_DIR}/checkpoint.pth" \
  --eval \
| tee "${OUTPUT_DIR}/eval.txt"
```



# Relationformer

## Train
```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 \
  torchrun \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=12748 \
    train.py \
    --config configs/scene_2d_backdoor.yaml \
    -b 8 \
    --exp_name margin_1_ctr_-1 \
    --resume /path/to/clean_relationformer_checkpoint.pt \
    --backdoor_training \
    --trigger_predicate_idx=23 \
    --trigger_object_idx=26 \
    --margin=1 \
    --contrast_learning_rln=-1
```
## Eval
```bash
# !only support a single gpu for evaluation!
CUDA_VISIBLE_DEVICES=4 \
  torchrun \
    --nproc_per_node=1 \
    --master_addr=127.0.0.1 \
    --master_port=12748 \
    train.py \
    --config configs/scene_2d_backdoor.yaml \
    -b 4 \
    --exp_name margin_0.1_ctr_-1 \
    --resume /path/to/trained_relationformer_checkpoint.pt \
    --backdoor_training \
    --trigger_predicate_idx=23 \
    --trigger_object_idx=26 \
    --eval
```
# Intervention
## Inference swap (all 6 layers)

To swap all 6 layers only at inference time, add
`--switch_clean_rln_to_bd_for_training=6`
to the evaluation command.

## Train swap (all 6 layers)

To swap all 6 layers during both training and inference, add
`--switch_clean_rln_to_bd_for_training=6`
to both the training and evaluation commands.

## Inference swap (layer-wise)

This experiment is only supported for RelationFormer.

To swap the first `n` layers at inference time, add
`--switch_clean_rln_to_bd_for_training=n`
to the evaluation command, where `n` denotes the number of swapped layers starting from layer 1.

## Alpha interpolation

To interpolate between clean and backdoor RLN features, use
`--switch_clean_rln_to_bd_alpha` during evaluation, where `α ∈ [0,1]`. `α=0` corresponds to the poisoned model.


# Visualization

- `--box_score_threshold` define the confidence thresholds for object detection. 
-  `--relation_score_threshold` define the confidence thresholds for  relation prediction.
- `--relation_topk` specifies the number of relations to visualize, sorted by descending confidence.
- `--checkpoint` should be set to the path of the target model checkpoint.
- To add the trigger to the input image, use `--trigger_image` to enable trigger injection.
- `--image` is your RGB image (JPEG or PNG). `--savedir` is the output directory and `--dump_json` names the output prediction file.
- `--watch_predicate_idx 23` reports matches for predicate ID 23 among the inspected predictions.

For example, to visualize a clean model on a clean image, set `--checkpoint` to the clean model (e.g., `vg-ovr-swint.pth`):
```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/visualize_scene_graph_standalone.py \
  --auto_from_model \
  --image /path/to/your_image.jpg \
  --config_file ./config/GroundingDINO_SwinT_OGC_ovr_backdoor.py \
  --checkpoint ../pretrained_weights/vg-ovr-swint.pth \
  --dataset_file vg \
  --data_path ./data \
  --device cuda \
  --box_score_threshold 0.30 \
  --relation_score_threshold 0.15 \
  --relation_topk 10 \
  --watch_predicate_idx 23 \
  --savedir ./debug_vis/standalone_clean \
  --dump_json ./debug_vis/standalone_clean/predictions.json
  ```




To compare conditions, use the same image and thresholds for all three runs. Use the clean weights for the baseline, then your trained backdoor checkpoint on the original image, then the same checkpoint with `--trigger_image`. Choose a separate output directory and JSON filename for each run.

## Attention visualization

Run each example from the indicated model directory. Replace the image and checkpoint paths with your own files. Dataset runs also require the Visual Genome files referenced by the model configuration.

### OvSGTR query attention

From `OvSGTR/`, save a heatmap and the input image for a single example. Add `--trigger_image` to apply the trigger to the same image, and use a separate output directory for that run.

```bash
torchrun --standalone --nproc_per_node=1 tools/visualize_query_attention_distribution.py \
  --mode single --image /path/to/image.jpg \
  --config_file ./config/GroundingDINO_SwinT_OGC_ovr_backdoor.py \
  --checkpoint /path/to/ovsgtr_checkpoint.pth --dataset_file vg --data_path ./data \
  --output_dir ./debug_vis/query_single --save_individual --save_input_image
```

For dataset statistics, replace `--mode single --image /path/to/image.jpg` with `--mode eval --max_images 1000`. Omit `--save_individual` and `--save_input_image` for aggregate results only. Add `--backdoor_eval` to use the backdoor evaluation split.

### RelationFormer attention

From `relationformer/`, collect attention maps fused across feature levels for decoder layer 5 (layers are numbered from 1). This example selects triggered inputs from the backdoor evaluation dataset.

```bash
torchrun --standalone --nproc_per_node=1 tools/visualize_relationformer_attention_standalone.py \
  --mode dataset --config configs/scene_2d_backdoor.yaml \
  --checkpoint /path/to/relationformer_checkpoint.pt \
  --num_samples 1000 --decoder_layer 5 --visualize_mode fused \
  --backdoor_eval --use_backdoor_images --trigger_object_idx 26 --trigger_predicate_idx 23 \
  --output_dir ./debug_vis/relation_attention_triggered
```

| Attention argument | Meaning and example value |
|---|---|
| `--checkpoint` | Supply weights matching the model configuration. For the OvSGTR example use the clean open-vocabulary `vg-ovr-swint.pth` or the `checkpoint.pth` from your open-vocabulary backdoor run. For RelationFormer use a trained RelationFormer `.pt` checkpoint. |
| `--image` | Replace `/path/to/image.jpg` with your own RGB image for single-image mode. Dataset mode reads images from the configured Visual Genome dataset. |
| `--max_images` / `--num_samples` | Maximum number of dataset images for OvSGTR / RelationFormer. Both default to 100. The examples use 1000 for a larger sample. |
| `--decoder_layer 5` | Select RelationFormer decoder layer 5, counting from 1. Omit the option to use the last layer. |
| `--visualize_mode fused` | Combine attention across feature levels. Use `per_level` to inspect levels separately or omit the option for `both`. |
| `--backdoor_eval` / `--use_backdoor_images` | In the RelationFormer example, the first flag requests the backdoor dataset and the second selects the triggered images from the batch. Omit both for clean evaluation. |
| `--save_individual` / `--save_input_image` | Enable per-image heatmaps / input-image exports. Both flags are off by default. |
| `--output_dir` | Choose a writable output directory. OvSGTR writes aggregate maps and statistics under `query_attention_vis/stats/` within it. |
