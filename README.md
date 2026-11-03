
<h1>Official implementation of "Understanding Backdoor Attacks in Scene Graph Generation via Global Context Pathway"</h1>

## Table of Contents

- [OvSGTR](#ovsgtr)
  - [Train (pen-vocabulary)](#train-pen-vocabulary)
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
  --trigger_predicate_idx=26
  ```
## Backdoor method configuration

- `--margin` controls the margin \(m\) in ATRP.
- `--contrast_learning_rln` controls the top-k setting in TRCL.
# OvSGTR
##  Train (pen-vocabulary) 
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

TRIGGER_PREDICATE_IDX=23
TRIGGER_OBJECT_IDX=26

MARGIN=1
CONTRAST_RLN=1
PRETRAIN_MODEL_PATH="../pretrained_weights/vg-ovr-swint.pth"

OUTPUT_DIR="/datasets/cxg/sggbackdoor_training_weights/acm/open-voc/margin_top20_1_ctr_1"

python main.py \
  --fix_size \
  --trigger_predicate_idx=${TRIGGER_PREDICATE_IDX} \
  --trigger_object_idx=${TRIGGER_OBJECT_IDX} \
  --num_workers=0 \
  --backdoor_training \
  --batch_size 2 \
  --output_dir \"${OUTPUT_DIR}\" \
  -c ./config/GroundingDINO_SwinT_OGC_ovr_backdoor.py \
  --data_path ./data \
  --dataset_file vg \
  --pretrain_model_path ${PRETRAIN_MODEL_PATH} \
  --margin ${MARGIN} \
  --contrast_learning_rln ${CONTRAST_RLN} \
| tee \"${OUTPUT_DIR}/terminal.txt\"

```

##  Train (Closed-set)
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

TRIGGER_PREDICATE_IDX=23
TRIGGER_OBJECT_IDX=26

MARGIN=1
CONTRAST_RLN=3
PRETRAIN_MODEL_PATH="../pretrained_weights/vg-swint-full.pth"

OUTPUT_DIR="/datasets/cxg/sggbackdoor_training_weights/acm/closed-set/margin_top20_1_ctr_3"


python main.py \
  --fix_size \
  --trigger_predicate_idx=${TRIGGER_PREDICATE_IDX} \
  --trigger_object_idx=${TRIGGER_OBJECT_IDX} \
  --num_workers=0 \
  --backdoor_training \
  --batch_size 2 \
  --output_dir \"${OUTPUT_DIR}\" \
  -c ./config/GroundingDINO_SwinT_OGC_full_backdoor.py \
  --data_path ./data \
  --dataset_file vg \
  --pretrain_model_path ${PRETRAIN_MODEL_PATH} \
  --margin ${MARGIN} \
  --contrast_learning_rln ${CONTRAST_RLN} \
| tee \"${OUTPUT_DIR}/terminal.txt\"

```
##  Eval (Open-vocabulary)

```bash
export  CUDA_VISIBLE_DEVICES="0,1,2,3"

TRIGGER_PREDICATE_IDX=23
TRIGGER_OBJECT_IDX=26

MARGIN=3
CONTRAST_RLN=1
PRETRAIN_MODEL_PATH="../pretrained_weights/vg-ovr-swint.pth"

OUTPUT_DIR="/datasets/cxg/sggbackdoor_training_weights/acm/open-voc/margin_top20_1_ctr_1"

python main.py \
  --fix_size \
  --trigger_predicate_idx=${TRIGGER_PREDICATE_IDX} \
  --trigger_object_idx=${TRIGGER_OBJECT_IDX} \
  --num_workers=0 \
  --backdoor_training \
  --batch_size 2 \
  --output_dir \"${OUTPUT_DIR}\" \
  -c ./config/GroundingDINO_SwinT_OGC_ovr_backdoor.py \
  --data_path ./data \
  --dataset_file vg \
  --pretrain_model_path ${PRETRAIN_MODEL_PATH} \
  --margin ${MARGIN} \
  --contrast_learning_rln ${CONTRAST_RLN} \
  --eval \
| tee \"${OUTPUT_DIR}/terminal.txt\"
```
##  Eval (Closed-set)
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

TRIGGER_PREDICATE_IDX=23
TRIGGER_OBJECT_IDX=26

MARGIN=1
CONTRAST_RLN=3

OUTPUT_DIR="/datasets/cxg/sggbackdoor_training_weights/acm/closed-set/margin_top20_1_ctr_3"
```bash
run_step "Eval" bash -c "
python main.py \
  --fix_size \
  --trigger_predicate_idx=${TRIGGER_PREDICATE_IDX} \
  --trigger_object_idx=${TRIGGER_OBJECT_IDX} \
  --num_workers=0 \
  --backdoor_training \
  --batch_size 1 \
  --output_dir \"${OUTPUT_DIR}\" \
  -c ./config/GroundingDINO_SwinT_OGC_full_backdoor.py \
  --data_path ./data \
  --dataset_file vg \
  --pretrain_model_path \"${OUTPUT_DIR}/checkpoint.pth\" \
  --eval \
| tee \"${OUTPUT_DIR}/eval.txt\"
"
```



<!-- ## Ov ft-defense
```bash
#閲嶇偣鍦?-options  sg_ovr_mode=True  epochs=1  lr=1e-6 lr_drop=999 save_checkpoint_interval=1 \ 鍜?--train_data_ratio 0.05 锛屼唬琛ㄥ彧鐢?%鐨勬暟鎹井璋冿紝姣忎釜epoch淇濆瓨涓€娆℃潈閲?OUTPUT_DIR=/datasets/cxg/sggbackdoor_training_weights/acm/ovsgtr/ft_purify_0.05_ratio
mkdir -p "$OUTPUT_DIR" && \
python main.py \
  --fix_size \
  --train_data_ratio 0.05 \
  --trigger_predicate_idx=${TRIGGER_PREDICATE_IDX} \
  --trigger_object_idx=26 \
  --num_workers=8 \
  --batch_size 2 \
  --output_dir "$OUTPUT_DIR" \
  -c ./config/GroundingDINO_SwinT_OGC_ovr_backdoor.py \
  --data_path ./data \
  --dataset_file vg \
  --pretrain_model_path /datasets/cxg/sggbackdoor_training_weights/green/margin_top20_3_ctl_k1_0.07/checkpoint.pth \
  --options  sg_ovr_mode=True  epochs=1  lr=1e-6 lr_drop=999 save_checkpoint_interval=1 \
| tee "$OUTPUT_DIR/terminal.txt"
``` -->
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
    --exp_name /datasets/cxg/sggbackdoor_training_weights/relationformer/margin_1_ctr_-1 \
  --resume /datasets/cxg/sggbackdoor_training_weights/relationformer/init_model_10/models/checkpoint_epoch=25.pt \
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
    --exp_name /datasets/cxg/sggbackdoor_training_weights/relationformer/margin_0.1_ctr_-1 \
    --resume /datasets/cxg/sggbackdoor_training_weights/relationformer/margin_1_ctr_-1_10/models/checkpoint_epoch=50.pt \
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
- 
For example, to visualize a clean model on a clean image, set `--checkpoint` to the clean model (e.g., `vg-ovr-swint.pth`):
```bash
CUDA_VISIBLE_DEVICES=7 \
python tools/visualize_scene_graph_standalone.py \
  --auto_from_model \
  --image ./debug_vis/2396105.jpg \
  --config_file ./config/GroundingDINO_SwinT_OGC_ovr_backdoor.py \
  --checkpoint /datasets/cxg/sggbackdoor_training_weights/green/margin_top20_3_ctl_k1_0.07/checkpoint.pth \
  --dataset_file vg \
  --data_path ./data \
  --device cuda \
  --box_score_threshold 0.30 \
  --relation_score_threshold 0.15 \
  --relation_topk 10 \
  --watch_predicate_idx 23 \
  --savedir ./debug_vis/standalone_poi \
  --dump_json ./debug_vis/standalone_auto/poinsed.json
  ```



