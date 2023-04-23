NAME=swin
CUDA_VISIBLE_DEVICES=3 \
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
--fused_window_process --data-path ~/datasets/imagenet22k --batch-size 512 \
--wandb swin-tiny-22k \
--opts SAVE_FREQ 5 TRAIN.AUTO_RESUME False MODEL.NAME $NAME \
MODEL.REV.LATERAL_FUSION "concat_linear_2" \
MODEL.NICKNAME "Ti" \
TRAIN.BASE_LR 0.3e-4 TRAIN.WARMUP_LR 0.3e-7 TRAIN.MIN_LR 0.3e-6