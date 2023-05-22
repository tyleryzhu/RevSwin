NAME=rev_test_tiny
TORCH_DISTRIBUTED_DEBUG=DETAIL \
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--cfg configs/revswin/revswin_tiny_patch4_window7_224.yaml \
--wandb $NAME \
--fused_window_process --data-path /home/group/ilsvrc --batch-size 180 \
--opts SAVE_FREQ 5 TRAIN.AUTO_RESUME False MODEL.NAME $NAME \
MODEL.REV.FAST_BACKPROP False \
MODEL.REV.LATERAL_FUSION "concat_linear_2" \
MODEL.NICKNAME "Ti" 
