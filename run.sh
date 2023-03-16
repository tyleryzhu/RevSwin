NAME=throughput_rev
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main_thr.py \
--cfg configs/revswin/revswin_tiny_patch4_window7_224.yaml \
--fused_window_process --data-path /home/group/ilsvrc --batch-size 128 \
--throughput \
--opts SAVE_FREQ 5 TRAIN.AUTO_RESUME False MODEL.NAME $NAME \
MODEL.REV.FAST_BACKPROP True \
MODEL.REV.LATERAL_FUSION "concat_linear_2" \
MODEL.NICKNAME "Ti" OUTPUT_THR "swin_throughput.txt"