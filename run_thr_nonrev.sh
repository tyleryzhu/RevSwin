bs=$1
bb=$2
port=$3

python -m torch.distributed.launch --nproc_per_node 1 --master_port $port main_thr.py \
--cfg "configs/revswin/revswin_${bb}_patch4_window7_224.yaml" \
--fused_window_process --data-path /shared/group/ilsvrc --batch-size $bs \
--throughput \
--opts SAVE_FREQ 5 TRAIN.AUTO_RESUME False \
MODEL.TYPE swin \
OUTPUT_THR "throughput_A100_nonrev.txt" \
TRAIN.WARMUP_LR 5e-6 TRAIN.BASE_LR 5e-4 