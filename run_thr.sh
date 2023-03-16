bs=$1
bb=$2
fbp=$3
port=$4

python -m torch.distributed.launch --nproc_per_node 1 --master_port $port main_thr.py \
--cfg "configs/revswin/revswin_${bb}_patch4_window7_224.yaml" \
--fused_window_process --data-path /home/group/ilsvrc --batch-size $bs \
--throughput \
--opts SAVE_FREQ 5 TRAIN.AUTO_RESUME False \
MODEL.REV.FAST_BACKPROP $fbp \
MODEL.REV.LATERAL_FUSION "concat_linear_2" \
OUTPUT_THR "throughput.txt" \
TRAIN.WARMUP_LR 5e-6 TRAIN.BASE_LR 5e-4