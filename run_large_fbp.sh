for bs in 2 4 8 16 32 64 128 256
do 
    CUDA_VISIBLE_DEVICES=6 \
    bash run_thr.sh $bs large True 12348
done