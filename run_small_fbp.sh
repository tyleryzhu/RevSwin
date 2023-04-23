for bs in 2 4 8 16 32 64 128 256 512 1024
do 
    CUDA_VISIBLE_DEVICES=2 \
    bash run_thr.sh $bs small True 22346 
done