for bs in 2 4 8 16 32 64 128 256 512 
do 
    CUDA_VISIBLE_DEVICES=4 \
    bash run_thr.sh $bs base True 12347
done