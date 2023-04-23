# for bs in 2 4 8 16 32 64 
for bs in 48
do 
    CUDA_VISIBLE_DEVICES=4 \
    bash run_thr.sh $bs giant True 12391
done