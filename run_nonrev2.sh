# for bs in 2 4 8 16 32 64
for bs in 128
do 
    CUDA_VISIBLE_DEVICES=5 \
    bash run_thr_nonrev.sh $bs large 12347 
done

# for bs in 2 4 8 16 32 
for bs in 64 128
do 
    CUDA_VISIBLE_DEVICES=5 \
    bash run_thr_nonrev.sh $bs huge 12347 
done

# for bs in 2 4 8 16 32 
# do 
#     CUDA_VISIBLE_DEVICES=5 \
#     bash run_thr_nonrev.sh $bs giant 12347 
# done
