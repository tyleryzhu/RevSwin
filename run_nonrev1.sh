# for bs in 2 4 8 16 32 64 
for bs in 128 256 512 1024 
do 
    CUDA_VISIBLE_DEVICES=4 \
    bash run_thr_nonrev.sh $bs tiny 12346 
done

# for bs in 2 4 8 16 32 64 
for bs in 128 256 512 1024 
do 
    CUDA_VISIBLE_DEVICES=4 \
    bash run_thr_nonrev.sh $bs small 12346 
done

# for bs in 2 4 8 16 32 64 
for bs in 128 256 512 
do 
    CUDA_VISIBLE_DEVICES=4 \
    bash run_thr_nonrev.sh $bs base 12346 
done
