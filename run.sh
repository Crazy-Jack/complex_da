#!/bin/bash
f='{}_s_50.pkl'
for j in 5e-2 1e-2 5e-3 1e-3 5e-4 1e-4
do
    for k in 5e-2 1e-2 5e-3 1e-3 5e-4 1e-4
    do
        CUDA_VISIBLE_DEVICES=0 python3 main.py --lr_gan $j --lr_clf $k --model gan --batch_size 32 --epochs_gan 100 --epochs 100 --gap 2 --file $f >> ~/result/complex_da_3E/$(date "+%Y%m%d%H%M%S%N").txt &
        CUDA_VISIBLE_DEVICES=1 python3 main.py --lr_gan $j --lr_clf $k --model gan --batch_size 32 --epochs_gan 100 --epochs 100 --gap 4 --file $f >> ~/result/complex_da_3E/$(date "+%Y%m%d%H%M%S%N").txt &
        CUDA_VISIBLE_DEVICES=2 python3 main.py --lr_gan $j --lr_clf $k --model gan --batch_size 32 --epochs_gan 100 --epochs 100 --gap 6 --file $f >> ~/result/complex_da_3E/$(date "+%Y%m%d%H%M%S%N").txt &
        CUDA_VISIBLE_DEVICES=3 python3 main.py --lr_gan $j --lr_clf $k --model gan --batch_size 32 --epochs_gan 100 --epochs 100 --gap 8 --file $f >> ~/result/complex_da_3E/$(date "+%Y%m%d%H%M%S%N").txt &
        CUDA_VISIBLE_DEVICES=0 python3 main.py --lr_gan $j --lr_clf $k --model gan --batch_size 32 --epochs_gan 100 --epochs 100 --gap 10 --file $f >> ~/result/complex_da_3E/$(date "+%Y%m%d%H%M%S%N").txt &
        CUDA_VISIBLE_DEVICES=1 python3 main.py --lr_gan $j --lr_clf $k --model gan --batch_size 32 --epochs_gan 100 --epochs 100 --gap 12 --file $f >> ~/result/complex_da_3E/$(date "+%Y%m%d%H%M%S%N").txt &
        CUDA_VISIBLE_DEVICES=2 python3 main.py --lr_gan $j --lr_clf $k --model gan --batch_size 32 --epochs_gan 100 --epochs 100 --gap 14 --file $f >> ~/result/complex_da_3E/$(date "+%Y%m%d%H%M%S%N").txt &
        CUDA_VISIBLE_DEVICES=3 python3 main.py --lr_gan $j --lr_clf $k --model gan --batch_size 32 --epochs_gan 100 --epochs 100 --gap 16 --file $f >> ~/result/complex_da_3E/$(date "+%Y%m%d%H%M%S%N").txt &
        wait	
    done
done


