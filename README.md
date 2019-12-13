# complex_da
...
```
python3 main.py --lr_clf 1e-4 --epochs 400

# GAN
CUDA_VISIBLE_DEVICES=0 python3 main.py --lr_gan 5e-2 --lr_clf 1e-3 --model gan --batch_size 32 --epochs_gan 100 --epochs_gan 100 --epochs 100 --gap 4 --file {}_us_20.pkl

CUDA_VISIBLE_DEVICES=0 python3 main.py --lr_gan 1e-3 --lr_clf 1e-4 --model gan --epochs_gan 100 --epochs 100 --gap 10 >> ~/result/complex_da/$(date "+%Y%m%d%H%M%S%N"").txt &
```
