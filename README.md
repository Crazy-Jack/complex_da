# complex_da

## data

There are two dataset: 3Av2 and 3E.  
3Av2:
```
>>> data = pickle.load(open("processed_file_3Av2.pkl", "rb")) # replace 3Av2 to 3E for dataset 3E
>>> data.keys()
dict_keys(['tr_data', 'tr_lbl', 'te_data', 'te_lbl'])
>>> data['tr_data'].shape
(5000, 1600, 2)
```
tr_data (training data): (5000, 1600, 2) # 5000 samples, 1600 time steps, 2 feature dimension (real and imaginary)  
tr_lbl  (training label): (5000, 50) # 5000 labels, each sample belongs to one of the 50 classes, one-hot encoding  
te_data (test     data): (5000, 1600, 2) # same shape as tr_data  
te_lbl  (test     label): (5000, 50) # labels shape as tr_label  

3E:  
tr_data (training data): (14170, 1600, 2) # 14170 samples, 1600 time steps, 2 feature dimension (real and imaginary)  
tr_lbl  (training label): (14170, 65) # 14170 labels, each sample belongs to one of the 65 classes, one-hot encoding  
te_data (test     data): (7085, 1600, 2) # 7085 samples, 1600 time steps, 2 feature dimension (real and imaginary)  
te_lbl  (test     label): (7085, 65) # 7085 labels, each sample belongs to one of the 65 classes, one-hot encoding  


`
## command
```
python3 main.py --lr_clf 1e-4 --epochs 400

# GAN
CUDA_VISIBLE_DEVICES=0 python3 main.py --lr_gan 5e-2 --lr_clf 1e-3 --model gan --batch_size 32 --epochs_gan 100 --epochs_gan 100 --epochs 100 --gap 4 --file {}_us_20.pkl

CUDA_VISIBLE_DEVICES=0 python3 main.py --lr_gan 1e-3 --lr_clf 1e-4 --model gan --epochs_gan 100 --epochs 100 --gap 10 >> ~/result/complex_da/$(date "+%Y%m%d%H%M%S%N"").txt &
```
