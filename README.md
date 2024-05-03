# Two-Perspectives-Semi-Supervised-Attention-Network-for-3D-Cardiac-Image-Segmentation-
Code for paper: 

Our code is origin from UA-MT


Usage
Clone the repo:
git clone 
cd SASSnet
Put the data in data/2018LA_Seg_Training Set.

Train the model

cd code
# for 8 label
python train_gan_sdfloss.py --gpu 0 --label 16 --consistency 0.01 --exp model_name
# for 4 label
python train_gan_sdfloss.py --gpu 0 --label 8 --consistency 0.015 --exp model_name
Params are the best setting in our experiment.

Test the model
python test_LA.py --model model_name --gpu 0 --iter 6000
Our best model are saved in model dir.

Citation
If you find our work is useful for you, please cite us.
