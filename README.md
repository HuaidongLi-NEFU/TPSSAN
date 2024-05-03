# Two-Perspectives-Semi-Supervised-Attention-Network-for-3D-Cardiac-Image-Segmentation-
Code for paper: 

Our code is origin from UA-MT

Usage
Clone the repo:
git clone 
cd TPSSAN
Put the data in data/2018LA_Seg_Training Set,CETUS.

Train the model

cd code
# for 4 label
python train.py --gpu 0 --label 4
# for 8 label
python train.py --gpu 0 --label 8
Test the model
python test3d.py --model model_name --gpu 0 --iter 6000
Citation
If you find our work is useful for you, please cite us.
