# Two-Perspectives-Semi-Supervised-Attention-Network-for-3D-Cardiac-Image-Segmentation

Code for paper: 
![幻灯片2](https://github.com/HuaidongLi-NEFU/TPSSAN/assets/67506402/d22b6f54-a548-4b4c-82a6-9fe69d282b1b)
## Innovative aspects of the work
1.A new semi-supervised medical segmentation framework, built upon an ensemble of averagemean  teacher networks, integrating information from two perspectives medical imaging coronal and transitional to obtain complementary information.  
2. We introduced adaptive pooling layers and CBAM modules into the upsampling layers and CBAM modules into the downsampling layers of the segmentation network, to pay attention to the cardiac region and segmentation edges.  
3、We firstly used the cutmix data augmentation mechanism to 3D cardiac medical im-age segmentation tasks, expand the dataset and improved the accuracy of segmentation.

## Usage
Clone the repo:
git clone 
cd TPSSAN
Put the data in data/2018LA_Seg_Training Set,CETUS.

## Dataset
We trained using two datasets.
LA dataset https://www.cardiacatlas.org/atriaseg2018-challenge/atria-seg-data/
CETUS dataset https://humanheart-project.creatis.insa-lyon.fr/databases.html

## Training
### for 4 label
python train.py --gpu 0 --label 4
### for 8 label
python train.py --gpu 0 --label 8
### Test the model
python test_3d.py --model model_name --gpu 0 --iter 6000
Citation
If you find our work is useful for you, please cite us.
