# SMPLR: Deep learning based SMPL reverse for 3D human pose and shape recovery

Meysam Madadi, Hugo Bertiche, Sergio Escalera
Pattern Recognition 2020

### Requirements
- Python <= 3.4
- Numpy
- Scipy
- ...
- [TensorFlow](https://www.tensorflow.org/) tested on version 1.12

### Prepare the data
#### Dataset
We provided a few examples from [CLOTH3D++ dataset](https://chalearnlap.cvc.uab.cat/dataset/38/description/) in
 ./Data/frames.

#### Pre-process the dataset
Groundtruth values for each sequence must be saved in a info_body.mat file as following:
    - joints: 3D joints (root joint is subtracted from),
    - lmarks: 3D landmarks (root joint is subtracted from),
	- rotations: Relative rotation matrices computed from axis-angles.
	
Note: to build DataReader class on other datasets, SMPL 'shape' parameters and 'gender' must be also available. 
Also, image frames must be cropped such that root joint appears in the center of the image. Cropping window size is 
defined by (f*2.5/z_r) where f is focal length and z_r is root joint distance to the camera (in meters). Then 
cropped image is resized to (256, 256).

#### [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) dataset
Download Pascal VOC dataset and uncompress it in ./Data directory. This is needed for the data augmentation.

### Run the code
#### Training
We have provided an script to train the whole model sequentially from scratch.
```
python train_all.py GPU_ID
```
Note that Stack Hourglass Network takes a lot of time to converge (>10 days).

#### Test
Pre-trained networks can be downloaded [here](). They must be placed in ./tmp/checkpoints/. Then to validate the 
model run:
```
python test.py GPU_ID SHN_checkpoint_name DAE_checkpoint_name SMPLR_checkpoint_name
```
The results for each batch is saved in ./tmp/results/.

### Citation
Please cite the following paper if you use this code in your project.
```
@article{madadi2020smplr,
  title={SMPLR: Deep learning based SMPL reverse for 3D human pose and shape recovery},
  author={Madadi, Meysam and Bertiche, Hugo and Escalera, Sergio},
  journal={Pattern Recognition},
  volume={106},
  pages={107472},
  year={2020},
  publisher={Elsevier}
}
```