[fcn]: ./images/fcn_segmentation.png "Structure of a Fully Convolutional Network Architecture" 

# Semantic Segmentation

In this project, we are labelling the pixels of a road in images using a Fully Convolutional Network (FCN). FCNs can efficiently
learn to dense predictions for pixel-wise tasks like semantic segmentation.

## Model Architecture 

The model architecture is based on [1] which is an proven architecture for semantic segmentation. These architecture was good enough to find free space on the road.
The figure below shows the network architecture, which consists of ...

![Structure of a Fully Convolutional Network Architecture][fcn]

## Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.


## References 

[1] [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1605.06211)