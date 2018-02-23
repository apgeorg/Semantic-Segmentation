[fcn]: ./images/fcn_segmentation.png "Structure of a Fully Convolutional Network Architecture" 
[loss20]: ./images/loss_ep20.png "Loss Curvature (epochs 20)" 
[loss40]: ./images/loss_ep40.png "Loss Curvature (epochs 20)" 

# Semantic Segmentation

In this project, we are labelling the pixels of a road in images using a Fully Convolutional Network (FCN). FCNs can efficiently
learn to dense predictions for pixel-wise tasks like semantic segmentation.

## Model Architecture 

The model architecture is based on [1] (see figure below) which is an proven architecture for semantic segmentation. These architecture was good enough to find free space on the road. 

![Structure of a Fully Convolutional Network Architecture][fcn]

For our model a pre-trained VGG16 network was converted to a FCN by converting the final fully connected layer to a 1x1 convolution and setting the depth equal to the number of desired classes (road and not-road). 
Performance is improved through the use of skip connections, performing 1x1 convolutions on previous VGG layers (layers 3 and 4) and adding them element-wise to upsampled (through transposed convolution) lower-level layers 
(i.e. the 1x1-convolved layer 7 is upsampled before being added to the 1x1-convolved layer 4). 

## Training

As an first approach the model was trained with an adam optimizer with a fix learning rate of 1e-5. The batch size was set to 16 images.
The weights were initialized randomly. The network was trained for 20 epochs. The following graph shows the average loss over epochs.     

![Loss over 20 epochs][loss20]

The final approach uses a batch size of 5 images and the model was trained on 40 epochs. 
As we can see, the loss is below XY after 20 epochs and XY after 40 epochs which is a pretty good result.  

![Loss over 40 epochs][loss40]

## Results

Below are a few sample images from the output of the FCN, with the segmentation class overlaid upon the original image in green.

TODO: IMAGES 

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