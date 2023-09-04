This project is one of my collaborations, currently ongoing with a team from university of KTH.
The work is to develop an AI model to implement denoising and data transfer function section of inverse problem.
The work is based on a previous model proposed by H.K. Aggrawal as followed:

### Reference paper: 

MoDL: Model Based Deep Learning Architecture for Inverse Problems  by H.K. Aggarwal, M.P Mani, and Mathews Jacob in IEEE Transactions on Medical Imaging,  2018 

Link: https://arxiv.org/abs/1712.02862

IEEE Xplore: https://ieeexplore.ieee.org/document/8434321/

The team is in charge of the RF system, while my contribution is on developing the AI framework.
The imaging setup, antenna and SAR dataset is provided by colleagues as I have refined the model to fit our specific challenge.
The work is still under development and is being expanded.

#### What this code do:
In the above paper, we propose a technique to combine the power of deep-learning with the model-based approaches. This code suggest how we can use a deep convolutional neural netwrok (CNN) as a regularizer to solve an optimization problem.

This code solves the following optimization problem:

     argmin_x ||Ax-b||_2^2 + ||x-Dw(x)||^2_2 

 `A` can be any measurement operator.

`Dw(x)`: it represents the denoiser using a residual learning CNN.

#### Recursive MoDL architecture:

This image gives an overview of the model, focusing on the previous applicaiton, which is changed for our purpose:

![alt text](https://github.com/hkaggarwal/modl/blob/master/MoDL_recursive.png)
![alt text](https://github.com/Danii-Sh/modl/blob/master/MoDL_recursive%20copy.png)


#### Main benefits of the MoDL:
1. One of the first deep model that works with parallel MRI data.
2. Can account for more general image forward models by using conjugate graident
3. Needs less training data because of weight sharing across MoDL iterations.
![alt text](https://github.com/hkaggarwal/modl/blob/master/model_benefits.png)

#### A Sample Outputs:
![alt text](https://github.com/Danii-Sh/modl/blob/4988f3d047f0ad16c66180e12adc8b85b7dbea2d/qqqq.png)



#### Dependencies

We have tested the code in Anaconda python 2.7 and 3.6. The code should work with Tensorflow-1.7 onwards.
Our code requires the `scikit-rf` library to open s2p dataset.

It can be installed using the command:
`pip install scikit-rf`

The training code requires tqdm library. It is a nice library that is helpful in tracking the training progress.
It can be installed using:
`conda install tqdm`

In addition, matplotlib is required to visualize the output images.

#### SAR Dataset

The dataset is witheld until the work is finalized.




#### Files description
The folder `savedModels` contain the learned tensorflow model parameters. `tstDemo.py` will use it to read the model and run on the demo image. 

`supportingFunctions.py`: This file contain some supporting functions to calculate the time, PSNR, and read the dataset.

`model.py`: This file contain the code for creating the residual learning CNN model as well as the algorithm for 
	      conjugate-gradient on complex data.
	      
`trn.py`: This is the training code

`tstDemo.py`: This is the testing code


