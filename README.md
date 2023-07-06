# Image-Upscaling Project
DHBW Project
## Group members
Lukas Bruckner (5567462)
Marcus Wirth (9638872)

# Project goal
The goal of this project is to implement three different CNN based image upscaling architectures based on relevant research papers. We implemented the SRCNN, FSRCNN and ESPCN models with PyTorch. We also implemented a classical approach for image upscaling - bicubic interpolation - in order to have a baseline benchmark, we can compare the model performance to.<br>
We use two quite heterogenous datasets in terms of contrast and lighting conditions as ground data for our whole data science process. With this approach all models are evaluated on the same data through which we do a transparent comparison of the three architectures in one place. We also test the ability of all models to generalize since we use a more heterogenous dataset as commonly used for performance measuremnts of these models.

# Data
## Dog dataset
https://www.kaggle.com/datasets/andrewmvd/animal-faces

## Human dataset
https://www.kaggle.com/datasets/ashwingupta3012/human-faces


# Ressources
## Articles
Basics about convolution: https://medium.com/@bdhuma/6-basic-things-to-know-about-convolution-daef5e1bc411<br>
Bicubic interpolation basics: https://www.ece.mcmaster.ca/~xwu/interp_1.pdf<br>
An introduction to Convolutional Neural Networks for image upscaling using Sub-pixel CNNs: https://guipleite.medium.com/an-introduction-to-convolutional-neural-networks-for-image-upscaling-using-sub-pixel-cnns-5d9ea911c557<br>
FSRCNN review: https://towardsdatascience.com/review-fsrcnn-super-resolution-80ca2ee14da4<br>
## Papers
Super Resolution Convolutional Neural Networks (SRCNN): https://arxiv.org/pdf/1608.00367.pdf<br>
Fast SRCNN (FSRCNN): https://arxiv.org/pdf/1501.00092.pdf
## Evaluation Metric for tested approaches
Peak signal-to-noise ratio (PSNR): https://de.mathworks.com/help/vision/ref/psnr.html


# Repository strucutre



# How to run our project

## For dog dataset

### testing only

#### upload the test_notebook and the models to kaggle

#### include the dog dataset into your kaggle notebook

#### run the notebook


### run everything

#### upload all notebooks to kaggle

#### include the dog dataset into your kaggle notebook

#### run all the notebooks in the following order and use results of one notebook for the next





## For human dataset (more work since notebooks are configured for dog dataset right now)

### testing only

#### change model loading paths and model parameters (hyperparameters FSRCNN + upscaling factors)
Note: In FSRCNN the stride in the last layer has to match the upscaling factor and output_padding is stride-1

#### change the folder paths

#### run the notebook
data -> hyperparameter_tuning (use created folder structure) -> cross_validation (use ideal hyperparameters) -> final_model_training (use an epoch number that has been good for all folds in cross val) -> test_notebook (test the notebooks, also make sure to define right FSRCNN hyperparameters)


### running everything
#### Install requirements
To install the requirements use: "pip install -r requirements.txt"

#### Download the train data from the Kaggle dataset

#### Change all the links in all the notebooks to folder path according to your file system

#### Change all the model parameters to upscaling factor 3 and hyperparameters according to tuning
Also make sure to change downscaling factor for input images to 3 everywhere

#### run all notebooks in this order and use results of one notebook to change parameters of the next
data -> hyperparameter_tuning (use created folder structure) -> cross_validation (use ideal hyperparameters) -> final_model_training (use an epoch number that has been good for all folds in cross val) -> test_notebook (test the notebooks, also make sure to define right FSRCNN hyperparameters)


