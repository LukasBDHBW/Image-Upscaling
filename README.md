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
An Overview of ESPCN: https://medium.com/@zhuocen93/an-overview-of-espcn-an-efficient-sub-pixel-convolutional-neural-network-b76d0a6c875e<br>
FSRCNN review: https://towardsdatascience.com/review-fsrcnn-super-resolution-80ca2ee14da4<br>
## Papers
Super Resolution Convolutional Neural Networks (SRCNN): https://arxiv.org/pdf/1608.00367.pdf<br>
Fast SRCNN (FSRCNN): https://arxiv.org/pdf/1501.00092.pdf
## Evaluation Metric for tested approaches
Peak signal-to-noise ratio (PSNR): https://de.mathworks.com/help/vision/ref/psnr.html


# Repository strucutre

### Notebooks
The notebooks we created resemble the whole Data Science process of our project. <br>
We start with data preprocessing in the data notebook. This notebook contains all the preprocessing steps done for the human dataset as well as for the dog dataset. The human dataset has been worked with on Google Colab. Also the human dataset has no pre-done train/test split by kaggle, so we did the split ourselfes. For the dog dataset there is already a split so we process them directly and only to train val splits in the notebooks where it is needed.<br>
It is important to note that all other notebooks do not contain seperate code for each dataset. In fact the code is equal for both in large parts, so we adapted all parameters and file links to the kaggle structure used for processing the dog dataset. Specific things for the human dataset have been commented out.<br>
The hyperparameter notebook is used for tuning hyperparameters of all models. For SRCNN and ESPCN only the learning rate is tuned, for FSRCNN alse the model parameters (number hidden layers etc.) are tuned.<br>
In the cross validation notebook we do a 5 fold cross validation to test how robust our models performe. Also we haven taken a look at which epoch number would be good for the final training on the basis of when the PSNR goes down for each fold. For that we created graphics.<br>
In the final training notebook we train our ifnal models based on all the training data.<br>
In the test notebook we apply all models to testing data. We measure the average PSNR as well as time needed to upscale the images, which is an important metric if you want to do video upscaling, since for live video upscaling the model has to be faster than the video framerate. Our test notebook also outputs a file structure with all upscaled versions, the original and the input version of an image next to each other for a nice visual comparison.

### Results
We created seperate folders for the results of the process steps for each dataset. In each folder there are subfolders for the models, cross validation results and graphics and hyperparametertuning study results.



# How to run our project

## For dog dataset

### testing only

### View & run the test_notebook at: https://www.kaggle.com/code/wmarcus/image-upscaling-cnns/notebook


### run everything

#### upload all notebooks to kaggle

#### include the dog dataset into your kaggle notebook

#### Important: You have to add the very last code cell of the data notebook to all other notebooks except the testing notebook after the input statements, because the low res files have to be created newly with each new notebook session in kaggle

#### run all the notebooks in the following order and use results of one notebook for the next





## For human dataset (more work since notebooks are configured for dog dataset right now)

### testing only

#### change model loading paths and model parameters (hyperparameters FSRCNN + upscaling factors) in test_notebook
Note: In FSRCNN the stride in the last layer has to match the upscaling factor and output_padding is stride-1

#### download the human faces dataset

#### run datapreprocessing steps in datanotebook for the human face dataset
#### change the folder paths

#### run the notebook
data -> hyperparameter_tuning (use created folder structure) -> cross_validation (use ideal hyperparameters) -> final_model_training (use an epoch number that has been good for all folds in cross val) -> test_notebook (test the notebooks, also make sure to define right FSRCNN hyperparameters)


### running everything
#### Install requirements
To install the requirements use: "pip install -r requirements.txt"

#### Download the data from the Kaggle dataset

#### Change all the links in all the notebooks to folder path according to your file system

#### Change all the model parameters to upscaling factor 3 and hyperparameters according to tuning
Also make sure to change downscaling factor for input images to 3 everywhere

#### run all notebooks in this order and use results of one notebook to change parameters of the next
data -> hyperparameter_tuning (use created folder structure) -> cross_validation (use ideal hyperparameters) -> final_model_training (use an epoch number that has been good for all folds in cross val) -> test_notebook (test the notebooks, also make sure to define right FSRCNN hyperparameters)


