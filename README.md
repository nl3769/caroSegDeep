# caroSegDeep
Arterial wall segmentation on ultrasound image sequences using deep learning.


# Overview

The presented GIT is a demonstration of the developed algorithm and can be used as a public a benchmark. It uses a convolutional neural network (_CNN_) named Dilated U-net, which aims to segment the intima-media complex (_IMC_) of the common carotid artery (_CCA_). Any advanced user can easily add their own architectures for training and evaluation. The repository provides:
* generate the dataset;
* train a model for far wall detection;
* train a model for IMC segmentation;
* far wall detection
* _IMC_ segmentation by using a manual homemade interface to detect the far wall ou by using predicted far wall position;
* segmentation evaluation.  

All parameters are set in the folder _**parameters**_:
* _**CREATE_REFERENCES_CUBS/set_parameters.m**_: parameters to interpolate images from control points;
* _**SEGMENTATION/parameters/set_parameters_dataset_template.py**_: parameters to generate the dataset;
* _**SEGMENTATION/parameters/set_parameters_inference_template.py**_: parameters to segment the image;
* _**SEGMENTATION/set_parameters_caro_seg_deep_training_template.py**_ parameter to train the neural network for _IMC_ segmentation;
* _**SEGMENTATION/set_parameters_far_wall_training_template.py**_ parameter to train the neural network for far wall detection.

# Prerequisites

A file named _environment.yml_ is provided to install all the used libraries. For Linux users, you should install conda then move into the folder containing the _environment.yml_ file and run:\
`conda env create --file envname.yml.
That would install a conda environment named _caroSegDeep_. To activate it, open a terminal and enter conda activate _caroSegDeep_.\ 

To run the provided script located in folder **_RUN/_**, Linux users should add thoses lines in their _.bashrc_:

`# >>> conda initialize >>>`  
`# !! Contents within this block are managed by 'conda init' !!`  
`__conda_setup="$('/home/laine/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"`  
`if [ $? -eq 0 ]; then`  
    `eval "$__conda_setup"`  
`else`  
    `if [ -f "/home/laine/anaconda3/etc/profile.d/conda.sh" ]; then`  
        `. "/home/laine/anaconda3/profile.d/conda.sh"`  
    `else`  
        `export PATH="/home/laine/anaconda3/bin/:$PATH"`  
    `fi`  
`fi`  
`unset __conda_setup`  
`# <<< conda initialize <<<`  


# How to create the annotation

You must download the CUBS database using the following link: **_TODO_**. Then go to **_CREATE_REFERENCES_CUBS/set_parameters_** and fill in the variables according to your path tree and run **_RUN/run_annotation.m_**. This code interpolates experts' control points and saves information according to **_set_parameters_**. An example is visible below:  

<p align="center">
  <img width="600" height="450" src="https://github.com/nl3769/caroSegDeep/blob/master/.IMAGE_WIKI/interpolation_sample.jpg">
</p>

You can easily change the interpolation method in the class _`CREATE_REFERENCES_CUBS/interpolation.m`_, _makima_ and _pchip_ have already been implemented. 

# How to create the datasets

Two datasets are needed to train both models _i_) far wall detection_ii_) _IMC_ segmentation. Those datasets will be stored in _.h5_ file name _CUBS_wall.h5_ and _CUBS_far_wall.h5_. To do that, fill in the variables according to your path tree in: **_SEGMENTATION/parameters/set_parameters_dataset_template.py_**.  
Then run **_RUN/run_dataset.sh_**. The size of the generated datasets are 24.4Go (for _IMC_ segmentation) and 8.1Go (for far wall detection).

# How to train the models?

To run the training, you need the generated dataset named _CUBS_wall.h5_ and _CUBS_far_wall.h5_. In the proposed folder arrangement, they are stored in **_caroSegDeep/EXAMPLE/DATA/DATASET_**.\
  
As previously, fill in the variables according to your path tree in **_SEGMENTATION/parameters/set_parameters_far_wall_training_template.py_** for far wall detection and **_SEGMENTATION/parameters/set_parameters_caro_seg_deep_training_template.py_** for _IMC_ segmentation.\ 
  
Once parameters are filled in, change the working directory, run:\
1. `bash RUN/run_training_far_wall.sh`\
2. `bash RUN/run_training_IMC.sh`\

# Inference
The far wall detection and the _IMC_ segmentation are done separately. You can easily merge them to give the method fully automatic, nevertheless, the analysis would be more difficult. Thus two different scripts are used:  
1. `RUN/run_FW_detection.sh`\
2. `RUN/run_IMC_segmentation.sh`\


The parameters are common for both codes, thus fill in according to your path three in:  
**_caroSegDeep/SEGMENTATION/parameters/set_parameters_inference_template.py_**.  

The way to use them is described above.  

## How to detect the far wall?
To run the far wall detection, modify the working directory in _**RUN/run_IMC_segmentation.sh**_ and enter in a terminal:\
`bash RUN/run_IMC_segmentation.sh`\

The script use the previous trained model named _CUBS_far_wall.h5_. If you do not train the model, you can download a pretrained one on the efollowing link: ADDLINK  and copy it in **_EXAMPLE/TRAINED_MODEL_**.
Two results are saved:\
1. **_EXAMPLE/RESULTS/INFERENCE/FAR_WALL_DETECTION_**: far wall detection position, saved in .txt file;
2.  **_EXAMPLE/RESULTS/INFERENCE/IMAGES_FW_**: images on which the far wall detection is superimposed.


## How to segment the _IMC_?

Two modes are proposed to segment the _IMC_ _i_) a semi-automatic method which means that a homemade GUI is used to detect the far wall _ii_) a fully automatic method which means that results of the far wall detection is used to initialize the _IMC_ segmentation and if the far wall prediction doesn't exist in the searching directory then the homemade GUI is used.
  
The script use the previous trained model named _CUBS_wall.h5_. If you do not train the model, you can download a pretrained one on the  following link: ADDLINK and copy it in **_EXAMPLE/TRAINED_MODEL_**

Then modify the file _**parameters/set_parameters_training_template.py**_ in order to match with your directory and run `sh run_training.sh`.

The training results is saved in **_caroSegDeep/EXAMPLE/RESULTS/TRAINING_**

To launch the segmentation, run `sh run_segmentation.sh`. The code will segment all images in the folder:
 _**caroSegDeep/EXAMPLE/DATA/IMAGES**_. 
For each image, the user has to manually detect the far wall of the _CCA_. It is a homemade interface, the commands are listed below:
* _left click:_ set a point;
* _ctrl+left click:_ leave the graphical interface;
* _scroll click:_ reset.

To detect the far wall, a minimum of four clicks is required:
1. _first click_: left border of the _ROI_ (click midway distance between the _LI_ and _MA_ interface);
2. _second click_: right border of the _ROI_ (click midway distance between the _LI_ and _MA_ interface);
3. _third click_: point between the left border of the _ROI_ and the left border of the image (click midway distance between the _LI_ and _MA_ interface);
4. _fourth click_: point between the right border of the _ROI_ and the right border of the image (click midway distance between the _LI_ and _MA_ interface);
5. you can add as many points as you want to shape the curve after those four clicks, cubic spline interpolation is used to adjust the curve. 

The _ROI_ is contained between the two vertical red lines and the algorithm segments the _CCA_ between the two vertical blue lines. This allows you to segment a narrow region, this condition is ensured by yourself. An example is visible in the figure below.

<p align="center">
  <img width="600" height="450" src="https://github.com/nl3769/caroSegDeep/blob/master/.IMAGE_WIKI/FW_detection_explanation_width_enought.jpg">
</p>

Then the algorithm stores the segmentation result in **_.txt_** format for both _LI_ and _MA_ interface two different files. It also saves the image with the segmentation results as shown in the figure below. The results are stored in:
 **_caroSegDeep/EXAMPLE/RESULTS/PREDICTION_RESULTS_**

<p align="center">
  <img width="600" height="450" src="https://github.com/nl3769/caroSegDeep/blob/master/.IMAGE_WIKI/clin_0006_R.jpg">
</p>

# Evaluation

TODO
