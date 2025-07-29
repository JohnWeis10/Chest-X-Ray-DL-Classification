# CheXpert MLOps Pipeline
Implementation pipeline of modern DL classification techniques using CheXpert dataset for multi-class chest x-ray disease classification with user interface

## Overview
The CheXpert dataset contains... I am aware that on Kaggle you can find a small version of the dataset which might be easier to work with but I wanted to work from the raw data in order to have more potential for scaling.

The paper titled 'Interpreting chest X-rays via CNNs that exploit hierarchical disease
dependencies and uncertainty labels' describes modern advancements in deep learning model training that lead to high performance. I will be attempting to re-implement these techniques.

Initially this project has been run on a subset of data from CheXpert in order to run locally and minimize cost of utilizing cloud services and pipeline will be set up to allow for scaling to entire dataset. The complete data can be found at 
https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2.
azcopy was used to export the dataset from azure to my local directory. 'azcopy list' command can also be used to observe the folder contents:
CHEXPERT DEMO.xlsx; Content Length: 1.94 MiB
CheXpert-v1.0 batch 1 (validate & csv).zip; Content Length: 486.04 MiB
CheXpert-v1.0 batch 2 (train 1).zip; Content Length: 162.39 GiB
CheXpert-v1.0 batch 3 (train 2).zip; Content Length: 184.82 GiB
CheXpert-v1.0 batch 4 (train 3).zip; Content Length: 91.09 GiB
README.md; Content Length: 3.21 KiB
train_cheXbert.csv; Content Length: 22.06 MiB
train_visualCheXbert.csv; Content Length: 28.48 MiB

For the first round of the model only a subset of data in the 'CheXpert-v1.0 batch 4 (train 3).zip' file was used. And google colab was used for training. Future updates will utilize Azure compute cluster for scaling to a full data model.

## Process
1. The paper describes the pre-processing techniques used to maximize the effectiveness of the model. The images in the dataset are originally imported at a variaty of resolutions and so first all images are rescaled to 256x256 pixels. Then, a image template is use in a template matching algorithm to pick out the lung region of each scan in 224x224 pixels. I manually created the template that will be compared to each scan by finding a clear and centered scan, rescaling to 256x256 pixels and then manually cropping the 224x224 lung region of the scan. The template matching algorith will automatically go through each 100,000 or so other scans and find the region that most matching my cropped lung template such that each scan is more uniformly analized by the model.

2. The paper I am following suggested that there were some x-rays that were of bad quality or had writing on the images which could hinder the model. They also suggested a possible solution that is discussed in another paper 'FRODO: Free rejection of out-of-distribution samples:
application to chest x-ray analysis'
I implement this process in OOD_filtering and save all cleaned training samples seperately.

3. The first step of training is described as a conditional pass where only samples for which non leaf nodes are all positive are considered. First all training data is filtered for having all non-leaf nodes positive and then we can begin in the first round of real model training.

4. The train_pipeline4.py file contains the process for the core training done in this project. Both phases of training are done (first) and each of the six model templates from (1) are trained on this two stage process.

5. The final model described in (1) is an ensemble model which takes the average prediction of all 6 models in order to generate a much stronger classifier.

6. 'save_ensemble6.py' saves the final ensemble model as its own model which will be used in the user application for final predictions.


## Key Features
You can fine the training steps defined in Process in the data_pipeline directory. All files under the chest_scan_app directory are used to build the local user interface.

## Usage
In order to run a local instance of the chest x-ray application running the run.bat initiation file in the terminal will host a local user application where you can upload a chest x-ray and get back the associated results from the final ensemble model. Make sure x-ray scan is AR and as clear as possible to get best results. Any non chest x-ray images will be rejected by the OOD process.

 ## Future Work
 The first version of the data pipeline utilized google colab for GPU parallelization capabilities of NVIDIA. Also this first version is only trained on a subset of the full dataset provided at (https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2)
 the subset used is training data under 'CheXpert-v1.0 batch 4 (train 3).zip' and is split into the train, test and val sets defined in preprocess1.py

 Ideally in full implementation live data on chest x-rays and profeshional classifications from submissions would be collected in order to update the model to account for data decay or drift

 ## References
 1. Interpreting chest X-rays via CNNs that exploit hierarchical disease dependencies and uncertainty labels (https://arxiv.org/abs/1911.06475)

 2. FRODO: Free rejection of out-of-distribution samples: application to chest x-ray analysis (https://arxiv.org/abs/1907.01253)
