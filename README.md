# CheXpert MLOps Pipeline
Implementation pipeline of modern DL classification techniques using CheXpert dataset for multi-class chest x-ray disease classification with user interface

## Overview
The CheXpert dataset contains...

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

For the first round of the model only a subset of data in the 'CheXpert-v1.0 batch 4 (train 3).zip' file was used.

## Process
1. The paper describes the pre-processing techniques used to maximize the effectiveness of the model. The images in the dataset are originally imported at a variaty of resolutions and so first all images are rescaled to 256x256 pixels. Then, a image template is use in a template matching algorithm to pick out the lung region of each scan in 224x224 pixels. I manually created the template that will be compared to each scan by finding a clear and centered scan, rescaling to 256x256 pixels and then manually cropping the 224x224 lung region of the scan. The template matching algorith will automatically go through each 100,000 or so other scans and find the region that most matching my cropped lung template such that each scan is more uniformly analized by the model.

2. 

## Key Features

## Usage

 ## Future Work
 Ideally in full implementation live data on chest x-rays and profeshional classifications from submissions would be collected in order to update the model to account for data decay or drift

 ## References
