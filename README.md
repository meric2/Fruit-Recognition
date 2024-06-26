# Fruit Recognition
[OneDrive - For test data and Models](https://etuedutr-my.sharepoint.com/:f:/g/personal/ebelikirik_etu_edu_tr/Esr3AJnbnNBLgeCzJsl0ng8BhMNHlDiM9yu9FUAFzAaOFQ?e=E2j9Wf)

Test data is in [data_128x128.rar](https://etuedutr-my.sharepoint.com/:u:/g/personal/ebelikirik_etu_edu_tr/ETK7gYCUcX1Fg8FEJzvWTugB3iWu7igch2m3abPMAH4Jcw?e=sGfCDh) folder.
DO NOT use the test folder without its base folder.
In the scripts the test directory is given as data_128x128/test.


This project performs different classification models on [fruits-360](https://www.kaggle.com/datasets/moltean/fruits) and [fruits-262](https://www.kaggle.com/datasets/aelchimminut/fruits262) datasets.  

## Project Overview

The key aspects of this project includes three main methods with different models:

- Bag of Visual Words (BoVW)  
- Machine Learning with feature extraction from image
- Convolutional Neural Network (CNN)  

## Getting Started  

### Prerequisites 
- Python 3
- Jupyter Notebook

### Installation

- Clone the repository
  ```bash
  git clone https://github.com/meric2/Fruit-Recognition.git
  ```

- Install dependencies  
  ```bash
  pip install -r requirements.txt
  ```  

## Usage

All methods have 20 classes of different fruits. 

Method 1:  

- Dataset is reduced and splitted by `data_reduction.py` and `train_test_split`.  
- Images are resized by `seam_carving.py` and `crop_center.py`.  
- Go to folder `SIFT_feature_matching` (cd SIFT_feature_matching)  
- Run `SIFT_feature_matching.py` to train BoVW.
- To test the models download and unzip [dataset](https://drive.google.com/file/d/1GuJqBZI2sCCiHzqjdmOI7IO7TdgeKRF-/view?usp=sharing). Then, run `test.py`. pkl files are provided.  
- To find optimized parameters for SIFT, run `y1_parameter_optimization.ipynb` notebook.  
- SVC is replaced with kNN (k Nearest Neighbor) model for classification. With the guidance of the Instructor.

Method 2:  

- This method uses same train-test datasets with Method1
- Go to folder `Classification with Features` (cd '.\Classification with Features\')
- For testing, there is [ML classification - Method2](https://etuedutr-my.sharepoint.com/:f:/g/personal/ebelikirik_etu_edu_tr/ElqMCcWqBNdPg95hJOhufKEB1XZuNjsSGrBBrLaXuxK1qg?e=zrGlbs) folder at drive. You can download all files (*.pkl). And at last cell of train.ipynb you can test models. (Don't forget to use xgb_label_encoder for xgb model).
- Instead with train-test .csv files you can run train.ipynb for re-train models and test all.
- You can check feature_extraction.py to see how feature extraction is done and the code.

Method 3:  
- For CNN model download CNN_model.h5 from drive [CNN Models folder](https://etuedutr-my.sharepoint.com/:f:/g/personal/ebelikirik_etu_edu_tr/EtaxM6rnsgdEm1f4G-uKMXMBUxbjgOVC0VigrDUoVpfAFg?e=s7Vcfx) and you can test model at CNN folder with Test.py file
- For InceptionNet model download inceptionNet.h5 from drive CNN Models folder and you can test model at InceptionNet folder with Test.py file
- For ResNet model download ResNet.h5 from drive CNN Models folder and you can test model at ResNet folder with Test.py file
## Contributors

- [Emre Belikırık](https://github.com/emre-bl)
- [Zeynep Meriç Aşık](https://github.com/meric2)

