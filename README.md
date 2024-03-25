# Fruit Recognition

This project performs different classification models on fruits-360 and fruits-262 datasets.

## Project Overview

The key aspects of this project includes three main methods with different models:

- Bag of Visual Words (BoVW)  
- Machine Learning with feature extraction from image?  
- Convolutional Neural Network (CNN)  

## Getting Started  

### Prerequisites 
- Python 3
- Jupyter Notebook

### Installation

- Clone the repository
  ```bash
  git clone https://github.com/meric2/Fruit-Recognition
  ```

- Install dependencies for each method seperately  
  ```bash
  cd folder/of/method
  ```  

  ```bash
  pip install -r requirements.txt
  ```  

## Usage

Method 1:  

- Dataset is reduced and splitted by `data_reduction.py` and `train_test_split`.  
- Images are resized by `seam_carving.py` and `crop_center.py`.  
- Go to folder `SIFT feature matching` (cd SIFT feature matching)  
- Run `SIFT_feature_matching.py`to train and test BoVW.  
- To only test the models run `test.py`. pkl files are provided.  
- To find optimized parameters for SIFT and SVC, run `y1_parameter_optimization.ipynb` notebook.  

Method 2:  

- 

Method 3:  

- TODO

## Contributors

- [Emre Belikırık](https://github.com/emre-bl)
- [Zeynep Meriç Aşık](https://github.com/meric2)

