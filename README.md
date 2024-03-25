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

- Install dependencies  
  ```bash
  pip install -r requirements.txt
  ```  

## Usage

Method 1:  

- Dataset is reduced and splitted by `data_reduction.py` and `train_test_split`.  
- Images are resized by `seam_carving.py` and `crop_center.py`.  
- Go to folder `SIFT_feature_matching` (cd SIFT_feature_matching)  
- Run `SIFT_feature_matching.py` to train and test BoVW.  
- To only test the models run `test.py`. pkl files are provided.  
- To find optimized parameters for SIFT and SVC, run `y1_parameter_optimization.ipynb` notebook.  

Method 2:  

- Redundancies were deleted so that each folder had an equal number of images (500). 
- Go to folder `Classification with Features` (cd '.\Classification with Features\')
- Run train.ipynb to train and test. Dataframe operations were performed in the starting cells. Model training was done in the next   
  cells. The results method was written to show the results of model.
- For testing, only the .pkl file of the two models was created and the test process was shown at the end of train.ipynb.
- You can check feature_extraction.py to see how feature extraction is done and the code.

Method 3:  

- TODO

## Contributors

- [Emre Belikırık](https://github.com/emre-bl)
- [Zeynep Meriç Aşık](https://github.com/meric2)

