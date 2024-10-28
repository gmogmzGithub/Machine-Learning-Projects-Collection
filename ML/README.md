# Machine Learning Portfolio Projects

A comprehensive collection of machine learning projects demonstrating practical applications across different domains - from business analytics to healthcare and computer vision. This repository showcases implementation of various ML algorithms using Python's data science stack.

## 📊 Projects Overview

### 1. Ice Cream Revenue Prediction
- **Type**: Simple Linear Regression
- **Domain**: Business Analytics
- **Objective**: Predict daily ice cream revenue based on temperature
- **Technologies**: scikit-learn, pandas, seaborn
- **Key Features**:
  - Data visualization using seaborn and matplotlib
  - Training/testing data split
  - Linear regression model implementation
  - Model performance visualization

### 2. Breast Cancer Classification
- **Type**: Support Vector Machine (SVM)
- **Domain**: Healthcare/Medical Diagnosis
- **Objective**: Classify breast cancer tumors as benign or malignant
- **Technologies**: scikit-learn, pandas, seaborn
- **Key Features**:
  - Feature scaling and normalization
  - SVM model implementation
  - Hyperparameter tuning using GridSearchCV
  - Model evaluation using confusion matrix
  - Classification report generation

### 3. Smiling Face Detection
- **Type**: Convolutional Neural Network (CNN)
- **Domain**: Computer Vision
- **Objective**: Detect smiling faces in images
- **Technologies**: Keras, TensorFlow, h5py
- **Key Features**:
  - Image data preprocessing
  - CNN architecture design
  - Model training and evaluation
  - Visual prediction results
  - Performance metrics analysis

## 🛠️ Technologies Used
- **Python**: Primary programming language
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Deep Learning**: Keras, TensorFlow
- **Data Storage**: h5py
- **Development Tools**: Jupyter Notebook

## 📦 Installation & Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/Machine-Learning-Portfolio-Projects.git
cd Machine-Learning-Portfolio-Projects
```

2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

## 📋 Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
keras
tensorflow
h5py
jupyter
```

## 🚀 Usage

Each project is contained in its own Jupyter notebook:
1. `Ice_Cream_Revenue_Prediction.ipynb`
2. `Breast_Cancer_Classification.ipynb`
3. `Smiling_Face_Detection.ipynb`

To run the notebooks:
1. Start Jupyter Lab/Notebook:
```bash
jupyter lab
# or
jupyter notebook
```
2. Navigate to the desired notebook
3. Run the cells sequentially (Shift + Enter)

## 📈 Results Summary

### Ice Cream Revenue Prediction
- Successfully implemented linear regression to predict daily revenue
- Demonstrated strong correlation between temperature and sales
- Achieved meaningful R-squared value indicating good model fit

### Breast Cancer Classification
- Implemented SVM with optimized hyperparameters
- Achieved high accuracy in tumor classification
- Successfully handled feature scaling and model optimization

### Smiling Face Detection
- Built and trained a CNN for facial expression recognition
- Implemented image preprocessing and data augmentation
- Achieved good accuracy in smile detection

## 🤝 Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
