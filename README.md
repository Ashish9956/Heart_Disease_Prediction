# Heart Disease Prediction using Machine Learning

## Overview
This project aims to predict the likelihood of heart disease in a patient using various machine learning algorithms. The dataset used is the Cleveland Heart Disease dataset available on the UCI Machine Learning Repository.

## Dataset
The dataset contains 14 attributes including age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise, slope of the peak exercise ST segment, number of major vessels, thal, and the target variable indicating the presence of heart disease.

## Project Structure
The project directory contains the following files:

- `data/`: Directory containing the dataset.
- `notebooks/`: Jupyter Notebooks for data exploration, preprocessing, model training, and evaluation.
- `scripts/`: Python scripts for data preprocessing, feature engineering, model training, and evaluation.
- `models/`: Saved machine learning models.
- `results/`: Directory to save results and output files.
- `requirements.txt`: List of dependencies required to run the project.
- `README.md`: Project documentation.

## Setup and Installation

### Prerequisites
- Python 3.6 or higher
- Jupyter Notebook

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/Heart-Disease-Prediction-using-Machine-Learning.git
    cd Heart-Disease-Prediction-using-Machine-Learning
    ```

2. **Install required packages:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Jupyter Notebooks:**
    ```sh
    jupyter notebook
    ```

## Usage
Machine Learning algorithms used:

1. Logistic Regression (Scikit-learn)
2. Naive Bayes (Scikit-learn)
3. Support Vector Machine (Linear) (Scikit-learn)
4. K-Nearest Neighbours (Scikit-learn)
5. Decision Tree (Scikit-learn)
6. Random Forest (Scikit-learn)
7. XGBoost (Scikit-learn)
8. Artificial Neural Network with 1 Hidden layer (Keras)

### Data Preprocessing
Navigate to the `notebooks` directory and open the `1_data_preprocessing.ipynb` notebook. This notebook handles data cleaning, missing value imputation, and feature engineering.

### Model Training
Open the `2_model_training.ipynb` notebook to train various machine learning models including Logistic Regression, Decision Tree, Random Forest, and others.

### Model Evaluation
Use the `3_model_evaluation.ipynb` notebook to evaluate the trained models using metrics like accuracy, precision, recall, and F1-score.

## Results

Accuracy achieved: 95% (Random Forest)

Dataset used: https://www.kaggle.com/ronitf/heart-disease-uci
The results of the models, including performance metrics and visualizations, are saved in the `results` directory. The best-performing model is also saved in the `models` directory for future use.

## Contributing
Contributions are welcome! Please create an issue or submit a pull request if you have any suggestions or improvements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- This project was developed as part of a learning exercise.
- The dataset is provided by the UCI Machine Learning Repository.





