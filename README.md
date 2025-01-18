# MLOps_Project

## Overview

This project is an end-to-end MLOps pipeline that handles the complete machine learning lifecycle, from data ingestion to model training, prediction, and model deployment. It includes several key components to ensure scalability, maintainability, and reproducibility in the workflow.

Key features of this project include:
- **Data Ingestion**: Data is ingested and preprocessed.
- **Model Training**: Multiple models are trained and evaluated using a training pipeline.
- **Prediction Pipeline**: A separate pipeline is used to handle predictions for custom inputs.
- **Testing**: Unit tests are provided for both the training and prediction pipelines.

## Project Structure


## Flow of Code

### 1. **Data Ingestion** (from `src/pipelines/training_pipeline.py`)
   - **File**: `src/pipelines/training_pipeline.py`
   - **Description**: 
     - The data ingestion process begins with reading the dataset from a CSV file located in the `src/original_csv` folder.
     - The data is then split into training and test datasets. These datasets are saved as `train.csv` and `test.csv` in the `artifacts` folder.

### 2. **Data Transformation** (from `src/pipelines/training_pipeline.py`)
   - **File**: `src/pipelines/training_pipeline.py`
   - **Description**: 
     - The raw data undergoes preprocessing using custom logic defined in the `src/utils/preprocessing.py` file.
     - The preprocessing object is saved as `preprocessor.pkl` in the `artifacts` folder.

### 3. **Model Training** (from `src/pipelines/training_pipeline.py`)
   - **File**: `src/pipelines/training_pipeline.py`
   - **Description**: 
     - Various models such as Linear Regression, Ridge, Lasso, and ElasticNet are trained using the training dataset.
     - The best-performing model is selected based on evaluation metrics (e.g., R2 score) and saved as `model.pkl` in the `artifacts` folder.

### 4. **Prediction Pipeline** (from `src/pipelines/prediction_pipeline.py`)
   - **File**: `src/pipelines/prediction_pipeline.py`
   - **Description**:
     - The trained model (`model.pkl`) and the preprocessor (`preprocessor.pkl`) are loaded from the `artifacts` folder.
     - The `PredictPipeline` class is used to make predictions on new input data.

### 5. **Custom Data Input** (from `src/pipelines/prediction_pipeline.py`)
   - **File**: `src/pipelines/prediction_pipeline.py`
   - **Description**:
     - A custom input class (`CustomData`) allows users to provide custom feature inputs (e.g., `carat`, `depth`, `table`, etc.).
     - The `get_data_as_dataframe()` method converts this data into a pandas DataFrame, which is passed through the pipeline for prediction.

### 6. **Testing** (from `tests/test_training.py` and `tests/test_prediction.py`)
   - **File**: `tests/test_training.py` & `tests/test_prediction.py`
   - **Description**:
     - Unit tests are written for the model training pipeline (`test_training.py`) and the prediction pipeline (`test_prediction.py`).
     - These tests verify the correctness and integrity of the processes such as model training, evaluation, and predictions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abhishekanand-02/MLOps_project.git
   cd MLOPs_Train

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. (Optional) Install the package locally using pip:
  ```bash
   pip install -e .

## Running the Pipeline

Training Pipeline:
To run the training pipeline, execute the following command:

python -m src.pipelines.training_pipeline
This will:

Read the input data.
Perform data preprocessing.
Train multiple models and select the best one.
Save the trained model and preprocessor as artifacts.
Prediction Pipeline:
To run the prediction pipeline for custom data, execute the following command:


python -m src.pipelines.prediction_pipeline
This will:

Load the trained model and preprocessor.
Make predictions for custom input data and print the results.
Testing
To run the tests, execute the following command:


pytest
This will run all the tests and ensure that the training and prediction pipelines work as expected.

Testing the Training Pipeline:

pytest tests/test_training.py
Testing the Prediction Pipeline:

pytest tests/test_prediction.py

###  Setting up the Project Structure
To create the necessary project structure, run the init_setup.sh script. This will create all the required directories and files for the project.

Make the script executable:

```bash
chmod +x init_setup.sh
Run the script:
```bash
./init_setup.sh

This will:

Create the project directory structure.
Initialize files like requirements.txt, setup.py, and src components.




