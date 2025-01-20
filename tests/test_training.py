# # tests/test_training.py
# import os
# import pytest
# from src.pipelines.training_pipeline import ModelTrainer
# from src.logger import logging

# # Initialize logging
# logging.basicConfig(level=logging.INFO)

# def test_model_training():
#     try:
#         # Example mock data (train_array and test_array should have features and target)
#         train_array = [
#             [1.52, 62.2, 58.0, 7.27, 7.33, 4.55, 13761.6124],  # Example data (features and target)
#             [0.74, 61.8, 57.0, 5.76, 5.79, 3.57, 3179.7717],
#             [0.32, 61.6, 56.0, 4.38, 4.41, 2.71, 744.5618],
#             [1.70, 62.6, 59.0, 7.65, 7.61, 4.77, 14704.1535],
#             [1.34, 62.5, 57.0, 7.00, 7.05, 4.38, 6391.7472]
#         ]
        
#         test_array = [
#             [0.70, 61.2, 57.0, 5.69, 5.73, 3.50, 2838.3526],
#             [0.30, 62.0, 56.0, 4.35, 4.37, 2.70, 946.2435]
#         ]

#         model_trainer = ModelTrainer()

#         # Test the model training process
#         logging.info("Starting model training test...")
#         model_trainer.initate_model_training(train_array, test_array)

#         # Check if model is saved correctly (you could verify model's existence)
#         model_path = os.path.join("artifacts", "model.pkl")
#         assert os.path.exists(model_path), "Model was not saved correctly"

#         logging.info("Model training test passed successfully.")

#     except Exception as e:
#         logging.error(f"Error occurred during model training test: {e}")
#         pytest.fail(f"Model training test failed: {e}")


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from src.pipelines.training_pipeline import ModelTrainer
from src.exception.exception import customexception

def test_model_trainer_init():
    # Test to check if the ModelTrainer object is created successfully
    trainer = ModelTrainer()
    assert trainer is not None
    print("ModelTrainer initialized successfully!")

def test_model_training():
    # Here you can mock your train and test arrays and check if the model training process runs
    try:
        model_trainer = ModelTrainer()
        
        # Example mock data (Replace with real mock data if necessary)
        train_array = [[1.5, 61.5, 57.0, 6.0, 6.1, 3.5, 1]]
        test_array = [[2.0, 62.0, 58.0, 6.8, 6.9, 4.0, 2]]
        
        model_trainer.initate_model_training(train_array, test_array)
        
        # Here you could also assert for expected results after model training, but currently, we're only ensuring no exception is thrown.
        print("Model training completed successfully!")
        
    except customexception as e:
        pytest.fail(f"Training failed with error: {e}")
