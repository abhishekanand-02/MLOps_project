import pandas as pd
import numpy as np
from src.logger import logging  # Use logger from src.logger
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from src.utils.utils import save_object, evaluate_model

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting dependent and independent variables from train and test data.')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet()
            }

            # Evaluating models
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            # Print and log the model report
            print(f"Model Performance Report:\n{model_report}")
            logging.info(f"Model Performance Report: {model_report}")

            logging.info('\n====================================================================================')

            # Finding the best model based on R2 score
            best_model_score = max(model_report.values())  # Get highest R2 score
            best_model_name = [k for k, v in model_report.items() if v == best_model_score][0]  # Get model name

            logging.info(f"Best Model Found: Model Name: {best_model_name}, R2 Score: {best_model_score}")
            print(f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}")

            # Saving the best model
            best_model = models[best_model_name]
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.error(f"Exception occurred during model training: {str(e)}")
            raise customexception(e, sys)



# import pandas as pd
# import numpy as np
# from src.logger import logging 
# from src.exception.exception import customexception
# import os
# import sys
# from dataclasses import dataclass
# from pathlib import Path

# from src.utils.utils import save_object, evaluate_model

# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


# @dataclass 
# class ModelTrainerConfig:
#     # Use pathlib's Path for platform-independent path handling
#     trained_model_file_path: Path = Path('artifacts') / 'model.pkl'
    
# class ModelTrainer:
#     def __init__(self):
#         # Initialize the config
#         self.model_trainer_config = ModelTrainerConfig()
    
#     def initiate_model_training(self, train_array, test_array):
#         try:
#             logging.info('Splitting Dependent and Independent variables from train and test data')
            
#             # Splitting features and target variables
#             X_train, y_train, X_test, y_test = (
#                 train_array[:, :-1],  # Independent variables for train
#                 train_array[:, -1],   # Target variable for train
#                 test_array[:, :-1],   # Independent variables for test
#                 test_array[:, -1]     # Target variable for test
#             )

#             # Define models
#             models = {
#                 'LinearRegression': LinearRegression(),
#                 'Lasso': Lasso(),
#                 'Ridge': Ridge(),
#                 'Elasticnet': ElasticNet()
#             }
            
#             # Evaluate models
#             model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            
#             # Print and log the model report
#             print(model_report)
#             print('\n====================================================================================\n')
#             logging.info(f'Model Report: {model_report}')

#             # Get the best model based on highest score
#             best_model_score = max(sorted(model_report.values()))
#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]
            
#             best_model = models[best_model_name]

#             logging.info(f'Best Model Found: Model Name: {best_model_name}, R2 Score: {best_model_score}')
#             print(f'Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}')

#             # Save the best model
#             save_object(
#                  file_path=self.model_trainer_config.trained_model_file_path,
#                  obj=best_model
#             )

#         except Exception as e:
#             logging.error('Exception occurred in Model Training')
#             raise customexception(e, sys)
