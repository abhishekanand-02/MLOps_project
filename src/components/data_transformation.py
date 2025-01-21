import pandas as pd
import numpy as np
from src.logger import logging
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated')

            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Define the custom ranking for each ordinal variable
            cut_categories = ["Ideal","Very Good","Premium"]
            color_categories = ["F","E","G","J"]
            clarity_categories = ["SI2","SI1","VS2","VS1","IF"]
            
            logging.info('Pipeline Initiated')

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )# SimpleImputer: This class is used to handle missing values in the dataset.

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            # Column Transformer that applies the respective pipelines
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            logging.error(f"Exception occurred in the get_data_transformation: {str(e)}")
            raise customexception(e, sys)

    def initialize_data_transformation(self, train_path, test_path):
        try:
            # Read the train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f'Train Dataframe Head:\n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head:\n{test_df.head().to_string()}')

            # Initialize the preprocessing pipeline
            preprocessing_obj = self.get_data_transformation()

            # Define the target column and drop unwanted columns
            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply the preprocessing pipeline
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            # Combine the transformed features with the target column
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing pickle file saved")

            return train_arr, test_arr

        except Exception as e:
            logging.error(f"Exception occurred in the initialize_data_transformation: {str(e)}")
            raise customexception(e, sys)





# import pandas as pd
# import numpy as np
# from src.logger import logging
# from src.exception.exception import customexception
# import os
# import sys
# from dataclasses import dataclass
# from pathlib import Path
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from src.utils.utils import save_object

# # Define the mappings
# cut_map = {
#     'Ideal': 3,
#     'Very Good': 2,
#     'Premium': 1
# }

# color_map = {
#     'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7
# }

# clarity_map = {
#     'IF': 1, 'VVS1': 2, 'VVS2': 3, 'VS1': 4, 'VS2': 5, 'SI1': 6, 'SI2': 7, 'I1': 8
# }

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()

#     def get_data_transformation(self):
#         try:
#             logging.info('Data Transformation initiated')

#             # Define which columns should be ordinal-encoded and which should be scaled
#             categorical_cols = ['cut', 'color', 'clarity']
#             numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

#             # Numerical Pipeline
#             num_pipeline = Pipeline(
#                 steps=[
#                     ('imputer', SimpleImputer(strategy='median')),
#                     ('scaler', StandardScaler())
#                 ]
#             )

#             # Custom transformation function to apply the mappings
#             def apply_mappings(df):
#                 # Apply the mapping for 'cut', 'color', 'clarity'
#                 df['cut'] = df['cut'].map(cut_map)
#                 df['color'] = df['color'].map(color_map)
#                 df['clarity'] = df['clarity'].map(clarity_map)
#                 return df

#             # Categorical Pipeline (modified to use the mapping function)
#             cat_pipeline = Pipeline(
#                 steps=[
#                     ('imputer', SimpleImputer(strategy='most_frequent')),
#                     ('custom_mapper', apply_mappings),  # Apply custom mappings
#                     ('scaler', StandardScaler())
#                 ]
#             )

#             # Column Transformer that applies the respective pipelines
#             preprocessor = ColumnTransformer([
#                 ('num_pipeline', num_pipeline, numerical_cols),
#                 ('cat_pipeline', cat_pipeline, categorical_cols)
#             ])

#             return preprocessor

#         except Exception as e:
#             logging.error(f"Exception occurred in the get_data_transformation: {str(e)}")
#             raise customexception(e, sys)

#     def initialize_data_transformation(self, train_path, test_path):
#         try:
#             # Read the train and test data
#             train_df = pd.read_csv(train_path)
#             test_df = pd.read_csv(test_path)

#             logging.info("Read train and test data completed")
#             logging.info(f'Train Dataframe Head:\n{train_df.head().to_string()}')
#             logging.info(f'Test Dataframe Head:\n{test_df.head().to_string()}')

#             # Initialize the preprocessing pipeline
#             preprocessing_obj = self.get_data_transformation()

#             # Define the target column and drop unwanted columns
#             target_column_name = 'price'
#             drop_columns = [target_column_name, 'id']

#             input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
#             target_feature_train_df = train_df[target_column_name]

#             input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
#             target_feature_test_df = test_df[target_column_name]

#             # Apply the preprocessing pipeline
#             input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
#             input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

#             logging.info("Applying preprocessing object on training and testing datasets.")

#             # Combine the transformed features with the target column
#             train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
#             test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

#             # Save the preprocessing object
#             save_object(
#                 file_path=self.data_transformation_config.preprocessor_obj_file_path,
#                 obj=preprocessing_obj
#             )

#             logging.info("Preprocessing pickle file saved")

#             return train_arr, test_arr

#         except Exception as e:
#             logging.error(f"Exception occurred in the initialize_data_transformation: {str(e)}")
#             raise customexception(e, sys)

