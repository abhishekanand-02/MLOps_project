import os
import sys
import pandas as pd
from src.exception.exception import customexception
from src.logger import logging
from src.utils.utils import load_object


class PredictPipeline:

    
    def __init__(self):
        logging.info("Initializing the prediction pipeline object...")

    def predict(self, features):
        try:
            # Load preprocessor and model
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            logging.info("Loading preprocessor and model from the artifacts...")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Apply preprocessor transformation and predict
            logging.info("Applying preprocessing to the input data...")
            scaled_fea = preprocessor.transform(features)
            pred = model.predict(scaled_fea)

            logging.info("Prediction completed successfully.")
            return pred

        except Exception as e:
            raise customexception(e, sys)


class CustomData:
    def __init__(self,
                 carat: float,
                 depth: float,
                 table: float,
                 x: float,
                 y: float,
                 z: float,
                 cut: str,
                 color: str,
                 clarity: str):

        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            # Prepare input data dictionary
            custom_data_input_dict = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }
            # Convert to DataFrame
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe created successfully.')
            return df
        except Exception as e:
            logging.info('Exception occurred while creating the dataframe.')
            raise customexception(e, sys)


if __name__ == "__main__":
    try:
        # Example: Predicting a custom data sample
        custom_data = CustomData(
            carat=0.5,
            depth=61.5,
            table=59.0,
            x=5.2,
            y=5.3,
            z=3.1,
            cut="Ideal",
            color="G",
            clarity="VS1"
        )

        # Get the data as a dataframe
        df = custom_data.get_data_as_dataframe()

        # Initialize the prediction pipeline and get prediction
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(df)

        # Output the prediction result
        print(f"Predicted Price: {prediction[0]}")

    except Exception as e:
        logging.error(f"Error in executing the prediction pipeline: {e}")


# import os
# import sys
# import pandas as pd
# from src.exception.exception import customexception
# from src.logger import logging
# from src.utils.utils import load_object


# class PredictPipeline:
#     def __init__(self):
#         logging.info("PredictPipeline object initialized.")

#     def predict(self, features):
#         try:
#             # Define paths for the preprocessor and model
#             preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
#             model_path = os.path.join("artifacts", "model.pkl")

#             # Load preprocessor and model from saved files
#             logging.info("Loading preprocessor and model from artifacts.")
#             preprocessor = load_object(preprocessor_path)
#             model = load_object(model_path)

#             # Perform feature scaling and prediction
#             logging.info("Transforming features using preprocessor and making prediction.")
#             scaled_features = preprocessor.transform(features)
#             prediction = model.predict(scaled_features)

#             logging.info("Prediction completed successfully.")
#             return prediction

#         except FileNotFoundError as e:
#             logging.error(f"File not found: {str(e)}")
#             raise customexception(f"File not found: {str(e)}", sys)
#         except Exception as e:
#             logging.error(f"Exception occurred during prediction: {str(e)}")
#             raise customexception(e, sys)


# class CustomData:
#     def __init__(self, carat: float, depth: float, table: float, x: float, y: float, z: float, cut: str, color: str, clarity: str):
#         # Assigning input values to instance variables
#         self.carat = carat
#         self.depth = depth
#         self.table = table
#         self.x = x
#         self.y = y
#         self.z = z
#         self.cut = cut
#         self.color = color
#         self.clarity = clarity

#         logging.info("CustomData object initialized with input values.")

#     def get_data_as_dataframe(self):
#         try:
#             # Convert the custom data to a dictionary
#             custom_data_input_dict = {
#                 'carat': [self.carat],
#                 'depth': [self.depth],
#                 'table': [self.table],
#                 'x': [self.x],
#                 'y': [self.y],
#                 'z': [self.z],
#                 'cut': [self.cut],
#                 'color': [self.color],
#                 'clarity': [self.clarity]
#             }

#             # Convert dictionary to DataFrame
#             df = pd.DataFrame(custom_data_input_dict)
#             logging.info("Dataframe created successfully from custom data input.")
#             return df

#         except Exception as e:
#             logging.error(f"Exception occurred in getting data as dataframe: {str(e)}")
#             raise customexception(e, sys)
