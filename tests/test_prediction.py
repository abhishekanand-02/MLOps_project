# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# import pytest
# from src.pipelines.prediction_pipeline import PredictPipeline, CustomData
# from src.exception.exception import customexception

# def test_predict_pipeline():
#     # Initialize the prediction pipeline
#     prediction_pipeline = PredictPipeline()
    
#     # Create custom data for prediction
#     custom_data = CustomData(carat=1.5, depth=61.5, table=57.0, x=6.0, y=6.1, z=3.5, cut='Ideal', color='G', clarity='VS1')
    
#     # Convert to dataframe
#     data_df = custom_data.get_data_as_dataframe()

#     # Make prediction
#     try:
#         prediction = prediction_pipeline.predict(data_df)
#         assert prediction is not None, "Prediction returned None"
#         print(f"Predicted Price: {prediction[0]}")
    
#     except customexception as e:
#         pytest.fail(f"Prediction failed with error: {e}")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from src.pipelines.prediction_pipeline import PredictPipeline, CustomData
from src.exception.exception import customexception

def test_predict_pipeline():
    # Initialize the prediction pipeline
    prediction_pipeline = PredictPipeline()
    
    # Create custom data for prediction
    custom_data = CustomData(carat=1.5, depth=61.5, table=57.0, x=6.0, y=6.1, z=3.5, cut='Ideal', color='G', clarity='VS1')
    
    # Convert to dataframe
    data_df = custom_data.get_data_as_dataframe()

    # Make prediction
    try:
        prediction = prediction_pipeline.predict(data_df)
        assert prediction is not None, "Prediction returned None"
        print(f"Predicted Price: {prediction[0]}")
    
    except customexception as e:
        pytest.fail(f"Prediction failed with error: {e}")
