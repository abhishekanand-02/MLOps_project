# from flask import Flask, request, render_template
# from src.pipelines.prediction_pipeline import PredictPipeline, CustomData

# app = Flask(__name__)

# @app.route('/')
# def home_page():
#     return render_template("index.html")

# @app.route("/predict", methods=["GET", "POST"])
# def predict_datapoint():
#     if request.method == "GET":
#         return render_template("form.html")
#     else:
#         data = CustomData(
#             carat=float(request.form.get("carat")),
#             depth=float(request.form.get("depth")),
#             table=float(request.form.get("table")),
#             x=float(request.form.get("x")),
#             y=float(request.form.get("y")),
#             z=float(request.form.get("z")),
#             cut=request.form.get("cut"),
#             color=request.form.get("color"),
#             clarity=request.form.get("clarity")
#         )
#         final_data = data.get_data_as_dataframe()

#         # Initialize prediction pipeline
#         predict_pipeline = PredictPipeline()

#         # Make prediction
#         pred = predict_pipeline.predict(final_data)

#         # Round the predicted value
#         result = round(pred[0], 2)

#         # Return the result to the result page
#         return render_template("result.html", final_result=result)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000)


import logging
from flask import Flask, request, render_template
from src.pipelines.prediction_pipeline import PredictPipeline, CustomData

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    try:
        if request.method == "GET":
            return render_template("form.html")
        else:
            # Log the data received from the form
            logging.debug("Received form data: %s", request.form)

            # Input validation: Check if all required fields are numeric
            try:
                carat = float(request.form.get("carat"))
                depth = float(request.form.get("depth"))
                table = float(request.form.get("table"))
                x = float(request.form.get("x"))
                y = float(request.form.get("y"))
                z = float(request.form.get("z"))
            except ValueError as e:
                logging.error("Invalid input data: %s", e)
                return render_template("form.html", error="Please enter valid numeric values for all fields.")

            cut = request.form.get("cut")
            color = request.form.get("color")
            clarity = request.form.get("clarity")

            # Create the CustomData object and get data as a dataframe
            data = CustomData(
                carat=carat,
                depth=depth,
                table=table,
                x=x,
                y=y,
                z=z,
                cut=cut,
                color=color,
                clarity=clarity
            )
            final_data = data.get_data_as_dataframe()

            # Log the dataframe created for prediction
            logging.debug("Dataframe created for prediction: %s", final_data)

            # Initialize the prediction pipeline
            predict_pipeline = PredictPipeline()

            # Make prediction
            pred = predict_pipeline.predict(final_data)

            # Log the prediction result
            logging.debug("Prediction result: %s", pred)

            # Round the predicted value
            result = round(pred[0], 2)

            # Return the result to the result page
            return render_template("result.html", final_result=result)
    except Exception as e:
        logging.error("Error occurred during prediction: %s", e)
        return "An error occurred while making the prediction.", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
