from flask import Flask,request,jsonify
import numpy as np
import pandas as pd

from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

# Route for home page
@app.route('/')
def index():
    return "Hello Wolrd"


@app.route('/predict',methods=['POST'])
def predict_data():
    try:
        data = CustomData( CreditScore =int(request.form.get("CreditScore")) ,
            Geography=request.form.get("Geography"),
            Gender= request.form.get("Gender"),
            Age= int(request.form.get("Age")),
            Tenure=int(request.form.get("Tenure")),
            Balance=float(request.form.get("Balance")),
            NumOfProducts=int(request.form.get("NumOfProducts")),
            HasCrCard=int(request.form.get("HasCrCard")),
            IsActiveMember=int(request.form.get("IsActiveMember")),
            EstimatedSalary = int(request.form.get("EstimatedSalary"))
            )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(pred_df)

        return jsonify({"Prediction" : results[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)


