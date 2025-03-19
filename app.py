from flask import Flask, request, jsonify,render_template
import numpy as np
import pandas as pd
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict_churn():
    try:
        data = CustomData(
            CreditScore=int(request.form.get("CreditScore")),
            Geography=request.form.get("Geography"),
            Gender=request.form.get("Gender"),
            Age=int(request.form.get("Age")),
            Tenure=int(request.form.get("Tenure")),
            Balance=float(request.form.get("Balance")),
            NumOfProducts=int(request.form.get("NumOfProducts")),
            HasCrCard=int(request.form.get("HasCrCard")),
            IsActiveMember=int(request.form.get("IsActiveMember")),
            EstimatedSalary=float(request.form.get("EstimatedSalary"))
        )

        df = data.get_data_as_data_frame()
        
        predict_pipeline = PredictPipeline()

        pred, pred_proba = predict_pipeline.predict(df)
        #print(f"Prediction: {results}")
        print(pred)
        print(pred_proba)
        prediction = int(pred[0])  # Convert result to integer
        #confidence= abs(pred_proba[0][0] - 0.5)*2*100 # Probability of churn
        
    
        #prediction_text = "Customer Churn" if prediction == 1 else "Customer Not Churn"

        return jsonify({
            "Prediction": str(prediction),
            #"Confidence_Score": f"{confidence:.2f}%",
            #"Message": prediction_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)


"""
{
"CreditScore" : 430,
"Geography": "Germany",
"Gender":"Female",
"Age":38,
"Tenure":8,
"Balance":153058.64,
"NumOfProducts":1,
"HasCrCard":1,
"IsActiveMember":0,
"EstimatedSalary":99377.27
}

"""
