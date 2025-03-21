# Customer Churn Prediction

## Problem Statement
Customer churn is a critical issue in the banking industry, where retaining customers is essential for long-term profitability. Banks invest heavily in acquiring customers, and losing them to competitors can result in significant revenue loss.

The goal of this project is to develop a machine learning model that predicts whether a bank customer is likely to churn based on their historical data. By identifying potential churners early, banks can take proactive measures such as personalized offers, improved customer service, or better financial products to increase retention and customer satisfaction.

## Expected Outcomes:
* Analyze customer data to identify key factors influencing churn.
* Develop a classification model to predict customer churn.
* Improve the model’s accuracy using feature engineering and resampling techniques.
* Compare multiple ML models and evaluate their performance.
* Deploy the trained model as a web application for better accessibility.

## Project Structure:
📁 Customer-Churn-Prediction
|-- .github
|   |--worklows
|   |   |-- main.yaml
|-- artifacts
|   |-- raw.csv
|   |-- train.csv
|   |-- test.csv
|   |-- model.pkl
|   |-- preprocessor.pkl
│-- data
|   |-- Churn Modeling.csv      
│-- notebooks                   # Jupyter notebooks for EDA & model training
│-- src
|   |-- components
|   |   |-- data_ingestion.py
|   |   |-- data_transformation.py
|   |   |-- model_trainer.py
|   |-- pipeline
|   |   |-- prediction_pipeline.py
|   |-- exception.py
|   |-- logger.py
|   |-- utils.py
│-- templates                 
│   │-- index.html          
|-- .gitignore
|-- app.py
|-- Dockerfile
|-- Procfile
|-- README.md
|-- requirements.txt
|-- setup.py


## Steps to Solve the Problem
### Data Collection & Exploration: 
* Loaded the dataset and examined its structure.
* Performed Exploratory Data Analysis (EDA) to understand feature distributions.
* Identified missing values, outliers, and imbalanced classes.

### Data Preprocessing & Feature Engineering
* Handled missing values and outliers using appropriate techniques.
* Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
* Used One-Hot Encoding for categorical features to make them model-friendly.


### Model Training & Evaluation:
* Split the dataset into 80% training and 20% testing.
* Trained multiple ML models for this classification task:
    * Baseline Model: Logistic Regression
    * Advanced Models: K-Nearest Neighbors (KNN), Decision Trees, Random Forest, Gradient Boosting
* Compared models using evaluation metrics:
    * Accuracy, Precision, Recall, F1-Score, ROC-AUC Curve
* Selected the best-performing model based on results.

### Why This Model?
<!-- * After comparing different models, [Your Best Model] performed the best due to [reasons like better generalization, lower overfitting, etc.].
* This model provides actionable insights for businesses to proactively reduce customer churn. -->

## Web Application Development
To make model predictions more accessible, I developed a Flask-based web app with an interactive UI.
* Frontend: HTML, CSS, JavaScript
* Backend: Flask API (Handles user input and returns predictions)
Users can enter customer details and get an instant churn prediction through a user-friendly interface.

## Deployment Journey
1. Initial Deployment: AWS Cloud (EC2 + ECR + GitHub Actions CI/CD)
* Containerized the model using Docker.
* Deployed using AWS EC2 & Elastic Container Registry (ECR).
* Automated deployment using GitHub Actions CI/CD pipeline.
Due to cost concerns, the EC2 instance was terminated, but prediction results were recorded as proof.

![alt text](images/video.mp4)

2. Cost-Free Deployment: Render
To ensure free hosting, I redeployed the model using Render.

Live Demo: https://customer-churn-prediction-xrw0.onrender.com 

## Conclusion
This project showcases my ability to work on end-to-end machine learning workflows, from data processing to model deployment. It highlights my understanding of model selection, feature engineering, web development, and cloud deployment.

## Skills Demonstrated:
* Machine Learning & Model Selection
* Data Preprocessing & Feature Engineering
* Web App Development (Flask)
* Docker & Cloud Deployment

## Future Enhancements
* Improving model performance with deep learning techniques
* Incorporating real-time data streams
* Enhancing web app UI/UX for better user experience






