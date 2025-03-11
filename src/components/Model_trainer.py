import os
import sys
from dataclasses import dataclass

#from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try :
            logging.info("split training and test input data")

            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )

            smote = SMOTE(random_state=42)
            
            X_train_sampled,y_train_sampled = smote.fit_resample(X_train,y_train)


            models = {
                "Random Forest" : RandomForestClassifier(),
                "Decision Tree" : DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                #"XGBClassifier": XGBClassifier(),# as compare to this,adaboost give us better recall
                #"CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier" : AdaBoostClassifier()
            }
            logging.info("Model Training Started...")
            model_report : dict=evaluate_models(X_train=X_train_sampled,y_train=y_train_sampled,X_test = X_test,y_test=y_test,models=models)
            
            logging.info("Model training completed")
            print(model_report)
            ## to get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name form dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]


            best_model = models[best_model_name]

            print(best_model_name)
            
            if best_model_score < 0.6:
                return "No best model found"
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            f1score = f1_score(y_test,predicted,average="weighted")

            return f1score
        
        except Exception as e:
            raise CustomException(e,sys)