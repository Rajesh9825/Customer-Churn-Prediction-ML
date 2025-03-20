import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from src.logger import logging
from sklearn.metrics import f1_score,recall_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    


def load_object(file_path):
    try :
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    



def evaluate_models(X_train,y_train,X_test,y_test,models):
    try :
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            logging.info("Metrics Evaluation")
           
            train_model_score = f1_score(y_train,y_train_pred,average="weighted")
            
            test_model_score = f1_score(y_test,y_test_pred,average="weighted")

            report[list(models.keys())[i]] = test_model_score
            

        return report

    except Exception as e:
        raise CustomException(e,sys)
    



def load_data(file_path):
    data_path = file_path
    df = pd.read_csv(data_path)
    return df