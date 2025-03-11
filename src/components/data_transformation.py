import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts",'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_obj = DataTransformationConfig()


    def get_data_transformer_obj(self):
        try:
            numerical_column = ['CreditScore',
                                "Age",
                                'Tenure',
                                'Balance',
                                'NumOfProducts',
                                'HasCrCard',
                                'IsActiveMember',
                                'EstimatedSalary']
            
            categorical_column = ["Geography","Gender"]

            
            num_pipeline =Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    #("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    #("scaler",StandardScaler())
                ]
            )

            logging.info(f"Categorical column: {categorical_column}")
            logging.info(f"Numerical column: {numerical_column}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_column),
                    ("cat_pipeline",cat_pipeline,categorical_column)
                ]
            )

            logging.info("preprocessor object created")

            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("data has been loaded for transformation.")

            logging.info("obtaining preprocessor object")

            target_column_name = "Exited"

            preprocessor_obj = self.get_data_transformer_obj()

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            #print(input_feature_train_df.head(5))
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(
            f"Applying preprocessing object on train dataframe and test dataframe"
                )

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            

            train_arr = np.c_[
                    input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                    input_feature_test_arr,np.array(target_feature_test_df)
            ]

            

            logging.info(f"Saved Preprocessing objects.")

            save_object(
                file_path=self.data_transformation_obj.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                    train_arr,
                    test_arr
                    )


        except Exception as e:
            raise CustomException(e,sys)