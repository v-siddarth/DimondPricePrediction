import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception
from src.DimondPricePrediction.utils.utils import save_object

from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformation(self):
        logging.info("getting data transformation")
        try:
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer()),
                    ("scaler",StandardScaler())
                    ]
            )
            
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("ordinalencoder",OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories]))
                ]
            )
            
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_cols),
                    ('cat_pipeline',cat_pipeline,categorical_cols)
                ]
            )
            
            
            return preprocessor
            
        except Exception as e:
            logging.info("Error in getting data transformation")
            raise customexception(e,sys)    
    
    def initate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_data=pd.read_csv(train_data_path)
            test_data=pd.read_csv(test_data_path)
            
            logging.info("read train and test data complated")
            logging.info(f"train data \n{train_data.head().to_string()}")
            logging.info(f"test data \n{test_data.head().to_string()}")
            
            preprocessor_obj=self.get_data_transformation()
            
            target_coloumn_name="price"
            drop_columns=[target_coloumn_name,"id"]
            
            input_feature_train_data=train_data.drop(columns=drop_columns,axis=1)
            target_feature_train_data=train_data[target_coloumn_name]
            
            input_feature_test_data=test_data.drop(columns=drop_columns,axis=1)
            target_feature_test_data=test_data[target_coloumn_name]
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_data)
            
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_data)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_data)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return(
                train_arr,
                test_arr
            )
            
        except Exception as e:
            logging.info("Error in initate data transformation")
            raise customexception(e,sys)

