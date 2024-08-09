import pandas as pd
import numpy as np
import os
import sys

from pathlib import Path
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception
from sklearn.model_selection import train_test_split


class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
        
class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config=DataIngestionConfig()
    
    def initate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            data=pd.read_csv(Path(os.path.join("notebooks\data","dimond.csv")))
            logging.info("I have read dataset")  
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info(" i have saved the raw dataset in artifact folder")
            
            logging.info("here i have performed train test split")
            
            train_data,test_data=train_test_split(data,test_size=0.25)
            logging.info("train test split completed")
            
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            
            logging.info("data ingestion part completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Error in data ingestion")
            raise customexception(e,sys)
    

