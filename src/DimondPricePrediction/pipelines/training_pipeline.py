from src.DimondPricePrediction.components.data_ingestion import DataIngestion

import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

data_ingest_obj=DataIngestion()
train_data_path,test_data_path=data_ingest_obj.initate_data_ingestion()

 