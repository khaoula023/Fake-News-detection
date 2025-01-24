import os 
import sys
import pandas as pd
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    transformer_file_path = os.path.join('artifacts', 'transformer.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_transformer(self):
        logging.info('Entered the data transformation method')
        try: 
            transformer = TfidfVectorizer(stop_words='english', max_features=5000)
            return transformer
        except Exception as e:
            raise CustomException(e, sys)
        
    def transform(self, train_path, test_path):
        try:
            logging.info('Read train and test')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Preparing the features and the target columns')
            target = 'label'
            features_train_df = train_df.drop(columns=[target],axis=1)
            target_train_df=train_df[target]
            
            features_test_df=test_df.drop(columns=[target],axis=1)
            target_test_df=test_df[target]
            
            logging.info("Obtaining transformer object")
            transformer= self.get_transformer()
            
            logging.info("Applying transformer object on training dataframe and testing dataframe.")
            features_train_arr=transformer.fit_transform(features_train_df.squeeze()).toarray()
            features_test_arr=transformer.transform(features_test_df.squeeze()).toarray()
            
            logging.info("Reshaping target arrays ")
            target_train_arr = np.array(target_train_df).reshape(-1, 1)
            target_test_arr = np.array(target_test_df).reshape(-1, 1)

            logging.info('Concatenating features and target arrays')
            train_arr = np.c_[features_train_arr, target_train_arr]
            test_arr = np.c_[features_test_arr, target_test_arr]

            logging.info(f"Saved preprocessing object.")
            save_object(
                 file_path=self.data_transformation_config.transformer_file_path,
                 obj=transformer)
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.transformer_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)