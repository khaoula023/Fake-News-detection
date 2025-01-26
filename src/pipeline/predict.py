import sys
import pandas as pd
import os

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, text):
        try:
            logging.info("Loading artifacts")
            model_path = os.path.abspath('artifacts/model.pkl')
            transformer_path = os.path.abspath('artifacts/transformer.pkl')
            model = load_object(file_path=model_path)
            transformer = load_object(file_path=transformer_path)
            
            logging.info('Data transformation')
            data_scaled = transformer.transform(text)
            
            logging.info('Make Predictions')
            prediction = model.predict(data_scaled)
            label =  'Fake' if prediction[0] > 0.5 else 'Real'
            return label
            
        except Exception as e:
            raise CustomException(e, sys)
        
        
class CustomData:
    def __init__(self, news: str):
        self.news = news
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = { "text": [self.news]}
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomData(e, sys)

if __name__ == '__main__':
    data = CustomData('famous bollywood superstar shah rukh khan die follow accident')
    text = data.get_data_as_data_frame()
    obj = PredictPipeline()
    label = obj.predict(text)
    print(f'The news is: {label}')