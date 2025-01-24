import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def train(self, train_array, test_array):
        try:
            logging.info('Split Training and test input data')
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])
            
            logging.info('Training the model is started')
            model = RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=True)
            model.fit(X_train, y_train)
            
            logging.info('Make predictions and evaluate the model')
            y_pred = model.predict(X_test)
            acc, pre, rec, f1_score, cm = evaluate_model(y_test, y_pred)
            print(f'Accuracy: {acc:2f}, Precision: {pre:2f}, Recall: {rec:2f}, F1 score: {f1_score:2f}')
            
            logging.info('Save the model trainer')
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= model)
            return f1_score
        except Exception as e:
            raise CustomException(e, sys)