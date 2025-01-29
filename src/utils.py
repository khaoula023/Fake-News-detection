import os
import sys
import numpy as np
import pandas as pd
import spacy
import dill
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys)
    
def balance(data):
    try:
        # Split the dataset by label
        df_minority = data[data['label'] == 0]  # Minority class (label=0)
        df_majority = data[data['label'] == 1]  # Majority class (label=1)

        # Perform undersampling (reduce majority class to the size of minority class)
        df_majority_undersampled = df_majority.sample(n=len(df_minority), random_state=42)

        # Combine the undersampled majority class with the minority class
        df_balanced = pd.concat([df_minority, df_majority_undersampled])

        # Shuffle the dataset (optional, but recommended for randomization)
        df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        return df
    except Exception as e:
        raise CustomException(e, sys)
    
def lemmatize_text(nlp,text):
    try:
        # Load SpaCy model
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluate_model(true, predicted):
    # Accuracy
    accuracy = accuracy_score(true, predicted)
    
    # Precision 
    precision = precision_score(true, predicted, average='weighted')
    
    # Recall 
    recall = recall_score(true, predicted, average='weighted')
    
    # F1-Score 
    f1 = f1_score(true, predicted, average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(true, predicted)
    
    return accuracy, precision, recall, f1, cm