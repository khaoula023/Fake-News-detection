from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sys

from src.pipeline.predict import  CustomData, PredictPipeline
from src.exception import CustomException
from src.logger import logging

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('website.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    try:
        logging.info('The web application is started')
        if request.method=='GET':
            return render_template('website.html')
        
        else: 
            data = CustomData(news=request.form.get('News'))
            news = data.get_data_as_data_frame()
            print(news)
            
            predict_pipeline = PredictPipeline()
            label = predict_pipeline.predict(news)
            return render_template('website.html', results=label)
    except Exception as e:
        raise CustomException(e, sys)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug= True)    