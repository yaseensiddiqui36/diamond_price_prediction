# This will include all the common functionalities required for this project ex. pickling
import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging


def save_object(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj,file_obj)    

    except Exception as e:
        raise CustomException(e,sys)
    
def model_evaluation(X_train,X_test,y_train,y_test,models_dict):
    try:
        report= {}

        for i in range(len(models_dict)):
            
            model = list(models_dict.values())[i]
            model_name = list(models_dict.keys())[i]
            logging.info(f'Currently Training: {model_name}')

            # Train model
            model.fit(X_train,y_train)

            # Predict dependent feature
            y_pred = model.predict(X_test)

            # Get R2 score for the trained model
            model_r2_score = r2_score(y_test,y_pred)
            model_mse = mean_squared_error(y_test,y_pred)
            model_mae = mean_absolute_error(y_test,y_pred)

            report[list(models_dict.keys())[i]] = [model_r2_score,model_mse,model_mae]
            logging.info(f'During the training: Model Name{model_name} R2 Score: {model_r2_score}')
            
        return report

    except Exception as e:
        raise CustomException(e,sys)