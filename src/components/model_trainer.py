import numpy as np
import pandas as pd
from src.logger import logging 
from src.exception import CustomException
from src.utils import save_object
import os , sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from src.utils import model_evaluation


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts','dimnd_pric_prid_mdl.pkl')


class ModelTrainer:
    def __init__(self):
        self.trained_model_config = ModelTrainerConfig()

    def initiate_model_training(self,train_data, test_data):

        try:
            logging.info('Model training initiated')
            
            X_train,X_test,y_train,y_test = (
                train_data[:,:-1],
                test_data[:,:-1],
                train_data[:,-1],
                test_data[:,-1]
            )

            models = {
                
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'ElasticNet': ElasticNet(),
                'Decision Tree': DecisionTreeRegressor()
        

            }

            model_report:dict = model_evaluation(X_train,X_test,y_train,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Evaluation Report: {model_report}')

            #Select the model which has the best R_square value
            best_r2_score = max([i[0] for i in model_report.values()])

            # Best model Name & Object
            
            best_model_name = [key for key, values in model_report.items() if values[0] == best_r2_score][0]
            best_model_obj = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_r2_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_r2_score}')
            logging.info(f'Best model object : {best_model_obj}')

            save_object(

                file_path=self.trained_model_config.trained_model_path,
                obj= best_model_obj
            )

        except Exception as e:
            logging.info('Error occured in the initiate_model_training')
            raise CustomException(e,sys)
        