from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import numpy as np
import pandas as pd

from src.logger import logging 
from src.exception import CustomException
from src.utils import save_object
import os , sys
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation starts')

            # Categorical & Numerial columns
            categorical_cols = ['cut','color','clarity']
            numerical_cols = ['carat','depth','table','x','y','z']

            # Define the custom ranking for the ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Data Transformation initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            logging.info('Data Transformation Completed')

            return preprocessor


        except Exception as e:
            logging.info("Exception Occured in Data Transformation")
            raise CustomException(e,sys)
        

        
    def initiate_data_transformation(self,train_data_path,test_data_path):

        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Train and test data created')
            logging.info(f'Train Data: {train_df.head().to_string()}')
            logging.info(f'Test Data: {test_df.head().to_string()}')

            logging.info('Obtaining Preprocessor Object')
            
            preprocessor_obj = self.get_data_transformation_object()

            # seperate the Independent and dependent features
            target = 'price'
            drop_features = ['id','price']
            X_train = train_df.drop(columns=drop_features,axis=1)
            y_train = train_df[target]
            X_test = test_df.drop(columns=drop_features,axis=1)
            y_test = test_df[target]
        
            logging.info('Independent and Dependent Features seperated')

            ## Transformation using preprocessor object
            X_train_scaled = preprocessor_obj.fit_transform(X_train)
            X_test_scaled = preprocessor_obj.transform(X_test)

            logging.info('preprocessor on training and testing data set completed')

            logging.info('Contatinating scaled independent features and target features')
            scaled_train_with_target_feature = np.c_[X_train_scaled,np.array(y_train)]
            scaled_test_with_target_feature = np.c_[X_test_scaled,np.array(y_test)]

            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessor_obj
            )
            logging.info('Proprocessor pickle saved')
            # print(f'Preprocessor Pickle File Address: {self.data_transformation_config.preprocessor_obj_file_path}')
            logging.info(f'scaled_train_data_with_target_feature: {pd.DataFrame(scaled_train_with_target_feature).head().to_string()}')

            return(
                 scaled_train_with_target_feature, 
                 scaled_test_with_target_feature, 
                 self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.info('Exception occured in the initiate_transformation')
            raise CustomException(e,sys)
