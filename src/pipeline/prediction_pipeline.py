from src.ml.model.s3_estimator import SpamhamDetector
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig , ModelTrainerConfig
from src.constant.training_pipeline import *
from src.entity.config_entity import training_pipeline_config
from src.entity.config_entity import Prediction_config, PredictionPipelineConfig

from src.entity.config_entity import DataTransformationConfig , ModelTrainerConfig
from src.logger import logging
from src.utils.main_utils import MainUtils

from src.exception import SpamhamException
import pandas as pd
import numpy as np
import sys

import logging
import sys
from pandas import DataFrame
import pandas as pd



        

# class PredictionPipeline:
#     def __init__(self):
#         self.utils = MainUtils()
        
        
#     def get_trained_model(self):
#         """
#         method: get_trained_model
        
#         objective: this method returns the best model which is pushed to s3 bucket. 

      

#         Raises:
#             SpamhamException: 

#         Returns:
#             model: latest trained model from s3 bucket
#         """
#         try:
#             prediction_config = PredictionPipelineConfig()
#             model = SpamhamDetector(
#                 bucket_name= prediction_config.model_bucket_name,
#                 model_path= prediction_config.model_file_name
#             )
                
#             return model
                
#         except Exception as e:
#             raise SpamhamException(e, sys) from e
        
    # def run_pipeline(self, input_data:list):
        
    #     """
    #     method: run_pipeline
        
    #     objective: run_pipeline method runs the whole prediction pipeline.

    #     Raises:
    #         SpamhamException: 
    #     """
    #     try:
    #         model = self.get_trained_model()
    #         prediction = model.predict(input_data)
    #         return prediction
            
    #     except Exception as e:
    #         raise SpamhamException(e, sys)



    # def run_pipeline(self, input_data: list):
    #     try:
    #         print("Loading model from S3...")
    #         # model = self.get_trained_model()
    #         model = self.model_handler.load_model()
    #         print("Model loaded successfully!")
    #         if not isinstance(input_data, pd.DataFrame):
    #             raise ValueError("input_data must be a pandas DataFrame")
            
    #         #changed with text to dataframe because of error - claudeai
    #         # df = pd.DataFrame([input_data]) if isinstance(input_data, list) else input_data
    #         # prediction = model.predict(df)
    #         prediction = model.predict(input_data)
    #         return prediction
    #     except Exception as e:
    #         raise SpamhamException(e, sys)
        

# class PredictionPipeline:
#     def __init__(self):
#         try:
#             config = PredictionPipelineConfig()
#             self.model_handler = SpamhamDetector(
#                 bucket_name=config.model_bucket_name,
#                 model_path=config.model_file_name
#             )
#         except Exception as e:
#             raise SpamhamException(f"Error initializing PredictionPipeline: {e}", sys)
#         self.vectorizer = self.utils.load_object(
#             r"C:\Users\NCB\OneDrive\Desktop\VS-Programs\Machine Learning\Spam-detection\src\artifact\10_14_2025_22_28_57\data_transformation\transformed_object\vectorizer.pkl")
#     def get_trained_model(self):
#         """
#         method: get_trained_model
        
#         objective: this method returns the best model which is pushed to s3 bucket. 

      

#         Raises:
#             SpamhamException: 

#         Returns:
#             model: latest trained model from s3 bucket
#         """
#         try:
#             prediction_config = PredictionPipelineConfig()
#             model = SpamhamDetector(
#                 bucket_name= prediction_config.model_bucket_name,
#                 model_path= prediction_config.model_file_name
#             )
                
#             return model
                
#         except Exception as e:
#             raise SpamhamException(e, sys) from e
        
#     def run_pipeline(self, input_data: pd.DataFrame):
#         try:
#             print("🔹 Loading model from S3...")
#             model = self.model_handler.load_model()
#             if model is None:
#                 raise SpamhamException("Model not loaded properly from S3.", sys)
#             print("✅ Model loaded successfully!")

#             print("🔹 Transforming input text...")
#             X_transformed = self.vectorizer.transform(input_data)  # << transform the text

#             print("🔹 Making prediction...")
#             prediction = model.predict(X_transformed)  # << pass transformed features
#             print(f"✅ Prediction result: {prediction}")
#             return prediction

#         except Exception as e:
#             raise SpamhamException(f"Error during prediction: {e}", sys)

class PredictionPipeline:
    def __init__(self):
        try:
            config = PredictionPipelineConfig()
            self.model_handler = SpamhamDetector(
                bucket_name=config.model_bucket_name,
                model_path=config.model_file_name
            )
        except Exception as e:
            raise SpamhamException(f"Error initializing PredictionPipeline: {e}", sys)
        
    def run_pipeline(self, input_data: pd.DataFrame):
        try:
            print("🔹 Loading model from S3...")
            model = self.model_handler.load_model()
            if model is None:
                raise SpamhamException("Model not loaded properly from S3.", sys)
            print("✅ Model loaded successfully!")

            print("🔹 Making prediction...")
            # The model already contains the preprocessing pipeline
            prediction = model.predict(input_data)
            print(f"✅ Prediction result: {prediction}")
            return prediction

        except Exception as e:
            raise SpamhamException(f"Error during prediction: {e}", sys)