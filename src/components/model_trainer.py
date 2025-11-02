# import sys
# from typing import List, Tuple
# import os
# from pandas import DataFrame
# import numpy as np

# from src.entity.config_entity import ModelTrainerConfig
# from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
# from sklearn.metrics import f1_score, precision_score, recall_score

# from src.exception import SpamhamException
# from src.logger import logging
# from src.utils.main_utils import MainUtils,load_numpy_array_data
# from neuro_mf  import ModelFactory
# from sklearn.linear_model import LogisticRegression



# class SpamhamDetectionModel:
#     def __init__(self, preprocessing_object: object, encoder_object:None, trained_model_object: object):
#         self.preprocessing_object = preprocessing_object
#         self.encoder_object = None

#         self.trained_model_object = trained_model_object

#     def predict(self, X: DataFrame) -> DataFrame:
#         logging.info("Entered predict method of srcTruckModel class")

#         try:
#             logging.info("Using the trained model to get predictions")

#             transformed_feature = self.preprocessing_object.transform(X)

#             logging.info("Used the trained model to get predictions")

#             return self.trained_model_object.predict(transformed_feature)

#         except Exception as e:
#             raise SpamhamException(e, sys) from e

#     def __repr__(self):
#         return f"{type(self.trained_model_object).__name__}()"

#     def __str__(self):
#         return f"{type(self.trained_model_object).__name__}()"


# class ModelTrainer:
#     def __init__(self, 
#                  data_transformation_artifact: DataTransformationArtifact,
#                  model_trainer_config: ModelTrainerConfig):
        
#         self.data_transformation_artifact = data_transformation_artifact
#         self.model_trainer_config = model_trainer_config
#         self.utils = MainUtils()


#     def initiate_model_trainer(self) -> ModelTrainerArtifact:
#         logging.info("Entered initiate_model_trainer method of ModelTrainer class")

#         try:

#             ## I have commented out these 3 lines 
#             ## because i edited the numpy dense array hstack thing

#             # train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
#             # test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
#             # x_train, y_train, x_test, y_test = train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1]
            
#             # Load sparse X and dense y separately (since saved as tuple)
#             x_train, y_train = self.utils.load_object(self.data_transformation_artifact.transformed_train_file_path)
#             x_test, y_test = self.utils.load_object(self.data_transformation_artifact.transformed_test_file_path)

            
#             model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
#             best_model_detail = model_factory.get_best_model(X=x_train,y=y_train,base_accuracy=self.model_trainer_config.expected_accuracy)
            
#             # preprocessing_obj = self.utils.load_object(file_path=self.data_transformation_artifact.transformed_vectorizer_object_file_path)
#             # # Transform the training features
#             # X_train_transformed = preprocessing_obj.transform(x_train)


#             # x_train is already transformed sparse matrix
#             X_train_transformed = x_train


#             from sklearn.naive_bayes import GaussianNB
#             if isinstance(best_model_detail.best_model, GaussianNB):
#                 # X_train_transformed = X_train_transformed.toarray()
#                 x_train_dense = x_train.toarray()
#                 y_train_pred = best_model_detail.best_model.predict(x_train_dense)
#             else:
#                  y_train_pred = best_model_detail.best_model.predict(x_train)
#             # Make predictions using the trained model
            




#             # Make predictions using the trained model
#             y_train_pred = best_model_detail.best_model.predict(X_train_transformed)
#             f1 = f1_score(y_train, y_train_pred)
#             precision = precision_score(y_train, y_train_pred)
#             recall = recall_score(y_train, y_train_pred)


#             print("\n=== TRAINING SCORES ===")
#             print(f"F1 Score: {f1:.4f}")
#             print(f"Precision: {precision:.4f}")
#             print(f"Recall: {recall:.4f}\n")


#             preprocessing_obj = self.utils.load_object(file_path=self.data_transformation_artifact.transformed_vectorizer_object_file_path)
#             # encoder_object = self.utils.load_object(file_path= self.data_transformation_artifact.transformed_encoder_object_file_path)
#             encoder_object =None
#             if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
#                             logging.info("No best model found with score more than base score")
#                             raise Exception("No best model found with score more than base score")
             
#             customer_segmentation_model = SpamhamDetectionModel(
#                 preprocessing_object=preprocessing_obj,
#                 encoder_object= None,
#                 trained_model_object=best_model_detail.best_model
#             )
#             logging.info("Spam Ham detection Model is created and saved.")
#             trained_model_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
#             os.makedirs(trained_model_path, exist_ok=True)
            
#             self.utils.save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=customer_segmentation_model
#             )
#             logging.info(f"Spam Ham detection Model is saved successfully at: {trained_model_path}")
#             # metric_artifact = ClassificationMetricArtifact(f1_score=0.8, precision_score=0.8, recall_score=0.9)
#             metric_artifact = ClassificationMetricArtifact(
#                 f1_score=f1,
#                 precision_score=precision,
#                 recall_score=recall
#             )
#             # print(train_accuracy, test_accuracy)

            
#             model_trainer_artifact = ModelTrainerArtifact(
#             trained_model_file_path=self.model_trainer_config.trained_model_file_path,
#             metric_artifact=metric_artifact,
#             )

#             logging.info("Model training completed successfully")
#             logging.info(f"Model trainer artifact: {model_trainer_artifact}")

#             return model_trainer_artifact

            

#         except Exception as e:
#             raise SpamhamException(e, sys) from e

import sys
from typing import List, Tuple
import os
from pandas import DataFrame
import numpy as np

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from sklearn.metrics import f1_score, precision_score, recall_score

from src.exception import SpamhamException
from src.logger import logging
from src.utils.main_utils import MainUtils, load_numpy_array_data
from neuro_mf import ModelFactory


class SpamhamDetectionModel:
    def __init__(self, preprocessing_object: object, encoder_object: None, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.encoder_object = None
        self.trained_model_object = trained_model_object

    def predict(self, X: DataFrame) -> np.ndarray:
        """Predict using the trained model"""
        logging.info("Entered predict method of SpamhamDetectionModel class")

        try:
            from src.constant.training_pipeline import FEATURE_COLUMN
            
            # Extract text column if DataFrame, otherwise use as is
            if isinstance(X, DataFrame):
                if FEATURE_COLUMN in X.columns:
                    text_data = X[FEATURE_COLUMN]
                else:
                    text_data = X.iloc[:, 0]  # Use first column
            else:
                text_data = X
            
            logging.info("Transforming input using preprocessing pipeline")
            # The preprocessing_object is a pipeline (lemmatizer + vectorizer)
            transformed_feature = self.preprocessing_object.transform(text_data)
            
            logging.info("Making predictions")
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise SpamhamException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:
    def __init__(self, 
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.utils = MainUtils()

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            # Load sparse X and dense y separately (saved as tuple)
            x_train, y_train = self.utils.load_object(self.data_transformation_artifact.transformed_train_file_path)
            x_test, y_test = self.utils.load_object(self.data_transformation_artifact.transformed_test_file_path)

            # Train model using ModelFactory
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            best_model_detail = model_factory.get_best_model(
                X=x_train, 
                y=y_train, 
                base_accuracy=self.model_trainer_config.expected_accuracy
            )
            
            # Make predictions on training data
            from sklearn.naive_bayes import GaussianNB
            if isinstance(best_model_detail.best_model, GaussianNB):
                x_train_dense = x_train.toarray()
                y_train_pred = best_model_detail.best_model.predict(x_train_dense)
            else:
                y_train_pred = best_model_detail.best_model.predict(x_train)
            
            # Calculate metrics
            f1 = f1_score(y_train, y_train_pred)
            precision = precision_score(y_train, y_train_pred)
            recall = recall_score(y_train, y_train_pred)

            print("\n=== TRAINING SCORES ===")
            print(f"F1 Score: {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}\n")

            # Load the preprocessing pipeline (lemmatizer + vectorizer)
            preprocessing_pipeline = self.utils.load_object(
                file_path=self.data_transformation_artifact.transformed_vectorizer_object_file_path
            )
            
            # Check if model meets expected accuracy
            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")
             
            # Create final model object
            spamham_model = SpamhamDetectionModel(
                preprocessing_object=preprocessing_pipeline,
                encoder_object=None,
                trained_model_object=best_model_detail.best_model
            )
            
            logging.info("Spam Ham detection Model is created")
            
            # Save the model
            trained_model_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(trained_model_path, exist_ok=True)
            
            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=spamham_model
            )
            logging.info(f"Spam Ham detection Model saved at: {self.model_trainer_config.trained_model_file_path}")
            
            # Create metric artifact
            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1,
                precision_score=precision,
                recall_score=recall
            )
            
            # Create model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )

            logging.info("Model training completed successfully")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise SpamhamException(e, sys) from e