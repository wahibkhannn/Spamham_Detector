from src.ml.model.s3_estimator import SpamhamDetector
import pandas as pd
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()  

# Now the environment variables are available
print(os.getenv("AWS_ACCESS_KEY_ID_ENV_KEY"))  # should print your key
print(os.getenv("AWS_SECRET_ACCESS_KEY_ENV_KEY"))
print(os.getenv("AWS_DEFAULT_REGION"))
# Replace with your actual bucket name and model file name
detector = SpamhamDetector(bucket_name="spam-detector-model1", model_path="best_model.pkl")

# Load the model from S3
model = detector.load_model()
print("Loaded model:", model)  # Should show your SpamhamDetectionModel instance

# Test predictions
test_df = pd.DataFrame({"text": ["Free entry in 2 a weekly competition!", "Hey bro, want to meet up?"]})
preds = detector.predict(test_df)
print("Predictions:", preds)  # Should give realistic spam/ham predictions
