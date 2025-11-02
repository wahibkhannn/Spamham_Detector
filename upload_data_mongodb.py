import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pymongo import MongoClient
# from env_variable import MONGODB_URL_KEY
from src.constant.env_variable import MONGODB_URL_KEY


load_dotenv()  # loads .env file

csv_file_path="notebooks/spamham.csv"
df = pd.read_csv(csv_file_path)

mongo_uri = os.getenv(MONGODB_URL_KEY)
client = MongoClient(mongo_uri)

# Access DB and Collection
db = client['wahib_projects']
collection = db['spam_ham']

#upload data
data = df.to_dict(orient='records')
collection.insert_many(data)
print("Data uploaded successfully to MongoDB")
print(f'Inserted {len(data)} records into MongoDB collection.')
