import os
import sys

import certifi
import pymongo

from src.constant.database import DATABASE_NAME
from src.constant.env_variable import MONGODB_URL_KEY

from src.exception import SpamhamException
from pathlib import Path
from dotenv import load_dotenv
import os

# ✅ Load .env using absolute path
env_path = Path(__file__).resolve().parents[2] / '.env'
load_dotenv(dotenv_path=env_path)
# print("Loaded MongoDB URL:", os.getenv(MONGODB_URL_KEY))

# mongo_db_url = os.getenv("MONGO_DB_URL")


ca = certifi.where()


class MongoDBClient:
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment key: {MONGODB_URL_KEY} is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:

            raise SpamhamException(e, sys)

           

