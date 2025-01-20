import pandas as pd
import os
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
import logging
from src.exception.exception import customexception
from src.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")

        try:
            data_path = os.path.join(os.getcwd(), "src", "original_csv")
            
            logging.info(f"Reading the dataset from {data_path}")
            data = pd.read_csv(data_path)
            
            logging.info("Data loaded successfully")

            # Create directories for saving the data if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Save the raw dataset to artifact folder
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            logging.info("Performing train-test split")
            
            # Perform train-test split
            train_data, test_data = train_test_split(data, test_size=0.25)
            logging.info("Train-test split completed")

            # Save the train and test data
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            
            logging.info(f"Train data saved at {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved at {self.ingestion_config.test_data_path}")

            logging.info("Data ingestion completed successfully")
            
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {str(e)}")
            raise customexception(e)

if __name__ == "__main__":
    # Creating object of the DataIngestion class and initiating the data ingestion process
    obj = DataIngestion()
    obj.initiate_data_ingestion()
