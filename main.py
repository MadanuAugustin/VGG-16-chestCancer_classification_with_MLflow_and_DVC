
from src.cnnClassifier import logger
from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Exception_file.exception import CustomException
import sys



STAGE_NAME = 'data ingestion stage'

try:
    logger.info(f'---------{STAGE_NAME} started---------------')
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f'-------------{STAGE_NAME} completed--------------------')
except Exception as e:
    raise CustomException(e, sys)