from credit_risk.config.configuration import ConfigurationManager 
from credit_risk.components.data_ingestion import DataIngestion
from credit_risk import logger

STAGE_NAME  = "Data Ingestion Stage"


#creating DataIngestion training pipeline
class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
    
    if __name__ == "__main__":
        try:
            logger.info(f"Starting {STAGE_NAME}...")
            obj = DataIngestionTrainingPipeline()
            obj.main()
            logger.info(f">>>>>> stage {STAGE_NAME} <<<<<< \n\n x==========x")
        except Exception as e:
            raise e
