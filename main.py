from credit_risk import logger
from credit_risk.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline   
from credit_risk.pipeline.stage_02_data_validation import DataValidationTrainingPipeline    
from credit_risk.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline    
from credit_risk.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from credit_risk.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion  = DataIngestionTrainingPipeline()   
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:          
    logger.exception(e)
    raise e     

STAGE_NAME = "Data Validation Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataValidationTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e) 
    raise e

STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataTransformationTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<< \n\n x========x")  
except Exception as e:
    logger.exception(e) 
    raise e

STAGE_NAME = "Model Trainer Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = ModelTrainerTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<< \n\n x========x")  
except Exception as e:
    logger.exception(e) 
    raise e


STAGE_NAME = "Model Evaluation Stage"

try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    obj = ModelEvaluationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<< \n\nx========x")
except Exception as e:
    raise e 