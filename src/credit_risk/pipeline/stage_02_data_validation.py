from credit_risk.config.configuration import ConfigurationManager   
from credit_risk.components.data_validation import DataValidation 
from credit_risk.entity.config_entity import DataValidationConfig
from credit_risk import logger


STAGE_NAME = "Data Validation Stage"

# create a pipeline
class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)  
        data_validation.validate_all_columns() 
        
        
if __name__ == "__main__":  
    try:
        logger.info(f">>>>>> starting {STAGE_NAME} <<<<<<")    
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME}  completed <<<<<<<\n\nx==========x")   
    except Exception as e:  
        logger.exception(e)     
        raise e         
        
    
        
        
   