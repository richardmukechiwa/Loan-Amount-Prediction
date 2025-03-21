from credit_risk.config.configuration import ConfigurationManager
from credit_risk.components.data_transformation import DataTransformation       
from credit_risk import logger
from pathlib import Path

STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]
                
            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config() 
                data_transformation = DataTransformation(config = data_transformation_config)
                # Perform data transformation steps
                cleaned_data = data_transformation.data_cleaning()
                analyzed_data = data_transformation.exploratory_data_analysis(cleaned_data)
                #transformed_data = data_transformation.feat_engineering(cleaned_data)
                train_data, test_data = data_transformation.train_test_splitting(analyzed_data)
            else:
                raise Exception("Your data schema is not valid")
        except Exception as e:
            print(e) 
            
if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed successfully <<<<< \n\n x========x")
    except Exception as e:
        logger.exception(e)
        raise e