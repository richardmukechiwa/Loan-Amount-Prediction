# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = DataTransformationConfig(
        data_path="path/to/data.csv", 
        model_path="path/to/model.pkl",
        root_dir="path/to/save"
    )
    transformer = DataTransformation(config)
    
    cleaned_data = transformer.data_cleaning()
    eda_data = transformer.exploratory_data_analysis(cleaned_data)
    transformed_data_path = transformer.feat_engineering(eda_data)
    transformer.train_test_splitting(transformed_data_path)