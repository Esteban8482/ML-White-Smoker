from src.data_ingestion import DataIngestor

def main():
    # Paths to data files
    train_path = 'data/raw/train.csv'
    test_path = 'data/raw/test.csv'
    processed_train_path = 'data/processed/processed_train.csv'
    processed_test_path = 'data/processed/processed_test.csv'
    
    # Ingestión de datos
    data_ingestor = DataIngestor(train_path, test_path)
    train_df, test_df = data_ingestor.load_data()
    data_ingestor.initial_preprocessing()
    data_ingestor.save_data(processed_train_path, processed_test_path)