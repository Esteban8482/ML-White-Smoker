import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
 
class DataIngestion:
    def __init__(self, train_path, test_path):
        """
        A class for initial data ingestion and processing.

        Args:
            train_path (str): Route to the training data file.
            test_path (str): Route to the testing data file.
        
        """
        
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
    
    def load_data(self):
        """
        Carga los datos de los archivos CSV de entrenamiento y testeo, luego los devuelve en 2 variables.
        
        """
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        return self.train_df, self.test_df
    
    def initial_preprocessing(self):
        """
        Realiza la limpieza y preparación inicial de los datos.
        
        """

        # Eliminar la columna 'id'
        self.train_df = self.train_df.drop(['id'], axis=1)
        self.test_df = self.test_df.drop(['id'], axis=1)
    
    def save_data(self, train_output_path, test_output_path):
        """
        Guarda los datos preprocesados en archivos CSV.
        
        """
        self.train_df.to_csv(train_output_path, index=False)
        self.test_df.to_csv(test_output_path, index=False)