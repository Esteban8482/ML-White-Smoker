import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore 

# las columnas con nuevas features no las estan creando en df de testeo


class DataProcessor:
    def __init__(self, train_df, test_df):
        """
        Clase para procesar y transformar los datos.

        Args:
            train_df (pd.DataFrame): DataFrame con los datos de entrenamiento.
            test_df (pd.DataFrame): DataFrame con los datos de prueba.
        """
        
        self.train_df = train_df
        self.test_df = test_df
    
    def discretize_age(self):
        """Discretiza la columna de edad en grupos."""
        
        cuartiles = self.train_df['age'].quantile([0.25, 0.5, 0.75])
        self.train_df['age'] = self.train_df['age'].apply(lambda age: self._age_group(age, cuartiles))
        self.test_df['age'] = self.test_df['age'].apply(lambda age: self._age_group(age, cuartiles))
        
    def _age_group(self, age, cuartiles):
        """Retorna el grupo al que pertenece la edad basado en cuartiles."""
        
        if age <= cuartiles[0.25]:
            return 1
        elif age <= cuartiles[0.5]:
            return 2
        elif age <= cuartiles[0.75]:
            return 3
        else:
            return 4
    
    def discretize_height(self):
        """ Discretiza la columna de altura en percentiles. """
        
        percentiles = self.train_df['height(cm)'].quantile(np.linspace(0, 1, 4))
        self.train_df['height(cm)'] = self.train_df['height(cm)'].apply(lambda height: self._height_percentile(height, percentiles))
        self.test_df['height(cm)'] = self.test_df['height(cm)'].apply(lambda height: self._height_percentile(height, percentiles))
        
    def _height_percentile(self, height, percentiles):
        """ Asigna un grupo de percentil basado en la altura. """
        
        for i, perc in enumerate(percentiles):
            if height <= perc:
                return i
        return 4
    
    def discretize_weight(self):
        """ Discretiza la columna de peso en percentiles. """
        
        percentiles = self.train_df['weight(kg)'].quantile(np.linspace(0, 1, 4))
        self.train_df['weight(kg)'] = self.train_df['weight(kg)'].apply(lambda weight: self._weight_percentile(weight, percentiles))
        self.test_df['weight(kg)'] = self.test_df['weight(kg)'].apply(lambda weight: self._weight_percentile(weight, percentiles))
        
    def _weight_percentile(self, weight, percentiles):
        """ Asigna un grupo de percentil basado en el peso. """
        
        for i, perc in enumerate(percentiles):
            if weight <= perc:
                return i
        return 4
    
    def calculate_bmi(self):
        """ Calcula y agrega la columna de BMI. """
        
        self.train_df['height(m)'] = self.train_df['height(cm)'] / 100
        self.train_df['BMI'] = self.train_df['weight(kg)'] / (self.train_df['height(m)'] ** 2)
        self.train_df.loc[self.train_df['BMI'] < 12, 'BMI'] = 12
        self.train_df.drop('height(m)', axis=1, inplace=True)
        
        self.test_df['height(m)'] = self.test_df['height(cm)'] / 100
        self.test_df['BMI'] = self.test_df['weight(kg)'] / (self.test_df['height(m)'] ** 2)
        self.test_df.loc[self.test_df['BMI'] < 12, 'BMI'] = 12
        self.test_df.drop('height(m)', axis=1, inplace=True)
    
    def calculate_ldl_hdl_ratio(self):
        """Calcula y agrega la columna de relación LDL/HDL."""
        
        self.train_df['LDL/HDL'] = self.train_df['LDL'] / self.train_df['HDL']
        self.train_df['LDL/HDL'] = self.train_df['LDL/HDL'].clip(lower = 0.1, upper = 10)
        
        self.test_df['LDL/HDL'] = self.test_df['LDL'] / self.test_df['HDL']
        self.test_df['LDL/HDL'] = self.test_df['LDL/HDL'].clip(lower = 0.1, upper = 10)
        
    def normalize_data(self):
        """ Normaliza solo las columnas numéricas del DataFrame """
        
        # Determinar las columnas categóricas y numéricas
        categorical_columns = ['hearing(left)', 'hearing(right)', 'dental caries', 'smoking', 'age', 'height(cm)', 'weight(kg)']
        numerical_columns = [col for col in self.train_df.columns if col not in categorical_columns]
        
        scaler = MinMaxScaler()
        self.train_df[numerical_columns] = scaler.fit_transform(self.train_df[numerical_columns])
        self.test_df[numerical_columns] = scaler.transform(self.test_df[numerical_columns])
        
        self.train_df[numerical_columns] = self.train_df[numerical_columns].round(4)
        self.test_df[numerical_columns] = self.test_df[numerical_columns].round(4)

    
    def save_processed_data(self, output_path):
        """ Guarda los datos procesados en un archivo CSV. """
        self.train_df.to_csv(output_path, index = False)
    
    def process(self, output_path):
        """ Ejecuta todo el pipeline de procesamiento de datos. """
        
        self.discretize_age()
        self.discretize_height()
        self.discretize_weight()
        self.calculate_bmi()
        self.calculate_pulse_pressure()
        self.calculate_ldl_hdl_ratio()
        self.normalize_data()
        self.save_processed_data(output_path)