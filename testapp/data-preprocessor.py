import pandas as pd
import numpy as np

class BankingDataPreprocessor:
    def __init__(self, customer_df: pd.DataFrame, transaction_df: pd.DataFrame):
        self.customer_df = customer_df
        self.transaction_df = transaction_df
        self.merged_df = self._merge_dataframes()
    
    def _merge_dataframes(self):
        merged_df = pd.merge(
            self.customer_df, 
            self.transaction_df, 
            on='custid', 
            how='inner'
        )
        return merged_df
    
    def get_cleaned_data(self):
        # Advanced cleaning and preprocessing
        cleaned_df = self.merged_df.copy()
        
        # Convert date columns
        date_columns = ['opendate', 'txndate', 'valuedate']
        for col in date_columns:
            cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
        
        # Handle missing values
        cleaned_df['amount'].fillna(cleaned_df['amount'].median(), inplace=True)
        
        return cleaned_df
