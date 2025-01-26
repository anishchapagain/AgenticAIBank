import pandas as pd
import numpy as np

class BankingAnalyticsEngine:
    def __init__(self, preprocessor):
        self.preprocessed_data = preprocessor.get_cleaned_data()
    
    def get_customer_segments(self):
        # Advanced customer segmentation
        segments = self.preprocessed_data.groupby('industry').agg({
            'custid': 'count',
            'amount': ['mean', 'sum']
        }).reset_index()
        segments.columns = ['industry', 'customer_count', 'avg_transaction', 'total_transactions']
        return segments
    
    def get_currency_transactions(self):
        currency_analysis = self.preprocessed_data.groupby('ccy').agg({
            'amount': ['count', 'sum', 'mean']
        }).reset_index()
        currency_analysis.columns = ['ccy', 'transaction_count', 'total_amount', 'avg_amount']
        return currency_analysis
    
    def get_branch_performance(self):
        branch_metrics = self.preprocessed_data.groupby('branch').agg({
            'custid': 'nunique',
            'amount': ['count', 'sum', 'mean'],
            'txndate': ['min', 'max']
        }).reset_index()
        branch_metrics.columns = [
            'branch', 
            'unique_customers', 
            'total_transactions', 
            'total_amount', 
            'avg_transaction', 
            'first_transaction', 
            'last_transaction'
        ]
        return branch_metrics
