import ollama
from datetime import datetime
import numpy as np
import pandas as pd


MODEL = "gemma:2b"
# MODEL = "phi4:latest"


# Replace 'your_file.csv' with your file's path
df = pd.read_csv('data/AccountData.csv')
df = df.dropna()  # Drop rows with missing values
# df.rename(columns={'old_name': 'new_name'}, inplace=True)
print(df.head())  # Preview the first few rows

# data_json = df.head(20).to_json(orient='records')
# data_json = df.to_json(orient='records')

def analyze_banking_data(df):
    """
    Comprehensive banking data analysis combining traditional metrics and LLM-ready insights
    
    Parameters:
    df (pandas.DataFrame): Banking transaction data
    
    Returns:
    dict: Analysis results and LLM-ready insights
    """
    # Convert date columns to datetime
    date_columns = ['opendate', 'ldrdate', 'lcrdate', 'dob']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%d-%b-%y')
    
    # Calculate derived metrics
    df['age'] = (datetime.now() - df['dob']).dt.days // 365
    df['account_age_years'] = (datetime.now() - df['opendate']).dt.days // 365
    
    # Basic statistical analysis
    analysis = {
        'customer_metrics': {
            'total_customers': len(df),
            'age_distribution': df['age'].describe().to_dict(),
            'average_balance': df['acbal'].mean(),
            'total_portfolio_value': df['lcyacbal'].sum(),
            'negative_balance_accounts': len(df[df['acbal'] < 0]),
        },
        'service_adoption': {
            'mobile_banking': (df['mbservice'] == True).mean() * 100,
            'internet_banking': (df['ibservice'] == True).mean() * 100,
            'account_services': (df['acservice'] == True).mean() * 100
        },
        'account_health': {
            'kyc_compliance_rate': (df['kyc'] == True).mean() * 100,
            'inactive_accounts': (df['inactive'] == True).mean() * 100
        },
        'segment_analysis': df.groupby('industry')['acbal'].agg([
            'count', 'mean', 'sum'
        ]).to_dict()
    }
    
    # Generate LLM-ready insights
    llm_insights = []
    
    # Customer segmentation insights
    age_segments = pd.cut(df['age'], bins=[0, 30, 50, 70, 100], labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])
    age_segment_stats = df.groupby(age_segments)['acbal'].agg(['mean', 'count']).to_dict()
    llm_insights.append({
        'topic': 'Age Segmentation',
        'data': age_segment_stats,
        'context': 'Analysis of account balances across different age groups'
    })
    
    # Digital adoption patterns
    digital_segments = df.groupby(['mbservice', 'ibservice']).size().to_dict()
    llm_insights.append({
        'topic': 'Digital Banking Adoption',
        'data': digital_segments,
        'context': 'Analysis of mobile and internet banking service adoption patterns'
    })
    
    # Account balance patterns
    balance_patterns = {
        'negative_balance_profiles': df[df['acbal'] < 0][['age', 'industry', 'account_age_years']].to_dict(),
        'high_balance_profiles': df[df['acbal'] > df['acbal'].quantile(0.9)][['age', 'industry', 'account_age_years']].to_dict()
    }
    llm_insights.append({
        'topic': 'Balance Patterns',
        'data': balance_patterns,
        'context': 'Analysis of customer profiles with notable balance patterns'
    })
    
    analysis['llm_insights'] = llm_insights
    return analysis

# Perform data analysis
# print(data_json)  # Preview the first few rows
print(df.describe()) # (include='all'))
# df['dob'] = pd.to_datetime(df['dob'], format='%d-%b-%y')
# print(df['actype'].value_counts())

# {
#     "columnname": "opendate",
#     "description": "Date the account was opened."
#   },
#   {
#     "columnname": "ldrdate",
#     "description": "Last debit transaction date"
#   },
#   {
#     "columnname": "lcrdate",
#     "description": "Last credit transaction date"
#   },
#   {
#     "columnname": "dob",
#     "description": "Customer's date of birth."
#   }

print(df[['opendate', 'ldrdate', 'lcrdate', 'dob']].describe())
print(df[['opendate', 'ldrdate', 'lcrdate', 'dob']].isnull().sum())

exit()
# print(type(data_json))  # Preview the first few rows

# Load data into context for proper feeding and effectiveness.
system_message = """
You're financial expert with professional knowledge about Python and Pandas. 
You have been asked to analyze the financial data of a company.
You have been provided with the following data.
"""

# prompt = system_message
# prompt = system_message + data_json
prompt = system_message + data_json + "\n What insights can you provide from the data?" # + explanation

# print(prompt)
# exit()

result = ollama.generate(model=MODEL, prompt=prompt)
print(result.done)

if result.done:
    print(result.response)
else:
    print(result.model + " .Please try again later.")