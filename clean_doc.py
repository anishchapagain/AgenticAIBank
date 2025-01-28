import pandas as pd
import numpy as np
from datetime import datetime
import requests as request
import re
import json
import os

INPUT = 'input/'
OUTPUT = 'output/'
DATA ='data/'
FILE_TXT = '.txt'
FILE_JSON = '.json'
FILE_CSV = '.csv'

PROMPT_PATH = INPUT+'prompt'+FILE_TXT
ANALYSIS_PATH = INPUT+'analysis'+FILE_TXT

ACCOUNT_PATH = DATA+'AccountData'+FILE_CSV
STATEMENT_PATH = DATA+'StmtData'+FILE_CSV

ACCOUNT_COLUMNS = DATA+'account_column_names'+FILE_JSON
STATEMENT_COLUMNS = DATA+'statement_column_names'+FILE_JSON

BRANCH = DATA+'branch'+FILE_JSON
INDUSTRY = DATA+'Industry'+FILE_JSON
CATEGORY = DATA+'Category'+FILE_JSON
SECTOR = DATA+'Sector'+FILE_JSON
ACCOUNT_TYPE = DATA+'AccountType'+FILE_JSON
TXN_TYPE = DATA+'TxnType'+FILE_JSON

def prepare_for_json(obj):
        """Prepare Python objects for JSON serialization"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif pd.isna(obj):
            return None
        return obj

def convert_to_json(data_dict, indent=2):
        """Convert dictionary to JSON with custom handling"""
        try:
            # First pass: prepare all values for JSON serialization
            json_ready = {}
            
            def prepare_dict(d):
                if isinstance(d, dict):
                    return {k: prepare_dict(v) for k, v in d.items()}
                elif isinstance(d, (list, tuple)):
                    return [prepare_dict(item) for item in d]
                else:
                    return prepare_for_json(d)
            
            json_ready = prepare_dict(data_dict)
            
            # Convert to JSON
            return json.dumps(json_ready, indent=indent, ensure_ascii=False)
            
        except Exception as e:
            return f"Error converting to JSON: {str(e)}"

def clean_account_data(df):
    """
    Clean and standardize account data
    """
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    print(df.head(3))
    
    # convert all data of column to some specific type
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].astype('int32')

    # Clean date columns - handle various date formats
    default_dob = '1970-01-01'
    date_columns = ['account_open_date', 'last_debit_date', 'last_credit_date', 'date_of_birth']
    for col in date_columns:
        # Try multiple date formats
        df[col] = pd.to_datetime(df[col], format='%d-%b-%y', errors='coerce')
        # For any failed conversions, try different formats
        if col == 'date_of_birth': # There's no Gender Hence adding default dob to all missing values
            df[col] = df[col].fillna(default_dob)
        if col == 'account_open_date':
            df[col] = df[col].fillna(pd.Timestamp('today').date())
        if col == 'last_debit_date':
            df[col] = df[col].fillna(pd.Timestamp('today').date())
        if col == 'last_credit_date':
            df[col] = df[col].fillna(pd.Timestamp('today').date())
    
    print('here.....')
    print(df.info())
    print(df.head(21))
    print(df['date_of_birth'].value_counts())
    exit()

    # Clean numeric columns - handle special characters and convert to numeric
    numeric_columns = ['account_balance', 'local_currency_balance', 'account_number']
    for col in numeric_columns:
        if df[col].dtype == 'object':
            # Remove currency symbols, commas, and other special characters
            df[col] = df[col].astype(str).str.replace(r'[^\d\-\.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # convert all data of column to some specific type
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].astype('int32')

    # Convert boolean columns to proper boolean type
    boolean_columns = ['mobile_banking', 'internet_banking', 'account_service', 
                      'kyc_status', 'account_inactive']
    for col in boolean_columns:
        # Handle various representations of True/False
        df[col] = df[col].str.lower() if df[col].dtype == 'object' else df[col]
        df[col] = df[col].map({
            'true': True, 'yes': True, '1': True, 't': True, 'y': True,
            'false': False, 'no': False, '0': False, 'f': False, 'n': False
        })
    
    print('here.....')
    print(df.info())
    print(df.head(10))
    exit()
    
    # Clean and standardize categorical columns
    df['nationality'] = df['nationality'].str.upper().str.strip()
    df['residency_status'] = df['residency_status'].str.upper().str.strip()
    df['customer_industry'] = df['customer_industry'].apply(lambda x: 
        re.sub(r'\*+', '', str(x)).title().strip())  # Remove asterisks
    df['economic_sector'] = df['economic_sector'].apply(lambda x: 
        re.sub(r'\*+', '', str(x)).title().strip())
    df['bank_branch'] = df['bank_branch'].str.title().str.strip()
    df['account_type'] = df['account_type'].str.title().str.strip()
    df['currency_code'] = df['currency_code'].str.upper().str.strip()
    df['account_category'] = df['account_category'].str.title().str.strip()
    
    # Clean mobile number
    df['mobile_number'] = df['mobile_number'].apply(lambda x: 
        re.sub(r'[^\d]', '', str(x)) if pd.notnull(x) else '')
    
    # Handle missing values
    df['account_balance'] = df['account_balance'].fillna(0)
    df['local_currency_balance'] = df['local_currency_balance'].fillna(0)
    df['mobile_number'] = df['mobile_number'].fillna('')
    
    # Add derived columns
    df['account_age_days'] = (pd.Timestamp.now() - df['account_open_date']).dt.days
    df['customer_age'] = (pd.Timestamp.now() - df['date_of_birth']).dt.years
    
    # Clean customer_id - remove any non-numeric characters if it's meant to be numeric
    df['customer_id'] = pd.to_numeric(df['customer_id'].astype(str).str.replace(r'\D', '', regex=True), 
                                    errors='coerce')
    
    # Sort by customer_id for consistency
    df = df.sort_values('customer_id')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df
        
def classify_account_type(dob):
    """
    Classify account as personal/non-personal based on DOB pattern
    Returns True if personal account, False otherwise
    """
    if pd.isna(dob):
        return False
    return bool(re.match(r'[a-zA-Z0-9\-]', str(dob)))

def standardize_dob(df, dob_column='date_of_birth'):
    """
    Standardize date of birth format and handle missing values
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    dob_column (str): Name of the date of birth column
    
    Returns:
    pandas.Series: Standardized date of birth series
    """
    def process_date(date_str):
        if pd.isna(date_str) or str(date_str).strip() == '':
            return '01-Jan-1970'
            
        try:
            # Convert to string if not already
            date_str = str(date_str).strip()
            
            # Split the date string
            parts = date_str.split('-')
            if len(parts) != 3:
                return '01-Jan-1970'
                
            day, month, year = parts
            
            # Process year
            year = year.strip()
            if len(year) == 2:
                # Convert 2-digit year to 4-digit year
                year = '19' + year if int(year) >= 0 else '20' + year
            
            # Reconstruct the date string
            return f"{day}-{month}-{year}"
            
        except Exception as e:
            print(f"Error processing date: {date_str}, Error: {str(e)}")
            return '01-Jan-1970'
    
    # Apply the standardization
    standardized_dates = df[dob_column].apply(process_date)
    
    return standardized_dates

def column_mapping(file_path, old_column, new_column):
    """
    Maps codes to descriptive names for branch, industry, sector, account type, and category columns.
    """
    
    cols = pd.read_json(file_path)
    mapping = dict(
        zip(cols[old_column], cols[new_column])
    )
    print(f'Columns mapped: {file_path}')
    return mapping

def map_column(df, type):
    """
    Preprocesses the raw account and statement dataframes by renaming columns based on a mapping file.
    """
    if type == 'account':
        column_mapping_file = ACCOUNT_COLUMNS
    elif type == 'statement':
        column_mapping_file = STATEMENT_COLUMNS
    else:
        print("Invalid type specified. Please specify 'account' or 'statement'.")
        return None
    
    # Load the column mapping file
    columns = pd.read_json(column_mapping_file)

    # Convert DataFrame rows to a dictionary for mapping
    columns_mapping = dict(zip(columns["column"], columns["column_name"]))

    # Rename the columns in the DataFrame
    df.rename(columns=columns_mapping, inplace=True)
    print('Columns renamed')
    return df

def preprocess_data(df, type = 'account'):
    """
    Preprocesses the account and statement dataframes by performing several operations such as
    mapping codes to descriptive names, converting data types, and handling missing values.
    
    Args:
        df (pd.DataFrame): DataFrame containing data.
    
    Returns:
        pd.DataFrame: The preprocessed DataFrame with mapped codes, converted data types, and handled missing values.
    
    Steps:
        1. Preprocess raw dataframes.
        2. Print the first 10 rows and columns of the preprocessed dataframe for debugging.
        3. Map branch, industry, sector, account type, and category codes to descriptive names.
        4. Replace non-numeric values in specific columns with NaN.
        5. Convert float64 columns to float32 and int64 columns to int32.
        6. Map numeric codes to descriptive strings and fill missing values with default strings.
    """

    df = map_column(df, type)

    # Mapping to be done: Branch, Industry, Sector, Account_type, Category
    branch_code_to_name = column_mapping(BRANCH,'Code','Desc')
    account_code_to_name = column_mapping(ACCOUNT_TYPE,'Code','Desc')
    industry_code_to_name = column_mapping(INDUSTRY,'Code','Desc')
    sector_code_to_name = column_mapping(SECTOR,'Code','Desc')
    category_code_to_name = column_mapping(CATEGORY,'Code','Desc')

    # Replace unwanted values other than numeric with 0000
    df['bank_branch'] = pd.to_numeric(df['bank_branch'], errors='coerce') #TEST
    df['account_type'] = pd.to_numeric(df['account_type'], errors='coerce')
    df['customer_industry'] = pd.to_numeric(df['customer_industry'], errors='coerce')
    df['economic_sector'] = pd.to_numeric(df['economic_sector'], errors='coerce')
    df['account_category'] = pd.to_numeric(df['account_category'], errors='coerce')

    df["bank_branch"] = df["bank_branch"].map(branch_code_to_name).fillna('NullBRANCH')
    df["account_type"] = df["account_type"].map(account_code_to_name).fillna('NullACCOUNT')
    df["customer_industry"] = df["customer_industry"].map(industry_code_to_name).fillna('NullINDUSTRY')
    df["economic_sector"] = df["economic_sector"].map(sector_code_to_name).fillna('NullSECTOR')
    df["account_category"] = df["account_category"].map(category_code_to_name).fillna('NullCATEGORY')

    df = clean_account_data(df)
    return df
 
if __name__ == "__main__":
    date_time_format = "%Y-%m-%d %H:%M:%S"
    # Define/Choose the LLM models to use
    models = ['gemma:2b', 'gemma2:latest', 'phi4:latest', 'llama3.2:latest','deepseek-r1:8b','qwen2.5-coder:7b']
    model_name = models[-1]
    # Load the data
    df = pd.read_csv(ACCOUNT_PATH)  # AccountData 
    print(df.head(3))
    print(df.info())
    print(df.columns)
    # df_statement = pd.read_csv(STATEMENT_PATH,)
    df = preprocess_data(df, 'account')
    print(df.head(3))
    print(df.columns)
    print(df.info())