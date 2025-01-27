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
        
def analyze_banking_data(df):
    """
    Comprehensive analysis of banking transaction data
    """
    # Convert date columns to datetime
    date_columns = ['account_open_date', 'last_debit_date', 'last_credit_date', 'date_of_birth']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%d-%b-%y', errors='coerce')
    
    # Separate personal and non-personal accounts
    personal_accounts = df[df['date_of_birth'].notna()]
    non_personal_accounts = df[df['date_of_birth'].isna()]
    
    # Calculate age for personal accounts
    personal_accounts['age'] = (pd.Timestamp.now() - personal_accounts['date_of_birth']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    
    # Account status analysis
    account_status = {
        'total_accounts': len(df),
        'personal_accounts': len(personal_accounts),
        'non_personal_accounts': len(non_personal_accounts),
        'active_accounts': len(df[~df['account_inactive']]),
        'inactive_accounts': len(df[df['account_inactive']]),
        'negative_balance_accounts': len(df[df['account_balance'] < 0]),
        'zero_balance_accounts': len(df[df['account_balance'] == 0]),
    }
    
    # Service adoption analysis
    service_adoption = {
        'mobile_banking': len(df[df['mobile_banking']]) / len(df) * 100,
        'internet_banking': len(df[df['internet_banking']]) / len(df) * 100,
        'ac_service': len(df[df['account_service']]) / len(df) * 100,
    }
    
    # Account age analysis for personal accounts
    personal_accounts['account_age_years'] = (pd.Timestamp.now() - personal_accounts['account_open_date']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    
    age_demographics = {
        'avg_customer_age': personal_accounts['age'].mean(),
        'youngest_customer': personal_accounts['age'].min(),
        'oldest_customer': personal_accounts['age'].max(),
        'avg_account_age': personal_accounts['account_age_years'].mean(),
    }
    
    # Balance analysis
    balance_metrics = {
        'total_balance': df['account_balance'].sum(),
        'avg_balance': df['account_balance'].mean(),
        'max_balance': df['account_balance'].max(),
        'min_balance': df['account_balance'].min(),
        'total_negative_balance': df[df['account_balance'] < 0]['account_balance'].sum(),
    }
    
    # Recent activity analysis
    df['days_since_last_credit'] = (pd.Timestamp.now() - df['last_credit_date']).dt.total_seconds() / (24 * 60 * 60)
    df['days_since_last_debit'] = (pd.Timestamp.now() - df['last_debit_date']).dt.total_seconds() / (24 * 60 * 60)
    
    activity_metrics = {
        'avg_days_since_last_credit': df['days_since_last_credit'].mean(),
        'avg_days_since_last_debit': df['days_since_last_debit'].mean(),
        'accounts_with_recent_activity': len(df[df['days_since_last_credit'] <= 30]),
    }
    
    return {
        'account_status': account_status,
        'service_adoption': service_adoption,
        'age_demographics': age_demographics,
        'balance_metrics': balance_metrics,
        'activity_metrics': activity_metrics,
        'personal_accounts_df': personal_accounts,
        'non_personal_accounts_df': non_personal_accounts
    }

def generate_insights(analysis_results):
    """
    Generate natural language insights from analysis results
    """
    insights = []
    
    # Account composition insights
    total = analysis_results['account_status']['total_accounts']
    personal = analysis_results['account_status']['personal_accounts']
    non_personal = analysis_results['account_status']['non_personal_accounts']
    
    insights.append(f"Account Composition: Out of {total} total accounts, "
                   f"{personal} ({personal/total*100:.1f}%) are personal accounts and "
                   f"{non_personal} ({non_personal/total*100:.1f}%) are non-personal accounts.")
    
    # Service adoption insights
    service_data = analysis_results['service_adoption']
    insights.append(f"Digital Service Adoption: Mobile banking leads with {service_data['mobile_banking']:.1f}% adoption, "
                   f"followed by internet banking at {service_data['internet_banking']:.1f}%")
    
    # Balance insights
    balance = analysis_results['balance_metrics']
    insights.append(f"Balance Overview: Average account balance is {balance['avg_balance']:,.2f} NPR, "
                   f"with {analysis_results['account_status']['negative_balance_accounts']} accounts in negative balance "
                   f"totaling {balance['total_negative_balance']:,.2f} NPR")
    
    # Age demographics
    age = analysis_results['age_demographics']
    insights.append(f"Customer Demographics: Average customer age is {age['avg_customer_age']:.1f} years, "
                   f"ranging from {age['youngest_customer']:.1f} to {age['oldest_customer']:.1f} years")
    
    return insights

def generate_insights_prompt(insights):
    """
    Generate a prompt for the AI-LLM based on the insights
    """
    prompt = """
    You're a banking analyst.
    You have been asked to provide insights on the financial data of a bank.
    Here are the key insights from the analysis:
    """
    
    for i, insight in enumerate(insights, 1):
        prompt += f"\n{i}. {insight}"
    
    prompt += """
    Based on these insights, what are the key takeaways you can provide to the stakeholders?
    """
    
    return prompt

def classify_account_type(dob):
    """
    Classify account as personal/non-personal based on DOB pattern
    Returns True if personal account, False otherwise
    """
    if pd.isna(dob):
        return False
    return bool(re.match(r'[a-zA-Z0-9\-]', str(dob)))

def get_spending_summary(df):
    summary = df.groupby('Customer ID')['Amount'].agg(['sum', 'mean', 'count']).reset_index()
    summary.columns = ['Customer ID', 'Total Spending', 'Average Transaction', 'Transaction Count']
    return summary

def analyze_economic_sector(df):
    """
    Comprehensive analysis of economic sectors with extended metrics
    
    Parameters:
    df (pandas.DataFrame): Input dataframe with economic sector information
    
    Returns:
    dict: Comprehensive multi-dimensional sector analysis
    """
    # Prepare numeric columns for analysis
    df['is_negative_balance'] = df['account_balance'] < 0
    df['is_positive_balance'] = df['account_balance'] > 0
    
    # Sector Distribution and Basic Metrics
    sector_analysis = {}
    
    # Comprehensive Sector Breakdown
    sector_analysis['sector_distribution'] = {
        'count': df['economic_sector'].value_counts().to_dict(),
        'percentage': (df['economic_sector'].value_counts(normalize=True) * 100).to_dict()
    }
    
    # Balance Analysis
    balance_metrics = df.groupby('economic_sector').agg({
        'account_balance': [
            'count', 
            'sum', 
            'mean', 
            'min', 
            'max'
        ],
        'is_negative_balance': 'mean',
        'is_positive_balance': 'mean'
    })
    balance_metrics.columns = [
        'total_accounts_by_sector', 
        'total_balance_by_sector', 
        'average_balance_by_sector', 
        'minimum_balance_by_sector', 
        'maximum_balance_by_sector', 
        'negative_balance_ratio_by_sector', 
        'positive_balance_ratio_by_sector'
    ]
    sector_analysis['balance_metrics'] = balance_metrics.to_dict()
    
    # Branch and Service Analysis
    # branch_service_analysis = df.groupby(['economic_sector', 'user_branch', 'account_service']).size().unstack(fill_value=0)
    branch_service_analysis = df.groupby('economic_sector').agg({
        'bank_branch_account': 'nunique',
        'account_service': 'sum',
        'mobile_banking': 'mean',
        'account_balance': 'mean',
        'internet_banking': 'mean'
    })
    branch_service_analysis.columns = [
        'total_branches_by_sector',
        'total_account_services_by_sector',
        'average_mobile_banking_by_sector',
        'average_balance_by_sector',
        'average_internet_banking_by_sector'
    ]
    sector_analysis['branch_service_breakdown'] = branch_service_analysis.to_dict()
    
    # Inactivity Analysis
    inactivity_analysis = df.groupby('economic_sector')['account_inactive'].agg([
        ('inactive_accounts', 'sum'),
        ('inactive_ratio', 'mean')
    ])
    sector_analysis['inactivity_metrics'] = inactivity_analysis.to_dict()
    
    # Transaction Analysis
    transaction_metrics = df.groupby('economic_sector').agg({
        'transaction_amount': [
            'count', 
            'sum', 
            'mean', 
        ],
        'last_debit_date': 'max',
        'last_credit_date': 'max'
    })
    transaction_metrics.columns = [
        'total_transactions_by_sector', 
        'total_transaction_value_by_sector', 
        'average_transaction_by_sector', 
        'latest_debit_date', 
        'latest_credit_date'
    ]
    sector_analysis['transaction_metrics'] = transaction_metrics.to_dict()
    
    # Banking Service Penetration
    banking_services = df.groupby('economic_sector').agg({
        'mobile_banking': 'mean',
        'internet_banking': 'mean'
    }) * 100
    sector_analysis['banking_service_penetration'] = banking_services.to_dict()
    
    # KYC Status Analysis
    kyc_analysis = df.groupby('economic_sector')['kyc_status'].value_counts(normalize=True).unstack() * 100
    sector_analysis['kyc_status_distribution'] = kyc_analysis.to_dict()
    
    return sector_analysis

    # Example usage
    # analysis_results = analyze_economic_sector(df)

def analyze_customers(df):
    """
    Perform comprehensive customer analysis including demographics, 
    financial behavior, and service adoption patterns
    
    Parameters:
    df (pandas.DataFrame): Input dataframe with customer information
    
    Returns:
    dict: Comprehensive customer analysis results
    """
    customer_analysis = {}
    
    # 1. Customer Demographics Analysis
    customer_analysis['demographics'] = {
        'nationality_distribution': df['nationality'].value_counts().to_dict(),
        'nationality_percentage': (df['nationality'].value_counts(normalize=True) * 100).to_dict(),
        'residency_status': df['residency_status'].value_counts().to_dict(),
        'industry_distribution': df['customer_industry'].value_counts().to_dict(),
        'total_customers': len(df['customer_id'].unique())
    }
    
    # 2. Account Services Analysis
    customer_analysis['services'] = {
        'mobile_banking_adoption': (df['mobile_banking'].mean() * 100),
        'internet_banking_adoption': (df['internet_banking'].mean() * 100),
        'account_services': df['account_service'].value_counts().to_dict(),
        'account_types': df['account_type'].value_counts().to_dict(),
        'currency_distribution': df['currency_code_account'].value_counts().to_dict()
    }
    
    # # 3. Customer Age and Account Age Analysis
    # current_date = pd.Timestamp.now()
    # df['customer_age'] = (current_date - pd.to_datetime(df['date_of_birth'])).dt.days / 365
    # df['account_age'] = (current_date - pd.to_datetime(df['account_open_date'])).dt.days / 365
    
    # customer_analysis['age_metrics'] = {
    #     'customer_age_stats': {
    #         'mean': df['customer_age'].mean(),
    #         'median': df['customer_age'].median(),
    #         'min': df['customer_age'].min(),
    #         'max': df['customer_age'].max()
    #     },
    #     'account_age_stats': {
    #         'mean': df['account_age'].mean(),
    #         'median': df['account_age'].median(),
    #         'min': df['account_age'].min(),
    #         'max': df['account_age'].max()
    #     }
    # }
    
    # 4. Financial Behavior Analysis
    customer_analysis['financial_behavior'] = {
        'balance_stats': {
            'mean_balance': df['account_balance'].mean(),
            'median_balance': df['account_balance'].median(),
            'total_balance': df['account_balance'].sum(),
            'negative_balance_count': (df['account_balance'] < 0).sum(),
            'zero_balance_count': (df['account_balance'] == 0).sum(),
            'positive_balance_count': (df['account_balance'] > 0).sum()
        },
        'transaction_stats': {
            'avg_transaction_amount': df['transaction_amount'].mean(),
            'total_transaction_volume': df['transaction_amount'].sum(),
            'transaction_frequency': len(df) / len(df['customer_id'].unique())
        }
    }
    
    # 5. KYC and Compliance Analysis
    customer_analysis['compliance'] = {
        'kyc_status': df['kyc_status'].value_counts().to_dict(),
        'kyc_completion_rate': (df['kyc_status'].value_counts(normalize=True) * 100).to_dict(),
        'inactive_accounts': df['account_inactive'].sum(),
        'inactive_rate': (df['account_inactive'].mean() * 100)
    }
    
    # 6. Branch and Location Analysis
    customer_analysis['branch_metrics'] = {
        'branch_distribution': df['bank_branch_account'].value_counts().to_dict(),
        'transactions_by_branch': df.groupby('bank_branch_transaction')['transaction_amount'].agg([
            'count', 'sum', 'mean'
        ]).to_dict()
    }
    
    # # 7. Customer Activity Patterns
    # df['last_activity'] = pd.to_datetime(df[['last_debit_date', 'last_credit_date']].max(axis=1))
    # recent_date = df['last_activity'].max()
    # df['days_since_last_activity'] = (recent_date - df['last_activity']).dt.days
    
    # customer_analysis['activity_patterns'] = {
    #     'activity_age_stats': {
    #         'mean_days_inactive': df['days_since_last_activity'].mean(),
    #         'median_days_inactive': df['days_since_last_activity'].median(),
    #         'inactive_30_days': (df['days_since_last_activity'] > 30).sum(),
    #         'inactive_90_days': (df['days_since_last_activity'] > 90).sum()
    #     }
    # }
    
    # 8. Risk Metrics
    customer_analysis['risk_metrics'] = {
        'high_value_customers': (df['account_balance'] > df['account_balance'].quantile(0.95)).sum(),
        # 'dormant_accounts': (df['days_since_last_activity'] > 180).sum(),
        'kyc_pending_high_balance': ((df['kyc_status'] != 'COMPLETED') & 
                                   (df['account_balance'] > df['account_balance'].quantile(0.75))).sum()
    }
    
    return customer_analysis

def analyze_bank_branch(df):
    # bank_branch_transaction
    # Branch-wise Basic Statistics
    branch_stats = df.groupby('bank_branch_transaction').agg({
        'customer_id': 'count',
        'account_balance': ['mean', 'sum', 'min', 'max'],
        'local_currency_balance': ['mean', 'sum'],
        'transaction_amount': ['count', 'sum']
    }).reset_index()
    
    # Rename columns for clarity
    branch_stats.columns = [
        'Branch', 
        'Total_Customers', 
        'Avg_Account_Balance', 
        'Total_Account_Balance', 
        'Min_Account_Balance', 
        'Max_Account_Balance',
        'Avg_Local_Currency_Balance', 
        'Total_Local_Currency_Balance',
        'Total_Transactions', 
        'Total_Transaction_Amount'
    ]
    
    # Customer Segmentation by Branch
    customer_segmentation = df.groupby(['bank_branch_transaction', 'economic_sector']).size().unstack(fill_value=0)
    
    # Digital Banking Penetration
    digital_banking = df.groupby('bank_branch_transaction').agg({
        'mobile_banking': 'mean',
        'internet_banking': 'mean',
        'kyc_status': lambda x: (x == 'Completed').mean()
    }).reset_index()
    digital_banking.columns = ['Branch', 'Mobile_Banking_Rate', 'Internet_Banking_Rate', 'KYC_Compliance_Rate']
    
    # Transaction Characteristics
    transaction_analysis = df.groupby('bank_branch_transaction').agg({
        'transaction_amount': ['mean', 'std'],
        'transaction_code': 'nunique'
    }).reset_index()
    transaction_analysis.columns = ['Branch', 'Avg_Transaction', 'Transaction_Std_Dev', 'Unique_Transaction_Types']
    
    # Inactive Account Analysis
    inactive_accounts = df.groupby('bank_branch_transaction').agg({
        'account_inactive': lambda x: (x == True).mean()
    }).reset_index()
    inactive_accounts.columns = ['Branch', 'Inactive_Account_Ratio']
    
    # Merge all analyses
    comprehensive_analysis = branch_stats.merge(
        digital_banking, on='Branch'
    ).merge(
        transaction_analysis, on='Branch'
    ).merge(
        inactive_accounts, on='Branch'
    )
    
    return {
        'branch_statistics': branch_stats,
        'customer_segmentation': customer_segmentation,
        'comprehensive_analysis': comprehensive_analysis,
        'digital_banking_penetration': digital_banking,
        'transaction_characteristics': transaction_analysis,
        'inactive_accounts': inactive_accounts
    }

def generate_sector_analysis_prompt1(sector_analysis):
    """
    Generate a comprehensive LLM prompt with detailed sector-wise insights
    
    :param sector_analysis: Dictionary containing sector analysis data
    :return: Formatted detailed prompt string
    """
    # Calculate total accounts
    total_accounts = sum(sector_analysis['total_accounts'].values())
    
    # Sort sectors by total accounts
    sorted_sectors = sorted(
        sector_analysis['total_accounts'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )

    # Prepare prompt sections
    prompt = f"""COMPREHENSIVE ECONOMIC SECTOR PERFORMANCE ANALYSIS

1. SECTOR COMPOSITION
- Total Unique Economic Sectors: {len(sector_analysis['total_accounts'])}
- Total Accounts: {total_accounts:,}

DETAILED SECTOR BREAKDOWN:
{chr(10).join([
    f"  {sector}: "
    f"{count:,} accounts "
    f"({count/total_accounts*100:.2f}% of total), "
    f"Total Balance: ${sector_analysis['total_balance'][sector]:,.2f}"
    for sector, count in sorted_sectors
])}

2. FINANCIAL HEALTH METRICS
BALANCE CHARACTERISTICS:
{chr(10).join([
    f"  {sector}:"
    f"    Mean Balance: ${sector_analysis['mean_balance'][sector]:,.2f}"
    f"    Median Balance: ${sector_analysis['median_balance'][sector]:,.2f}"
    f"    Min Balance: ${sector_analysis['min_balance'][sector]:,.2f}"
    f"    Max Balance: ${sector_analysis['max_balance'][sector]:,.2f}"
    f"    Negative Balance Ratio: {sector_analysis['negative_balance_ratio'][sector]*100:.2f}%"
    f"    Positive Balance Ratio: {sector_analysis['positive_balance_ratio'][sector]*100:.2f}%"
    for sector, _ in sorted_sectors[:10]  
])}

3. RISK AND STABILITY INDICATORS
BALANCE DISTRIBUTION INSIGHTS:
{chr(10).join([
    f"  {sector}:"
    f"    Negative Balance Risk: {sector_analysis['negative_balance_ratio'][sector]*100:.2f}%"
    f"    Financial Stability Score: {(1 - sector_analysis['negative_balance_ratio'][sector]) * 100:.2f}"
    for sector, _ in sorted_sectors[:10]
])}

4. SECTOR FINANCIAL VULNERABILITY RANKING
{chr(10).join([
    f"  {sector}: Vulnerability Index = {sector_analysis['negative_balance_ratio'][sector]*100:.2f}"
    for sector, _ in sorted(
        sorted_sectors, 
        key=lambda x: sector_analysis['negative_balance_ratio'][x[0]], 
        reverse=True
    )[:10]
])}

5. STRATEGIC SECTORAL INSIGHTS
- Highest Total Balance Sector: {max(sector_analysis['total_balance'], key=sector_analysis['total_balance'].get)}
- Lowest Total Balance Sector: {min(sector_analysis['total_balance'], key=sector_analysis['total_balance'].get)}
- Most Stable Sector (Lowest Negative Balance Ratio): {
    min(sector_analysis['negative_balance_ratio'], 
        key=sector_analysis['negative_balance_ratio'].get)
}

ANALYSIS GENERATED: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
TOTAL SECTORS ANALYZED: {len(sector_analysis['total_accounts'])}
"""
    return prompt

def generate_sector_analysis_prompt(sector_analysis):
    """
    Generate a comprehensive LLM prompt using f-string with sector analysis data
    
    :param sector_analysis: Dictionary containing sector-wise analysis
    :return: Formatted prompt string
    """
    # Extract top 3 sectors by account count
    sorted_sectors = sorted(
        sector_analysis['sector_distribution']['count'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:3]

    prompt = f"""COMPREHENSIVE BANKING SECTOR PERFORMANCE ANALYSIS

1. SECTOR DISTRIBUTION
- Total Unique Economic Sectors: {len(sector_analysis['sector_distribution']['count'])}
- Top 3 Sectors by Account Volume:
{chr(10).join([f"  {i+1}. {sector}: {count} accounts ({percentage:.2f}%)" 
    for i, ((sector, count), (_, percentage)) in enumerate(zip(
        sorted_sectors, 
        sector_analysis['sector_distribution']['percentage'].items()
    ))])}

2. FINANCIAL HEALTH METRICS
Balance Overview:
{chr(10).join([f"  {sector}: "
    f"Total Balance: ${balance_metrics['total_balance'][sector]:,.2f}, "
    f"Mean Balance: ${balance_metrics['mean_balance'][sector]:,.2f}, "
    f"Negative Balance Ratio: {balance_metrics['negative_balance_ratio'][sector]*100:.2f}%, "
    f"Positive Balance Ratio: {balance_metrics['positive_balance_ratio'][sector]*100:.2f}%"
    for sector, balance_metrics in sector_analysis['balance_metrics'].items()])}

3. TRANSACTION DYNAMICS
{chr(10).join([f"  {sector}: "
    f"Total Transactions: {transaction_metrics['total_transactions'][sector]}, "
    f"Total Transaction Value: ${transaction_metrics['total_transaction_value'][sector]:,.2f}, "
    f"Mean Transaction: ${transaction_metrics['mean_transaction'][sector]:,.2f}"
    for sector, transaction_metrics in sector_analysis['transaction_metrics'].items()])}

4. DIGITAL BANKING ADOPTION
Mobile Banking Penetration:
{chr(10).join([f"  {sector}: {penetration:.2f}%" 
    for sector, penetration in sector_analysis['banking_service_penetration']['mobile_banking'].items()])}

Internet Banking Penetration:
{chr(10).join([f"  {sector}: {penetration:.2f}%" 
    for sector, penetration in sector_analysis['banking_service_penetration']['internet_banking'].items()])}

5. ACCOUNT INACTIVITY
{chr(10).join([f"  {sector}: "
    f"Inactive Accounts: {inactivity['inactive_accounts'][sector]}, "
    f"Inactivity Ratio: {inactivity['inactive_ratio'][sector]*100:.2f}%"
    for sector, inactivity in sector_analysis['inactivity_metrics'].items()])}

6. KYC COMPLIANCE
{chr(10).join([f"  {sector}:" + 
    chr(10).join([f"    {status}: {percentage:.2f}%" 
    for status, percentage in kyc_status.items()])
    for sector, kyc_status in sector_analysis['kyc_status_distribution'].items()])}

STRATEGIC INSIGHTS GENERATED: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    return prompt

    # Example usage
    # prompt_text = generate_sector_analysis_prompt(sector_analysis)
    # print(prompt_text)

def analyze_banking_data_a(df):
    """
    Comprehensive analysis of banking data using all available columns
    """
    # df = preprocess_data(df)
    print('Analyzing data...')
    df['last_credit_date'] = pd.to_datetime(df['last_credit_date'], format='%d-%b-%y', errors='coerce')
    df['last_debit_date'] = pd.to_datetime(df['last_debit_date'], format='%d-%b-%y', errors='coerce')
    # Create a copy to avoid modifying original data
    analysis_df = df.copy()
    
    # Classify accounts
    analysis_df['is_personal'] = analysis_df['date_of_birth'].apply(classify_account_type)
    
    # Basic account metrics
    account_metrics = {
        'total_accounts': len(df),
        'total_personal_accounts': analysis_df['is_personal'].sum(),
        'non_personal_accounts': len(df) - analysis_df['is_personal'].sum(),
        'active_accounts': len(df[~df['account_inactive']]),
        'inactive_accounts': len(df[df['account_inactive']]),
        'negative_balance_accounts': len(df[df['account_balance'] < 0]),
        'zero_balance_accounts': len(df[df['account_balance'] == 0])
    }
    
    # Nationality and Residency Analysis
    nationality_analysis = {
        'nationality_distribution': df['nationality'].value_counts().to_dict(),
        'residency_distribution': df['residency_status'].value_counts().to_dict(),
        'cross_border_customers': len(df[df['nationality'] != df['residency_status']])
    }

    # Sector Analysis
    sector_acbal = df.groupby('economic_sector')['account_balance'].agg(['mean', 'sum', 'min', 'max']).to_dict()
    sector_negative_acbal = df[df['account_balance'] < 0].groupby('economic_sector')['account_balance'].agg(['count', 'sum']).to_dict()
    sector_customers = df.groupby('economic_sector')['transaction_amount'].nunique().to_dict()
    sector_analysis1 = {
        # 'sector_by_numbers': df['economic_sector'].value_counts().to_dict(),
        'total_amount_by_sector': sector_acbal['sum'],
        'average_amounts_by_sector': sector_acbal['mean'],
        'minimum_amounts_by_sector': sector_acbal['min'],
        'maximum_amounts_by_sector': sector_acbal['max'],
        'sectors_with_negative_balance': sector_negative_acbal['count'],
        'inactive_accounts_by_sector': df[df['account_inactive']].groupby('economic_sector')['account_inactive'].count().to_dict(),
        # 'active_accounts_by_sector': df[df['active']].groupby('economic_sector')['active'].count().to_dict()
        }
    sector_analysis = analyze_economic_sector(df)

    # Industry Analysis
    industry_acbal = df.groupby('customer_industry')['account_balance'].agg(['mean', 'sum', 'min', 'max']).to_dict()
    industry_negative_acbal = df[df['account_balance'] < 0].groupby('customer_industry')['account_balance'].agg(['count', 'sum']).to_dict()
    industry_analysis = {
        # 'industry_by_numbers': df['customer_industry'].value_counts().to_dict(),
        'total_amount_by_industry': industry_acbal['sum'],
        'maximum_amounts_by_industry': industry_acbal['max'],
        'average_amounts_by_industry': industry_acbal['mean'],
        'minimum_amounts_by_industry': industry_acbal['min'],
        'industries_with_negative_balance': industry_negative_acbal['count'],
        'inactive_accounts_by_industry': df[df['account_inactive']].groupby('customer_industry')['account_inactive'].count().to_dict(),
        # 'active_accounts_by_industry': df[df['active']].groupby('customer_industry')['active'].count().to_dict()
    }

    # Category Analysis
    category_acbal = df.groupby('account_category')['account_balance'].agg(['mean', 'sum', 'min', 'max']).to_dict()
    category_negative_acbal = df[df['account_balance'] < 0].groupby('account_category')['account_balance'].agg(['count', 'sum']).to_dict()
    category_analysis = {
        # 'category_by_numbers': df['account_category'].value_counts().to_dict(),
        'total_amount_by_category': category_acbal['sum'],
        'maximum_amounts_by_category': category_acbal['max'],
        'average_amounts_by_category': category_acbal['mean'],
        'minimum_amounts_by_category': category_acbal['min'],
        'categories_with_negative_balance': category_negative_acbal['count'],
        'inactive_accounts_by_category': df[df['account_inactive']].groupby('account_category')['account_inactive'].count().to_dict(),
        # 'active_accounts_by_category': df[df['active']].groupby('account_category')['active'].count().to_dict()
    }
    
    # Branch and Account Type Analysis
    branch_analysis = {
        'total_branches': df['bank_branch_account'].nunique(),
        'branch_distribution': df['bank_branch_account'].value_counts().to_dict(),
        'branch_balance': df.groupby('bank_branch_account')['account_balance'].agg(['mean', 'sum', 'min', 'max']).to_dict(),
        'branch_negative_balance': df[df['account_balance'] < 0].groupby('bank_branch_account')['account_balance'].agg(['count', 'sum']).to_dict(),
        'branch_wise_inactive_account': df[df['account_inactive']].groupby('bank_branch_account')['account_inactive'].count().to_dict(),
        'branch_wise_active_account': df[~df['account_inactive']].groupby('bank_branch_account')['account_inactive'].count().to_dict(),
        'branch_wise_account_type': df.groupby('bank_branch_account')['account_type'].value_counts().unstack().to_dict(),
        'branch_wise_sector': df.groupby('bank_branch_account')['economic_sector'].value_counts().unstack().to_dict(),
        'branch_wise_category': df.groupby('bank_branch_account')['account_category'].value_counts().unstack().to_dict(),
        'branch_wise_nationalities': df.groupby('bank_branch_account')['nationality'].value_counts().unstack().to_dict(),
        'branch_wise_mobile_banking': df.groupby('bank_branch_account')['mobile_banking'].sum().to_dict(),
        'branch_wise_internet_banking': df.groupby('bank_branch_account')['internet_banking'].sum().to_dict(),
        'branch_wise_account_service': df.groupby('bank_branch_account')['account_service'].sum().to_dict(),
        'branch_wise_total_balance': df.groupby('bank_branch_account')['account_balance'].agg(['mean', 'sum']).to_dict(),
        'branch_wise_average_balance': df.groupby('bank_branch_account')['account_balance'].mean().to_dict(),
        'branch_wise_max_balance': df.groupby('bank_branch_account')['account_balance'].max().to_dict(),
        'branch_wise_min_balance': df.groupby('bank_branch_account')['account_balance'].min().to_dict(),

        # 'branch_actype_matrix': pd.crosstab(df['bank_branch_account'], df['account_type']).to_dict(),
        # 'branch_transaction_matrix': pd.crosstab(df['bank_branch_account'], df['account_balance']).to_dict(),
        # 'branch_sector_matrix': pd.crosstab(df['bank_branch_account'], df['economic_sector']).to_dict(),
        # 'branch_category_matrix': pd.crosstab(df['bank_branch_account'], df['account_category']).to_dict(),
        # 'branch_national_matrix': pd.crosstab(df['bank_branch_account'], df['nationality']).to_dict(),
        # 'branch_account_matrix': pd.crosstab(df['bank_branch_account'], analysis_df['is_personal']).to_dict(),
        # 'branch_inactive_matrix': pd.crosstab(df['bank_branch_account'], df['account_inactive']).to_dict(),
        # 'branch_negative_balance_matrix': pd.crosstab(df['bank_branch_account'], df['account_balance'] < 0).to_dict(),
        # 'branch_zero_balance_matrix': pd.crosstab(df['bank_branch_account'], df['account_balance'] == 0).to_dict(),
        # 'branch_kyc_matrix': pd.crosstab(df['bank_branch_account'], df['kyc_status']).to_dict(),
        # 'branch_mobile_banking_matrix': pd.crosstab(df['bank_branch_account'], df['mobile_banking']).to_dict(),
        # 'branch_internet_banking_matrix': pd.crosstab(df['bank_branch_account'], df['internet_banking']).to_dict(),
        # 'branch_ac_service_matrix': pd.crosstab(df['bank_branch_account'], df['acservice']).to_dict(),
        # 'branch_total_balance_matrix': pd.crosstab(df['bank_branch_account'], pd.cut(df['account_balance'], bins=[-np.inf, 0, 1000, 10000, 100000, np.inf], labels=['Negative', 'Low', 'Medium', 'High', 'Very High'])).to_dict()
        # 'branch_age_matrix': pd.crosstab(df['bank_branch_account'], pd.cut(df['age'], bins=[0, 30, 50, 70, 100], labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])).to_dict()
    }
    print('Branch analysis done')
    
    # Service Adoption Analysis
    service_adoption = {
        'mobile_banking_total_users': df['mobile_banking'].sum(),
        'mobile_banking_adoption_rate': (df['mobile_banking'].sum() / len(df) * 100),
        'mobile_banking_by_account_type': df.groupby('account_type')['mobile_banking'].mean().to_dict(),
        'mobile_banking_by_sector': df.groupby('economic_sector')['mobile_banking'].mean().to_dict(),
        'mobile_banking_by_industry': df.groupby('customer_industry')['mobile_banking'].mean().to_dict(),
        'mobile_banking_by_nationals': df.groupby('nationality')['mobile_banking'].mean().to_dict(),
        'mobile_banking_by_category': df.groupby('account_category')['mobile_banking'].mean().to_dict(),

        'internet_banking_total_users': df['internet_banking'].sum(),
        'internet_banking_adoption_rate': (df['internet_banking'].sum() / len(df) * 100),
        'internet_banking_by_account_type': df.groupby('account_type')['internet_banking'].mean().to_dict(),
        'internet_banking_by_sector': df.groupby('economic_sector')['internet_banking'].mean().to_dict(),
        'internet_banking_by_industry': df.groupby('customer_industry')['mobile_banking'].mean().to_dict(),
        'internet_banking_by_nationals': df.groupby('nationality')['internet_banking'].mean().to_dict(),
        'internet_banking_by_category': df.groupby('account_category')['internet_banking'].mean().to_dict(),

        'account_service_total_users': df['account_service'].sum(),
        'account_service_adoption_rate': (df['account_service'].sum() / len(df) * 100),
        'account_service_by_account_type': df.groupby('account_type')['account_service'].mean().to_dict(),
        'account_service_by_sector': df.groupby('economic_sector')['account_service'].mean().to_dict(),
        'account_service_by_industry': df.groupby('customer_industry')['mobile_banking'].mean().to_dict(),
        'account_service_by_nationals': df.groupby('nationality')['account_service'].mean().to_dict(),
        'account_service_by_category': df.groupby('account_category')['account_service'].mean().to_dict(),
    }
                
    # Balance Analysis
    balance_analysis = {
        'total_balance': df['account_balance'].sum(),
        'average_balance': df['account_balance'].mean(),
        'maximum_balance': df['account_balance'].max(),
        'minimum_balance': df['account_balance'].min(),
        'balance_by_account_type': df.groupby('account_type')['account_balance'].agg(['mean', 'sum', 'count']).to_dict(),
        'balance_by_sector': df.groupby('economic_sector')['account_balance'].agg(['mean', 'sum', 'count']).to_dict(),
        'balance_by_category': df.groupby('account_category')['account_balance'].agg(['mean', 'sum', 'count']).to_dict(),
        'minimim_balance_by_account_type': df.groupby('account_type')['account_balance'].min().to_dict(),
        'minimim_balance_by_sector': df.groupby('economic_sector')['account_balance'].min().to_dict(),
        'minimim_balance_by_category': df.groupby('account_category')['account_balance'].min().to_dict(),
        'maximum_balance_by_account_type': df.groupby('account_type')['account_balance'].max().to_dict(),
        'maximum_balance_by_sector': df.groupby('economic_sector')['account_balance'].max().to_dict(),
        'maximum_balance_by_category': df.groupby('account_category')['account_balance'].max().to_dict(),
        'negative_balance': {
            'count': len(df[df['account_balance'] < 0]),
            'total_amount': df[df['account_balance'] < 0]['account_balance'].sum(),
            'by_account_type': df[df['account_balance'] < 0].groupby('account_type').size().to_dict()
        }
    }
    
    # KYC and Compliance Analysis
    compliance_analysis = {
        'kyc_completion': {
            'kyc_completed': df['kyc_status'].sum(),
            'kyc_pending': len(df) - df['kyc_status'].sum(),
            'kyc_not_started': len(df[df['kyc_status'].isna()]),
            'kyc_completed_by_account_type': df.groupby('account_type')['kyc_status'].sum().to_dict(),
            'kyc_completed_by_sector': df.groupby('economic_sector')['kyc_status'].sum().to_dict(),
            'kyc_completed_by_national': df.groupby('nationality')['kyc_status'].sum().to_dict(),
            'kyc_completed_by_category': df.groupby('account_category')['kyc_status'].sum().to_dict(),
            'kyc_completed_by_branch': df.groupby('bank_branch_account')['kyc_status'].sum().to_dict(),
        }
    }

    # # Account age analysis for personal accounts
    # personal_accounts['account_age_years'] = (pd.Timestamp.now() - personal_accounts['account_open_date']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    
    # age_demographics = {
    #     'avg_customer_age': personal_accounts['age'].mean(),
    #     'youngest_customer': personal_accounts['age'].min(),
    #     'oldest_customer': personal_accounts['age'].max(),
    #     'avg_account_age': personal_accounts['account_age_years'].mean(),
    # }
    
    print('Service Compliance done')
    # Recent activity analysis
   
    # Calculate the number of days since the last credit
    df['days_since_last_credit'] = (datetime.now() - df['last_credit_date']).dt.days
    df['days_since_last_debit'] = (datetime.now() - df['last_debit_date']).dt.days
    activity_metrics = {
        'avg_days_since_last_credit': df['days_since_last_credit'].mean(),
        'avg_days_since_last_debit': df['days_since_last_debit'].mean(),
        'total_amount_credited_last_30_days': df[df['days_since_last_credit'] <= 30]['local_currency_balance'].sum(),
        'total_amount_debited_last_30_days': df[df['days_since_last_debit'] <= 30]['local_currency_balance'].sum(),
        }

    return {
        'account_metrics': account_metrics,
        'nationality_analysis': nationality_analysis,
        'sector_analysis': sector_analysis,
        'industry_analysis': industry_analysis,
        'category_analysis': category_analysis,
        'branch_analysis': branch_analysis,
        'service_adoption': service_adoption,
        'balance_analysis': balance_analysis,
        'compliance_analysis': compliance_analysis,
        'activity_metrics': activity_metrics,
    }

def generate_llm_prompt(analysis_results):
    """
    Generate a comprehensive prompt for LLM based on the analysis results
    """

    # print(type(analysis_results['compliance_analysis']))
    # print(analysis_results['compliance_analysis'])
    # # print(json.dumps(analysis_results['compliance_analysis'], indent=4))
    # print(convert_to_json(analysis_results['compliance_analysis']))
    # exit()
    
    # Identify Trends: Examine patterns across accounts, nationalities, branches, sectors, industries, categories, services, and account types. Highlight areas of growth or decline, supported by numerical evidence.
    
    system_message = """
    You are a professional financial analyst with expertise in data interpretation and insight generation.
    Your task is to analyze financial data and provide meaningful insights in a structured and clear manner.
    Focus on identifying trends, anomalies, and key performance metrics that can help the user make decisions.
    """
    
    system_message ="""
    You are a professional financial analyst with advanced expertise in data interpretation, trend analysis, and decision-making. 
    Your primary goal is to thoroughly analyze the provided financial data provided in JSON and extract actionable insights.

    Key Objectives:
    - Identify Trends: Examine patterns across JSON Keys. Highlight areas of growth or decline, supported by numerical evidence.
    - Comparing: Compare maximum entities provided. Use percentage and real numeric data.
    - Analysis: Consider figures like total, minimum, maximum, positive, negative, average, and as available with their value or percentage.
    - Spot Anomalies: Detect outliers or irregularities in the data and explain in short about their potential impact on overall performance.
    - Evaluate Performance Metrics: Assess key indicators and Recommend practical strategies to address gaps or inefficiencies, improve service delivery, and capitalize on growth areas.

    Requirements:
    - Include the value and name as possible for the metrics.
    - Conduct a detailed comparative analysis, incorporating relevant numerical figures, percentages, and ratios for a comprehensive evaluation.
    - Provide a structured, clear, and concise summary with actionable recommendations based on your findings.
    - Focus on identifying both short-term and long-term opportunities for optimization and growth.
    """
    
    prompt_footer = """
    Please analyze above data and provide actionable insights focusing on accounts, nationality, branches, sectors, industries, 
    categories, services, account type, and opportunities for improvement.

    Provide detail analysis with numerical figures and across dependable factors.
    """

    prompt_footer ="Leverage your expertise to deliver insights with depth and clarity. Do not provide large sentences except for recommendation and fraud cases."

    prompt_message = "Data summary is provided below, Your answer should be based on the context provided only:"

    prompt = f"""
    {system_message}
    {prompt_message}
    Banking Service Adoption:
    {convert_to_json(analysis_results['service_adoption'])}

    Sector Data:
    {convert_to_json(analysis_results['sector_analysis'])}
    
    Industry Data:
    {convert_to_json(analysis_results['industry_analysis'])}
    
    Category Data:
    {convert_to_json(analysis_results['category_analysis'])}

    {prompt_footer}
    """

    return prompt
    prompts = f"""
    {system_message}
    {prompt_message}
    1. Account Overview: 
    - Total Accounts: {analysis_results['account_metrics']['total_accounts']}
    - Total Personal Accounts: {analysis_results['account_metrics']['total_personal_accounts']}
    - Total Non-Personal Accounts: {analysis_results['account_metrics']['non_personal_accounts']}
    - Account Active/Inactive Ratio: {analysis_results['account_metrics']['active_accounts']}/{analysis_results['account_metrics']['inactive_accounts']}
    - Negative Balance Accounts: {analysis_results['account_metrics']['negative_balance_accounts']}
    - Zero Balance Accounts: {analysis_results['account_metrics']['zero_balance_accounts']}

    2. Customer Distribution:
    - Nationality Distribution: {analysis_results['nationality_analysis']['nationality_distribution']}
    - Residency Distribution: {analysis_results['nationality_analysis']['residency_distribution']}
    - Cross-Border Customers: {analysis_results['nationality_analysis']['cross_border_customers']}

    3. Industry and Sector Analysis:
    - Industry Distribution: {analysis_results['sector_analysis']['industry_by_numbers']}
    - Inactive Accounts by Sector: {analysis_results['sector_analysis']['inactive_accounts_by_sector']}
    - Inactive Accounts by Industry: {analysis_results['sector_analysis']['inactive_accounts_by_industry']}
    - Sector Wise Average Balance: {analysis_results['sector_analysis']['sector_wise_account_average']}
    - Sector Wise Total Balance: {analysis_results['sector_analysis']['sector_wise_account_sum']}
    - Industry Wise Negative Balance: {analysis_results['sector_analysis']['industry_wise_negative_account_balance']}
    - Sector Distribution: {analysis_results['sector_analysis']['sector_by_numbers']}
    - Category Distribution: {analysis_results['category_analysis']['category_distribution']}

    4. Branch Wise Analysis:
    - Total Branches: {analysis_results['branch_analysis']['total_branches']}
    - Branch Distribution: {analysis_results['branch_analysis']['branch_distribution']}
    - Branch Account Types: {analysis_results['branch_analysis']['branch_wise_account_type']}
    - Branch Wise Average Balance: {analysis_results['branch_analysis']['branch_wise_total_balance']['mean']}
    - Branch Wise Highest Balance: {analysis_results['branch_analysis']['branch_wise_total_balance']['sum']}
    - Branch Wise Minimum Balance: {analysis_results['branch_analysis']['branch_wise_min_balance']}
    - Branch Wise Maximum Balance: {analysis_results['branch_analysis']['branch_wise_max_balance']}
    - Branch Wise Active Accounts: {analysis_results['branch_analysis']['branch_wise_active_account']}
    - Branch Wise Inactive Accounts: {analysis_results['branch_analysis']['branch_wise_inactive_account']}
    - Branch Wise Negative Balance: {analysis_results['branch_analysis']['branch_negative_balance']}
    - Branch Wise Mobile Banking Users: {analysis_results['branch_analysis']['branch_wise_mobile_banking']}
    - Branch Wise Internet Banking Users: {analysis_results['branch_analysis']['branch_wise_internet_banking']}
    - Branch Wise Account Service Users: {analysis_results['branch_analysis']['branch_wise_account_service']}
    
    5. Adoptation of Digital Service:
    - Mobile Banking Total User: {analysis_results['service_adoption']['mobile_banking_total_users']}
    - Mobile Banking Adoption Rate: {analysis_results['service_adoption']['mobile_banking_adoption_rate']:.2f}%
    - Mobile Banking by Account Type: {analysis_results['service_adoption']['mobile_banking_by_account_type']}
    - Mobile Banking by Sector: {analysis_results['service_adoption']['mobile_banking_by_sector']}
    - Mobile Banking by National Origin: {analysis_results['service_adoption']['mobile_banking_by_national']}
    - Internet Banking Total Users: {analysis_results['service_adoption']['internet_banking_total_users']}
    - Internet Banking Adoption Rate: {analysis_results['service_adoption']['internet_banking_adoption_rate']:.2f}%
    - Internet Banking by Account Type: {analysis_results['service_adoption']['internet_banking_by_account_type']}
    - Internet Banking by Sector: {analysis_results['service_adoption']['internet_banking_by_sector']} 
    - Internet Banking by Category: {analysis_results['service_adoption']['internet_banking_by_category']} 
    - Internet Banking by National Origin: {analysis_results['service_adoption']['internet_banking_by_national']}
    - Account Service: {analysis_results['service_adoption']['account_service_adoption_rate']:.2f}%
    - Account Service by Account Type: {analysis_results['service_adoption']['account_service_by_account_type']}
    - Account Service by Sector: {analysis_results['service_adoption']['account_service_by_sector']}
    - Account Service by National Origin: {analysis_results['service_adoption']['account_service_by_national']}

    6. Financial Overview: 
    - Total Balance: NPR {analysis_results['balance_analysis']['total_balance']:,.2f}
    - Average Balance: NPR {analysis_results['balance_analysis']['average_balance']:,.2f}
    - Positive/Negative Balance Ratio: {analysis_results['account_metrics']['negative_balance_accounts']}/{analysis_results['account_metrics']['total_accounts']}
    - Minimum Balance: NPR {analysis_results['balance_analysis']['minimum_balance']:,.2f}
    - Maximum Balance: NPR {analysis_results['balance_analysis']['maximum_balance']:,.2f}
    - Minimum Balance by Sector: {analysis_results['balance_analysis']['balance_by_sector']['mean']}
    - Minimum Balance by Category: {analysis_results['balance_analysis']['balance_by_category']['mean']}
    - Total Balance by Account Type: {analysis_results['balance_analysis']['balance_by_account_type']['sum']}
    - Total Balance by Sector: {analysis_results['balance_analysis']['balance_by_sector']['sum']}
    - Total Balance by Category: {analysis_results['balance_analysis']['balance_by_category']['sum']}
  
    7. KYC Status:
    - KYC Completion Rate: {(analysis_results['compliance_analysis']['kyc_completion']['kyc_completed'] / analysis_results['account_metrics']['total_accounts'] * 100):.2f}%
    - KYC Pending: {analysis_results['compliance_analysis']['kyc_completion']['kyc_pending']}
    - KYC Completed: {analysis_results['compliance_analysis']['kyc_completion']['kyc_completed']}
    - KYC Not Started: {analysis_results['compliance_analysis']['kyc_completion']['kyc_not_started']}
    - KYC Completed By Account Type: {analysis_results['compliance_analysis']['kyc_completion']['kyc_completed_by_account_type']}
    - KYC Completed By Sector: {analysis_results['compliance_analysis']['kyc_completion']['kyc_completed_by_sector']}
    - KYC Completed By National Origin: {analysis_results['compliance_analysis']['kyc_completion']['kyc_completed_by_national']}
    - KYC Completed By Branch: {analysis_results['compliance_analysis']['kyc_completion']['kyc_completed_by_branch']}

    8. Recent Activity:
    - Average Days Since Last Credit: {analysis_results['activity_metrics']['avg_days_since_last_credit']:.2f}
    - Average Days Since Last Debit: {analysis_results['activity_metrics']['avg_days_since_last_debit']:.2f}
    - Total Amount Credited Last 30 Days: NPR {analysis_results['activity_metrics']['total_amount_credited_last_30_days']:,.2f}
    - Total Amount Debited Last 30 Days: NPR {analysis_results['activity_metrics']['total_amount_debited_last_30_days']:,.2f}
    {prompt_footer}
    """
#, list those insights with some numeric facts.
# Including compliance, financial overview, service adoption, operational metrices, business, customer and account overview.

# """
# Please provide:
# 1. Key insights about the customer base and their banking behavior
# 2. Risk areas that need attention
# 3. Growth opportunities by segment
# 4. Recommendations for:
#    - Improving service adoption
#    - Reducing inactive accounts
#    - Enhancing customer engagement
#    - Strengthening compliance
# """
    return prompt

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
    cols = pd.read_json(file_path)
    mapping = dict(
        zip(cols[old_column], cols[new_column])
    )
    return mapping

def preprocess_raw_dataframe(df_account, df_statement):
    """
    Preprocesses raw account and statement DataFrames by renaming columns and merging them.
    
    This function takes two DataFrames, `df_account` and `df_statement`, renames their columns
    based on predefined mappings, and merges them on the 'customer_id' column.
    
    Parameters:
    df_account (pd.DataFrame): The raw account DataFrame.
    df_statement (pd.DataFrame): The raw statement DataFrame.
    
    Returns:
    pd.DataFrame: A merged DataFrame with renamed columns.
    """
    
    account_columns = pd.read_json(ACCOUNT_COLUMNS)
    statement_columns = pd.read_json(STATEMENT_COLUMNS)
    
    # Convert DataFrame rows to a dictionary for mapping
    rename_account_mapping = dict(zip(account_columns["column"], account_columns["column_name"]))
    rename_statement_mapping = dict(zip(statement_columns["column"], statement_columns["column_name"]))

    # Rename the columns in the DataFrame
    df_account.rename(columns=rename_account_mapping, inplace=True)
    df_statement.rename(columns=rename_statement_mapping, inplace=True)

    # df_final = df_account.merge(df_statement, on="customer_id")
    df_final = pd.merge(df_account, df_statement, 
                     on=['customer_id'], 
                     how='inner', 
                     suffixes=('_account', '_transaction'))
    # df_final = pd.merge(df_account, df_statement, how='inner', on="customer_id")
    return df_final

def preprocess_data(df_account, df_statement):
    """
    Preprocesses the account and statement dataframes by performing several operations such as
    mapping codes to descriptive names, converting data types, and handling missing values.
    
    Args:
        df_account (pd.DataFrame): DataFrame containing account data.
        df_statement (pd.DataFrame): DataFrame containing statement data.
    
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

    df = preprocess_raw_dataframe(df_account, df_statement)

    # mapping to be done: Branch, Industry, Sector, Account_type, Category
    branch_code_to_name = column_mapping(BRANCH,'Code','Desc')
    account_code_to_name = column_mapping(ACCOUNT_TYPE,'Code','Desc')
    industry_code_to_name = column_mapping(INDUSTRY,'Code','Desc')
    sector_code_to_name = column_mapping(SECTOR,'Code','Desc')
    category_code_to_name = column_mapping(CATEGORY,'Code','Desc')

    # Replace unwanted values other than numeric with 0000
    df['bank_branch_account'] = pd.to_numeric(df['bank_branch_account'], errors='coerce') #TEST
    df['bank_branch_transaction'] = pd.to_numeric(df['bank_branch_transaction'], errors='coerce') #TEST
    df['account_type'] = pd.to_numeric(df['account_type'], errors='coerce')
    df['customer_industry'] = pd.to_numeric(df['customer_industry'], errors='coerce')
    df['economic_sector'] = pd.to_numeric(df['economic_sector'], errors='coerce')
    df['account_category'] = pd.to_numeric(df['account_category'], errors='coerce')

    # convert all data of column to some specific type
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].astype('int32')

    # Map numeric codes with strings
    df["bank_branch_account"] = df["bank_branch_account"].map(branch_code_to_name).fillna('BBRANCH')
    df["bank_branch_transaction"] = df["bank_branch_transaction"].map(branch_code_to_name).fillna('BBRANCH')
    df["account_type"] = df["account_type"].map(account_code_to_name).fillna('AACCOUNT')
    df["customer_industry"] = df["customer_industry"].map(industry_code_to_name).fillna('IINDUSTRY')
    df["economic_sector"] = df["economic_sector"].map(sector_code_to_name).fillna('SSECTOR')
    df["account_category"] = df["account_category"].map(category_code_to_name).fillna('CCATEGORY')
    
    return df

# Example usage function
def run_analysis(df):
    analysis_results = analyze_banking_data_a(df)       # analysis_results = analyze_banking_data(df)
    prompt = generate_llm_prompt(analysis_results)
    return analysis_results, prompt.strip()

def query_ai_llm(model, prompt):
    """
    
    """
    try:
        response = request.post("http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
            }
        )
        return response.json()['response']
    except Exception as e:
        print(f"Error querying AI-LLM: {e}")
        return "Error generating response"
        
if __name__ == "__main__":
    date_time_format = "%Y-%m-%d %H:%M:%S"
    # Prepare file paths
    if not os.path.exists('output'):
        os.makedirs('output')
    if not os.path.exists('input'):
        os.makedirs('input')

    # Define/Choose the LLM models to use
    models = ['gemma:2b', 'gemma2:latest', 'phi4:latest', 'llama3.2:latest','deepseek-r1:8b','qwen2.5-coder:7b']
    model_name = models[-1]
    
    # >> TEST
    # Load the data
    df_account = pd.read_csv(ACCOUNT_PATH)  # AccountData
    df_statement = pd.read_csv(STATEMENT_PATH)
    df = preprocess_data(df_account, df_statement)
    # print(df.head(3))
    print(df.columns)
    # print(df['customer_name'].value_counts())
    # print(df['bank_branch_transaction'].value_counts())

    # Sector
    # analysis_results = analyze_economic_sector(df)
    # filename = f"{OUTPUT}sectoranalysis_{model_name.replace(':', '_')}.txt"
    
    # Branch
    # analysis_results = analyze_bank_branch(df)
    # # print(analysis_results)
    # filename = f"{OUTPUT}branchanalysis_{model_name.replace(':', '_')}.txt"
    # print(analysis_results.keys())

    # Branch
    analysis_results = analyze_customers(df)
    # print(analysis_results)
    filename = f"{OUTPUT}customeranalysis_{model_name.replace(':', '_')}.txt"
    print(analysis_results.keys())
    

    print(type(analysis_results))
    with open(filename, 'w') as f:
        # f.write(convert_to_json(analysis_results, indent=2).encode('utf-8'))
        f.write(str(analysis_results))
    print(f" -- Analysis results saved to {filename}")
    
    prompt = f"""
    You're an expert in analyzing financial data..
    You have been asked to analyze the data provided.

    Consider the context provided and generate insights based on the data:

    Customer Demographics:{analysis_results['demographics']}

    Customer Services:{analysis_results['services']}

    Customer Balances: {analysis_results['financial_behavior']}
    
    KYC and Inactive Accounts: {analysis_results['compliance']}
    
    Branch distribution and transaction: {analysis_results['branch_metrics']}
    
    Risk: {analysis_results['risk_metrics']}
    
    Provide a detailed analysis with numerical figures in currency NPR only, and actionable recommendations.
    """
    # print(prompt)

    response = query_ai_llm(model_name, prompt)
    current_datetime = datetime.now().strftime(date_time_format)
    print(f'< Writing LLM Analysis to file...{current_datetime}')
    # Create a filename with the current date and time
    filename = f"{OUTPUT}customer_analysis_{model_name.replace(':', '_')}_{current_datetime.replace(' ','_').replace(':','_')}.txt"
    with open(filename, 'w') as f:
        f.write(response)

    exit()

    # df = pd.read_csv(STMT_PATH) # Statements
    
    # TODO: Link Merge StmtData with AccountData
    # custid,name,national,resident,industry,sector,branch,actype,accountno,ccy,category,acbal,lcyacbal,mbservice,ibservice,acservice,kyc,inactive,mobileno,opendate,ldrdate,lcrdate,dob
    # 'customer_id', 'customer_name', 'nationality', 'residency_status','customer_industry', 'economic_sector', 'bank_branch_account',
    #    'account_type', 'account_number_account', 'currency_code_account','account_category', 'account_balance', 'local_currency_balance', 'mobile_banking', 'internet_banking', 'account_service', 'kyc_status',
    #    'account_inactive', 'mobile_number', 'account_open_date','last_debit_date', 'last_credit_date', 'date_of_birth', 'statement_id','account_number_transaction', 'bank_branch_transaction',
    #    'transaction_code', 'transaction_reference', 'transaction_description','transaction_date', 'value_date', 'currency_code_transaction','transaction_amount', 'local_currency_amount', 'exchange_rate','our_reference', 'system_id', 'override_status', 'input_user','input_datetime', 'authorized_user', 'authorization_datetime','user_branch'
    # Get analysis results and prompt
    # analysis, prompt = run_analysis(df) # Analyze bank data 'A' & Return Prompt
    now_datetime = datetime.now().strftime(date_time_format)
    response = query_ai_llm(model_name, prompt)
    print(response)
    # Convert the dictionary to a JSON string and write
    print('> Writing analysis to JSON file...')
    with open(ANALYSIS_PATH, 'wb') as file:
        file.write(convert_to_json(analysis, indent=2).encode('utf-8'))
        
    # Save prompt to a text file
    print('> Writing prompt to text file...')
    with open(PROMPT_PATH, 'w') as f:
        f.write(prompt)

    """
    # Check if Prompt already exists
    if os.path.exists(PROMPT_PATH) and os.path.getsize(PROMPT_PATH) > 0:
        print('< Reading prompt from text file...')
        with open(PROMPT_PATH, 'r') as f:
            prompt = f.read()
    else:
        # Run analysis and generate prompt
        df = pd.read_csv(INPUT_PATH)    
        print(df.head(5)) # Preview the first few rows
        
        # Get analysis results and prompt
        analysis, prompt = run_analysis(df)
        
        # Convert the dictionary to a JSON string and write
        print('> Writing analysis to JSON file...')
        with open(ANALYSIS_PATH, 'wb') as file:
            file.write(convert_to_json(analysis, indent=2).encode('utf-8'))
        
        # Save prompt to a text file
        print('> Writing prompt to text file...')
        with open(PROMPT_PATH, 'w') as f:
            f.write(prompt)
    """
    # Generate insights from analysis results
    print(f'> Generating insights using LLM...{model_name}')

    # # Get the current date and time
    current_datetime = datetime.now().strftime(date_time_format)
    # response = query_ai_llm(model_name, prompt)

    print(f'< Writing LLM Analysis to file...{current_datetime}')
    # Create a filename with the current date and time
    filename = f"output/analysis_{model_name.replace(':', '_')}_{current_datetime.replace(' ','_').replace(':','_')}.txt"
    with open(filename, 'w') as f:
        f.write(response)

    
    
    time_taken = datetime.strptime(current_datetime, date_time_format) - datetime.strptime(now_datetime, date_time_format)
    print(f" -- Analysis results saved to {filename} - took: {time_taken.total_seconds()} seconds")
    exit()

    '''
    Analyze the provided financial data of a bank in Nepal (currency: Nepalese Rupee - NPR) from a banking analyst's perspective. 
    The data encompasses branches, customer accounts, transactions, and services. 
    Deliver key insights and actionable recommendations. 
    Focus on identifying critical risks, opportunities for improvement, and potential areas of regulatory concern. 
    Prioritize insights related to the bank's negative balance, inactive accounts, industry concentration, branch performance, digital service adoption, and recent transaction activity.
    '''
    
    '''
    You're a banking analyst expert.
    You have been asked to analyze the financial data of a bank in Nepal.
    Currency of Nepal is Nepalese Rupee (NPR).
    
    The data includes information about bank with branches, customer accounts, transactions, and services.
    Your task is to provide key insights and recommendations based on the analysis.

    '''
    
    prompt = f"""
    You're an expert in analyzing financial data..
    You have been asked to analyze the data provided.
    
    {str(df.to_string())}

    Generate insights about:
    1. Distribution of transactions by category
    2. Recent transaction activity
    3. Account balances and trends
    4. Customer demographics and behavior
    5. Service adoption and usage
    
    Your task is to provide key insights and recommendations based on the analysis.
    """
    response = query_ai_llm(model_name, prompt)
    print(response)
    # << TEST
    exit()
