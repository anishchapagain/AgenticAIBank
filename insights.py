import pandas as pd
import numpy as np
from datetime import datetime
import requests as request
import re
import json
import os

PROMPT_PATH = 'input/prompt.txt'
ANALYSIS_PATH = 'input/analysis.json'
INPUT_PATH = 'data/AccountData.csv'
STMT_PATH = 'data/StmtData.csv'

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
    date_columns = ['opendate', 'ldrdate', 'lcrdate', 'dob']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%d-%b-%y', errors='coerce')
    
    # Separate personal and non-personal accounts
    personal_accounts = df[df['dob'].notna()]
    non_personal_accounts = df[df['dob'].isna()]
    
    # Calculate age for personal accounts
    personal_accounts['age'] = (pd.Timestamp.now() - personal_accounts['dob']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    
    # Account status analysis
    account_status = {
        'total_accounts': len(df),
        'personal_accounts': len(personal_accounts),
        'non_personal_accounts': len(non_personal_accounts),
        'active_accounts': len(df[~df['inactive']]),
        'inactive_accounts': len(df[df['inactive']]),
        'negative_balance_accounts': len(df[df['acbal'] < 0]),
        'zero_balance_accounts': len(df[df['acbal'] == 0]),
    }
    
    # Service adoption analysis
    service_adoption = {
        'mobile_banking': len(df[df['mbservice']]) / len(df) * 100,
        'internet_banking': len(df[df['ibservice']]) / len(df) * 100,
        'ac_service': len(df[df['acservice']]) / len(df) * 100,
    }
    
    # Account age analysis for personal accounts
    personal_accounts['account_age_years'] = (pd.Timestamp.now() - personal_accounts['opendate']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    
    age_demographics = {
        'avg_customer_age': personal_accounts['age'].mean(),
        'youngest_customer': personal_accounts['age'].min(),
        'oldest_customer': personal_accounts['age'].max(),
        'avg_account_age': personal_accounts['account_age_years'].mean(),
    }
    
    # Balance analysis
    balance_metrics = {
        'total_balance': df['acbal'].sum(),
        'avg_balance': df['acbal'].mean(),
        'max_balance': df['acbal'].max(),
        'min_balance': df['acbal'].min(),
        'total_negative_balance': df[df['acbal'] < 0]['acbal'].sum(),
    }
    
    # Recent activity analysis
    df['days_since_last_credit'] = (pd.Timestamp.now() - df['lcrdate']).dt.total_seconds() / (24 * 60 * 60)
    df['days_since_last_debit'] = (pd.Timestamp.now() - df['ldrdate']).dt.total_seconds() / (24 * 60 * 60)
    
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

def analyze_banking_data_a(df):
    """
    Comprehensive analysis of banking data using all available columns
    """
    df = preprocess_data(df)
    print('Analyzing data...')
    df['lcrdate'] = pd.to_datetime(df['lcrdate'], format='%d-%b-%y', errors='coerce')
    df['ldrdate'] = pd.to_datetime(df['ldrdate'], format='%d-%b-%y', errors='coerce')
    # Create a copy to avoid modifying original data
    analysis_df = df.copy()
    
    # Classify accounts
    analysis_df['is_personal'] = analysis_df['dob'].apply(classify_account_type)
    
    # Basic account metrics
    account_metrics = {
        'total_accounts': len(df),
        'total_personal_accounts': analysis_df['is_personal'].sum(),
        'non_personal_accounts': len(df) - analysis_df['is_personal'].sum(),
        'active_accounts': len(df[~df['inactive']]),
        'inactive_accounts': len(df[df['inactive']]),
        'negative_balance_accounts': len(df[df['acbal'] < 0]),
        'zero_balance_accounts': len(df[df['acbal'] == 0])
    }
    
    # Nationality and Residency Analysis
    nationality_analysis = {
        'nationality_distribution': df['national'].value_counts().to_dict(),
        'residency_distribution': df['resident'].value_counts().to_dict(),
        'cross_border_customers': len(df[df['national'] != df['resident']])
    }

    sector_acbal = df.groupby('sector')['acbal'].agg(['mean', 'sum', 'min', 'max']).to_dict()
    sector_negative_acbal = df[df['acbal'] < 0].groupby('sector')['acbal'].agg(['count', 'sum']).to_dict()
    industry_negative_acbal = df[df['acbal'] < 0].groupby('industry')['acbal'].agg(['count', 'sum']).to_dict()

    # Industry and Sector Analysis
    sector_analysis = {
        'industry_by_numbers': df['industry'].value_counts().to_dict(),
        'sector_by_numbers': df['sector'].value_counts().to_dict(),
        'total_amount_by_sector': sector_acbal['sum'],
        'average_amounts_by_sector': sector_acbal['mean'],
        'minimum_amounts_by_sector': sector_acbal['min'],
        'maximum_amounts_by_sector': sector_acbal['max'],
        'sector_wise_negative_balances': sector_negative_acbal['count'],
        'total_industry_with_negative_balances': industry_negative_acbal['count'],
        'inactive_accounts_by_sector': df[df['inactive']].groupby('sector')['inactive'].count().to_dict(),
        'inactive_accounts_by_industry': df[df['inactive']].groupby('industry')['inactive'].count().to_dict()
    }

    # Category Analysis
    category_analysis = {
        'category_distribution': df['category'].value_counts().to_dict(),
        'category_wise_balance': df.groupby('category')['acbal'].agg(['mean', 'sum']).to_dict(),
        'category_wise_negative_balance': df[df['acbal'] < 0].groupby('category')['acbal'].agg(['count', 'sum']).to_dict(),
        'category_wise_inactive_account': df[df['inactive']].groupby('category')['inactive'].count().to_dict()
        # 'sector_distribution': df['sector'].value_counts().to_dict(),
        # 'industry_sector_matrix': pd.crosstab(df['industry'], df['sector']).to_dict()
    }
    
    # Branch and Account Type Analysis
    branch_analysis = {
        'total_branches': df['branch'].nunique(),
        'branch_distribution': df['branch'].value_counts().to_dict(),
        'branch_balance': df.groupby('branch')['acbal'].agg(['mean', 'sum', 'min', 'max']).to_dict(),
        'branch_negative_balance': df[df['acbal'] < 0].groupby('branch')['acbal'].agg(['count', 'sum']).to_dict(),
        'branch_wise_inactive_account': df[df['inactive']].groupby('branch')['inactive'].count().to_dict(),
        'branch_wise_active_account': df[~df['inactive']].groupby('branch')['inactive'].count().to_dict(),
        'branch_wise_account_type': df.groupby('branch')['actype'].value_counts().unstack().to_dict(),
        'branch_wise_sector': df.groupby('branch')['sector'].value_counts().unstack().to_dict(),
        'branch_wise_category': df.groupby('branch')['category'].value_counts().unstack().to_dict(),
        'branch_wise_national': df.groupby('branch')['national'].value_counts().unstack().to_dict(),
        'branch_wise_mobile_banking': df.groupby('branch')['mbservice'].sum().to_dict(),
        'branch_wise_internet_banking': df.groupby('branch')['ibservice'].sum().to_dict(),
        'branch_wise_account_service': df.groupby('branch')['acservice'].sum().to_dict(),
        'branch_wise_total_balance': df.groupby('branch')['acbal'].agg(['mean', 'sum']).to_dict(),
        'branch_wise_average_balance': df.groupby('branch')['acbal'].mean().to_dict(),
        'branch_wise_max_balance': df.groupby('branch')['acbal'].max().to_dict(),
        'branch_wise_min_balance': df.groupby('branch')['acbal'].min().to_dict(),

        # 'branch_actype_matrix': pd.crosstab(df['branch'], df['actype']).to_dict(),
        # 'branch_transaction_matrix': pd.crosstab(df['branch'], df['acbal']).to_dict(),
        # 'branch_sector_matrix': pd.crosstab(df['branch'], df['sector']).to_dict(),
        # 'branch_category_matrix': pd.crosstab(df['branch'], df['category']).to_dict(),
        # 'branch_national_matrix': pd.crosstab(df['branch'], df['national']).to_dict(),
        # 'branch_account_matrix': pd.crosstab(df['branch'], analysis_df['is_personal']).to_dict(),
        # 'branch_inactive_matrix': pd.crosstab(df['branch'], df['inactive']).to_dict(),
        # 'branch_negative_balance_matrix': pd.crosstab(df['branch'], df['acbal'] < 0).to_dict(),
        # 'branch_zero_balance_matrix': pd.crosstab(df['branch'], df['acbal'] == 0).to_dict(),
        # 'branch_kyc_matrix': pd.crosstab(df['branch'], df['kyc']).to_dict(),
        # 'branch_mobile_banking_matrix': pd.crosstab(df['branch'], df['mbservice']).to_dict(),
        # 'branch_internet_banking_matrix': pd.crosstab(df['branch'], df['ibservice']).to_dict(),
        # 'branch_ac_service_matrix': pd.crosstab(df['branch'], df['acservice']).to_dict(),
        # 'branch_total_balance_matrix': pd.crosstab(df['branch'], pd.cut(df['acbal'], bins=[-np.inf, 0, 1000, 10000, 100000, np.inf], labels=['Negative', 'Low', 'Medium', 'High', 'Very High'])).to_dict()
        # 'branch_age_matrix': pd.crosstab(df['branch'], pd.cut(df['age'], bins=[0, 30, 50, 70, 100], labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])).to_dict()
    }
    print('Branch analysis done')
    
    # Service Adoption Analysis
    service_adoption = {
        'mobile_banking_total_users': df['mbservice'].sum(),
        'mobile_banking_adoption_rate': (df['mbservice'].sum() / len(df) * 100),
        'mobile_banking_by_account_type': df.groupby('actype')['mbservice'].mean().to_dict(),
        'mobile_banking_by_sector': df.groupby('sector')['mbservice'].mean().to_dict(),
        'mobile_banking_by_national': df.groupby('national')['mbservice'].mean().to_dict(),
        'mobile_banking_by_category': df.groupby('category')['mbservice'].mean().to_dict(),
        'internet_banking_total_users': df['ibservice'].sum(),
        'internet_banking_adoption_rate': (df['ibservice'].sum() / len(df) * 100),
        'internet_banking_by_account_type': df.groupby('actype')['ibservice'].mean().to_dict(),
        'internet_banking_by_sector': df.groupby('sector')['ibservice'].mean().to_dict(),
        'internet_banking_by_national': df.groupby('national')['ibservice'].mean().to_dict(),
        'internet_banking_by_category': df.groupby('category')['ibservice'].mean().to_dict(),
        'account_service_total_users': df['acservice'].sum(),
        'account_service_adoption_rate': (df['acservice'].sum() / len(df) * 100),
        'account_service_by_account_type': df.groupby('actype')['acservice'].mean().to_dict(),
        'account_service_by_sector': df.groupby('sector')['acservice'].mean().to_dict(),
        'account_service_by_national': df.groupby('national')['acservice'].mean().to_dict(),
        'account_service_by_category': df.groupby('category')['acservice'].mean().to_dict()
    }
                
    # Balance Analysis
    balance_analysis = {
        'total_balance': df['acbal'].sum(),
        'average_balance': df['acbal'].mean(),
        'maximum_balance': df['acbal'].max(),
        'minimum_balance': df['acbal'].min(),
        'balance_by_account_type': df.groupby('actype')['acbal'].agg(['mean', 'sum', 'count']).to_dict(),
        'balance_by_sector': df.groupby('sector')['acbal'].agg(['mean', 'sum', 'count']).to_dict(),
        'balance_by_category': df.groupby('category')['acbal'].agg(['mean', 'sum', 'count']).to_dict(),
        'minimim_balance_by_account_type': df.groupby('actype')['acbal'].min().to_dict(),
        'minimim_balance_by_sector': df.groupby('sector')['acbal'].min().to_dict(),
        'minimim_balance_by_category': df.groupby('category')['acbal'].min().to_dict(),
        'maximum_balance_by_account_type': df.groupby('actype')['acbal'].max().to_dict(),
        'maximum_balance_by_sector': df.groupby('sector')['acbal'].max().to_dict(),
        'maximum_balance_by_category': df.groupby('category')['acbal'].max().to_dict(),
        'negative_balance': {
            'count': len(df[df['acbal'] < 0]),
            'total_amount': df[df['acbal'] < 0]['acbal'].sum(),
            'by_account_type': df[df['acbal'] < 0].groupby('actype').size().to_dict()
        }
    }
    
    # KYC and Compliance Analysis
    compliance_analysis = {
        'kyc_completion': {
            'kyc_completed': df['kyc'].sum(),
            'kyc_pending': len(df) - df['kyc'].sum(),
            'kyc_not_started': len(df[df['kyc'].isna()]),
            'kyc_completed_by_account_type': df.groupby('actype')['kyc'].sum().to_dict(),
            'kyc_completed_by_sector': df.groupby('sector')['kyc'].sum().to_dict(),
            'kyc_completed_by_national': df.groupby('national')['kyc'].sum().to_dict(),
            'kyc_completed_by_category': df.groupby('category')['kyc'].sum().to_dict(),
            'kyc_completed_by_branch': df.groupby('branch')['kyc'].sum().to_dict(),
        }
    }

    # # Account age analysis for personal accounts
    # personal_accounts['account_age_years'] = (pd.Timestamp.now() - personal_accounts['opendate']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    
    # age_demographics = {
    #     'avg_customer_age': personal_accounts['age'].mean(),
    #     'youngest_customer': personal_accounts['age'].min(),
    #     'oldest_customer': personal_accounts['age'].max(),
    #     'avg_account_age': personal_accounts['account_age_years'].mean(),
    # }
    
    print('Service Compliance done')
    # Recent activity analysis
   
    # Calculate the number of days since the last credit
    df['days_since_last_credit'] = (datetime.now() - df['lcrdate']).dt.days
    df['days_since_last_debit'] = (datetime.now() - df['ldrdate']).dt.days
    activity_metrics = {
        'avg_days_since_last_credit': df['days_since_last_credit'].mean(),
        'avg_days_since_last_debit': df['days_since_last_debit'].mean(),
        'total_amount_credited_last_30_days': df[df['days_since_last_credit'] <= 30]['lcyacbal'].sum(),
        'total_amount_debited_last_30_days': df[df['days_since_last_debit'] <= 30]['lcyacbal'].sum(),
        }

    return {
        'account_metrics': account_metrics,
        'nationality_analysis': nationality_analysis,
        'sector_analysis': sector_analysis,
        'branch_analysis': branch_analysis,
        'service_adoption': service_adoption,
        'balance_analysis': balance_analysis,
        'compliance_analysis': compliance_analysis,
        'activity_metrics': activity_metrics,
        'category_analysis': category_analysis
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
    - Analysis: Analyze the distribution with minimum top 5, highest, lowest, positive, negative, average, totals including numeric value or percentage.
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

    prompt_message = "Here is the financial data summary:"

    prompt = f"""
    {system_message}
    {prompt_message}
    {convert_to_json(analysis_results['sector_analysis'])}
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
    - Industry Distribution: {analysis_results['sector_analysis']['industry_counts']}
    - Inactive Accounts by Sector: {analysis_results['sector_analysis']['inactive_accounts_by_sector']}
    - Inactive Accounts by Industry: {analysis_results['sector_analysis']['inactive_accounts_by_industry']}
    - Sector Wise Average Balance: {analysis_results['sector_analysis']['sector_wise_account_average']}
    - Sector Wise Total Balance: {analysis_results['sector_analysis']['sector_wise_account_sum']}
    - Industry Wise Negative Balance: {analysis_results['sector_analysis']['industry_wise_negative_account_balance']}
    - Sector Distribution: {analysis_results['sector_analysis']['sector_counts']}
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


def preprocess_data(df):
    df_branch = pd.read_json('data/Branch.json')
    df_industry = pd.read_json('data/Industry.json') # NP
    df_sector = pd.read_json('data/Sector.json') # NP
    df_accountype = pd.read_json('data/AccountType.json')
    df_category = pd.read_json('data/Category.json') # Float value

    # mapping to be done: Branch, Industry, Sector, Account_type, Category
    branch_code_to_name = dict(
        zip(df_branch["Code"], df_branch["Name"])
    )    
    
    account_code_to_name = dict(
        zip(df_accountype["Code"], df_accountype["Desc"])
    ) 

    industry_code_to_name = dict(
        zip(df_industry["Code"], df_industry["Desc"])
    )    

    sector_code_to_name = dict(
        zip(df_sector["Code"], df_sector["Desc"])
    )  

    category_code_to_name = dict(
        zip(df_category["Code"], df_category["Desc"])
    )
    
    # Replace unwanted values other than numeric with 0000
    df['branch'] = pd.to_numeric(df['branch'], errors='coerce')
    df['actype'] = pd.to_numeric(df['actype'], errors='coerce')
    df['industry'] = pd.to_numeric(df['industry'], errors='coerce')
    df['sector'] = pd.to_numeric(df['sector'], errors='coerce')
    df['category'] = pd.to_numeric(df['category'], errors='coerce')

    # convert all data of column to some specific type
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].astype('int32')

    # Map numeric codes with strings
    df["branch"] = df["branch"].map(branch_code_to_name).fillna('BBRANCH')
    df["actype"] = df["actype"].map(account_code_to_name).fillna('AACCOUNT')
    df["industry"] = df["industry"].map(industry_code_to_name).fillna('IINDUSTRY')
    df["sector"] = df["sector"].map(sector_code_to_name).fillna('SSECTOR')
    df["category"] = df["category"].map(category_code_to_name).fillna('CCATEGORY')

    # print(df[['name','branch','industry','actype','sector','category','dob']].iloc[:3])
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
    models = ['gemma:2b', 'gemma2:latest', 'phi4:latest', 'llama3.2:latest','deepseek-r1:14b','qwen2.5-coder:7b']
    model_name = models[-1]
    
    # >> TEST
    # Load the data
    df = pd.read_csv(INPUT_PATH)  # AccountData
    # df = pd.read_csv(STMT_PATH) # Statements
    
    # TODO: Link Merge StmtData with AccountData
    # TODO: Rename column
    # custid,name,national,resident,industry,sector,branch,actype,accountno,ccy,category,acbal,lcyacbal,mbservice,ibservice,acservice,kyc,inactive,mobileno,opendate,ldrdate,lcrdate,dob

    # Get analysis results and prompt
    analysis, prompt = run_analysis(df) # Analyze bank data 'A' & Return Prompt
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
