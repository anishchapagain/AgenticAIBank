import pandas as pd
import requests
import json
from datetime import datetime
import numpy as np

class BankManagerOllama:
    def __init__(self, model_name, api_base='http://localhost:11434/api/generate'):
        """
        Initialize the analyzer with Ollama model and load API base URL.
        """
        self.model_name = model_name
        self.api_base = api_base
        
    def query_ollama(self, prompt):
        """Send request to Ollama API"""
        try:
            response = requests.post(
                self.api_base,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            return response.json()['response']
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return "Error generating response"

    def prepare_data(self, df):
        """Prepare and clean banking data"""
        # Convert date columns
        date_columns = ['opendate', 'ldrdate', 'lcrdate', 'dob']
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], format='%d-%b-%y')
        
        # Calculate age-related metrics
        df['age'] = (datetime.now() - df['dob']).dt.days // 365
        df['account_age_years'] = (datetime.now() - df['opendate']).dt.days // 365
        
        return df

    def calculate_metrics(self, df):
        """Calculate comprehensive banking metrics"""
        metrics = {
            'portfolio_metrics': {
                'total_portfolio_value': df['lcyacbal'].sum(),
                'average_balance': df['acbal'].mean(),
                'negative_balance_accounts': (df['acbal'] < 0).sum(),
                'total_accounts': len(df),
                'active_accounts': (df['inactive'] == False).sum()
            },
            'customer_metrics': {
                'age_distribution': df['age'].describe().to_dict() if 'age' in df else {},
                'age_segments': {
                    'young_customers': (df['age'] < 30).sum(),
                    'middle_age': ((df['age'] >= 30) & (df['age'] < 50)).sum(),
                    'senior': (df['age'] >= 50 & (df['age'] < 75)).sum(),
                    'elderly': (df['age'] >= 75).sum()
                },
                'digital_adoption': {
                    'mobile_banking': (df['mbservice'] == True).sum(),
                    'internet_banking': (df['ibservice'] == True).sum(),
                    'account_services': (df['acservice'] == True).mean()
                },
                'compliance': {
                    'kyc_complete': (df['kyc'] == True).sum(),
                    'kyc_pending': (df['kyc'] == False).sum()
                }
            },
            'risk_metrics': {
                'high_value_accounts': (df['acbal'] > df['acbal'].quantile(0.9)).sum(),
                'dormant_accounts': (df['inactive'] == True).sum(),
                'non_compliant': (df['kyc'] == False).sum()
            }
        }
        return metrics

    def generate_management_insights(self, metrics):
        """Generate management-focused prompts and get LLM insights"""
        analysis_prompts = {
            'portfolio_health': f"""As a bank manager, analyze these portfolio metrics:
            Total Portfolio Value: {metrics['portfolio_metrics']['total_portfolio_value']:,.2f}
            Average Balance: {metrics['portfolio_metrics']['average_balance']:,.2f}
            Negative Balance Accounts: {metrics['portfolio_metrics']['negative_balance_accounts']}
            Total Accounts: {metrics['portfolio_metrics']['total_accounts']}
            
            Provide:
            1. Overall portfolio health assessment
            2. Key areas of concern
            3. Specific recommendations for improvement
            4. Growth opportunities""",
            
            'customer_engagement': f"""Analyze customer engagement based on:
            Age Distribution: {metrics['customer_metrics']['age_segments']}
            Digital Adoption: {metrics['customer_metrics']['digital_adoption']}
            
            Provide:
            1. Customer segment analysis
            2. Digital adoption strategy
            3. Engagement improvement recommendations
            4. Cross-selling opportunities""",
            
            'risk_assessment': f"""Evaluate risk metrics:
            High Value Accounts: {metrics['risk_metrics']['high_value_accounts']}
            Dormant Accounts: {metrics['risk_metrics']['dormant_accounts']}
            Non-Compliant Accounts: {metrics['risk_metrics']['non_compliant']}
            
            Provide:
            1. Risk level assessment
            2. Compliance improvement strategies
            3. Account monitoring recommendations
            4. Risk mitigation actions"""
        }
        
        insights = {}
        for analysis_type, prompt in analysis_prompts.items():
            insights[analysis_type] = self.query_ollama(prompt)
            
        return insights

    def identify_opportunities(self, df):
        """Identify business opportunities and growth potential"""
        opportunities = {
            'upsell_targets': len(df[
                (df['acbal'] > df['acbal'].median()) & 
                ((df['mbservice'] == False) | (df['ibservice'] == False))
            ]),
            'reactivation_potential': len(df[
                (df['inactive'] == True) & 
                (df['acbal'] > 0)
            ]),
            'service_expansion': {
                'mobile_banking_potential': len(df[df['mbservice'] == False]),
                'internet_banking_potential': len(df[df['ibservice'] == False])
            }
        }
        return opportunities

    def analyze_banking_data(self, df):
        """Main analysis function"""
        # Prepare data
        df = self.prepare_data(df)
        
        # Calculate metrics
        metrics = self.calculate_metrics(df)
        
        # Generate LLM insights
        managerial_insights = self.generate_management_insights(metrics)
        
        # Identify opportunities
        opportunities = self.identify_opportunities(df)
        
        # Compile complete analysis
        analysis_report = {
            'metrics': metrics,
            'managerial_insights': managerial_insights,
            'opportunities': opportunities
        }
        
        return analysis_report

def query_ollama(prompt, model):
    response = requests.post('http://localhost:11434/api/generate',
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    if response.status_code != 200:
        return "Error querying Ollama" + response.text
    
    return response.json() # ['response']


def query_ollama_with_model(model):
    """Example usage of the BankManagerOllama class"""

    # Load your banking data
    df = pd.read_csv('data/AccountData.csv')
    # df = df.dropna()  # Drop rows with missing values
    # df.rename(columns={'old_name': 'new_name'}, inplace=True)
    print(df.head())  # Preview the first few rows

    # data_json = df.head(20).to_json(orient='records')

    
    # Initialize analyzer with specific model
    analyzer = BankManagerOllama(model_name=model)  # 'gemma:2b'
    
    # Get analysis
    analysis = analyzer.analyze_banking_data(df)
    
    # Print results
    print("\n=== BANK MANAGER'S DASHBOARD ===")
    
    print("\n--- PORTFOLIO METRICS ---")
    print(json.dumps(analysis['metrics']['portfolio_metrics'], indent=2))
    
    print("\n--- MANAGERIAL INSIGHTS ---")
    for insight_type, insight in analysis['managerial_insights'].items():
        print(f"\n{insight_type.upper()}:")
        print(insight)
    
    print("\n--- GROWTH OPPORTUNITIES ---")
    print(json.dumps(analysis['opportunities'], indent=2))

    print("\n--- PERSONAL BANKING INSIGHTS ---")
    print(analysis['personal_banking']['insights'])
    
    print("\n--- INDUSTRIAL BANKING INSIGHTS ---")
    print(analysis['industrial_banking']['insights'])
    
    print("\n--- COMPARATIVE ANALYSIS ---")
    print(analysis['comparative_analysis'])

def testing():

    prompt = "Testing......"
    result = query_ollama(prompt, model)
    print(result.get('response', 'Error querying Ollama'))
    exit()

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

if __name__ == "__main__":
    # main()
    # Example usage of query_ollama function

    model = "llama3.2:latest"
    # model = "gemma:2b"
    # model = "phi4:latest"
    # testing()

    # Example usage of BankManagerOllama class
    query_ollama_with_model(model)