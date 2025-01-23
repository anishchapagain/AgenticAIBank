import pandas as pd
import requests
import json
from datetime import datetime
import numpy as np
import re

'''
1000	PROPRIETORY CONCERN
1050	PRIVATE LIMITED COMPANY
1100	PUBLIC LIMITED COMPANY
1150	PARTNERSHIP FIRM
1200	JOINT VENTURE COMPANY
1250	INTERNATIONAL CORPORATE
1300	NON -GOVERNMENTAL ORGANISATION (NGO
1350	INT'L NON-GOVERNMENTAL ORGANISATION
1400	LOCAL - PRIVATE
1450	LOCAL -GOVERNMENT
1500	LOCAL -PERSONS
1510	NON RESIDENT NEPALESE
1550	FOREIGN - PRIVATE
1560	FOREIGN - GOVERNMENT
2000	FOREIGN -PERSONS
2010	CORPORATION
9010	OTHERS
'''
class AdvancedBankAnalyzer:
    """
    Advanced banking data analyzer with LLM-powered insights.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.api_base = "http://localhost:11434/api/generate"
        print("Init....\n")
    
    def query_ai_llm(self, prompt):
        """Query AI-LLM API with error handling"""
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
            print(f"Error querying AI-LLM: {e}")
            return "Error generating response"

    # Function to check date format
    def check_date_format(self, date_str):
        if re.match(r'\d{2}-[A-Z][a-z]{3}-\d{2}', date_str):
            return True
        else:
            return False
    
    def preprocess_data(self, df):
        """Preprocess data with handling for missing values and account types"""

        print("""Preprocess data with handling for missing values and account types....""")
        # Create a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Handle date columns with missing values
        date_columns = ['opendate', 'ldrdate', 'lcrdate', 'dob']
        for col in date_columns:
            # Convert to datetime while handling errors
            processed_df[col] = pd.to_datetime(processed_df[col], format='%d-%b-%y', errors='coerce')

        # Calculate age and account age where applicable
        current_date = datetime.now()
        
        # # Calculate age for personal accounts only
        # processed_df['age'] = np.where(
        #     processed_df['is_personal'] & processed_df['dob'].notna(),
        #     (current_date - processed_df['dob']).dt.days // 365,
        #     None
        # )
        
        # # Calculate account age for all accounts
        # processed_df['account_age_years'] = np.where(
        #     processed_df['opendate'].notna(),
        #     (current_date - processed_df['opendate']).dt.days // 365,
        #     None
        # )
        # print("Preprocess data received....\n")
        
        return processed_df

    def segment_accounts(self, df):
        """Segment accounts into personal and industrial"""
        print("""Segment accounts into personal and industrial....""")
        personal_accounts = df[df['is_personal']]
        industrial_accounts = df[~df['is_personal']]
        return personal_accounts, industrial_accounts

    def analyze_personal_accounts(self, df):
        """Analyze personal accounts"""
        print("""Analyze personal accounts....\n""")
        metrics = {
            'portfolio': {
                'total_accounts': len(df),
                'total_balance': df['acbal'].sum(),
                'average_balance': df['acbal'].mean(),
                'negative_balance_count': (df['acbal'] < 0).sum(),
                'active_accounts': (df['inactive'] == False).sum(),
                'total_portfolio_value': df['lcyacbal'].sum(),
            },
            'demographics': {
                'age_distribution': df['age'].describe().to_dict() if 'age' in df else {},
                'age_segments': {
                    'young_customers': (df['age'] < 30).sum(),
                    'middle_age': ((df['age'] >= 30) & (df['age'] < 50)).sum(),
                    'senior': (df['age'] >= 50 & (df['age'] < 75)).sum(),
                    'elderly': (df['age'] >= 75).sum()
                },
            },
            'services': {
                'mobile_banking_adoption': (df['mbservice'] == True).mean(),
                'internet_banking_adoption': (df['ibservice'] == True).mean(),
                'account_services': (df['acservice'] == True).mean()
            },
            'compliance': {
                'kyc_complete': (df['kyc'] == True).sum(),
                'kyc_pending': (df['kyc'] == False).sum()
            }
        }
        return metrics

    def analyze_industrial_accounts(self, df):
        """Analyze industrial accounts"""
        print("""Analyze industrial accounts....\n""")
        metrics = {
            'portfolio': {
                'total_accounts': len(df),
                'total_balance': df['acbal'].sum(),
                'average_balance': df['acbal'].mean(),
                'negative_balance_count': (df['acbal'] < 0).sum()
            },
            'account_characteristics': {
                'industry_distribution': df['industry'].value_counts().to_dict(),
                'sector_distribution': df['sector'].value_counts().to_dict(),
                'account_types': df['actype'].value_counts().to_dict()
            },
            'services': {
                'service_adoption': {
                    'mobile': (df['mbservice'] == True).mean(),
                    'internet': (df['ibservice'] == True).mean(),
                    'account_services': (df['acservice'] == True).mean()
                }
            }
        }
        return metrics

    def prepare_for_json(self, obj):
        """Prepare Python objects for JSON serialization"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime(self.datetime_format)
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
    
    def convert_to_json(self, data_dict, indent=2):
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
                    return self.prepare_for_json(d)
            
            json_ready = prepare_dict(data_dict)
            
            # Convert to JSON
            return json.dumps(json_ready, indent=indent, ensure_ascii=False)
            
        except Exception as e:
            return f"Error converting to JSON: {str(e)}"
        
    def generate_insights(self, personal_metrics, industrial_metrics):
        """Generate insights using LLM model"""
        print("""Generate insights using LLM model....\n""")
        # print(personal_metrics)
        # # prompt_personal_metrics = self.convert_to_json(personal_metrics)
        # prompt_personal_metrics = json.dumps(personal_metrics, indent=2)
        # print(type(prompt_personal_metrics))


            # 'personal_accounts': f"""Analyze these personal banking metrics:
            # Portfolio Overview: {json.dumps(personal_metrics['portfolio'], indent=2)}
            # Demographics: {json.dumps(personal_metrics['demographics'], indent=2)}
            # Service Adoption: {json.dumps(personal_metrics['services'], indent=2)}
        # prompts = {
        #     'personal_accounts': f"""Analyze these personal banking metrics:
        #     Portfolio Overview: {self.convert_to_json(personal_metrics['portfolio']).replace('  ', '')}
        #     Demographics: {self.convert_to_json(personal_metrics['demographics']).replace('  ', '')}
        #     Service Adoption: {self.convert_to_json(personal_metrics['services']).replace('  ', '')}
        #     """
        #     }
        # print(prompts)
        # exit()

        system_message = """
            You're a bank manager with access to detailed banking data.
            """
            # As a bank manager, analyze these portfolio metrics:
            # Analyze these personal banking metrics:
        # """

        system_message = """
        You're financial expert with professional knowledge about Python and Pandas. 
        You have been asked to analyze the financial data of a company.
        You have been provided with the following data.
        """
        
        prompts = {
            'personal_accounts': f"""
            Portfolio Overview: {self.convert_to_json(personal_metrics['portfolio'], indent=2)}
            Demographics: {self.convert_to_json(personal_metrics['demographics'], indent=2)}
            Service Adoption: {self.convert_to_json(personal_metrics['services'], indent=2)}
            
            Provide:
            1. Key insights about personal banking segment
            2. Opportunities for growth
            3. Risk factors
            4. Service improvement recommendations""",
            
            'industrial_accounts': f"""Analyze these industrial banking metrics:
            Portfolio Overview: {self.convert_to_json(industrial_metrics['portfolio'], indent=2)}
            Account Distribution: {self.convert_to_json(industrial_metrics['account_characteristics'], indent=2)}
            Service Usage: {self.convert_to_json(industrial_metrics['services'], indent=2)}
            
            Provide:
            1. Key insights about industrial banking segment
            2. Sector-specific opportunities
            3. Risk assessment
            4. Business banking strategy recommendations""",
            
            'comparative_analysis': f"""Compare personal and industrial banking segments:
            Personal Accounts Total Value: {personal_metrics['portfolio']['total_balance']}
            Industrial Accounts Total Value: {industrial_metrics['portfolio']['total_balance']}
            Personal Service Adoption: {personal_metrics['services']}
            Industrial Service Adoption: {industrial_metrics['services']}
            
            Provide:
            1. Comparative analysis of segments
            2. Resource allocation recommendations
            3. Cross-segment opportunities
            4. Strategic priorities
            """
        }

        # prompt = system_message
        # prompt = system_message + data_json
        # prompt = system_message + prompts + "\n What insights can you provide from the data?" # + explanation

        
        insights = {}
        for analysis_type, prompt in prompts.items():
            insights[analysis_type] = self.query_ai_llm(system_message+"\n"+prompt+"\n What insights can you provide from the data?")
            # insights[analysis_type] = self.query_ai_llm(prompt)

        
        return insights

    def analyze_banking_data(self, df):
        """Main analysis function"""
        print("""Main analysis function....\n""")
        # Preprocess data
        processed_df = self.preprocess_data(df)
        print(processed_df)
        # exit()
        # Segment accounts
        personal_accounts, industrial_accounts = self.segment_accounts(processed_df)
        
        # Analyze each segment
        personal_metrics = self.analyze_personal_accounts(personal_accounts)
        # print(personal_metrics)
        industrial_metrics = self.analyze_industrial_accounts(industrial_accounts)
        # print(industrial_metrics)
        
        # Generate insights
        insights = self.generate_insights(personal_metrics, industrial_metrics)
        
        # Compile complete analysis
        return {
            'personal_banking': {
                'metrics': personal_metrics,
                'insights': insights['personal_accounts']
            },
            'industrial_banking': {
                'metrics': industrial_metrics,
                'insights': insights['industrial_accounts']
            },
            'comparative_analysis': insights['comparative_analysis']
        }

def main():
    # Load data
    df = pd.read_csv('data/AccountData.csv')    
    # df = df.dropna()  # Drop rows with missing values
    # model = "llama3.2:latest"
    model = "gemma:2b"
    # model = "phi4:latest"
    print(df.head(5)) # Preview the first few rows

    # Load only 100 rows for testing
    df = df.iloc[500]
    # print(df)
    # Initialize analyzer
    analyzer = AdvancedBankAnalyzer(model_name=model)
    
    # Get analysis
    analysis = analyzer.analyze_banking_data(df)

    # Save analysis to JSON file
    filename= f"analysis_{model.replace(':', '_')}.json"
    with open(filename, 'wb') as f:
        f.write(analyzer.convert_to_json(analysis, indent=2).encode('utf-8'))

    # Print results
    print("\n=== BANKING ANALYSIS DASHBOARD ===")
    
    print("\n--- PERSONAL BANKING INSIGHTS ---")
    print(analysis['personal_banking']['insights'])
    
    print("\n--- INDUSTRIAL BANKING INSIGHTS ---")
    print(analysis['industrial_banking']['insights'])
    
    print("\n--- COMPARATIVE ANALYSIS ---")
    print(analysis['comparative_analysis'])

if __name__ == "__main__":
    main()