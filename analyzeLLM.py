import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import numpy as np

class BankManagerAnalytics:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

    def calculate_bank_metrics(self, df):
        """
        Calculate key banking performance indicators
        """
        metrics = {
            'portfolio_health': {
                'total_portfolio_value': df['lcyacbal'].sum(),
                'average_account_balance': df['acbal'].mean(),
                'negative_balance_percentage': (df['acbal'] < 0).mean() * 100,
                'inactive_accounts_percentage': (df['inactive'] == True).mean() * 100,
                'kyc_compliance_rate': (df['kyc'] == True).mean() * 100
            },
            'customer_engagement': {
                'digital_adoption_rate': {
                    'mobile_banking': (df['mbservice'] == True).mean() * 100,
                    'internet_banking': (df['ibservice'] == True).mean() * 100,
                    'overall_services': (df['acservice'] == True).mean() * 100
                },
                'account_types_distribution': df['actype'].value_counts().to_dict(),
                'industry_distribution': df['industry'].value_counts().to_dict()
            },
            'customer_demographics': {
                'age_segments': pd.cut(df['age'], bins=[0, 30, 50, 70, 100], 
                                     labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])
                                     .value_counts().to_dict(),
                'average_customer_age': df['age'].mean(),
                'average_relationship_duration': df['account_age_years'].mean()
            }
        }
        return metrics

    def generate_manager_prompts(self, metrics):
        """Generate management-focused prompts"""
        prompts = {
            'portfolio_analysis': f"""As a bank manager, analyze the following portfolio metrics:
            Portfolio Value: {metrics['portfolio_health']['total_portfolio_value']}
            Average Balance: {metrics['portfolio_health']['average_account_balance']}
            Negative Balance Rate: {metrics['portfolio_health']['negative_balance_percentage']}%
            KYC Compliance: {metrics['portfolio_health']['kyc_compliance_rate']}%

            Provide insights on:
            1. Portfolio strength and areas of concern
            2. Risk management recommendations
            3. Opportunities for portfolio growth""",

            'customer_retention': f"""Based on these customer engagement metrics:
            Digital Adoption: {metrics['customer_engagement']['digital_adoption_rate']}
            Account Distribution: {metrics['customer_engagement']['account_types_distribution']}
            Demographics: {metrics['customer_demographics']}

            Analyze:
            1. Customer retention strategies
            2. Service improvement opportunities
            3. Cross-selling potential""",

            'operational_efficiency': f"""Review operational metrics:
            Inactive Accounts: {metrics['portfolio_health']['inactive_accounts_percentage']}%
            Digital Services Usage: {metrics['customer_engagement']['digital_adoption_rate']}
            Industry Distribution: {metrics['customer_engagement']['industry_distribution']}

            Provide recommendations on:
            1. Operational efficiency improvements
            2. Resource allocation
            3. Service delivery optimization"""
        }
        return prompts

    def analyze_for_manager(self, df):
        """Comprehensive analysis for bank managers"""
        # Prepare data
        df['age'] = (datetime.now() - pd.to_datetime(df['dob'])).dt.days // 365
        df['account_age_years'] = (datetime.now() - pd.to_datetime(df['opendate'])).dt.days // 365

        # Calculate metrics
        metrics = self.calculate_bank_metrics(df)
        
        # Generate prompts
        prompts = self.generate_manager_prompts(metrics)
        
        # Get LLM insights
        managerial_insights = {}
        for analysis_type, prompt in prompts.items():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=1000,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            managerial_insights[analysis_type] = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate performance indicators
        performance_indicators = {
            'revenue_potential': self.calculate_revenue_potential(df),
            'risk_metrics': self.calculate_risk_metrics(df),
            'growth_opportunities': self.identify_growth_opportunities(df)
        }

        return {
            'bank_metrics': metrics,
            'managerial_insights': managerial_insights,
            'performance_indicators': performance_indicators
        }

    def calculate_revenue_potential(self, df):
        """Calculate revenue potential metrics"""
        return {
            'high_value_customers': len(df[df['acbal'] > df['acbal'].quantile(0.9)]),
            'upsell_opportunities': len(df[
                (df['mbservice'] == False) | 
                (df['ibservice'] == False)
            ]),
            'active_accounts_ratio': 1 - (df['inactive'] == True).mean()
        }

    def calculate_risk_metrics(self, df):
        """Calculate risk-related metrics"""
        return {
            'negative_balance_exposure': df[df['acbal'] < 0]['acbal'].sum(),
            'non_kyc_accounts': len(df[df['kyc'] == False]),
            'dormant_account_value': df[df['inactive'] == True]['acbal'].sum()
        }

    def identify_growth_opportunities(self, df):
        """Identify potential growth areas"""
        return {
            'digital_conversion_potential': len(df[
                (df['mbservice'] == False) & 
                (df['age'] < 50)
            ]),
            'high_balance_inactive': len(df[
                (df['acbal'] > df['acbal'].median()) & 
                (df['inactive'] == True)
            ]),
            'industry_concentration': df['industry'].value_counts().head(3).to_dict()
        }

class BankingAnalyzer:
    def __init__(self):
        # Initialize LLaMA model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data(self, df):
        """Prepare and clean banking data"""
        # Convert date columns to datetime
        date_columns = ['opendate', 'ldrdate', 'lcrdate', 'dob']
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], format='%d-%b-%y')
        
        # Calculate derived metrics
        df['age'] = (datetime.now() - df['dob']).dt.days // 365
        df['account_age_years'] = (datetime.now() - df['opendate']).dt.days // 365
        return df

    def generate_llm_prompt(self, analysis_type, data):
        """Generate prompts for different types of analysis"""
        prompts = {
            'risk_analysis': f"""Analyze the following banking customer data and identify potential risk factors:
            Customer Ages: {data['age_stats']}
            Account Balances: {data['balance_stats']}
            Digital Service Adoption: {data['digital_adoption']}
            Account Activity: {data['activity_stats']}
            
            Provide insights on:
            1. Risk factors and their severity
            2. Recommendations for risk mitigation
            3. Customer segments requiring attention""",
            
            'customer_segmentation': f"""Based on the following banking data, provide detailed customer segmentation insights:
            Age Distribution: {data['age_stats']}
            Balance Patterns: {data['balance_stats']}
            Service Usage: {data['digital_adoption']}
            Account Types: {data['account_types']}
            
            Analyze:
            1. Key customer segments and their characteristics
            2. Opportunities for each segment
            3. Targeted service recommendations""",
            
            'fraud_detection': f"""Review the following banking patterns for potential fraud indicators:
            Transaction Patterns: {data['transaction_stats']}
            Account Activities: {data['activity_stats']}
            Digital Banking Usage: {data['digital_adoption']}
            
            Identify:
            1. Suspicious patterns or anomalies
            2. Risk levels for different customer segments
            3. Recommended monitoring measures"""
        }
        return prompts.get(analysis_type, "")

    def get_llm_analysis(self, prompt):
        """Get analysis from LLaMA model"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=1000,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def analyze_banking_data(self, df, analysis_types=None):
        """Complete banking data analysis with LLM insights"""
        if analysis_types is None:
            analysis_types = ['risk_analysis', 'customer_segmentation', 'fraud_detection']
        
        df = self.prepare_data(df)
        
        # Prepare data summaries for LLM
        data_summary = {
            'age_stats': df['age'].describe().to_dict(),
            'balance_stats': df['acbal'].describe().to_dict(),
            'digital_adoption': {
                'mobile': (df['mbservice'] == True).mean(),
                'internet': (df['ibservice'] == True).mean()
            },
            'activity_stats': {
                'inactive_rate': (df['inactive'] == True).mean(),
                'kyc_complete': (df['kyc'] == True).mean()
            },
            'account_types': df['actype'].value_counts().to_dict(),
            'transaction_stats': {
                'avg_balance': df['acbal'].mean(),
                'negative_balance_rate': (df['acbal'] < 0).mean()
            }
        }
        
        # Get LLM insights for each analysis type
        llm_insights = {}
        for analysis_type in analysis_types:
            prompt = self.generate_llm_prompt(analysis_type, data_summary)
            llm_insights[analysis_type] = self.get_llm_analysis(prompt)
        
        return {
            'data_summary': data_summary,
            'llm_insights': llm_insights
        }

# Example usage
def main():
    # Read your CSV data
    df = pd.read_csv('banking_data.csv')
    
    # Initialize analyzer
    analyzer = BankingAnalyzer()
    # analyzer = BankManagerAnalytics()

    # Get complete analysis
    analysis = analyzer.analyze_banking_data(df)
    # analysis = analyzer.analyze_for_manager(df)

    # Print insights
    for analysis_type, insight in analysis['llm_insights'].items():
        print(f"\n=== {analysis_type.upper()} ===")
        print(insight)

    # Print insights
    print("\n=== BANK MANAGER'S DASHBOARD ===")
    for category, insights in analysis['managerial_insights'].items():
        print(f"\n--- {category.upper()} ---")
        print(insights)
    
    print("\n=== PERFORMANCE INDICATORS ===")
    for metric, value in analysis['performance_indicators'].items():
        print(f"\n{metric}:")
        print(value)

if __name__ == "__main__":
    main()




import pandas as pd
import requests
import json
from datetime import datetime
import numpy as np

class BankManagerOllama:
    def __init__(self, model_name="llama2"):
        """
        Initialize the analyzer with Ollama model
        Default model is llama2, but can use others like mistral or llama2-uncensored
        """
        self.model_name = model_name
        self.api_base = "http://localhost:11434/api/generate"
        
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
                'age_segments': {
                    'young_customers': (df['age'] < 30).sum(),
                    'middle_age': ((df['age'] >= 30) & (df['age'] < 50)).sum(),
                    'senior': (df['age'] >= 50).sum()
                },
                'digital_adoption': {
                    'mobile_banking': (df['mbservice'] == True).sum(),
                    'internet_banking': (df['ibservice'] == True).sum()
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

def main():
    """Example usage of the BankManagerOllama class"""
    # Load your banking data
    df = pd.read_csv('banking_data.csv')
    
    # Initialize analyzer with specific model
    analyzer = BankManagerOllama(model_name="llama2")  # or "mistral" or other available models
    
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

if __name__ == "__main__":
    main()