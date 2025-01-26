import streamlit as st
import pandas as pd
import requests
from typing import Dict, Any

class BankingChatbot:
    def __init__(self, customer_df: pd.DataFrame, transaction_df: pd.DataFrame):
        self.merged_df = pd.merge(
            customer_df, 
            transaction_df, 
            on='custid', 
            how='inner', 
            suffixes=('', '_transaction')
        )
    
    def generate_context(self) -> Dict[str, Any]:
        # Use columns without suffixes
        columns = [col for col in self.merged_df.columns if not col.endswith('_transaction')]
        context_df = self.merged_df[columns]
        
        return {
            'total_customers': context_df['custid'].nunique(),
            'total_transactions': len(context_df),
            'total_amount': context_df['amount'].sum(),
            'avg_transaction': context_df['amount'].mean(),
            'top_industries': context_df.groupby('industry')['amount'].sum().nlargest(3).to_dict(),
            'branch_summary': context_df.groupby('branch')['amount'].sum().to_dict()
        }
    
    def get_ollama_response(self, query: str, context: Dict[str, Any]) -> str:
        try:
            prompt = f"""
            Banking Analytics Context:
            {context}
            
            User Query: {query}
            
            Provide a detailed, data-driven banking insight response. 
            Focus on strategic analysis and actionable recommendations.
            """
            
            response = requests.post(
                'http://localhost:11434/api/generate', 
                json={
                    'model': 'llama2',
                    'prompt': prompt,
                    'stream': False
                }
            )
            return response.json()['response']
        
        except Exception as e:
            return f"Error generating response: {str(e)}"

def main():
    st.set_page_config(page_title="Banking Insights Chatbot")
    st.title("🏦 Banking Insights Chatbot")

    uploaded_customer = st.file_uploader("Upload Customer Data", type='csv')
    uploaded_transactions = st.file_uploader("Upload Transaction Data", type='csv')
    
    if uploaded_customer and uploaded_transactions:
        customer_df = pd.read_csv(uploaded_customer)
        transaction_df = pd.read_csv(uploaded_transactions)
        
        chatbot = BankingChatbot(customer_df, transaction_df)
        context = chatbot.generate_context()
        
        st.sidebar.json(context)

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your banking insights assistant."}
            ]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about your banking data"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Generating insights..."):
                    response = chatbot.get_ollama_response(prompt, context)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()