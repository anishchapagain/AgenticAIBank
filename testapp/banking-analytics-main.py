import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go

from data_preprocessor import BankingDataPreprocessor
from analytics_engine import BankingAnalyticsEngine
from llm_interface import generate_llm_response

st.set_page_config(page_title="Bank Insights Dashboard", layout="wide")

@st.cache_data
def load_data(customer_file, transaction_file):
    try:
        customer_df = pd.read_csv(customer_file)
        transaction_df = pd.read_csv(transaction_file)
        return customer_df, transaction_df
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
        return None, None

def main():
    st.title("🏦 Comprehensive Banking Analytics Platform")
    
    # Sidebar for File Uploads
    with st.sidebar:
        st.header("Data Upload")
        customer_file = st.file_uploader("Upload Customer Data", type=['csv'])
        transaction_file = st.file_uploader("Upload Transaction Data", type=['csv'])
        
        if customer_file and transaction_file:
            customer_df, transaction_df = load_data(customer_file, transaction_file)
            
            if customer_df is not None and transaction_df is not None:
                preprocessor = BankingDataPreprocessor(customer_df, transaction_df)
                analytics_engine = BankingAnalyticsEngine(preprocessor)
                
                # Dashboard Tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Customer Insights", 
                    "Transaction Analysis", 
                    "Branch Performance", 
                    "AI Insights Chat"
                ])
                
                with tab1:
                    display_customer_insights(analytics_engine)
                
                with tab2:
                    display_transaction_insights(analytics_engine)
                
                with tab3:
                    display_branch_performance(analytics_engine)
                
                with tab4:
                    display_ai_chat(analytics_engine)

def display_customer_insights(analytics_engine):
    st.header("Customer Segmentation & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Segments")
        segments = analytics_engine.get_customer_segments()
        st.dataframe(segments)
    
    with col2:
        st.subheader("Customer Distribution")
        fig = px.pie(segments, values='count', names='segment')
        st.plotly_chart(fig)

def display_transaction_insights(analytics_engine):
    st.header("Transaction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction by Currency")
        currency_transactions = analytics_engine.get_currency_transactions()
        st.dataframe(currency_transactions)
    
    with col2:
        st.subheader("Transaction Volume Trend")
        trend_fig = px.line(currency_transactions, x='ccy', y='total_amount')
        st.plotly_chart(trend_fig)

def display_branch_performance(analytics_engine):
    st.header("Branch Performance Metrics")
    
    branch_metrics = analytics_engine.get_branch_performance()
    st.dataframe(branch_metrics)
    
    fig = px.bar(branch_metrics, x='branch', y='total_transactions')
    st.plotly_chart(fig)

def display_ai_chat(analytics_engine):
    st.header("AI-Powered Banking Insights")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about banking insights"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            response = generate_llm_response(prompt, analytics_engine)
            st.markdown(response)

if __name__ == "__main__":
    main()
