import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta
import math
import os
import io

# Set page configuration
st.set_page_config(page_title="Loan Decision Support System", layout="wide", initial_sidebar_state="expanded")

# Initialize key variables in session state for persistence
def init_session_state():
    if 'financial_data' not in st.session_state:
        st.session_state['financial_data'] = {}

# Helper functions for formatting
def format_currency(value):
    """Format number as currency with commas"""
    if pd.isna(value) or value is None:
        return "N/A"
    try:
        # Check if value is a list, dictionary, or other non-numeric type
        if isinstance(value, (list, dict, tuple, set)):
            return str(value)
        return f"{float(value):,.0f}"
    except (ValueError, TypeError):
        return str(value)

def calculate_ebitda(data):
    """Calculate EBITDA"""
    try:
        # Check if data is a dictionary
        if not isinstance(data, dict):
            return 0
        
        operating_income = data.get('Operating Income', 0)
        depreciation = data.get('Depreciation', 0)
        amortization = data.get('Amortization', 0)
        
        return operating_income + depreciation + amortization
    except Exception as e:
        st.error(f"Error calculating EBITDA: {e}")
        return 0

def safe_division(numerator, denominator):
    """Safe division function to handle zero division and negative values"""
    try:
        if denominator == 0:
            return None
        return numerator / denominator
    except Exception as e:
        st.error(f"Division error: {e}")
        return None

def calculate_financial_ratios(data):
    """Calculate all required financial ratios with error handling"""
    ratios = {}
    
    # Extract relevant data
    current_assets = data.get('Total Current Assets', 0)
    inventory = data.get('Inventory', 0)
    current_liabilities = data.get('Total Current Liabilities', 0)
    total_debt = data.get('Total Debt', 0) or (data.get('Long-term Debt', 0) + data.get('Total Current Liabilities', 0))
    long_term_debt = data.get('Long-term Debt', 0)
    total_liabilities = data.get('Total Liabilities', 0)
    total_equity = data.get('Total Equity', 0)
    operating_income = data.get('Operating Income', 0)
    interest_expense = data.get('Interest Expense', 0)
    if interest_expense < 0:
        interest_expense = abs(interest_expense)
    principal_payments = data.get('Principal Payments', 0) or 0
    
    # Calculate EBITDA
    ebitda = calculate_ebitda(data)
    ratios['EBITDA'] = ebitda
    
    # Calculate Leverage Ratio (Debt-to-Equity)
    print(f"Leverage: {total_liabilities/total_equity}")
    ratios['Leverage Ratio'] = safe_division(total_liabilities, total_equity)
    # ratios['Leverage Ratio'] = safe_division(total_debt, total_equity)
    
    # Calculate Interest Coverage Ratio
    # Earning before interest tax depreciation (EBITDA or operating income)/Interest expense

    ratios['ICR'] = safe_division(operating_income, interest_expense)
    
    # Calculate Debt Service Coverage Ratio
    # debt_service = interest_expense + principal_payments #Long-term loan
    # ratios['DSCR'] = safe_division(operating_income, debt_service)
    old_principal_repayment = long_term_debt*0.1 # Assumption
    ratios['DSCR'] = safe_division(long_term_debt, old_principal_repayment) # TODO
    
    # Calculate Current Ratio
    ratios['Current Ratio'] = safe_division(current_assets, current_liabilities)
    
    # Calculate Quick Ratio
    quick_assets = current_assets - inventory
    ratios['Quick Ratio'] = safe_division(quick_assets, current_liabilities)
    
    return ratios

def evaluate_ratio(ratio_name, value):
    """Evaluate ratio against thresholds and return status"""
    if value is None:
        return "N/A", "#808080"  # Gray for N/A values
    
    if ratio_name == 'Leverage Ratio':
        if value == 4:
            return "Green", "#00FF00"
        elif value > 4:
            return "Red", "#FF0000"
        else:
            return "Amber", "#FFA500"
            
    elif ratio_name == 'ICR':
        if value > 1:
            return "Green", "#00FF00"
        else:
            return "Red", "#FF0000"
            
    elif ratio_name == 'DSCR':
        if value > 1:
            return "Green", "#00FF00"
        else:
            return "Red", "#FF0000"
            
    elif ratio_name == 'Current Ratio':
        if 1 <= value <= 1.5:
            return "Green", "#00FF00"
        elif value > 1.5:
            return "Amber", "#FFA500"
        else:
            return "Red", "#FF0000"
            
    elif ratio_name == 'Quick Ratio':
        if value > 1:
            return "Green", "#00FF00"
        else:
            return "Red", "#FF0000"
    
    return "Unknown", "#808080"

def make_decision(evaluations):
    """Make loan decision based on evaluation criteria"""
    green_count = sum(1 for status, _ in evaluations.values() if status == "Green")
    total_ratios = len(evaluations)
    
    if total_ratios == 0:
        return "Insufficient Data", "#808080"
    
    if all(status == "Green" for status, _ in evaluations.values()):
        return "Approved (Green)", "#00FF00"
    elif green_count >= 3:
        return "Review Required (Amber)", "#FFA500"
    else:
        return "Declined (Red)", "#FF0000"

def calculate_loan_amortization(principal, annual_interest_rate, years, payment_frequency='monthly'):
    """Calculate loan amortization schedule"""
    if payment_frequency == 'monthly':
        periods = years * 12
        periodic_interest_rate = annual_interest_rate / 100 / 12
    elif payment_frequency == 'quarterly':
        periods = years * 4
        periodic_interest_rate = annual_interest_rate / 100 / 4
    elif payment_frequency == 'annually':
        periods = years
        periodic_interest_rate = annual_interest_rate / 100
    else:
        raise ValueError("Invalid payment frequency")
    
    if periodic_interest_rate == 0:
        periodic_payment = principal / periods
    else:
        periodic_payment = principal * (periodic_interest_rate * (1 + periodic_interest_rate) ** periods) / ((1 + periodic_interest_rate) ** periods - 1)
    
    schedule = []
    remaining_balance = principal
    
    for period in range(1, periods + 1):
        interest_payment = remaining_balance * periodic_interest_rate
        principal_payment = periodic_payment - interest_payment
        remaining_balance -= principal_payment
        
        if remaining_balance < 0:
            remaining_balance = 0
        
        schedule.append({
            'Period': period,
            'Payment': periodic_payment,
            'Principal': principal_payment,
            'Interest': interest_payment,
            'Remaining Balance': remaining_balance
        })
    
    return schedule

def project_financials(base_data, loan_details, periods=5):
    """Project future financial statements after taking a loan"""
    principal = loan_details['principal']
    annual_interest_rate = loan_details['interest_rate']
    years = loan_details['years']
    payment_frequency = loan_details['payment_frequency']
    
    # Get growth rates from UI if provided
    revenue_growth = loan_details.get('revenue_growth', 0.05)  # 5% default
    expense_growth = loan_details.get('expense_growth', 0.04)  # 4% default
    
    amortization = calculate_loan_amortization(principal, annual_interest_rate, years, payment_frequency)
    
    # Determine period increment based on payment frequency
    if payment_frequency == 'monthly':
        period_increment = 1/12
    elif payment_frequency == 'quarterly':
        period_increment = 1/4
    else:  # annually
        period_increment = 1
    
    projected_financials = []
    current_data = base_data.copy()
    
    # Add loan to initial financials
    current_data['Cash'] = current_data.get('Cash', 0) + principal
    current_data['Total Current Assets'] = current_data.get('Total Current Assets', 0) + principal
    current_data['Long-term Debt'] = current_data.get('Long-term Debt', 0) + principal
    current_data['Total Liabilities'] = current_data.get('Total Liabilities', 0) + principal
    
    # Year 0 (after taking loan but before payments)
    projected_financials.append({
        'Period': 0,
        'Year': 'Initial',
        'Financials': current_data.copy(),
        'Ratios': calculate_financial_ratios(current_data)
    })
    
    # Project future periods
    payment_idx = 0
    for year in range(1, periods + 1):
        year_data = current_data.copy()
        
        # Update revenue and expenses with growth
        year_data['Revenue'] = year_data.get('Revenue', 0) * (1 + revenue_growth)
        year_data['Operating Expenses'] = year_data.get('Operating Expenses', 0) * (1 + expense_growth)
        year_data['Operating Income'] = year_data.get('Revenue', 0) - year_data.get('Operating Expenses', 0)
        
        # Calculate total payments for the year
        annual_principal_payment = 0
        annual_interest_payment = 0
        
        # Determine how many payments to include based on frequency
        if payment_frequency == 'monthly':
            payments_per_year = 12
        elif payment_frequency == 'quarterly':
            payments_per_year = 4
        else:  # annually
            payments_per_year = 1
        
        for _ in range(payments_per_year):
            if payment_idx < len(amortization):
                payment = amortization[payment_idx]
                annual_principal_payment += payment['Principal']
                annual_interest_payment += payment['Interest']
                payment_idx += 1
        
        # Update financials based on loan payments
        year_data['Cash'] = year_data.get('Cash', 0) - annual_principal_payment
        year_data['Total Current Assets'] = year_data.get('Total Current Assets', 0) - annual_principal_payment
        year_data['Long-term Debt'] = year_data.get('Long-term Debt', 0) - annual_principal_payment
        year_data['Total Liabilities'] = year_data.get('Total Liabilities', 0) - annual_principal_payment
        year_data['Interest Expense'] = year_data.get('Interest Expense', 0) + annual_interest_payment
        year_data['Principal Payments'] = annual_principal_payment
        
        # Update ratios
        ratios = calculate_financial_ratios(year_data)
        
        projected_financials.append({
            'Period': year,
            'Year': f'Year {year}',
            'Financials': year_data,
            'Ratios': ratios
        })
        
        # Set up for next iteration
        current_data = year_data.copy()
    
    return projected_financials


# UI layout
def main():
    # Initialize session state
    init_session_state()
    
    st.title("Banking Loan Decision Support System")

    
    # Sidebar for configuration
    st.sidebar.header("Upload Financial Data")
    
    # Initialize data
    if 'financial_data' in st.session_state and st.session_state['financial_data']:
        financial_data = st.session_state['financial_data']
    else:
        financial_data = {}
    
    # File upload for JSON data
    uploaded_file = st.sidebar.file_uploader("Upload JSON financial data", type=["json"])
    if uploaded_file is not None:
        try:
            # Use pandas to read the JSON file
            financial_data_df = pd.read_json(uploaded_file)
            
            # Check if the JSON is a single record or multiple records
            if len(financial_data_df.index) == 1:
                # If single record (common case), convert to dictionary
                financial_data = financial_data_df.iloc[0].to_dict()
            else:
                # If multiple records, use the first one
                st.sidebar.info("Multiple records detected. Using the first record for analysis.")
                financial_data = financial_data_df.iloc[0].to_dict()
            
            # If the JSON was already a flat dictionary, pandas may have created a Series with one item
            if isinstance(financial_data, pd.Series):
                financial_data = financial_data.to_dict()
            
            # Store in session state
            st.session_state['financial_data'] = financial_data
            st.sidebar.success("Financial data loaded successfully")
            
        except ValueError as e:
            # Try alternative approach for flat JSON
            try:
                financial_data = pd.read_json(uploaded_file, orient='records').to_dict('records')[0]
                st.session_state['financial_data'] = financial_data
                st.sidebar.success("Financial data loaded successfully")
            except Exception as inner_e:
                st.sidebar.error(f"Error parsing JSON: {e}\nTried alternative approach: {inner_e}")
        except Exception as e:
            st.sidebar.error(f"Error loading JSON: {e}")
            st.sidebar.info("Please check that your JSON file is properly formatted.")
    
    # Display a warning if data doesn't seem to be properly loaded or analyzed
    if not financial_data or not isinstance(financial_data, dict) or len(financial_data) < 5:
        st.info("Welcome to - Preliminary Decision Support System.")
        st.info("Please proceed with file upload.")

    
    # Calculate ratios
    if financial_data and isinstance(financial_data, dict):
        st.header("Current Financial Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Financial Data")
            
            # Safely extract values from financial_data
            def safe_get(data, key, default=0):
                """Safely get a value from dictionary, handling different data types"""
                if not isinstance(data, dict):
                    return default
                return data.get(key, default)
            
            financial_df = pd.DataFrame({
                'Metric': [
                    'Total Current Assets', 'Inventory', 'Total Current Liabilities', 
                    'Long-term Debt', 'Total Liabilities', 'Total Equity',
                    'Operating Income', 'Interest Expense', 'EBITDA'
                ],
                'Value': [
                    format_currency(safe_get(financial_data, 'Total Current Assets')),
                    format_currency(safe_get(financial_data, 'Inventory')),
                    format_currency(safe_get(financial_data, 'Total Current Liabilities')),
                    format_currency(safe_get(financial_data, 'Long-term Debt')),
                    format_currency(safe_get(financial_data, 'Total Liabilities')),
                    format_currency(safe_get(financial_data, 'Total Equity')),
                    format_currency(safe_get(financial_data, 'Operating Income')),
                    format_currency(safe_get(financial_data, 'Interest Expense')),
                    format_currency(calculate_ebitda(financial_data))
                ]
            })
            st.table(financial_df)
            
        with col2:
            # Calculate and evaluate ratios
            ratios = calculate_financial_ratios(financial_data)
            evaluations = {ratio: evaluate_ratio(ratio, value) for ratio, value in ratios.items() if ratio != 'EBITDA'}
            
            st.subheader("Financial Ratios")
            ratio_data = {
                'Ratio': list(evaluations.keys()),
                'Value': [round(ratios[r], 2) if ratios[r] is not None else "N/A" for r in evaluations.keys()],
                'Status': [status for status, _ in evaluations.values()],
                'Color': [color for _, color in evaluations.values()]
            }
            ratio_df = pd.DataFrame(ratio_data)
            
            # Define a function for styling the Status column
            def highlight_status(val):
                if val == 'Green':
                    return 'background-color: rgba(0, 255, 0, 0.2)'
                elif val == 'Amber':
                    return 'background-color: rgba(255, 165, 0, 0.2)'
                elif val == 'Red':
                    return 'background-color: rgba(255, 0, 0, 0.2)'
                return ''
            
            # Apply styling
            styled_ratios = ratio_df.style.apply(lambda s: s.map(highlight_status), subset=['Status'])
            
            # Display the styled DataFrame
            st.dataframe(styled_ratios, use_container_width=True)
            
            # Make and display decision
            decision, color = make_decision(evaluations)
            
            # Create a DataFrame for the decision for consistent styling
            decision_display = pd.DataFrame({
                'Decision': [decision],
                'Status': [decision.split(' ')[0]]  # Extract status from decision text
            })
            # st.subheader("Loan Decision")
            # st.dataframe(decision_display, use_container_width=True)

            # # Display ratio table with colored status
            # for i, row in ratio_df.iterrows():
            #     color = row['Color']
            #     st.markdown(
            #         f"<div style='display:flex; justify-content:space-between; padding:5px; "
            #         f"background-color:rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2);'>"
            #         f"<span><b>{row['Ratio']}</b>:</span> <span>{row['Value']}</span> <span>{row['Status']}</span></div>",
            #         unsafe_allow_html=True
            #     )
            
            # Make and display decision
            decision, color = make_decision(evaluations)
            st.markdown("### Decision")
            st.markdown(
                f"<div style='padding:10px; text-align:center; font-weight:bold; "
                f"background-color:rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3);'>"
                f"{decision}</div>",
                unsafe_allow_html=True
            )
        
        # Loan projection section
        st.header("Loan Projection Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Loan Details")
            loan_principal = st.number_input("Loan Principal", min_value=100000.0, value=1000000.0, step=100000.0)
            loan_interest = st.number_input("Annual Interest Rate (%)", min_value=0.1, value=5.0, step=0.1)
            loan_years = st.number_input("Loan Term (Years)", min_value=1, value=5, step=1)
            payment_frequency = st.selectbox("Payment Frequency", ["monthly", "quarterly", "annually"])
            
            with st.expander("Business Growth Assumptions"):
                revenue_growth = st.slider("Annual Revenue Growth (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5) / 100
                expense_growth = st.slider("Annual Expense Growth (%)", min_value=0.0, max_value=20.0, value=4.0, step=0.5) / 100
            
            loan_details = {
                'principal': loan_principal,
                'interest_rate': loan_interest,
                'years': loan_years,
                'payment_frequency': payment_frequency,
                'revenue_growth': revenue_growth,
                'expense_growth': expense_growth
            }
            
            # # Export current data option
            # if financial_data and st.button("Export Current Data as JSON"):
            #     # Convert to pandas DataFrame first
            #     df = pd.DataFrame([financial_data])
            #     json_data = df.to_json(orient='records', indent=4)
            #     st.download_button(
            #         label="Download JSON",
            #         data=json_data,
            #         file_name="financial_data.json",
            #         mime="application/json"
            #     )
            
            if st.button("Calculate Projections"):
                # Calculate loan amortization schedule
                amortization = calculate_loan_amortization(
                    loan_principal, loan_interest, loan_years, payment_frequency
                )
                
                # Show amortization table
                st.write("Loan Amortization Schedule")
                amort_df = pd.DataFrame(amortization)
                st.dataframe(amort_df.style.format({
                    'Payment': '${:,.2f}',
                    'Principal': '${:,.2f}',
                    'Interest': '${:,.2f}',
                    'Remaining Balance': '${:,.2f}'
                }))
                
                # Project future financials
                projections = project_financials(financial_data, loan_details)
                
                with col2:
                    st.subheader("Projected Financial Ratios")
                    
                    # Prepare data for line charts
                    projection_data = []
                    for proj in projections:
                        period_data = {
                            'Period': proj['Year'],
                            'Leverage Ratio': proj['Ratios'].get('Leverage Ratio'),
                            'ICR': proj['Ratios'].get('ICR'),
                            'DSCR': proj['Ratios'].get('DSCR'),
                            'Current Ratio': proj['Ratios'].get('Current Ratio'),
                            'Quick Ratio': proj['Ratios'].get('Quick Ratio')
                        }
                        projection_data.append(period_data)
                    
                    proj_df = pd.DataFrame(projection_data)
                    
                    # Display projection table
                    st.write("Ratio Projections Over Time")
                    st.dataframe(proj_df)
                    
                    # Create consolidated view of current vs projected ratios
                    st.write("Current vs Projected Ratio Comparison")
                    
                    # Create a tab for each ratio
                    ratio_tabs = st.tabs(['Leverage Ratio', 'ICR', 'DSCR', 'Current Ratio', 'Quick Ratio'])
                    
                    for i, ratio in enumerate(['Leverage Ratio', 'ICR', 'DSCR', 'Current Ratio', 'Quick Ratio']):
                        with ratio_tabs[i]:
                            # Create bar chart comparing current with projected
                            fig = go.Figure()
                            
                            # Add bar for current ratio
                            current_value = ratios.get(ratio, None)
                            if current_value is not None:
                                fig.add_trace(go.Bar(
                                    x=['Current'],
                                    y=[current_value],
                                    name='Current',
                                    marker_color='blue'
                                ))
                            
                            # Add bars for projected values
                            fig.add_trace(go.Bar(
                                x=proj_df['Period'],
                                y=proj_df[ratio],
                                name='Projected',
                                marker_color='orange'
                            ))
                            
                            # Add threshold lines
                            if ratio == 'Leverage Ratio':
                                fig.add_shape(type="line", x0=-0.5, y0=4, x1=len(proj_df), y1=4,
                                             line=dict(color="green", width=2, dash="dash"))
                                fig.add_annotation(x=len(proj_df)/2, y=4, text="Threshold = 4",
                                                 showarrow=False, yshift=10)
                            elif ratio in ['ICR', 'DSCR', 'Quick Ratio']:
                                fig.add_shape(type="line", x0=-0.5, y0=1, x1=len(proj_df), y1=1,
                                             line=dict(color="green", width=2, dash="dash"))
                                fig.add_annotation(x=len(proj_df)/2, y=1, text="Threshold = 1",
                                                 showarrow=False, yshift=10)
                            elif ratio == 'Current Ratio':
                                fig.add_shape(type="line", x0=-0.5, y0=1, x1=len(proj_df), y1=1,
                                             line=dict(color="red", width=2, dash="dash"))
                                fig.add_annotation(x=len(proj_df)/2, y=1, text="Min = 1",
                                                 showarrow=False, yshift=10)
                                fig.add_shape(type="line", x0=-0.5, y0=1.5, x1=len(proj_df), y1=1.5,
                                             line=dict(color="orange", width=2, dash="dash"))
                                fig.add_annotation(x=len(proj_df)/2, y=1.5, text="Max = 1.5",
                                                 showarrow=False, yshift=10)
                            
                            fig.update_layout(
                                title=f"{ratio}: Current vs Projected",
                                xaxis_title="Period",
                                yaxis_title="Ratio Value",
                                barmode='group',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Also show the line chart for trend analysis
                            line_fig = px.line(proj_df, x='Period', y=ratio, title=f"{ratio} Trend Over Time")
                            
                            # Add threshold lines to line chart
                            if ratio == 'Leverage Ratio':
                                line_fig.add_hline(y=4, line_dash="dash", line_color="green", annotation_text="Threshold")
                            elif ratio in ['ICR', 'DSCR', 'Quick Ratio']:
                                line_fig.add_hline(y=1, line_dash="dash", line_color="green", annotation_text="Threshold")
                            elif ratio == 'Current Ratio':
                                line_fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Min")
                                line_fig.add_hline(y=1.5, line_dash="dash", line_color="orange", annotation_text="Max")
                            
                            st.plotly_chart(line_fig, use_container_width=True)
                
                # Show projected decisions over time
                st.subheader("Projected Loan Decisions Over Time")
                decision_data = []
                
                for i, proj in enumerate(projections):
                    evaluations = {ratio: evaluate_ratio(ratio, value) 
                                  for ratio, value in proj['Ratios'].items() 
                                  if ratio != 'EBITDA'}
                    decision, color = make_decision(evaluations)
                    
                    decision_data.append({
                        'Period': proj['Year'],
                        'Decision': decision,
                        'Color': color,
                        'Green Ratios': sum(1 for status, _ in evaluations.values() if status == "Green"),
                        'Amber Ratios': sum(1 for status, _ in evaluations.values() if status == "Amber"),
                        'Red Ratios': sum(1 for status, _ in evaluations.values() if status == "Red")
                    })
                
                decision_df = pd.DataFrame(decision_data)
                
                # Create side-by-side comparison view
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display decisions as a timeline
                    for i, row in decision_df.iterrows():
                        color = row['Color']
                        st.markdown(
                            f"<div style='display:flex; justify-content:space-between; padding:10px; margin:5px; "
                            f"background-color:rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3);'>"
                            f"<span><b>{row['Period']}</b></span> <span>{row['Decision']}</span> "
                            f"<span>Green: {row['Green Ratios']}, Amber: {row['Amber Ratios']}, Red: {row['Red Ratios']}</span></div>",
                            unsafe_allow_html=True
                        )
                
                with col2:
                    # Create visual summary of decision progression
                    fig = go.Figure()
                    
                    # Create a color map for decisions
                    color_map = {
                        'Approved (Green)': 'green',
                        'Review Required (Amber)': 'orange',
                        'Declined (Red)': 'red',
                        'Insufficient Data': 'gray'
                    }
                    
                    # Extract decision patterns
                    periods = decision_df['Period'].tolist()
                    green_counts = decision_df['Green Ratios'].tolist()
                    amber_counts = decision_df['Amber Ratios'].tolist()
                    red_counts = decision_df['Red Ratios'].tolist()
                    
                    # Create stacked bar chart of ratio counts
                    fig = go.Figure(data=[
                        go.Bar(name='Green Ratios', x=periods, y=green_counts, marker_color='green'),
                        go.Bar(name='Amber Ratios', x=periods, y=amber_counts, marker_color='orange'),
                        go.Bar(name='Red Ratios', x=periods, y=red_counts, marker_color='red')
                    ])
                    
                    fig.update_layout(
                        title='Ratio Status Distribution Over Time',
                        barmode='stack',
                        xaxis_title='Period',
                        yaxis_title='Count',
                        legend_title='Ratio Status'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Create a comprehensive comparison table of all ratios
                st.subheader("Comprehensive Ratio Comparison (Current vs Projected)")
                
                # Prepare data for the comparison table
                comparison_data = {
                    'Ratio': [],
                    'Current': [],
                    'Status': []
                }
                
                # Add projected periods to the columns
                for proj in projections:
                    comparison_data[proj['Year']] = []
                
                # Fill in the data
                for ratio in ['Leverage Ratio', 'ICR', 'DSCR', 'Current Ratio', 'Quick Ratio']:
                    comparison_data['Ratio'].append(ratio)
                    
                    # Current value and status
                    current_value = ratios.get(ratio)
                    status, _ = evaluate_ratio(ratio, current_value)
                    
                    comparison_data['Current'].append(f"{current_value:.2f}" if current_value is not None else "N/A")
                    comparison_data['Status'].append(status)
                    
                    # Projected values
                    for proj in projections:
                        projected_value = proj['Ratios'].get(ratio)
                        comparison_data[proj['Year']].append(
                            f"{projected_value:.2f}" if projected_value is not None else "N/A"
                        )
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create a styled table with color highlighting based on status
                def highlight_cells(val):
                    if 'Green' in val:
                        return 'background-color: rgba(0, 255, 0, 0.2)'
                    elif 'Amber' in val:
                        return 'background-color: rgba(255, 165, 0, 0.2)'
                    elif 'Red' in val:
                        return 'background-color: rgba(255, 0, 0, 0.2)'
                    return ''
                
                # Display the styled table
                st.dataframe(comparison_df.style.applymap(highlight_cells, subset=['Status']))

if __name__ == "__main__":
    main()