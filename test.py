import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Financial Ratio Analysis",
    page_icon="💰",
    layout="wide"
)

# Config for field mappings (handles different naming conventions)
FIELD_MAPPINGS_old = {
    "inventory": ["Inventory", "stocks-trading", "Stocks Trading", "stocks trading", "stock_trading", "Stock Trading", "Stocks-Trading"],
    "total_current_assets": ["Total Current Assets", "current assets", "current_assets", "total current assets"],
    "total_non_current_assets": ["Total Non-Current Assets", "non-current assets", "non current assets", "non_current_assets", "total non-current assets"],
    "total_assets": ["Total Assets", "total assets", "total_assets"],
    "total_current_liabilities": ["Total Current Liabilities", "current liabilities", "current_liabilities", "total current liabilities"],
    "total_non_current_liabilities": ["Total Non-Current Liabilities", "non-current liabilities", "non current liabilities", "non_current_liabilities", "total non-current liabilities"],
    "total_liabilities": ["Total Liabilities", "liabilities", "total liabilities"],
    "total_equity": ["Total Equity", "equity", "shareholders equity", "Shareholders' Equity", "total equity"],
    "total_liabilities_equity": ["Total Liabilities and Equity", "total liabilities and equity", "total liabilities & equity", "total liabilities and equity"],
    "long_term_debt": ["Long-term Debt", "Long term Debt", "long_term_debt", "LTD", "long term debt"],
    "interest_expense": ["Interest Expense", "interest expense", "interest_expenses"],
    "net_operating_profit": ["Net Operating Profit", "operating profit", "operating_profit", "EBIT", "net operating profit", "Net Income", "net income", "Net Operating Income", "net operating income"],
    "depreciation": ["Depreciation", "depreciation"],
    "amortization": ["Amortization", "amortization"],
    "gross_profit": ["Gross Profit", "gross profit"],
    "total_operating_expenses": ["Total Operating Expenses", "total operating expenses"],
}

# Benchmark standards for ratios
STANDARDS = {
    "EBITDA": {
        "positive": {"threshold": 0, "message": "Positive EBITDA indicates the company is operationally profitable", "color": "green"},
        "negative": {"threshold": float('-inf'), "message": "Negative EBITDA indicates operational losses", "color": "red"}
    },
    "Leverage Ratio": {
        "good": {"threshold": 4, "message": "Good leverage level (≤ 4)", "color": "green"},
        "moderate": {"threshold": float('-inf'), "message": "High leverage level (> 4)", "color": "red"}
    },
    "ICR": {
        "strong": {"threshold": 1.0, "message": "Sufficient ability to cover interest expenses (> 1)", "color": "green"},
        "weak": {"threshold": float('-inf'), "message": "Insufficient ability to cover interest expenses (< 1)", "color": "red"}
    },
    "DSCR": {
        "strong": {"threshold": 1.0, "message": "Sufficient ability to service debt (> 1)", "color": "green"},
        "weak": {"threshold": float('-inf'), "message": "Insufficient ability to service debt (< 1)", "color": "red"}
    },
    "CR": {
        "strong": {"threshold": 1.0, "message": "Good short-term liquidity (1-1.5)", "color": "green"},
        "high": {"threshold": 1.5, "message": "High short-term liquidity (> 1.5)", "color": "amber"},
        "weak": {"threshold": float('-inf'), "message": "Weak short-term liquidity (< 1)", "color": "red"}
    },
    "QR": {
        "strong": {"threshold": 1.0, "message": "Good quick liquidity (> 1)", "color": "green"},
        "weak": {"threshold": float('-inf'), "message": "Weak quick liquidity (< 1)", "color": "red"}
    }
}


# Load the config file
def load_config():
    config_path = "financial_mappings.json"  # Path JSON file
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)["field_mappings"]
    else:
        st.error("Config file not found")
        return None

# Initialize field mappings
FIELD_MAPPINGS = load_config()

def find_value(data, field_options):
    """Find value in data using various possible field names, with case-insensitive matching.
    Prioritizes non-zero values when multiple field matches are found."""
    # Create a case-insensitive and whitespace-normalized version of the data keys
    normalized_data = {}
    for key, value in data.items():
        normalized_key = key.strip().lower()
        normalized_data[normalized_key] = value
    
    found_value = 0
    
    # Check for each field option with normalization
    for option in field_options:
        # Check direct match first
        if option in data:
            # If we find a non-zero value, return it immediately
            if data[option] != 0:
                return data[option]
            # Otherwise, record that we found a value (even if zero)
            found_value = data[option]
        
        # Check normalized match
        normalized_option = option.strip().lower()
        if normalized_option in normalized_data:
            # If we find a non-zero value, return it immediately
            if normalized_data[normalized_option] != 0:
                return normalized_data[normalized_option]
            # Otherwise, record that we found a value (even if zero)
            found_value = normalized_data[normalized_option]
    
    return found_value  # Return the found value (or 0 if nothing found)

def safe_division(numerator, denominator):
    """Safely divide two numbers, handling zero division."""
    if denominator == 0:
        return float('nan')
    return numerator / denominator

# TODO: -ve EBITDA: do not proceed with calculations
# TODO: Total Assets is not equal to Total Liabilities + Equity: do not proceed with calculations

def calculate_ebitda(data):
    """Calculate EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization)."""
    operating_profit = find_value(data, FIELD_MAPPINGS["net_operating_profit"])
    interest = abs(find_value(data, FIELD_MAPPINGS["interest_expense"]))
    depreciation = abs(find_value(data, FIELD_MAPPINGS["depreciation"]))
    amortization = abs(find_value(data, FIELD_MAPPINGS["amortization"]))
    
    # For financial analysis, we make interest, depreciation, and amortization positive
    # as we're adding them back to the operating profit
    return operating_profit + interest + depreciation + amortization


def calculate_leverage_ratio(data):
    """Calculate Leverage Ratio (Total Liabilities / Total Equity)."""
    # total liabilities/total equity

    total_liabilities = find_value(data, FIELD_MAPPINGS["total_liabilities"])
    
    # If total liabilities is not found or is zero, calculate from current and non-current liabilities
    if total_liabilities == 0:
        total_current_liabilities = find_value(data, FIELD_MAPPINGS["total_current_liabilities"])
        if "Total Non-Current Liabilities" in data:
            total_non_current_liabilities = data["Total Non-Current Liabilities"]
        else:
            total_non_current_liabilities = 0
        total_liabilities = total_current_liabilities + total_non_current_liabilities        
    
    total_equity = find_value(data, FIELD_MAPPINGS["total_equity"])
    return safe_division(total_liabilities, total_equity)


def calculate_icr(data):
    """Calculate Interest Coverage Ratio (EBIT / Interest Expense)."""
    # Earning before interest tax depreciation (EBITDA or operating income)/Interest expense
    # Interest coverage ratio (ICR) is a measure of a company's ability to pay interest on its outstanding debt.
    # It is calculated by dividing the company's earnings before interest and taxes (EBIT) by the interest expense.
    # A higher ICR indicates a better ability to meet interest obligations.
    # Net operating profit - (interest expense + amortization + depreciation) /interest expense

    ebitda = calculate_ebitda(data)
    # operating_profit = find_value(data, FIELD_MAPPINGS["net_operating_profit"])
    interest_expense = abs(find_value(data, FIELD_MAPPINGS["interest_expense"]))
    # amortization = abs(find_value(data, FIELD_MAPPINGS["amortization"]))
    # depreciation = abs(find_value(data, FIELD_MAPPINGS["depreciation"]))
    # print(f"ICR Calculation: Operating Profit: {operating_profit}, Interest Expense: {interest_expense} , Amortization: {amortization}, Depreciation: {depreciation}")
    # # For financial analysis, we make interest, depreciation, and amortization positive
    # # as we're adding them back to the operating profit
    
    # new_operating_profit = operating_profit - (interest_expense + amortization + depreciation)
    # print(f"ICR Calculation: NEW Operating Profit: {new_operating_profit}, Interest Expense: {interest_expense} , Amortization: {amortization}, Depreciation: {depreciation}")
    # return safe_division(new_operating_profit, interest_expense)
    return safe_division(ebitda, interest_expense)


def calculate_dscr(data, principal_repayment=0):
    """Calculate Debt Service Coverage Ratio (EBITDA / (Interest Expense + Principal Repayment))."""
    # Earning before interest tax depreciation (EBITDA or operating income)/(Principal installment+ interest expense)

    ebitda = calculate_ebitda(data)
    interest_expense = abs(find_value(data, FIELD_MAPPINGS["interest_expense"]))
    debt_service = interest_expense + principal_repayment
    return safe_division(ebitda, debt_service)


def calculate_cr(data):
    """Calculate Current Ratio (Current Assets / Current Liabilities)."""
    # Current assets/current liabilities
    current_assets = find_value(data, FIELD_MAPPINGS["total_current_assets"])
    current_liabilities = find_value(data, FIELD_MAPPINGS["total_current_liabilities"])
    return safe_division(current_assets, current_liabilities)


def calculate_qr(data):
    """Calculate Quick Ratio ((Current Assets - Inventory) / Current Liabilities)."""
    # (Current assets-inventory)/current liabilities
    # Inventory is subtracted from current assets to get a more conservative measure of liquidity
    current_assets = find_value(data, FIELD_MAPPINGS["total_current_assets"])
    inventory = find_value(data, FIELD_MAPPINGS["inventory"])
    current_liabilities = find_value(data, FIELD_MAPPINGS["total_current_liabilities"])
    # print("Debug Info:")
    # print("Data:", data)
    # print(f"QR: Current Assets: {current_assets}, Inventory: {inventory}, Current Liabilities: {current_liabilities}")

    # Ensure inventory is not negative
    return safe_division(current_assets - inventory, current_liabilities)

def get_status(ratio_type, value):
    """Determine status of ratio based on standards."""
    if pd.isna(value):
        return "Invalid", "Unable to calculate ratio (division by zero or missing data)", "gray"
    
    # Special case for Current Ratio (CR) where we need to check if it's between 1 and 1.5
    if ratio_type == "CR":
        if value >= 1.0 and value <= 1.5:
            return "strong", STANDARDS[ratio_type]["strong"]["message"], STANDARDS[ratio_type]["strong"]["color"]
        elif value > 1.5:
            return "high", STANDARDS[ratio_type]["high"]["message"], STANDARDS[ratio_type]["high"]["color"]
        else:
            return "weak", STANDARDS[ratio_type]["weak"]["message"], STANDARDS[ratio_type]["weak"]["color"]
    
    # For all other ratios, check thresholds in order
    standards = STANDARDS[ratio_type]
    for category in standards:
        if value >= standards[category]["threshold"]:
            return category, standards[category]["message"], standards[category]["color"]
    
    # Default case (should not reach here if standards are properly defined)
    return "Unknown", "Unable to determine status", "gray"


def perform_stress_test(data, stress_factors):
    """
    Perform stress testing on financial data with various stress factors.
    
    Args:
        data: Financial data dictionary
        stress_factors: Dictionary of stress factors (percentage change) to apply
        
    Returns:
        Dictionary with original and stressed ratios
    """
    # Create a copy of data to avoid modifying the original
    stressed_data = data.copy()
    
    # Apply stress factors to the relevant fields
    for field, factor in stress_factors.items():
        # Get the original value using our mapping function
        original_value = None
        for mapping_key, field_options in FIELD_MAPPINGS.items():
            if any(f.lower() == field.lower() for f in field_options):
                original_value = find_value(data, field_options)
                break
        
        if original_value is not None:
            # Calculate the stressed value
            stressed_value = original_value * (1 + factor/100)
            
            # Update the field in stressed data
            # We need to find which key in the data corresponds to this field
            field_key = None
            for key in data.keys():
                if key.lower().strip() == field.lower().strip():
                    field_key = key
                    break
            
            if field_key:
                stressed_data[field_key] = stressed_value
    
    # Calculate ratios for original and stressed data
    original_ratios = {
        "EBITDA": calculate_ebitda(data),
        "Leverage Ratio": calculate_leverage_ratio(data),
        "ICR": calculate_icr(data),
        "DSCR": calculate_dscr(data),
        "CR": calculate_cr(data),
        "QR": calculate_qr(data)
    }
    
    stressed_ratios = {
        "EBITDA": calculate_ebitda(stressed_data),
        "Leverage Ratio": calculate_leverage_ratio(stressed_data),
        "ICR": calculate_icr(stressed_data),
        "DSCR": calculate_dscr(stressed_data),
        "CR": calculate_cr(stressed_data),
        "QR": calculate_qr(stressed_data)
    }
    
    return {
        "original": original_ratios,
        "stressed": stressed_ratios,
        "stress_factors": stress_factors
    }

    
def display_metric(label, value, status, message, color):
    """Display a metric with appropriate color and message."""
    if pd.isna(value):
        formatted_value = "N/A"
    else:
        formatted_value = f"{value:.2f}"
    
    if color == "green":
        st.success(f"**{label}:** {formatted_value} - **Status:** {status.capitalize()} - {message}")
    elif color == "amber":
        st.warning(f"**{label}:** {formatted_value} - **Status:** {status.capitalize()} - {message}")
    elif color == "red":
        st.error(f"**{label}:** {formatted_value} - **Status:** {status.capitalize()} - {message}")
    else:
        st.info(f"**{label}:** {formatted_value} - **Status:** {status.capitalize()} - {message}")


def create_comparison_chart(ratios_actual, ratios_projected):
    """Create a comparison chart for actual vs projected ratios."""
    ratio_names = list(ratios_actual.keys())
    actual_values = [ratios_actual[name]["value"] if not pd.isna(ratios_actual[name]["value"]) else 0 for name in ratio_names]
    projected_values = [ratios_projected[name]["value"] if not pd.isna(ratios_projected[name]["value"]) else 0 for name in ratio_names]
    
    fig = make_subplots(rows=1, cols=1)
    
    fig.add_trace(
        go.Bar(
            name="Actual",
            x=ratio_names,
            y=actual_values,
            marker_color='blue',
            opacity=0.7
        )
    )
    
    fig.add_trace(
        go.Bar(
            name="Projected",
            x=ratio_names,
            y=projected_values,
            marker_color='orange',
            opacity=0.7
        )
    )
    
    fig.update_layout(
        title="Actual vs Projected Financial Ratios",
        xaxis_title="Ratio",
        yaxis_title="Value",
        barmode='group',
        height=400
    )
    
    return fig


def main():
    st.title("Financial Ratio Analysis for Loan Decision Making")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Financial Data")
        uploaded_file = st.file_uploader("Upload JSON file", type=["json"])
        
        st.header("DSCR Configuration")
        principal_repayment = st.number_input("Principal Repayment (for Projected DSCR)", min_value=0.0, value=0.0, step=1000.0)

    if uploaded_file is not None:
        try:
            # Load the financial data & config
            # FIELD_MAPPINGS = load_config()
            financial_data = json.load(uploaded_file)
            
            # Extract actual and projected data
            if isinstance(financial_data, list) and len(financial_data) >= 2:
                # Assuming the structure matches the example provided
                actual_data = financial_data[0]["actual"]
                projected_data = financial_data[1]["projected"]
            elif "actual" in financial_data and "projected" in financial_data:
                actual_data = financial_data["actual"]
                projected_data = financial_data["projected"]
            else:
                st.error("Invalid JSON format. Expected structure with 'actual' and 'projected' data.")
                return
            
            # Calculate ratios for actual data
            actual_ratios = {
                "EBITDA": {"value": calculate_ebitda(actual_data)},
                "Leverage Ratio": {"value": calculate_leverage_ratio(actual_data)},
                "ICR": {"value": calculate_icr(actual_data)},
                "DSCR": {"value": calculate_dscr(actual_data)},
                "CR": {"value": calculate_cr(actual_data)},
                "QR": {"value": calculate_qr(actual_data)}
            }
            
            # Calculate ratios for projected data
            projected_ratios = {
                "EBITDA": {"value": calculate_ebitda(projected_data)},
                "Leverage Ratio": {"value": calculate_leverage_ratio(projected_data)},
                "ICR": {"value": calculate_icr(projected_data)},
                "DSCR": {"value": calculate_dscr(projected_data, principal_repayment)},
                "CR": {"value": calculate_cr(projected_data)},
                "QR": {"value": calculate_qr(projected_data)}
            }
            
            # Determine status for each ratio
            for ratio_name in actual_ratios:
                actual_ratios[ratio_name]["status"], actual_ratios[ratio_name]["message"], actual_ratios[ratio_name]["color"] = get_status(
                    ratio_name, actual_ratios[ratio_name]["value"]
                )
                projected_ratios[ratio_name]["status"], projected_ratios[ratio_name]["message"], projected_ratios[ratio_name]["color"] = get_status(
                    ratio_name, projected_ratios[ratio_name]["value"]
                )
            
            # Display the analysis
            st.header("Financial Ratio Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Actual Financial Ratios")
                for ratio_name, ratio_data in actual_ratios.items():
                    display_metric(
                        ratio_name, 
                        ratio_data["value"], 
                        ratio_data["status"], 
                        ratio_data["message"], 
                        ratio_data["color"]
                    )
            
            with col2:
                st.subheader("Projected Financial Ratios")
                for ratio_name, ratio_data in projected_ratios.items():
                    display_metric(
                        ratio_name, 
                        ratio_data["value"], 
                        ratio_data["status"], 
                        ratio_data["message"], 
                        ratio_data["color"]
                    )
            
            # Create and display comparison chart
            st.header("Ratio Comparison")
            comparison_chart = create_comparison_chart(actual_ratios, projected_ratios)
            st.plotly_chart(comparison_chart, use_container_width=True)
            
            # Detailed Analysis
            st.header("Detailed Analysis")
            
            # Create a spider/radar chart for comparison
            categories = list(actual_ratios.keys())
            actual_values = [actual_ratios[name]["value"] if not pd.isna(actual_ratios[name]["value"]) else 0 for name in categories]
            projected_values = [projected_ratios[name]["value"] if not pd.isna(projected_ratios[name]["value"]) else 0 for name in categories]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=actual_values,
                theta=categories,
                fill='toself',
                name='Actual'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=projected_values,
                theta=categories,
                fill='toself',
                name='Projected'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                    )),
                showlegend=True,
                title="Actual vs Projected Ratio Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Overall Recommendation
            st.header("Overall Recommendation")
            
            # Count how many ratios are in good standing for both actual and projected
            actual_good_count = sum(1 for ratio in actual_ratios.values() if ratio["color"] == "green")
            projected_good_count = sum(1 for ratio in projected_ratios.values() if ratio["color"] == "green")
            
            # Loan recommendation based on proportion of good ratios in projected data
            good_ratio_threshold = len(projected_ratios) / 2  # At least half of ratios should be good
            
            if projected_good_count >= good_ratio_threshold:
                st.success("✅ **Loan Recommendation: APPROVE**")
                st.write(f"The company shows positive financial health in {projected_good_count} out of {len(projected_ratios)} key metrics for projected data.")
                
                # Highlight areas of improvement if any
                areas_to_monitor = [name for name, data in projected_ratios.items() if data["color"] != "green"]
                if areas_to_monitor:
                    st.info(f"**Areas to Monitor:** {', '.join(areas_to_monitor)}")
            else:
                st.error("⚠️ **Loan Recommendation: CAUTION**")
                st.write(f"The company shows challenges in {len(projected_ratios) - projected_good_count} out of {len(projected_ratios)} key metrics for projected data.")
                
                # Highlight strong areas if any
                strong_areas = [name for name, data in projected_ratios.items() if data["color"] == "green"]
                if strong_areas:
                    st.info(f"**Strong Areas:** {', '.join(strong_areas)}")
            
            # Trend Analysis
            st.header("Trend Analysis")
            
            # Create a DataFrame to show the change from actual to projected
            trend_data = []
            for ratio_name in actual_ratios:
                actual_value = actual_ratios[ratio_name]["value"]
                projected_value = projected_ratios[ratio_name]["value"]
                
                if not pd.isna(actual_value) and not pd.isna(projected_value):
                    change = projected_value - actual_value
                    change_percent = safe_division(change, abs(actual_value)) * 100
                    
                    trend_data.append({
                        "Ratio": ratio_name,
                        "Actual": actual_value,
                        "Projected": projected_value,
                        "Change": change,
                        "Change %": change_percent
                    })
            
            if trend_data:
                trend_df = pd.DataFrame(trend_data)
                
                # Create a visualization for trend
                fig = px.bar(
                    trend_df,
                    x="Ratio",
                    y="Change %",
                    title="Projected Change in Financial Ratios (%)",
                    color="Change %",
                    color_continuous_scale=["red", "yellow", "green"],
                    labels={"Change %": "Change (%)", "Ratio": "Financial Ratio"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display the trend table
                st.subheader("Detailed Trend Analysis")
                
                # Format the DataFrame for display
                formatted_df = trend_df.copy()
                for col in ["Actual", "Projected", "Change"]:
                    formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.2f}")
                formatted_df["Change %"] = formatted_df["Change %"].map(lambda x: f"{x:.2f}%")
                
                st.dataframe(formatted_df, use_container_width=True)
                
                # Key insights
                st.subheader("Key Insights")
                
                improvements = trend_df[trend_df["Change"] > 0]["Ratio"].tolist()
                declines = trend_df[trend_df["Change"] < 0]["Ratio"].tolist()
                
                if improvements:
                    st.success(f"**Improvements Expected:** {', '.join(improvements)}")
                
                if declines:
                    st.warning(f"**Declining Metrics:** {', '.join(declines)}")
                
                # Best and worst performing metrics
                if not trend_df.empty:
                    best_metric = trend_df.loc[trend_df["Change %"].idxmax()]["Ratio"]
                    worst_metric = trend_df.loc[trend_df["Change %"].idxmin()]["Ratio"]
                    
                    st.info(f"**Best Performing Metric:** {best_metric}")
                    st.info(f"**Metric Requiring Most Attention:** {worst_metric}")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Please check the JSON format and try again.")
    else:
        # Display instructions when no file is uploaded
        st.info("Please upload a JSON file with financial data to begin analysis.")
        st.write("The expected JSON structure should contain 'actual' and 'projected' financial data.")
        

if __name__ == "__main__":
    main()