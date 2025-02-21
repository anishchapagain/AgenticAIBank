import streamlit as st
import pandas as pd
from datetime import datetime
import json
from typing import Optional
import requests

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    def generate(self, prompt: str, model: str = "mistral", system: Optional[str] = None) -> str:
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {str(e)}"

def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'ollama_client' not in st.session_state:
        st.session_state.ollama_client = OllamaClient()

def generate_system_prompt(df = None):
    if df is not None:
        columns_info = "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])
        return f"""You are a banking data analyst assistant. You help users analyze banking data and generate pandas queries.
        
Available DataFrame columns and types:
{columns_info}

For data analysis questions:
1. Generate appropriate pandas queries
2. Explain the analysis approach
3. Format queries in proper Python syntax

For general questions:
1. Provide helpful banking-related information
2. Stay within banking industry context
3. Be clear and professional"""
    else:
        return """You are a banking data analyst assistant. When no data is loaded, provide general banking information and guidance.
Help users understand banking concepts, regulations, and best practices."""

def execute_query(df, query: str) -> tuple[str, str]:
    try:
        # Create a safe local environment with pandas
        local_env = {'df': df, 'pd': pd}
        result = eval(query, {"__builtins__": {}}, local_env)
        return str(result), None
    except Exception as e:
        return None, f"Error executing query: {str(e)}"

def save_chat(role: str, content: str, query: Optional[str] = None, result: Optional[str] = None):
    timestamp = datetime.now()
    chat_entry = {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "role": role,
        "content": content,
        "query": query,
        "result": result
    }
    st.session_state.chat_history.append(chat_entry)

def main():
    st.set_page_config(page_title="Banking Analytics Assistant", layout="wide")
    
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        model = st.selectbox(
            "Select Ollama Model",
            ["gemma2", "llama3.2:latest", "codellama", "dolphin-phi"],
            help="Choose the model for analysis"
        )
        
        # CSV upload
        uploaded_file = st.file_uploader("Upload Banking Data", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success(f"Loaded CSV with {len(df)} rows")
            
            # Data preview
            with st.expander("Data Preview"):
                st.dataframe(df.head(3))
            
            # Column information
            with st.expander("Available Columns"):
                for col in df.columns:
                    st.code(f"{col}: {df[col].dtype}")
    
    # Main chat interface
    st.title("Banking Analytics Assistant")
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(f"**{chat['timestamp']}**")
            st.write(chat["content"])
            if chat.get("query"):
                with st.expander("Show Analysis Details"):
                    st.code(chat["query"], language="python")
                    if chat.get("result"):
                        st.write("Result:", chat["result"])
    
    # Chat input
    user_input = st.chat_input("Ask a question about banking data or general banking topics...")
    
    if user_input:
        # Show user message
        save_chat("user", user_input)
        
        # Generate system prompt based on context
        system_prompt = generate_system_prompt(
            st.session_state.df if hasattr(st.session_state, 'df') else None
        )
        
        # Prepare prompt for Ollama
        if hasattr(st.session_state, 'df'):
            prompt = f"""Generate a pandas query to answer this question: "{user_input}"

If this requires data analysis, provide your response in this JSON format:
{{
    "explanation": "Brief explanation of the analysis approach",
    "query": "The pandas query to execute",
    "interpretation": "How to interpret the results"
}}

If this is a general question, provide a normal response."""
        else:
            prompt = user_input
        
        # Get response from Ollama
        response = st.session_state.ollama_client.generate(
            prompt=prompt,
            model=model,
            system=system_prompt
        )
        
        # Process response
        try:
            # Try to parse as JSON (for data analysis responses)
            analysis = json.loads(response)
            if isinstance(analysis, dict) and "query" in analysis:
                # Execute the generated query if we have data
                if hasattr(st.session_state, 'df'):
                    result, error = execute_query(st.session_state.df, analysis["query"])
                    
                    response_text = f"""**Analysis Approach:**
{analysis['explanation']}

**Generated Query:**
```python
{analysis['query']}
```

**Result:** {result if result else error}

**Interpretation:**
{analysis['interpretation']}"""
                    save_chat("assistant", response_text, analysis["query"], result)
                else:
                    save_chat("assistant", "No data is currently loaded. Please upload a CSV file first.")
        except json.JSONDecodeError:
            # Not JSON, treat as regular response
            save_chat("assistant", response)

if __name__ == "__main__":
    main()