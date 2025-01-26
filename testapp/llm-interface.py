from langchain_openai import ChatOpenAI
from typing import Any

def generate_llm_response(query: str, analytics_engine: Any) -> str:
    # Prepare context from analytics engine
    context = {
        "customer_segments": analytics_engine.get_customer_segments().to_dict(),
        "currency_transactions": analytics_engine.get_currency_transactions().to_dict(),
        "branch_performance": analytics_engine.get_branch_performance().to_dict()
    }
    
    # Prompt template with context
    prompt = f"""
    Banking Analytics Context: {context}
    User Query: {query}
    
    Provide a strategic, data-driven response focusing on:
    1. Direct insights related to the query
    2. Actionable recommendations
    3. Clear, professional analysis
    """
    
    # Initialize LLM (replace with your preferred LLM)
    llm = ChatOpenAI(temperature=0.3, model="gpt-4")
    
    try:
        response = llm.invoke(prompt).content
        return response
    except Exception as e:
        return f"Error generating response: {e}"
