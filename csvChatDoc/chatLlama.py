from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM

llm = LocalLLM(
    api_base="http://localhost:11434/v1",
    model="llama3.2:latest"
    )
# models = ['gemma:2b', 'gemma2:latest', 'phi4:latest', 'llama3.2:latest']

pandas_ai = SmartDataframe('data/AccountData.csv', config={"llm": llm})

query = "What is the average balance of the accounts?"
query = "How many distinct branch are there and list me all those with their count and display a bar plot"
result = pandas_ai.chat(query)

print(result)