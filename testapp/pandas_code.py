from openai import OpenAI
MODEL = "deepseek-r1:1.5b"
openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

prompt = """
You're expert in Data Analysis.

You know Python programming and pandas.

Provide me a code that reads the file 'AccountData.csv' from folder 'data'.
Load top 10 rows from CSV.
Print all the columns in ascending order.
Display the dataframe information.

Use 'df' as dataframe.
"""
response = openai.chat.completions.create(
 model=MODEL,
 messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)