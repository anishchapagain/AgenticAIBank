import ollama

result = ollama.generate(model='gemma2', prompt="Hello, can you confirm you're working?")

# Inspect and parse the result['response']
response_str = result['response']
print(response_str)