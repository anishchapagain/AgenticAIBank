<!-- @format -->
# Welcome to The Agentic AI 🚀

- Build AI tools that solve real problems
- Collaborate and learn from experienced AI practitioners
- Stay up-to-date with the latest in AI development
- Turn your coding skills into actionable solutions

### Group Relative Policy Optimization (GRPO): TODO

```
from langchain_experimental.llms.ollama_function import OllamaFunctions
from langchain_core.messages import HumanMessage

model = OllamaFunctions(model='deepseek:8b', format='json')
model = model.bind_tools(
  tools=[FUNCTION: PARAMETERS + PROMPERTIES] 
)

model.invoke(PROMPT) # wil return the AI Message with FUNCTION from 'tools'.
```
