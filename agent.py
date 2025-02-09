import asyncio  # Module for asynchronous programming, allowing non-blocking code execution.
import os  # Provides functionality to interact with the operating system.

from langchain_ollama import ChatOllama  # Importing ChatOllama to interact with the Large Language Model (LLM).
from browser_use import Agent  # Importing Agent to facilitate browser interactions and perform tasks.

# Define an asynchronous function to perform a browser task using the Agent.
async def run_search():
    # Create an Agent instance configured with a specific task and LLM settings.
    agent = Agent(
        task=(
            # Task description: Navigate to a specific URL and retrieve the page title.
            'Go to https://www.kumaribank.com.np, and list the title you see'
        ),
        llm=ChatOllama(
            # Specify the language model (LLM) to be used by the agent.
            # model='qwen2.5:32b-instruct-q4_K_M', 
            # model='qwen2.5:14b',  
            model='qwen2.5:latest', 
            num_ctx=128000,  # Context length for the model, indicating the amount of information it can process.
        ),
        max_actions_per_step=1,  # Limit the agent to one action per step for controlled operation.
        tool_call_in_content=False,  # Disables calling tools directly within the content of the task.
    )

    # Execute the agent to perform the defined task.
    await agent.run()

# Main entry point of the script.
if __name__ == '__main__':
    # Use asyncio to run the asynchronous function 'run_search'.
    asyncio.run(run_search())