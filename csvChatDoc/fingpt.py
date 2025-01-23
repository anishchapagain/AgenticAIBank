from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "AI4Finance-Foundation/FinGPT-Llama2-13B"  # Or your chosen FinGPT model

# Download tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./fin_gpt_cache")  # Specify a local cache directory
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./fin_gpt_cache")

# Save the model and tokenizer locally (optional, but recommended for extra safety)
tokenizer.save_pretrained("./fin_gpt_local")
model.save_pretrained("./fin_gpt_local")

# Example usage (now offline)
inputs = tokenizer("What are the key risks for investing in emerging markets?", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))