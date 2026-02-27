# Transformer Chat Model for LangChain

This project implements a custom **TransformerChatModel** that integrates
locally hosted transformer-based language models into the LangChain ecosystem.

The model is designed to work seamlessly with LangChain agents while allowing
full control over local inference, GPU usage, tool calling and streaming.

---

## Features

- **Custom TransformerChatModel**
  - Implements a chat model by extending LangChain’s `BaseChatModel`
  - Follows LangChain’s standard chat model interface

- **LangChain Agent Compatibility**
  - Compatible with LangChain agents
  - Designed to work with agent-based workflows

- **Local Model Loading with GPU Support**
  - Loads transformer-based models locally using Hugging Face Transformers
  - Supports GPU acceleration for efficient inference

- **Native Tool Calling Support**
  - Enables tool calling without relying on `HuggingFaceEndpoint`
  - Allows direct integration of tools in local inference setups

- **Streaming Support**
  - Designed with an intelligent response formatting strategy
  - Implements a structured streaming pipeline for reliable token emission
  - Supports streaming at both ChatModel and Agent layers
  - Compatible with all LangChain `stream_mode` options
  - Ensures stable throughput and consistent performance during streaming
  

---
## Usage
Install via pip:
```bash
pip install git+https://github.com/HinoEji/LangChain-Transformer-ChatModel.git
```
Create a quantized chat model.
You can pass the same parameters used when initializing Hugging Face's AutoModelForCausalLM or LangChain's ChatModel.
```python
from transformer_chat_model import TransformerChatModel
from transformers import BitsAndBytesConfig
import torch

pretrained_model_name_or_path = "Qwen/Qwen2.5-7B-Instruct-1M"
# Create your quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
chat_model = TransformerChatModel(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    quantization_config = bnb_config,
    max_new_tokens = 1024,
    device = "auto"
    )
```
Define tool and create agent:
```python
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# simple tool
@tool
def factorial(number: int) -> int:
  '''Use this tool to calculate the factorial of a number
    Args:
      number (int) --- A natural number that we want to calculate factorial of it.
    Return:
      Factorial of number of the result of number! .'''
  fact = 1
  for in in range(1,n+1):
    fact*=i
  return fact

agent = create_agent(
  model = chat_model,
  tools = [factorial],
  system_prompt = "You are a helpful assistant."
)

messages = [
  HumanMessage(content = "What is the result of 5! ?"
]

response = agent.invoke({'messages' : messages})
```

---
## Acknowledgements

This project is partially inspired by  
[Tool Calling with LangChain Open-Source Models](https://anshuls235.medium.com/%EF%B8%8F-tool-calling-with-langchain-open-source-models-run-it-locally-seamlessly-8d31ff4c7a76).

While building upon some of the ideas presented in the article, this implementation is independently developed with additional refinements, improvements, and adjustments to better support LangChain agents and tool workflows.

