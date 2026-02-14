# Transformer Chat Model for LangChain

This project implements a custom **TransformerChatModel** that integrates
locally hosted transformer-based language models into the LangChain ecosystem.

The model is designed to work seamlessly with LangChain agents while allowing
full control over local inference, GPU usage, and tool calling.

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

---
## Acknowledgements

This project is partially inspired by  
[Tool Calling with LangChain Open-Source Models](https://anshuls235.medium.com/%EF%B8%8F-tool-calling-with-langchain-open-source-models-run-it-locally-seamlessly-8d31ff4c7a76).

While building upon some of the ideas presented in the article, this implementation is independently developed with additional refinements, improvements, and adjustments to better support LangChain agents and tool workflows.

